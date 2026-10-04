# Copyright 2024 <org> Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import pickle
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
import pandas as pd
import gc
from typing import Type, Dict
from verl.utils.py_functional import append_to_dict
import wandb
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.topk_buffer import BufferNode
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance, get_reverse_idx

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean, masked_sum


def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # kl_ctrl.update is skipped; it only matters for AdaptiveKLController
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {
        'critic/current_kl': current_kl,
    }

    return data, metrics


def compute_reward_rate_surrogate(
    exec_time_tensor: torch.Tensor,
    reward_tensor: torch.Tensor,
    reward_rate: float,
    valid_submission_tensor: torch.Tensor,
    penalty_coef: float = 1.0,
    penalize_invalid_time: bool = False,
    invalid_reward_rate: float = None,
    ) -> torch.Tensor:
    """
    Time-penalised reward: r' = r - penalty_coef * delta * rho_hat, where
    delta is the charged execution time and rho_hat the reward-rate estimate.

    Returns:
        torch.Tensor: The adjusted token-level scores.
    """
    time_penalty = penalty_coef * (exec_time_tensor * reward_rate)

    if reward_tensor.dim() == 2 and time_penalty.dim() == 1:
        raise ValueError("[Error] Dimension mismatch for reward rate penalty function")
    
    if not penalize_invalid_time:
        if valid_submission_tensor.dim() == 1 and time_penalty.dim() == 2:
            valid_submission_tensor = valid_submission_tensor.unsqueeze(-1)
        time_penalty = time_penalty * valid_submission_tensor
    elif invalid_reward_rate is not None:
        valid_mask = valid_submission_tensor
        if valid_mask.dim() == 1 and time_penalty.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        invalid_penalty = penalty_coef * (exec_time_tensor * invalid_reward_rate)
        time_penalty = time_penalty * valid_mask + invalid_penalty * (1.0 - valid_mask)

    adjusted_reward_tensor = reward_tensor - time_penalty
    return adjusted_reward_tensor


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    else:
        raise NotImplementedError
    
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        try:
            metrics[key] = np.mean(val)
        except:
            pass
    return metrics


def compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_total_response_length(batch):
    response_length = batch.batch['responses'].shape[-1]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    return response_mask.sum().item()


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def save_stats(batch, reward_fn_dict, reward_rate, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response_info = compute_response_info(batch)
    response_length = response_info['response_length']
    response_mask = response_info['response_mask'].bool()
    
    values = {
        "raw_reward": reward_fn_dict['raw_reward_tensor'].cpu().tolist(),
        "raw_exec_time": reward_fn_dict['raw_exec_time_tensor'].cpu().tolist(),
        "values": batch.batch['values'].cpu().tolist(),
        "returns": batch.batch['returns'].cpu().tolist(),
        "advantages": batch.batch['advantages'].cpu().tolist(),
        "reward": reward_fn_dict['reward_tensor'].cpu().tolist(),
        "exec_time": reward_fn_dict['exec_time_tensor'].cpu().tolist(),
        'response_mask': response_mask.cpu().tolist(),
        "response_length": response_length.cpu().tolist(),
        'reward_rate': reward_rate,
    }
    torch.save(values, path)
    

def restore_original_row_order(reward_fn_dict, global_idx):
    """
    Undo `_balance_batch`'s seqlen-based reorder for the row-level metadata
    fields only.
    """
    idx = global_idx.tolist() if torch.is_tensor(global_idx) else list(global_idx)
    if sorted(idx) != list(range(len(idx))):
        raise RuntimeError(
            "restore_original_row_order: global_idx is not a permutation of "
            f"0..{len(idx) - 1} -- row identity cannot be restored. Every "
            "reorder of the batch must be captured in the global_idx passed "
            "here, or self-improve continuation and the reward-rate "
            "accumulators pair rows across different samples.")
    n_rows = reward_fn_dict['raw_reward_tensor'].shape[0]
    if n_rows != len(idx):
        raise RuntimeError(
            f"restore_original_row_order: reward_fn_dict has {n_rows} rows "
            f"but global_idx has {len(idx)} entries -- the permutation does "
            "not describe this batch (rollout.n repeat or a batch-size "
            "change upstream?).")
    revert_indices = get_reverse_idx(idx)
    for key in (
        'raw_code_lst', 'error_lst', 'uid_lst', 'ml_algo_str_lst', 'num_feature_lst',
        'raw_exec_time_tensor', 'raw_reward_tensor', 'raw_grader_score_tensor',
    ):
        val = reward_fn_dict[key]
        if torch.is_tensor(val):
            reward_fn_dict[key] = val[revert_indices]
        else:
            reward_fn_dict[key] = [val[i] for i in revert_indices]
    # exec_time_tensor and valid_submission_tensor keep their balanced order
    # for the PPO update; original-order copies are added for the multi-step
    # reward-rate accumulators, which index by original dataset row.
    for key in ('exec_time_tensor', 'valid_submission_tensor'):
        val = reward_fn_dict.get(key)
        if val is not None:
            reward_fn_dict[f'{key}_orig_order'] = val[revert_indices]
    return reward_fn_dict





def trim_batch_padding(batch, align: int = 8):
    """
    Drop padding columns shared by EVERY row of the post-rollout batch.
    """
    tb = batch.batch
    prompt_len = tb['prompts'].size(1)
    resp_len = tb['responses'].size(1)
    am = tb['attention_mask']
    tokens_before = int(am.sum())

    prompt_cols = am[:, :prompt_len].any(dim=0)
    nz = torch.nonzero(prompt_cols)
    left = int(nz[0].item()) if len(nz) else prompt_len
    left = (left // align) * align

    resp_cols = am[:, prompt_len:].any(dim=0)
    nz = torch.nonzero(resp_cols)
    keep_resp = int(nz[-1].item()) + 1 if len(nz) else 0
    keep_resp = min(resp_len, ((keep_resp + align - 1) // align) * align)

    if left == 0 and keep_resp == resp_len:
        return 0, prompt_len + resp_len

    # Slice every 2D tensor in the batch by its width class (full sequence,
    # prompt-width, or response-width).
    assert prompt_len != resp_len, \
        'trim_batch_padding cannot distinguish prompt- from response-width ' \
        'tensors when max_prompt_length == max_response_length; set them apart'
    for key in list(tb.keys()):
        t = tb[key]
        if t.dim() < 2:
            continue
        width = t.size(1)
        if width == prompt_len + resp_len:
            tb[key] = t[:, left:prompt_len + keep_resp].contiguous()
        elif width == prompt_len:
            tb[key] = t[:, left:].contiguous()
        elif width == resp_len:
            tb[key] = t[:, :keep_resp].contiguous()

    assert int(tb['attention_mask'].sum()) == tokens_before, \
        'trim_batch_padding dropped real tokens -- refusing to continue'
    return left, (prompt_len - left) + keep_resp


def update_metrics(metrics, batch, reward_fn_dict, do_multi_step, k, batch_size):
    batch_size = len(reward_fn_dict['raw_reward_tensor'])
    table_df = pd.DataFrame({
        "ml_algo_str": reward_fn_dict["ml_algo_str_lst"],
        "num_features": reward_fn_dict["num_feature_lst"],
        "raw_exec_time": reward_fn_dict['raw_exec_time_tensor'].cpu().tolist(),
        "raw_reward": reward_fn_dict['raw_reward_tensor'].cpu().tolist(),
        "raw_grader_score": reward_fn_dict['raw_grader_score_tensor'].cpu().tolist(),
        "raw_code": reward_fn_dict['raw_code_lst'],
        "uid": reward_fn_dict['uid_lst'],
        "error": reward_fn_dict['error_lst'],
    })
    table = wandb.Table(dataframe=table_df)
    
    response_info = compute_response_info(batch)
    response_length = response_info['response_length']
    response_mask = response_info['response_mask'].bool()
    prompt_length = response_info['prompt_length']
    
    advantages = batch.batch['advantages']
    valid_adv = torch.masked_select(advantages, response_mask)
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    returns = batch.batch['returns']
    
    metrics.update({
        'train/raw_rewards/mean': torch.mean(reward_fn_dict['raw_reward_tensor']).detach().item(),
        'train/raw_rewards/max': torch.max(reward_fn_dict['raw_reward_tensor']).detach().item(),
        'train/raw_rewards/min': torch.min(reward_fn_dict['raw_reward_tensor']).detach().item(),
        'train/raw_rewards/std': torch.std(reward_fn_dict['raw_reward_tensor']).detach().item(),
        'train/raw_exec_time/mean': torch.mean(reward_fn_dict['raw_exec_time_tensor']).item(),
        'train/raw_exec_time/max': torch.max(reward_fn_dict['raw_exec_time_tensor']).item(),
        'train/raw_exec_time/min': torch.min(reward_fn_dict['raw_exec_time_tensor']).item(),
        'train/raw_exec_time/std': torch.std(reward_fn_dict['raw_exec_time_tensor']).item(),
        'train/raw_grader_score/mean': torch.mean(reward_fn_dict['raw_grader_score_tensor']).detach().item(),
        'train/raw_grader_score/max': torch.max(reward_fn_dict['raw_grader_score_tensor']).detach().item(),
        'train/raw_grader_score/min': torch.min(reward_fn_dict['raw_grader_score_tensor']).detach().item(),
        'train/raw_grader_score/std': torch.std(reward_fn_dict['raw_grader_score_tensor']).detach().item(),
        "train_2/adv/mean": torch.mean(valid_adv).detach().item(),
        "train_2/adv/std": torch.std(valid_adv).detach().item(),
        "train_2/returns/mean": torch.mean(returns).detach().item(),
        "train_2/returns/std": torch.std(returns).detach().item(),
        "train_2/score/mean": torch.mean(sequence_score).detach().item(),
        "train_2/score/std": torch.std(sequence_score).detach().item(),
        "train_2/reward/mean": torch.mean(sequence_reward).detach().item(),
        "train_2/reward/std": torch.std(sequence_reward).detach().item(),
        'train/valid_submission_perc': reward_fn_dict['valid_submission'] / batch_size,
        'train/crashed_error_perc': reward_fn_dict['crashed_error_count'] / batch_size,
        'train/unknown_error_perc': reward_fn_dict['unknown_error_count'] / batch_size,
        'train/timeout_error_perc': reward_fn_dict['timeout_error_count'] / batch_size,
        'token/response_length/mean': torch.mean(response_length).detach().item(),
        'token/response_length/std': torch.std(response_length).detach().item(),
        'token/prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'token/prompt_length/std': torch.std(prompt_length).detach().item(),
        'train_2/num_feature/mean': np.mean(reward_fn_dict["num_feature_lst"]),
        'train_2/num_feature/std': np.std(reward_fn_dict["num_feature_lst"]),
        'table': table,
    })
    metrics.update({"b_score": reward_fn_dict["b_score"]})
        
    return metrics


def compute_log_reward_rate_surrogate(batch, avg_sum_reward, avg_sum_time):
    batch.batch['token_level_scores'] = \
        batch.batch['token_level_scores'] / avg_sum_reward - \
        batch.batch['exec_time_tensor'] / avg_sum_time
    return batch

    
class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        start_global_step=1,
        max_reward_rate=0.0,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        
        self.start_global_step = start_global_step
        self.max_reward_rate = max_reward_rate
        # Running max of the per-step penalty reward rate, used as the
        # invalid-sample rate when actor.invalid_penalty_use_max_rate is on
        # (see compute_reward_rate_surrogate). Persisted in niw_state.pt.
        self.invalid_penalty_rate = 0.0
        # Online NIW log-time reward-rate estimator (set up in fit() when
        # actor.use_niw_reward_rate is on); _niw_last caches its latest
        # {mean, lo, hi} for logging.
        self.niw_estimator = None
        self._niw_last = {}

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
                
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, \
                    f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                    
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()


    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, SelfImproveDataset
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
        )
        # Prompts are cycled indefinitely; there are no epochs.
        self.train_dataloader = infinite_loader(DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        ))
        
        if self.config.trainer.multi_step:
            self.self_improve_dataset = SelfImproveDataset(
                parquet_files=self.config.data.train_files,
                batch_size=self.config.data.train_batch_size,
                tokenizer=self.tokenizer,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='error',
            )
            self.self_improve_dataloader = infinite_loader(DataLoader(
                dataset=self.self_improve_dataset,
                batch_size=self.config.data.train_batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn,
            ))
        print(f'Total training steps: {self.config.trainer.total_training_steps}')


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        # All roles share the single global pool defined in main_ppo.py.
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)

            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout',
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], 
                config=self.config.critic,
            )
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref',
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, 
                ray_cls_with_init=worker_dict_cls,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

 
    def _build_niw_estimator(self):
        """Construct a fresh NIW estimator from config."""
        from verl.utils.reward_rate_niw import OnlineNIWRewardRate
        return OnlineNIWRewardRate(
            forgetting_factor=self.config.actor_rollout_ref.actor.get('niw_forgetting_factor', 0.3),
            log_time=True,
            pool_samples=False,
            nu_cap=self.config.actor_rollout_ref.actor.get('niw_nu_cap', None),
        )

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
        
        global_step_path = os.path.join(actor_local_path, 'global_step.txt')
        with open(global_step_path, 'w') as f:
            f.write(str(self.global_steps))

        
        self.reward_fn.save_reward_buffer(os.path.join(actor_local_path, 'reward_buffer.pkl'))
        
        if getattr(self, '_si_prev_code_snapshot', None) is not None:
            si_state_path = os.path.join(actor_local_path, 'self_improve_state.pkl')
            with open(si_state_path, 'wb') as f:
                pickle.dump({
                    'prev_code_lst': self._si_prev_code_snapshot,
                    'prev_error_lst': self._si_prev_error_snapshot,
                    'global_step': self.global_steps,
                }, f)

        # Top-k buffer: saved as a start-of-step snapshot, like the
        # self-improve lists above, since resume re-runs the step recorded
        # in global_step.txt.
        if getattr(self, '_topk_buffer_snapshot', None) is not None:
            with open(os.path.join(actor_local_path, 'topk_buffer.pkl'), 'wb') as f:
                pickle.dump(self._topk_buffer_snapshot, f)
        
        max_reward_rate_path = os.path.join(actor_local_path, 'max_reward_rate.txt')
        with open(max_reward_rate_path, 'w') as f:
            f.write(str(self.max_reward_rate))

        # Persist the online NIW estimator and the per-slot accumulators it
        # fits on, so a resumed job continues the same posterior.
        if self.niw_estimator is not None:
            niw_state_path = os.path.join(actor_local_path, 'niw_state.pt')
            torch.save({
                'estimator': self.niw_estimator.state_dict(),
                'accum_reward': self.accum_reward,
                'accum_time': self.accum_time,
                'ever_valid': self.ever_valid,
                'invalid_penalty_rate': self.invalid_penalty_rate,
            }, niw_state_path)


    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst,
            k_partitions=world_size,
            equal_size=True,
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        append_to_dict(metrics, global_balance_stats)
        # Returned so the caller can restore the original row order
        # (see restore_original_row_order()).
        return global_idx
    
    
    def compute_reward_rate(self):
        """
        compute reward rate and average time
        """
        total_rewards = 0.0
        total_delays = 0.0
        total_numels = 0
        n = self.config.actor_rollout_ref.rollout.n

        for k in self.multi_step_iterator:
            if k >= 2:
                batch_dict = next(self.self_improve_dataloader)
            else:
                batch_dict = next(self.train_dataloader)
                
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                            
            gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            
            # Build the exact batch structure expected by the reward function
            batch.non_tensor_batch['uid'] = np.array(
                [f"step{self.global_steps}_k={k}_{uuid.uuid4().hex}" for _ in range(len(batch.batch))], 
                dtype=object,
            )
            batch = batch.repeat(repeat_times=n, interleave=True)
            batch = batch.union(gen_output)
            
            # Calculate rewards and execution times (delays)
            reward_fn_dict = self.reward_fn(batch, self.global_steps, k)
            self.update_self_improve_dataset(reward_fn_dict["raw_code_lst"])

            # Accumulate the sums across the batch for \hat{r} and \hat{\delta}
            total_rewards += reward_fn_dict['reward_tensor'].sum().item()
            total_delays += reward_fn_dict['exec_time_tensor'].sum().item()
            total_numels += compute_total_response_length(batch)
        
        if self.config.actor_rollout_ref.actor.reward_rate_div_avg_time == "batch_avg":
            avg_time = total_delays / (len(batch.batch) * k)
        elif self.config.actor_rollout_ref.actor.reward_rate_div_avg_time == "token_avg":
            avg_time = total_delays / total_numels
        else:
            avg_time = 1.0
            
        return total_rewards, total_delays, avg_time
    
    
    def compute_and_record_old_reward_rate(self, reward_fn_dict, batch):
        raise NotImplementedError()

    
    def compute_reward_rate_if_needed(self):
        if self.config.trainer.use_old_reward_rate:
            
            raise NotImplementedError()

        if self.config.actor_rollout_ref.actor.use_reward_rate_penalty or \
            self.config.actor_rollout_ref.actor.log_reward_rate or \
            self.config.actor_rollout_ref.actor.use_log_reward_rate_penalty:
            
            total_rewards, total_delays, avg_time = self.compute_reward_rate()
            reward_rate = (total_rewards + self.total_rewards) / (total_delays + self.total_delays)

        else:
            total_rewards, total_delays, reward_rate, avg_time = \
                None, None, None, 1.0
        
        return total_rewards, total_delays, reward_rate, avg_time

    
    def update_self_improve_dataset(self, raw_code_lst, error_lst):
        if hasattr(self, "self_improve_dataloader"):
            self.self_improve_dataset.set_prev_code_lst(raw_code_lst)
            self.self_improve_dataset.set_prev_error_lst(error_lst)
    
    
    def optim_zero_grad(self):
        # Actor only: the critic's optimizer is stepped and zeroed in fit()
        # right after update_critic, freeing its gradients before update_actor.
        self.actor_rollout_wg.optim_zero_grad()


    def optim_step(self, metrics):
        # Actor only; see optim_zero_grad().
        actor_grad_norm = self.actor_rollout_wg.optim_step()
        data = {
            'actor/grad_norm': actor_grad_norm,
        }
        append_to_dict(metrics, data)
        return metrics
    
    
    @property
    def multi_step_iterator(self):
        """
        Return multi-step pbar if using multi-step, otherwise single-step.
        """
        if self.config.trainer.multi_step:
            return range(1, self.config.trainer.multi_step + 1)
        else:
            return range(1, 2)
        
    
    def _niw_reward_rate(self, reward_vec, time_vec, valid_submission_tensor):
        """
        Update the online NIW log-time estimator with this step's per-slot
        (reward, time) snapshot and return the scalar reward rate.
        """
        actor_cfg = self.config.actor_rollout_ref.actor
        valid_mask = None
        if actor_cfg.get('reward_rate_exclude_invalid', False):
            if valid_submission_tensor is not None:
                valid_mask = valid_submission_tensor.detach().cpu().numpy()

        est = self.niw_estimator.update(
            reward_vec.detach().cpu().numpy(),
            time_vec.detach().cpu().numpy(),
            valid_mask=valid_mask,
        )
        self._niw_last = est
        # rho-hat = 95th percentile of the posterior-predictive batch rates
        self.max_reward_rate = max(0.0, float(est['p95']))
        return self.max_reward_rate

    def _restore_rate_accumulators(self, ckpt):
        """
        Reload every reward-rate accumulator from a saved niw_state.pt.
        """
        self.accum_reward = ckpt['accum_reward']
        self.accum_time = ckpt['accum_time']
        self.ever_valid = ckpt.get('ever_valid', torch.zeros_like(self.accum_reward))
        self.invalid_penalty_rate = ckpt.get('invalid_penalty_rate', 0.0)


    def get_max_reward_rate(self, reward_fn_dict, do_multi_step):
        actor_cfg = self.config.actor_rollout_ref.actor
        use_expected_reward_rate = actor_cfg.get('expected_reward_rate', False)
        use_niw_reward_rate = self.niw_estimator is not None

        if do_multi_step:
            reward = reward_fn_dict['raw_reward_tensor']
            time = reward_fn_dict['exec_time_tensor_orig_order'].sum(dim=-1)
            self.accum_reward += reward
            self.accum_time += time

            step_valid = reward_fn_dict.get('valid_submission_tensor_orig_order')
            if step_valid is not None:
                sv = step_valid.detach().cpu().float()
                if sv.dim() > 1:
                    sv = sv.reshape(sv.shape[0], -1).amax(dim=-1)
                self.ever_valid = torch.maximum(self.ever_valid, sv)

            if use_niw_reward_rate:
                if actor_cfg.get('reward_rate_valid_thus_far', False):
                    fit_mask = self.ever_valid
                else:
                    fit_mask = step_valid
                return self._niw_reward_rate(
                    self.accum_reward, self.accum_time, fit_mask)

            if use_expected_reward_rate:
                self.max_reward_rate = (self.accum_reward.mean() / self.accum_time.mean()).item()
                return self.max_reward_rate

            reward_rate = self.accum_reward / self.accum_time
            max_reward_rate = torch.max(reward_rate).item()

            if max_reward_rate > self.max_reward_rate:
                self.max_reward_rate = max_reward_rate

            return self.max_reward_rate

        else:
            # Single-step: no cross-step accumulators, so the balanced order
            # is used directly.
            reward = reward_fn_dict['reward_tensor'].sum(dim=-1)
            time = reward_fn_dict['exec_time_tensor'].sum(dim=-1)

            if actor_cfg.get('reward_rate_valid_thus_far', False):
                raise ValueError(
                    'reward_rate_valid_thus_far requires trainer.multi_step: '
                    'without cross-step accumulators there is no "thus far" '
                    '-- the single-step fit sees only this step\'s rewards.')

            if use_niw_reward_rate:
                return self._niw_reward_rate(
                    reward, time, reward_fn_dict.get('valid_submission_tensor'))

            if use_expected_reward_rate:
                self.max_reward_rate = (reward.mean() / time.mean()).item()
                return self.max_reward_rate

            reward_rate = reward / time
            max_reward_rate = torch.max(reward_rate).item()

            if max_reward_rate > self.max_reward_rate:
                self.max_reward_rate = max_reward_rate

            return self.max_reward_rate


    def get_accum_time_reward_metrics(self):
        min_time_idx = torch.argmin(self.accum_time).item()
        return {
            'train/min_accum_time_sample/accum_time': self.accum_time[min_time_idx].item(),
            'train/min_accum_time_sample/accum_reward': self.accum_reward[min_time_idx].item(),
            'train/accum_time_mean': self.accum_time.mean().item(),
            'train/accum_reward_mean': self.accum_reward.mean().item(),
            # Fraction of slots that have produced a valid submission at least
            # once (the mask reward_rate_valid_thus_far fits on).
            'train/ever_valid_perc': self.ever_valid.mean().item(),
        }


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = self.start_global_step
        self.total_rewards = 0.0
        self.total_delays = 0.0
        self.total_numels = 0
        self.accum_reward = torch.zeros(self.config.data.train_batch_size)
        self.accum_time = torch.zeros(self.config.data.train_batch_size)
        self.ever_valid = torch.zeros(self.config.data.train_batch_size)
        self.niw_estimator = None
        self._niw_last = {}
        if self.config.actor_rollout_ref.actor.get('use_niw_reward_rate', False):
            self.niw_estimator = self._build_niw_estimator()

            niw_state_path = os.path.join(self.config.actor_rollout_ref.model.path, 'niw_state.pt')
            if os.path.exists(niw_state_path):
                try:
                    ckpt = torch.load(niw_state_path, map_location='cpu', weights_only=False)
                except TypeError:  # older torch without weights_only kwarg
                    ckpt = torch.load(niw_state_path, map_location='cpu')
                self.niw_estimator.load_state_dict(ckpt['estimator'])
                self._restore_rate_accumulators(ckpt)
                self._niw_last = self.niw_estimator.last or {}
                print(f'[NIW] Resumed estimator + accumulators from {niw_state_path}.')
            else:
                print(f'[NIW] No saved estimator state at {niw_state_path}; starting fresh.')

        reward_buffer_path = os.path.join(
            self.config.trainer.default_local_dir, 'actor', 'reward_buffer.pkl')
        if not os.path.exists(reward_buffer_path):
            bootstrap_path = os.path.join(
                self.config.actor_rollout_ref.model.path, 'reward_buffer.pkl')
            if os.path.exists(bootstrap_path):
                reward_buffer_path = bootstrap_path
        self.reward_fn.load_reward_buffer(reward_buffer_path)

        if hasattr(self, 'self_improve_dataloader'):
            si_state_path = os.path.join(
                self.config.trainer.default_local_dir, 'actor', 'self_improve_state.pkl')
            if not os.path.exists(si_state_path):
                si_state_path = os.path.join(
                    self.config.actor_rollout_ref.model.path, 'self_improve_state.pkl')
            if os.path.exists(si_state_path):
                with open(si_state_path, 'rb') as f:
                    si_state = pickle.load(f)
                if si_state.get('prev_code_lst') is not None:
                    self.self_improve_dataset.set_prev_code_lst(si_state['prev_code_lst'])
                    self.self_improve_dataset.set_prev_error_lst(si_state['prev_error_lst'])
                    print(f'[Self-improve] Resumed prev code/error lists '
                          f'(saved at global step {si_state.get("global_step")}) from {si_state_path}.')
            elif self.global_steps >= 2:
                print(f'[Self-improve] WARNING: resuming at global step {self.global_steps} '
                      f'but no self_improve_state.pkl found; the first batch will crash '
                      f'unless prev lists are set elsewhere.')

        # -- Top-k self-improve buffer (trainer.use_topk_buffer) -----------
        self.topk_buffer = None
        if self.config.trainer.get('use_topk_buffer', False):
            if not self.config.trainer.multi_step:
                raise ValueError(
                    'use_topk_buffer requires trainer.multi_step: the buffer '
                    'replaces the self-improve prompt chain, and without '
                    'multi_step there are no self-improve prompts to fill.')
            from verl.trainer.ppo.topk_buffer import TopKBuffer
            self.topk_buffer = TopKBuffer(
                use_reward_rate=bool(
                    self.config.actor_rollout_ref.actor.use_reward_rate_penalty),
                capacity=int(self.config.data.train_batch_size)
                         * int(self.config.trainer.get('topk_buffer_capacity_mult', 1) or 1))
            buf_path = os.path.join(
                self.config.trainer.default_local_dir, 'actor', 'topk_buffer.pkl')
            if not os.path.exists(buf_path):
                buf_path = os.path.join(
                    self.config.actor_rollout_ref.model.path, 'topk_buffer.pkl')
            if os.path.exists(buf_path):
                with open(buf_path, 'rb') as f:
                    self.topk_buffer.load_state_dict(pickle.load(f))
                print(f'[TopK buffer] Resumed from {buf_path}: '
                      f'{self.topk_buffer.stats()}')
            else:
                print('[TopK buffer] Starting empty (base prompts until the '
                      'first valid solution).')


        self.reward_fn.init_gpu_heartbeat_actor()

        while self.global_steps <= self.config.trainer.total_training_steps:
            metrics = {}
            print(f'GLOBAL_STEP={self.global_steps}')

            if hasattr(self, 'self_improve_dataloader'):
                self._si_prev_code_snapshot = self.self_improve_dataset.prev_code_lst
                self._si_prev_error_snapshot = self.self_improve_dataset.prev_error_lst
            if self.topk_buffer is not None:
                self._topk_buffer_snapshot = self.topk_buffer.state_dict()

            use_si_prompts = self.global_steps >= 2 and self.config.trainer.multi_step
            if self.topk_buffer is not None:
                use_si_prompts = (use_si_prompts and
                                  self.self_improve_dataset.prev_code_lst is not None)
            if use_si_prompts:
                batch_dict = next(self.self_improve_dataloader)
            else:
                batch_dict = next(self.train_dataloader)
                
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            
            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            # create uid's for each sample
            batch.non_tensor_batch['uid'] = np.array(
                [f"step{self.global_steps}_{uuid.uuid4().hex}" for _ in range(len(batch.batch))],
                dtype=object,
            )

            # repeat to align with repeated responses in rollout
            # the batch doesn't have "response" field yet
            batch = batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n,
                interleave=True,
            )

            # union the two, add "response" into batch
            batch = batch.union(gen_batch_output)

            # cut columns that are padding in every row
            trim_left, trim_width = trim_batch_padding(batch)
            metrics['batch/width_after_trim'] = trim_width
            metrics['batch/trim_left_cols'] = trim_left

            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch;
            # restore_original_row_order() undoes it for the reward metadata
            batch.non_tensor_batch['orig_row_idx'] = np.arange(len(batch), dtype=np.int64)
            global_idx = self._balance_batch(batch, metrics=metrics)
            stamped_order = batch.non_tensor_batch.pop('orig_row_idx')
            if not np.array_equal(stamped_order, global_idx.cpu().numpy()):
                raise RuntimeError(
                    "_balance_batch permuted the batch differently from the "
                    "global_idx it returned -- row identity downstream "
                    "(self-improve continuation, reward-rate accumulators, "
                    "wandb tables) would be scrambled. Fix _balance_batch / "
                    "global_idx before training.")

            # compute global_valid tokens, this is total number of valid tokens
            batch.meta_info['global_token_num'] = torch.sum(
                batch.batch['attention_mask'], dim=-1).tolist()

            # compute reference log_prob, this adds "ref_log_prob" key into batch
            if self.use_reference_policy:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

            # compute values, this adds "value" key into batch
            if self.use_critic:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

            # we combine with rule-based rm, adds "token_level_scores" key into batch
            reward_fn_dict = self.reward_fn(batch, self.global_steps, 1)    # k=1: self-improve rounds span global steps

            reward_fn_dict = restore_original_row_order(reward_fn_dict, global_idx)

            valid_subm_perc = reward_fn_dict['valid_submission'] / len(batch.batch)

            if self.config.trainer.use_submission_rate_reweighting:
                batch.batch['token_level_scores'] = reward_fn_dict['reward_tensor'] / (valid_subm_perc + 1e-8)
            else:
                batch.batch['token_level_scores'] = reward_fn_dict['reward_tensor']
                
            batch.batch['exec_time_tensor'] = reward_fn_dict['exec_time_tensor']
            batch.batch['penalty_time_tensor'] = reward_fn_dict.get(
                'penalty_time_tensor', reward_fn_dict['exec_time_tensor'])
            batch.batch['valid_submission_tensor'] = reward_fn_dict['valid_submission_tensor']

            if self.config.trainer.use_old_reward_rate:
                self.compute_and_record_old_reward_rate(reward_fn_dict, batch)
            
            if self.config.trainer.multi_step and self.topk_buffer is None:
                self.update_self_improve_dataset(
                    reward_fn_dict["raw_code_lst"], reward_fn_dict["error_lst"])

            reward_rate = self.get_max_reward_rate(
                reward_fn_dict=reward_fn_dict,
                do_multi_step=self.config.trainer.multi_step
            )
            use_max_invalid_rate = self.config.actor_rollout_ref.actor.get(
                'invalid_penalty_use_max_rate', False)
            if use_max_invalid_rate:
                self.invalid_penalty_rate = max(self.invalid_penalty_rate, reward_rate)

            if self.topk_buffer is not None:
                rr_on = bool(
                    self.config.actor_rollout_ref.actor.use_reward_rate_penalty)
                self.topk_buffer.set_rho(reward_rate if rr_on else 0.0)
                
                bonus = float(self.config.trainer.get('self_improve_bonus', 0.0) or 0.0)
                n_improving = 0
                if bonus > 0.0:
                    raw_r_bal = reward_fn_dict['raw_reward_tensor'][global_idx]
                    v_bal = batch.batch['valid_submission_tensor'].reshape(-1) > 0.5
                    best = self.topk_buffer.best_raw_reward()
                    improving = v_bal if best is None else (v_bal & (raw_r_bal > best))
                    rows = improving.nonzero(as_tuple=True)[0]
                    n_improving = int(rows.numel())
                    
                    if n_improving:
                        pos = batch.batch['exec_time_tensor'].argmax(dim=-1)
                        batch.batch['token_level_scores'][rows, pos[rows]] += bonus
                        
                metrics['train/topk_buffer/improving_rows'] = n_improving

                v_orig = reward_fn_dict['valid_submission_tensor_orig_order'].reshape(-1)
                t_orig = reward_fn_dict['exec_time_tensor_orig_order'].sum(dim=-1)
                r_orig = reward_fn_dict['raw_reward_tensor']
                for i in (v_orig > 0.5).nonzero(as_tuple=True)[0].tolist():
                    self.topk_buffer.add(BufferNode(
                        id=f'step{self.global_steps}_row{i}',
                        code=reward_fn_dict['raw_code_lst'][i],
                        error=reward_fn_dict['error_lst'][i],
                        raw_reward=float(r_orig[i]),
                        exec_time=float(t_orig[i]),
                        step=int(self.global_steps)))
                self.topk_buffer.trim()

                # Next step's prompts: top batch_size by the objective,
                # cycling from the top when fewer exist. An empty buffer
                # leaves the prev lists unset, keeping base prompts in use.
                if self.topk_buffer.n_nodes > 0:
                    sel = self.topk_buffer.select(self.config.data.train_batch_size)
                    self.update_self_improve_dataset(
                        [n.code for n in sel], [n.error for n in sel])
                    metrics['train/topk_buffer/distinct_selected'] = \
                        len({n.id for n in sel})
                for _k, _v in self.topk_buffer.stats().items():
                    metrics[f'train/topk_buffer/{_k}'] = _v
            if self.config.actor_rollout_ref.actor.use_reward_rate_penalty and \
                not self.config.actor_rollout_ref.actor.use_log_reward_rate_penalty:

                reward_tensor = compute_reward_rate_surrogate(
                    exec_time_tensor=batch.batch['penalty_time_tensor'],
                    reward_tensor=batch.batch['token_level_scores'],
                    reward_rate=reward_rate,
                    valid_submission_tensor=batch.batch['valid_submission_tensor'],
                    penalty_coef=self.config.actor_rollout_ref.actor.reward_rate_penalty_coeff,
                    penalize_invalid_time=self.config.actor_rollout_ref.actor.get(
                        'penalize_invalid_time', False),
                    invalid_reward_rate=(
                        self.invalid_penalty_rate if use_max_invalid_rate else None),
                )
                batch.batch['token_level_scores'] = reward_tensor



            if not self.config.actor_rollout_ref.actor.use_kl_loss:
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                append_to_dict(metrics, kl_metrics)
            else:
                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

            # compute advantages, executed on the driver process
            # adds "advantages" and "returns" key into batch
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
            )

            # update critic
            if self.use_critic:
                critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                append_to_dict(metrics, critic_output_metrics)
                critic_grad_norm = self.critic_wg.optim_step()
                self.critic_wg.optim_zero_grad()
                append_to_dict(metrics, {'critic/grad_norm': critic_grad_norm})

            # update actor if after critic_warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # Divisor of pg_loss when actor.divide_avg_time is set; fixed at 1 (no rescaling).
                batch.meta_info["avg_time"] = 1
                actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                append_to_dict(metrics, actor_output_metrics)
            
            metrics = update_metrics(
                metrics=metrics, 
                batch=batch,
                reward_fn_dict=reward_fn_dict, 
                do_multi_step=self.config.trainer.multi_step,
                k=1,
                batch_size=self.config.data.train_batch_size,
            )
            
            metrics.update({
                'train/reward_rate': reward_rate,
            })
            if use_max_invalid_rate:
                metrics['train/invalid_penalty_rate'] = self.invalid_penalty_rate
            if self.niw_estimator is not None and self._niw_last:
                metrics.update({
                    'train/niw/reward_rate_mean': self._niw_last['mean'],
                    'train/niw/reward_rate_p2_5': self._niw_last['lo'],
                    'train/niw/reward_rate_p97_5': self._niw_last['hi'],
                    'train/niw/reward_rate_p25': self._niw_last['p25'],
                    'train/niw/reward_rate_p75': self._niw_last['p75'],
                    'train/niw/reward_rate_p95': self._niw_last['p95'],
                    'train/niw/reward_rate_max': self._niw_last['max'],
                })
            if self.config.trainer.multi_step:
                metrics.update(self.get_accum_time_reward_metrics())
            metrics = self.optim_step(metrics)
            self.optim_zero_grad()
            
            if self.config.trainer.save_freq > 0 and \
                    self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

            # collect metrics
            metrics = reduce_metrics(metrics)
            logger.log(data=metrics, step=self.global_steps)

            self.actor_rollout_wg.log_memory_snapshot(f'end of step {self.global_steps}')
            self.critic_wg.log_memory_snapshot(f'end of step {self.global_steps}')

            self.global_steps += 1
