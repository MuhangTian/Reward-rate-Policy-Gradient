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

import json
import os
import pickle
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
import pandas as pd
import gc
from typing import Type, Dict
import copy
from verl.utils.py_functional import append_to_dict
import wandb
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance, get_reverse_idx

WorkerType = Type[Worker]

VAL_TASKS = ['random-acts-of-pizza',
             'tweet-sentiment-extraction',
             'spooky-author-identification',
             'learning-agency-lab-automated-essay-scoring-2',
             'leaf-classification',
             'aerial-cactus-identification',
             'detecting-insults-in-social-commentary',
             'the-icml-2013-whale-challenge-right-whale-redux',
             'spaceship-titanic']

TRAIN_TASKS = ['tweet-sentiment-extraction',
               'spooky-author-identification',
               'learning-agency-lab-automated-essay-scoring-2',
               'leaf-classification',
               'aerial-cactus-identification',
               'detecting-insults-in-social-commentary',
               'the-icml-2013-whale-challenge-right-whale-redux']


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
    # batch_size = data.batch.batch_size[0]
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
    # Disabled for now, only used for AdaptiveKLController
    # kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
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
    anchor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
    """
    Reward-rate shaping: r' = r - penalty_coef * rho * t.

    By default the time term is masked out for samples without a valid
    submission, so failing quickly is not rewarded.

    Args:
        exec_time_tensor (torch.Tensor): Execution times for the batch.
        reward_tensor (torch.Tensor): The raw token-level scores.
        reward_rate (float): The reward rate rho used for the penalty.
        valid_submission_tensor (torch.Tensor): Per-sample (bsz,) mask, 1.0
            for a valid submission and 0.0 otherwise.
        penalty_coef (float): Scale of the penalty.
        penalize_invalid_time (bool): Also charge invalid samples for time.
        invalid_reward_rate (float): When set (requires penalize_invalid_time),
            invalid samples are charged this rate instead of reward_rate.
        anchor_mask (torch.Tensor): Optional per-sample (bsz,) gate in the
            same row order as reward_tensor. 1.0 applies the time terms;
            0.0 zeroes them so the reward passes through unshaped.

    Returns:
        torch.Tensor: The adjusted token-level scores.
    """
    time_penalty = penalty_coef * (exec_time_tensor * reward_rate)

    if reward_tensor.dim() == 2 and time_penalty.dim() == 1:
        raise ValueError("[Error] Dimension mismatch for reward rate penalty function")

    # penalize_invalid_time lifts the mask; invalid samples are then charged
    # for their (timeout) execution time as well.
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

    # Per-sample gate, applied last so it covers both valid and invalid terms.
    if anchor_mask is not None:
        gate = anchor_mask.detach().to(time_penalty.device, time_penalty.dtype)
        if gate.dim() == 1 and time_penalty.dim() == 2:
            gate = gate.unsqueeze(-1)
        time_penalty = time_penalty * gate

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
    # Identify the response portion of the mask
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
    Undo `_balance_batch`'s reorder for the row-level metadata fields.

    `global_idx[k]` is the original row placed at balanced position k. Tensors
    that feed the PPO update stay in balanced order; metadata used by the
    self-improve dataset and logging is restored to original row order, and
    original-order copies of exec_time/valid_submission are added under
    `*_orig_order` keys.
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
    # Original-order copies for the per-slot reward-rate accumulators.
    for key in ('exec_time_tensor', 'valid_submission_tensor'):
        val = reward_fn_dict.get(key)
        if val is not None:
            reward_fn_dict[f'{key}_orig_order'] = val[revert_indices]
    return reward_fn_dict


# Keys kept in balanced order because the PPO update reads them positionally.
_BALANCED_ORDER_KEYS = (
    'reward_tensor', 'exec_time_tensor', 'penalty_time_tensor',
    'valid_submission_tensor',
)


def align_prescored_row_order(reward_fn_dict, global_idx):
    """
    Bring a dict scored before `_balance_batch` (e.g. by rejection sampling)
    into the same row-order convention `restore_original_row_order` produces.

    Only the tensors in `_BALANCED_ORDER_KEYS` are permuted into balanced order
    (t_balanced = t_original[global_idx]); everything else stays in original
    order.
    """
    idx = global_idx.tolist() if torch.is_tensor(global_idx) else list(global_idx)
    if sorted(idx) != list(range(len(idx))):
        raise RuntimeError(
            "align_prescored_row_order: global_idx is not a permutation of "
            f"0..{len(idx) - 1} -- row identity cannot be aligned.")
    n_rows = reward_fn_dict['raw_reward_tensor'].shape[0]
    if n_rows != len(idx):
        raise RuntimeError(
            f"align_prescored_row_order: reward_fn_dict has {n_rows} rows but "
            f"global_idx has {len(idx)} entries -- the permutation does not "
            "describe this batch.")
    fwd = torch.as_tensor(idx, dtype=torch.long)
    for key in ('exec_time_tensor', 'valid_submission_tensor'):
        val = reward_fn_dict.get(key)
        if val is not None:
            reward_fn_dict[f'{key}_orig_order'] = val.clone()
    for key in _BALANCED_ORDER_KEYS:
        val = reward_fn_dict.get(key)
        if val is not None:
            reward_fn_dict[key] = val[fwd]
    return reward_fn_dict


def trim_batch_padding(batch, align: int = 8):
    """
    Drop padding columns shared by every row of the post-rollout batch.

    Removes left columns where no row has a prompt token (prompts are
    left-padded) and right columns where no row has a response token
    (responses are right-padded). Widths stay aligned to `align` columns.

    Returns (left_cols_removed, new_total_width).
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

    # Slice every 2D tensor in the batch according to its width
    # (full, prompt-only, or response-only).
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
        # invalid-sample rate when actor.invalid_penalty_use_max_rate is on.
        self.invalid_penalty_rate = 0.0
        # Online NIW reward-rate estimator (set up in fit() when
        # actor.use_niw_reward_rate is on); _niw_last caches its latest
        # posterior summary for logging.
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
        # Cycle over the training prompts indefinitely.
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

        # NOTE: since we only have one key, which is global_pool_id in main_ppo.py, this will only have one entry
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            # NOTE: since the mapping are all to global_pool_id (see main_ppo.py), they are on the same thing
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            
            # NOTE: role_worker_mapping maps to actual class (Ray actor)
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

 
    # ------------------------------------------------------------------
    # Staged objective switch (trainer.rr_switch_valid_threshold).
    #
    # The run trains with the vanilla objective until batch validity clears
    # the threshold, then switches to the reward-rate objective for the rest
    # of the run. The switch toggles both the reward-rate surrogate in the fit
    # loop (self.rr_active) and reward.invalid_zero_reward_full_time, which
    # selects how invalid samples are scored by the reward manager.
    # ------------------------------------------------------------------
    def _build_niw_estimator(self):
        """Construct a fresh NIW estimator from config."""
        from verl.utils.reward_rate_niw import OnlineNIWRewardRate
        return OnlineNIWRewardRate(
            forgetting_factor=self.config.actor_rollout_ref.actor.get('niw_forgetting_factor', 0.3),
            log_time=True,
            pool_samples=False,
            # Optional cap on the posterior degrees of freedom (None = no cap).
            nu_cap=self.config.actor_rollout_ref.actor.get('niw_nu_cap', None),
        )

    def _rr_switch_enabled(self):
        return self.config.trainer.get('rr_switch_valid_threshold', None) is not None

    def _reset_reward_rate_state(self):
        """
        Reset all reward-rate estimation state at the switch, so rho is
        estimated only from reward-rate-phase batches. Mirrors the
        initialization in fit().
        """
        bsz = self.config.data.train_batch_size
        self.accum_reward = torch.zeros(bsz)
        self.accum_time = torch.zeros(bsz)
        self.ever_valid = torch.zeros(bsz)
        self.accum_valid_reward = 0.0
        self.accum_valid_time = 0.0
        self.n_valid_samples = 0
        self.total_rewards = 0.0
        self.total_delays = 0.0
        self.total_numels = 0
        self.max_reward_rate = 0.0
        self.invalid_penalty_rate = 0.0
        self._niw_last = {}
        if self.niw_estimator is not None:
            self.niw_estimator = self._build_niw_estimator()
        print('[RR-SWITCH] Reward-rate state reset: accumulators, NIW '
              'posterior, max_reward_rate and invalid_penalty_rate all start '
              'from phase 2.')

    def _save_phase1_snapshot(self):
        """
        Save a full checkpoint (actor, critic, optimizers, step, prompt chain)
        to trainer.rr_switch_snapshot_dir at the switch, once.
        """
        dst = self.config.trainer.get('rr_switch_snapshot_dir', None)
        if not dst:
            return
        dst = os.path.join(dst, f'{self.config.trainer.experiment_name}_step{self.global_steps}')
        if os.path.exists(dst):
            print(f'[RR-SWITCH] Phase-1 snapshot already exists at {dst}; keeping it.')
            return
        # Write through the normal checkpoint path into a temp dir, then
        # rename it into place.
        tmp = dst + '.tmp'
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        real_dir = self.config.trainer.default_local_dir
        try:
            with open_dict(self.config.trainer):
                self.config.trainer.default_local_dir = tmp
            self._save_checkpoint()
        finally:
            with open_dict(self.config.trainer):
                self.config.trainer.default_local_dir = real_dir
        # Drop the vanilla-phase reward-rate state from the snapshot.
        for stale in ('max_reward_rate.txt', 'niw_state.pt'):
            p = os.path.join(tmp, 'actor', stale)
            if os.path.exists(p):
                os.remove(p)
        os.rename(tmp, dst)
        print(f'[RR-SWITCH] Phase-1 snapshot (actor + CRITIC + optimizers + '
              f'chain, global step {self.global_steps}) saved to {dst}.')

    def _sync_rr_reward_config(self):
        """Point the reward manager at the phase's invalid-scoring scheme."""
        if not self._rr_switch_enabled():
            return
        with open_dict(self.reward_fn.reward_config):
            self.reward_fn.reward_config.invalid_zero_reward_full_time = bool(
                self.rr_active)

    def _init_rr_switch_state(self):
        """
        Set up the switch state, restoring it from rr_switch_state.json in
        the checkpoint dir (or model.path) when resuming.
        """
        self.rr_active = False
        self.rr_switch_step = None
        self._rr_consecutive = 0
        if not self._rr_switch_enabled():
            # Not staged: rr_active follows the config from step 1.
            self.rr_active = bool(
                self.config.actor_rollout_ref.actor.use_reward_rate_penalty)
            return
        if not self.config.actor_rollout_ref.actor.use_reward_rate_penalty:
            raise ValueError(
                'trainer.rr_switch_valid_threshold is set but '
                'actor.use_reward_rate_penalty is False: there is no '
                'reward-rate objective to switch into, so the run would '
                'stay vanilla forever. Set the flag or drop the threshold.')
        thr = float(self.config.trainer.rr_switch_valid_threshold)
        if not 0.0 < thr <= 1.0:
            raise ValueError(
                f'trainer.rr_switch_valid_threshold must be in (0, 1], got {thr}.')
        for d in (os.path.join(self.config.trainer.default_local_dir, 'actor'),
                  self.config.actor_rollout_ref.model.path):
            p = os.path.join(d, 'rr_switch_state.json')
            if os.path.exists(p):
                with open(p) as f:
                    st = json.load(f)
                self.rr_active = bool(st.get('rr_active', False))
                self.rr_switch_step = st.get('rr_switch_step')
                self._rr_consecutive = int(st.get('consecutive', 0))
                print(f'[RR-SWITCH] Restored latch from {p}: '
                      f'active={self.rr_active} switch_step={self.rr_switch_step}.')
                break
        else:
            print(f'[RR-SWITCH] No saved latch; starting VANILLA, switching to '
                  f'reward-rate after validity > {thr:.3f} for '
                  f'{int(self.config.trainer.get("rr_switch_valid_consecutive", 1))} '
                  f'consecutive step(s).')
        self._sync_rr_reward_config()

    def _maybe_switch_to_rr(self, valid_perc, metrics):
        """
        Update the switch on this step's validity and log its state. The
        reward-rate objective takes effect from the next step.
        """
        if not self._rr_switch_enabled():
            return
        thr = float(self.config.trainer.rr_switch_valid_threshold)
        need = int(self.config.trainer.get('rr_switch_valid_consecutive', 1))
        if not self.rr_active:
            self._rr_consecutive = self._rr_consecutive + 1 if valid_perc > thr else 0
            if self._rr_consecutive >= need:
                # Snapshot before switching, so the snapshot records the
                # vanilla phase.
                self._save_phase1_snapshot()
                self.rr_active = True
                self.rr_switch_step = self.global_steps + 1
                self._sync_rr_reward_config()
                self._reset_reward_rate_state()
                print(f'[RR-SWITCH] Batch validity {valid_perc:.4f} > {thr:.3f} '
                      f'for {need} consecutive step(s) at global step '
                      f'{self.global_steps}. Reward-rate objective ACTIVE from '
                      f'step {self.rr_switch_step}.')
        metrics['train/rr_switch/active'] = float(self.rr_active)
        metrics['train/rr_switch/consecutive'] = float(self._rr_consecutive)
        if self.rr_switch_step is not None:
            metrics['train/rr_switch/step'] = float(self.rr_switch_step)

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

        # Staged objective switch state, so a resume continues in the same phase.
        if self._rr_switch_enabled():
            with open(os.path.join(actor_local_path, 'rr_switch_state.json'), 'w') as f:
                json.dump({'rr_active': bool(self.rr_active),
                           'rr_switch_step': self.rr_switch_step,
                           'consecutive': int(self._rr_consecutive)}, f)
        
        self.reward_fn.save_reward_buffer(os.path.join(actor_local_path, 'reward_buffer.pkl'))

        # Persist the self-improve prompt state (each row's previous code and
        # error) as of the start of this step, since a resume re-runs the step
        # recorded in global_step.txt.
        if getattr(self, '_si_prev_code_snapshot', None) is not None:
            si_state_path = os.path.join(actor_local_path, 'self_improve_state.pkl')
            with open(si_state_path, 'wb') as f:
                pickle.dump({
                    'prev_code_lst': self._si_prev_code_snapshot,
                    'prev_error_lst': self._si_prev_error_snapshot,
                    'global_step': self.global_steps,
                }, f)
        
        max_reward_rate_path = os.path.join(actor_local_path, 'max_reward_rate.txt')
        with open(max_reward_rate_path, 'w') as f:
            f.write(str(self.max_reward_rate))

        # Persist the NIW estimator state and the reward-rate accumulators
        # (per-slot sums, ever-valid mask, pooled valid-only sums) for resume.
        if (self.niw_estimator is not None
                or self.config.actor_rollout_ref.actor.get('global_valid_reward_rate', False)
                or self.config.actor_rollout_ref.actor.get('hybrid_reward_rate', False)):
            niw_state_path = os.path.join(actor_local_path, 'niw_state.pt')
            torch.save({
                'estimator': (self.niw_estimator.state_dict()
                              if self.niw_estimator is not None else None),
                'accum_reward': self.accum_reward,
                'accum_time': self.accum_time,
                'ever_valid': self.ever_valid,
                'accum_valid_reward': self.accum_valid_reward,
                'accum_valid_time': self.accum_valid_time,
                'n_valid_samples': self.n_valid_samples,
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
        # Returned so the caller can restore original row order
        # (see restore_original_row_order).
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

        # compute reward rate if needed
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
        # Actor only; the critic's optimizer is stepped and zeroed in fit()
        # right after update_critic to free its gradients early.
        self.actor_rollout_wg.optim_zero_grad()


    def optim_step(self, metrics):
        # Actor only (see optim_zero_grad).
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
        Update the online NIW estimator with this step's per-slot (reward, time)
        and return the reward rate for the time penalty: the 95th percentile
        of the posterior-predictive batch rates, clamped at 0. The full
        estimate is cached on self._niw_last. When reward_rate_exclude_invalid is set,
        invalid slots are dropped before the fit.

        valid_submission_tensor must be in the same row order as reward_vec
        and time_vec.
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
        self.max_reward_rate = max(0.0, float(est['p95']))
        return self.max_reward_rate

    def _restore_rate_accumulators(self, ckpt):
        """
        Reload the reward-rate accumulators from a saved niw_state.pt.
        Missing keys fall back to empty defaults.
        """
        self.accum_reward = ckpt['accum_reward']
        self.accum_time = ckpt['accum_time']
        self.ever_valid = ckpt.get('ever_valid', torch.zeros_like(self.accum_reward))
        self.accum_valid_reward = float(ckpt.get('accum_valid_reward', 0.0))
        self.accum_valid_time = float(ckpt.get('accum_valid_time', 0.0))
        self.n_valid_samples = int(ckpt.get('n_valid_samples', 0))
        self.invalid_penalty_rate = ckpt.get('invalid_penalty_rate', 0.0)

    def _accumulate_global_valid(self, reward, time, valid_mask):
        """
        Add this step's valid submissions to the pooled running sums.

        reward/time/valid_mask are per-slot vectors in a common order; only
        entries with a truthy mask contribute to the reward and time sums.
        """
        # Initialize lazily for callers that bypass fit().
        if not hasattr(self, 'accum_valid_reward'):
            self.accum_valid_reward = 0.0
            self.accum_valid_time = 0.0
            self.n_valid_samples = 0
        m = valid_mask.detach().cpu().float()
        r = reward.detach().cpu().float()
        t = time.detach().cpu().float()
        self.accum_valid_reward += float((r * m).sum())
        self.accum_valid_time += float((t * m).sum())
        self.n_valid_samples += int(m.sum())

    def _global_valid_reward_rate(self):
        """
        Pooled rate = sum(valid rewards) / sum(valid times) over all steps.
        Returns 0.0 until at least one valid submission exists.
        """
        if self.accum_valid_time <= 0.0:
            return 0.0
        rate = self.accum_valid_reward / self.accum_valid_time
        self.max_reward_rate = rate
        return rate

    def hybrid_anchor_snapshot(self):
        """
        Anchor mask (original slot order) for this step's hybrid shaping.

        Must be taken before get_max_reward_rate() folds the current step's
        validity into self.ever_valid, so a slot switches to the reward-rate
        objective only from the step after its first valid submission.
        """
        return self.ever_valid.detach().clone()


    def get_max_reward_rate(self, reward_fn_dict, do_multi_step):
        # Reward-rate options, in order of precedence:
        #   global_valid_reward_rate: pooled valid-only rate,
        #   NIW estimator (when enabled),
        #   expected_reward_rate: mean(reward) / mean(time) over the batch,
        #   default: running max of per-slot reward / time.
        actor_cfg = self.config.actor_rollout_ref.actor
        use_expected_reward_rate = actor_cfg.get('expected_reward_rate', False)
        use_niw_reward_rate = self.niw_estimator is not None
        use_global_valid = actor_cfg.get('global_valid_reward_rate', False)

        if do_multi_step:
            # Per-slot accumulators are indexed by original row order.
            reward = reward_fn_dict['raw_reward_tensor']
            time = reward_fn_dict['exec_time_tensor_orig_order'].sum(dim=-1)
            self.accum_reward += reward
            self.accum_time += time

            # Running "ever valid" mask: slot i is 1.0 once it has produced at
            # least one valid submission.
            step_valid = reward_fn_dict.get('valid_submission_tensor_orig_order')
            if step_valid is not None:
                sv = step_valid.detach().cpu().float()
                if sv.dim() > 1:
                    sv = sv.reshape(sv.shape[0], -1).amax(dim=-1)
                self.ever_valid = torch.maximum(self.ever_valid, sv)

            # Pooled valid-only accumulation, masked by this step's validity.
            if step_valid is not None:
                self._accumulate_global_valid(reward, time, sv)

            if use_global_valid:
                return self._global_valid_reward_rate()

            if use_niw_reward_rate:
                # reward_rate_valid_thus_far: mask the fit of the cumulative
                # accumulators by ever-valid rather than this step's validity.
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
            # Single-step: no cross-step accumulators; balanced order is used.
            reward = reward_fn_dict['reward_tensor'].sum(dim=-1)
            time = reward_fn_dict['exec_time_tensor'].sum(dim=-1)

            sv = reward_fn_dict.get('valid_submission_tensor')
            if sv is not None:
                sv = sv.detach().cpu().float()
                if sv.dim() > 1:
                    sv = sv.reshape(sv.shape[0], -1).amax(dim=-1)
                self._accumulate_global_valid(reward, time, sv)

            if use_global_valid:
                return self._global_valid_reward_rate()

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
        """
        Diagnostics over the per-slot multi-step accumulators: the slot with
        the least accumulated time and its reward, batch-wide averages, the
        ever-valid fraction, and the pooled valid-only sums.
        """
        min_time_idx = torch.argmin(self.accum_time).item()
        return {
            'train/min_accum_time_sample/accum_time': self.accum_time[min_time_idx].item(),
            'train/min_accum_time_sample/accum_reward': self.accum_reward[min_time_idx].item(),
            'train/accum_time_mean': self.accum_time.mean().item(),
            'train/accum_reward_mean': self.accum_reward.mean().item(),
            # Fraction of slots that have produced a valid submission at least once.
            'train/ever_valid_perc': self.ever_valid.mean().item(),
            # Pooled valid-only accumulators (actor.global_valid_reward_rate).
            'train/global_valid/reward_sum': self.accum_valid_reward,
            'train/global_valid/time_sum': self.accum_valid_time,
            'train/global_valid/n_samples': self.n_valid_samples,
            'train/global_valid/reward_rate': (
                self.accum_valid_reward / self.accum_valid_time
                if self.accum_valid_time > 0 else 0.0),
        }


    # Per-row fields of a reward_fn_dict, regathered together when rejection
    # sampling stitches rounds; other fields are batch-level scalars.
    _REWARD_ROW_TENSOR_KEYS = (
        'raw_reward_tensor', 'raw_exec_time_tensor', 'raw_grader_score_tensor',
        'valid_submission_tensor', 'reward_tensor', 'exec_time_tensor',
        'penalty_time_tensor',
    )
    _REWARD_ROW_LIST_KEYS = (
        'ml_algo_str_lst', 'num_feature_lst', 'raw_code_lst', 'uid_lst',
        'error_lst',
    )

    def _rejection_enabled(self):
        return bool(self.config.trainer.get('rejection_sample_all_valid', False))

    @staticmethod
    def _subset_dataproto(data: DataProto, slots) -> DataProto:
        """Return a copy of `data` restricted to (and ordered by) `slots`."""
        sub = copy.deepcopy(data)
        sub.reorder(torch.as_tensor(list(slots), dtype=torch.long))
        return sub

    def _generate_until_all_valid(self, batch: DataProto, gen_batch: DataProto,
                                  metrics: dict):
        """
        Rejection-sample per slot until every row has a valid submission, for
        at most trainer.rejection_sample_max_rounds rounds (the last attempt is
        kept for slots that never become valid).

        Returns (batch, reward_fn_dict), both in original slot order: row i is
        always a sample generated from row i's prompt.
        """
        bsz = len(gen_batch)
        max_rounds = max(1, int(self.config.trainer.get(
            'rejection_sample_max_rounds', 4)))

        kept_batches, kept_slots, kept_dicts = [], [], []
        counts = {'valid_submission': 0, 'crashed_error_count': 0,
                  'unknown_error_count': 0, 'timeout_error_count': 0}
        pending = list(range(bsz))
        rounds_used = 0
        resampled_rows = 0

        while pending:
            rounds_used += 1
            is_last = rounds_used >= max_rounds
            if rounds_used > 1:
                resampled_rows += len(pending)

            sub_gen = self._subset_dataproto(gen_batch, pending)
            sub_out = self.actor_rollout_wg.generate_sequences(sub_gen)

            sub_batch = self._subset_dataproto(batch, pending)
            # uid is per draw, so a resampled slot gets a new id.
            sub_batch.non_tensor_batch['uid'] = np.array(
                [f"step{self.global_steps}_r{rounds_used}_{uuid.uuid4().hex}"
                 for _ in range(len(pending))],
                dtype=object,
            )
            sub_batch = sub_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n,
                interleave=True,
            )
            sub_batch = sub_batch.union(sub_out)

            rd = self.reward_fn(sub_batch, self.global_steps, 1)
            for key in counts:
                counts[key] += rd[key]

            valid = rd['valid_submission_tensor']
            keep_local = [j for j in range(len(pending))
                          if bool(valid[j] > 0) or is_last]
            still = [pending[j] for j in range(len(pending))
                     if not (bool(valid[j] > 0) or is_last)]

            if keep_local:
                kept_batches.append(self._subset_dataproto(sub_batch, keep_local))
                kept_slots.append(np.asarray([pending[j] for j in keep_local],
                                             dtype=np.int64))
                kept_dicts.append((rd, keep_local))

            pending = still

        # Stitch: concat the per-round keeps, then permute into slot order.
        order = np.concatenate(kept_slots)
        if sorted(order.tolist()) != list(range(bsz)):
            raise RuntimeError(
                "rejection sampling produced a row set that is not exactly "
                f"slots 0..{bsz - 1} (got {len(order)} rows) -- slot identity "
                "would be scrambled.")
        perm = torch.as_tensor(np.argsort(order, kind='stable'), dtype=torch.long)

        widths = {b.batch['responses'].shape[1] for b in kept_batches}
        if len(widths) > 1:
            raise RuntimeError(
                f"rejection rounds returned different response widths {widths}; "
                "cannot concat without repadding.")

        full_batch = DataProto.concat(kept_batches)
        full_batch.reorder(perm)

        stitched = {}
        for key in self._REWARD_ROW_TENSOR_KEYS:
            stitched[key] = torch.cat(
                [rd[key][keep] for rd, keep in kept_dicts], dim=0)[perm]
        for key in self._REWARD_ROW_LIST_KEYS:
            flat = [rd[key][j] for rd, keep in kept_dicts for j in keep]
            stitched[key] = [flat[i] for i in perm.tolist()]
        stitched.update(counts)
        # valid_submission counts the kept batch, not every draw.
        stitched['valid_submission'] = int(
            stitched['valid_submission_tensor'].sum().item())
        stitched['b_score'] = kept_dicts[-1][0]['b_score']

        unfilled = int(bsz - stitched['valid_submission'])
        metrics['train/rejection/rounds'] = rounds_used
        metrics['train/rejection/resampled_rows'] = resampled_rows
        metrics['train/rejection/unfilled_slots'] = unfilled
        if unfilled:
            print(f'[REJECT] step {self.global_steps}: {unfilled} slot(s) still '
                  f'invalid after {rounds_used} rounds; keeping last attempt.')

        return full_batch, stitched


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
        # Per-slot "has ever produced a valid submission" flag.
        self.ever_valid = torch.zeros(self.config.data.train_batch_size)
        # Pooled valid-only accumulators (actor.global_valid_reward_rate).
        self.accum_valid_reward = 0.0
        self.accum_valid_time = 0.0
        self.n_valid_samples = 0
        if (self.config.actor_rollout_ref.actor.get('global_valid_reward_rate', False)
                and self.config.actor_rollout_ref.actor.get('use_niw_reward_rate', False)):
            raise ValueError(
                'global_valid_reward_rate and use_niw_reward_rate are mutually '
                'exclusive: the pooled rate is a single (reward, time) pair per '
                'step, which has no cross-section for the NIW model to fit.')
        if self.config.actor_rollout_ref.actor.get('hybrid_reward_rate', False):
            if not self.config.trainer.multi_step:
                raise ValueError(
                    'hybrid_reward_rate requires trainer.multi_step: the anchor '
                    'is per persistent dataset slot, and without multi-step '
                    'self-improve there is no slot identity to anchor.')
            if not self.config.actor_rollout_ref.actor.use_reward_rate_penalty:
                raise ValueError(
                    'hybrid_reward_rate gates the reward-rate time penalty '
                    'per slot, so use_reward_rate_penalty must be on -- with '
                    'it off there is nothing to gate and the flag would '
                    'silently run plain vanilla.')
        # Set up the online NIW reward-rate estimator if enabled.
        self.niw_estimator = None
        self._niw_last = {}
        if self.config.actor_rollout_ref.actor.get('use_niw_reward_rate', False):
            self.niw_estimator = self._build_niw_estimator()
            # Resume the estimator and accumulators from the actor checkpoint
            # if present.
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

        elif self.config.actor_rollout_ref.actor.get('global_valid_reward_rate', False):
            # Resume the pooled valid-only sums.
            state_path = os.path.join(self.config.actor_rollout_ref.model.path, 'niw_state.pt')
            if os.path.exists(state_path):
                try:
                    ckpt = torch.load(state_path, map_location='cpu', weights_only=False)
                except TypeError:
                    ckpt = torch.load(state_path, map_location='cpu')
                self._restore_rate_accumulators(ckpt)
                print(f'[RATE] Resumed pooled valid accumulators from {state_path}: '
                      f'reward={self.accum_valid_reward:.4f} time={self.accum_valid_time:.1f}s '
                      f'n={self.n_valid_samples}.')
            else:
                print(f'[RATE] No saved accumulator state at {state_path}; starting fresh.')

        elif self.config.actor_rollout_ref.actor.get('hybrid_reward_rate', False):
            # Resume the hybrid's per-slot anchor state (ever_valid).
            state_path = os.path.join(self.config.actor_rollout_ref.model.path, 'niw_state.pt')
            if os.path.exists(state_path):
                try:
                    ckpt = torch.load(state_path, map_location='cpu', weights_only=False)
                except TypeError:
                    ckpt = torch.load(state_path, map_location='cpu')
                self._restore_rate_accumulators(ckpt)
                print(f'[HYBRID] Resumed anchor state from {state_path}: '
                      f'{int(self.ever_valid.sum().item())}/{len(self.ever_valid)} '
                      f'slots anchored.')
            else:
                print(f'[HYBRID] No saved anchor state at {state_path}; all slots '
                      f'start under the vanilla objective.')

        reward_buffer_path = os.path.join(
            self.config.trainer.default_local_dir, 'actor', 'reward_buffer.pkl')
        if not os.path.exists(reward_buffer_path):
            # Resuming from another run's checkpoint: look under model.path.
            bootstrap_path = os.path.join(
                self.config.actor_rollout_ref.model.path, 'reward_buffer.pkl')
            if os.path.exists(bootstrap_path):
                reward_buffer_path = bootstrap_path
        self.reward_fn.load_reward_buffer(reward_buffer_path)

        # Restore the self-improve prompt state (previous code/error per row)
        # saved by _save_checkpoint: this run's dir first, then model.path.
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

        # Staged objective switch: set the phase before the first step.
        self._init_rr_switch_state()

        self.reward_fn.init_gpu_heartbeat_actor()

        while self.global_steps <= self.config.trainer.total_training_steps:
            metrics = {}
            print(f'GLOBAL_STEP={self.global_steps}')

            # Snapshot the self-improve prompt state at the start of the step
            # for checkpointing (a resume re-runs this step).
            if hasattr(self, 'self_improve_dataloader'):
                self._si_prev_code_snapshot = self.self_improve_dataset.prev_code_lst
                self._si_prev_error_snapshot = self.self_improve_dataset.prev_error_lst

            if self.global_steps >= 2 and self.config.trainer.multi_step:
                batch_dict = next(self.self_improve_dataloader)
            else:
                batch_dict = next(self.train_dataloader)
                
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            
            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                
            # generate a batch, this adds "response" and "old_log_probs" to batch
            # input_ids become the tokens for the whole squence including prompt and response
            # With rejection sampling the batch comes back already scored
            # (in original slot order), and the reward call below is skipped.
            prescored_reward_dict = None
            if self._rejection_enabled():
                batch, prescored_reward_dict = self._generate_until_all_valid(
                    batch, gen_batch, metrics)
            else:
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
            # restore_original_row_order() undoes it for row-level metadata below.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            #
            # Stamp each row with its pre-balance index and check that the
            # observed permutation matches global_idx.
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
            if prescored_reward_dict is not None:
                reward_fn_dict = prescored_reward_dict
            else:
                reward_fn_dict = self.reward_fn(batch, self.global_steps, 1)    # k=1: multi-step spans global steps

            # PPO-update tensors stay in balanced order to match `batch.batch`;
            # row-level metadata is restored to original row order.
            if prescored_reward_dict is not None:
                reward_fn_dict = align_prescored_row_order(reward_fn_dict, global_idx)
            else:
                reward_fn_dict = restore_original_row_order(reward_fn_dict, global_idx)

            valid_subm_perc = reward_fn_dict['valid_submission'] / len(batch.batch)

            if self.config.trainer.use_submission_rate_reweighting:
                batch.batch['token_level_scores'] = reward_fn_dict['reward_tensor'] / (valid_subm_perc + 1e-8)
            else:
                batch.batch['token_level_scores'] = reward_fn_dict['reward_tensor']
            batch.batch['exec_time_tensor'] = reward_fn_dict['exec_time_tensor']
            # Time used for the time penalty only. Equals exec_time_tensor except
            # for invalid submissions under reward.invalid_zero_reward_full_time,
            # which are charged the full timeout.
            batch.batch['penalty_time_tensor'] = reward_fn_dict.get(
                'penalty_time_tensor', reward_fn_dict['exec_time_tensor'])
            batch.batch['valid_submission_tensor'] = reward_fn_dict['valid_submission_tensor']

            if self.config.trainer.use_old_reward_rate:
                self.compute_and_record_old_reward_rate(reward_fn_dict, batch)
            
            if self.config.trainer.multi_step:
                self.update_self_improve_dataset(
                    reward_fn_dict["raw_code_lst"], reward_fn_dict["error_lst"])


            # Hybrid per-slot objective: snapshot the anchor state before the
            # accumulator update below folds in this step's validity.
            use_hybrid = self.config.actor_rollout_ref.actor.get(
                'hybrid_reward_rate', False)
            if use_hybrid:
                anchor_snapshot = self.hybrid_anchor_snapshot()

            reward_rate = self.get_max_reward_rate(
                reward_fn_dict=reward_fn_dict,
                do_multi_step=self.config.trainer.multi_step
            )
            use_max_invalid_rate = self.config.actor_rollout_ref.actor.get(
                'invalid_penalty_use_max_rate', False)
            if use_max_invalid_rate:
                self.invalid_penalty_rate = max(self.invalid_penalty_rate, reward_rate)
            # self.rr_active mirrors use_reward_rate_penalty unless the
            # staged switch is on (see _init_rr_switch_state).
            if self.rr_active and \
                self.config.actor_rollout_ref.actor.use_reward_rate_penalty and \
                not self.config.actor_rollout_ref.actor.use_log_reward_rate_penalty:

                # The snapshot is in original slot order; permute it into the
                # balanced order of the surrogate tensors.
                anchor_mask = anchor_snapshot[global_idx] if use_hybrid else None
                if use_hybrid:
                    metrics['train/hybrid/anchored_share'] = \
                        anchor_snapshot.mean().item()
                    metrics['train/hybrid/pre_anchor_rows'] = \
                        int((1.0 - anchor_snapshot).sum().item())

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
                    anchor_mask=anchor_mask,
                )
                batch.batch['token_level_scores'] = reward_tensor

                # -- Risk-seeking exponential tilt (optional) --------------
                # r'_i = exp(beta*s_i) / mean_j exp(beta*s_j) = B * softmax(beta*s)_i,
                # where s_i is sample i's total shaped reward after the
                # surrogate above. The batch mean of r' is 1.
                rs_ess_loo = self.config.actor_rollout_ref.actor.get(
                    'risk_seeking_ess_loo_target', 0.0) or 0.0
                rs_ess_v = self.config.actor_rollout_ref.actor.get(
                    'risk_seeking_ess_valid_target', 0.0) or 0.0
                rs_ess = self.config.actor_rollout_ref.actor.get(
                    'risk_seeking_ess_target', 0.0) or 0.0
                rs_beta = self.config.actor_rollout_ref.actor.get(
                    'risk_seeking_beta', 0.0) or 0.0
                # Minimum number of valid rows for the adaptive solves.
                min_valid = int(self.config.actor_rollout_ref.actor.get(
                    'risk_seeking_min_valid', 4) or 4)
                if rs_ess_loo > 0:
                    # -- Leave-one-out entropic weight -----------------------
                    # w_n = exp(beta*(s_n - s_max)) / (Zhat_{-n} + eps),
                    #   Zhat_{-n} = mean_{m != n} exp(beta*(s_m - s_max))
                    # over the valid rows, on the raw (unstandardized) shaped
                    # reward; invalid rows get risk_seeking_invalid_replace.
                    # beta is solved by bisection so the valid-row ESS
                    # fraction hits risk_seeking_ess_loo_target. No "- 1"
                    # centering is applied since the critic supplies the
                    # baseline.
                    scores = batch.batch['token_level_scores']
                    s = scores.sum(-1).float()
                    v = batch.batch['valid_submission_tensor']
                    v = v.sum(-1) if v.dim() == 2 else v
                    vmask = (v.float().reshape(-1) > 0.5)
                    nv = int(vmask.sum().item())
                    inv_r = float(self.config.actor_rollout_ref.actor.get(
                        'risk_seeking_invalid_replace', -10.0))
                    beta_max = float(self.config.actor_rollout_ref.actor.get(
                        'risk_seeking_beta_max', 100.0))
                    eps = float(self.config.actor_rollout_ref.actor.get(
                        'risk_seeking_loo_eps', 1e-6))
                    a_valid, beta_solved, clamped = None, 0.0, 0
                    sigma, ess_ach, d_valid, z_min = 0.0, 1.0, 0.0, 0.0
                    # LOO needs at least one other row in the baseline
                    if nv >= max(2, min_valid):
                        sv = s[vmask]
                        sigma = float(sv.std(unbiased=False).item())
                        c = sv - sv.max()          # <= 0, raw scale, no sigma
                        spread = float((-c.min()).item())
                        if spread > 1e-9:

                            def _ess_v(b):
                                p = torch.softmax(b * c, dim=0)
                                return float((1.0 / (nv * p.pow(2).sum())).item())
                            if _ess_v(beta_max) > rs_ess_loo:
                                beta_solved, clamped = beta_max, 1
                            else:
                                lo, hi = 0.0, beta_max
                                for _ in range(50):
                                    mid = 0.5 * (lo + hi)
                                    if _ess_v(mid) > rs_ess_loo:
                                        lo = mid
                                    else:
                                        hi = mid
                                beta_solved = 0.5 * (lo + hi)
                            e = torch.exp(beta_solved * c)   # in (0, 1]
                            z_loo = (e.sum() - e) / (nv - 1)  # leave-one-out
                            a_valid = e / (z_loo + eps)
                            ess_ach = _ess_v(beta_solved)
                            z_min = float(z_loo.min().item())
                            qs = torch.tensor([0.1, 0.9], device=sv.device,
                                              dtype=sv.dtype)
                            q = torch.quantile(sv, qs)
                            d_valid = float((q[1] - q[0]).item())
                    if a_valid is None and nv > 0:
                        # too few valid rows or no spread: flat weights of 1
                        a_valid = torch.ones(nv, device=s.device, dtype=s.dtype)
                    applied = torch.full_like(s, inv_r)
                    if nv > 0:
                        applied[vmask] = a_valid.to(applied.dtype)
                    resp_len = batch.batch['responses'].shape[-1]
                    resp_mask = batch.batch['attention_mask'][:, -resp_len:]
                    last_idx = (resp_mask.sum(-1).long() - 1).clamp(min=0)
                    new_scores = torch.zeros_like(scores)
                    new_scores[torch.arange(scores.shape[0],
                                            device=scores.device), last_idx] = \
                        applied.to(scores.dtype)
                    batch.batch['token_level_scores'] = new_scores
                    # beta is on the raw shaped-reward scale, so beta_raw == beta_solved
                    metrics['train/risk_seeking/beta_solved'] = beta_solved
                    metrics['train/risk_seeking/beta_raw'] = beta_solved
                    metrics['train/risk_seeking/sigma_valid'] = sigma
                    metrics['train/risk_seeking/ess_valid'] = ess_ach
                    metrics['train/risk_seeking/clamped'] = clamped
                    metrics['train/risk_seeking/n_valid'] = nv
                    metrics['train/risk_seeking/min_valid'] = min_valid
                    metrics['train/risk_seeking/invalid_replace'] = inv_r
                    metrics['train/risk_seeking/d_valid'] = d_valid
                    metrics['train/risk_seeking/s_spread'] = \
                        (s.max() - s.min()).item()
                    # weight stats over valid rows; loo_z_min is the smallest
                    # leave-one-out denominator
                    if nv > 0:
                        av = applied[vmask]
                        metrics['train/risk_seeking/w_max'] = av.max().item()
                        metrics['train/risk_seeking/w_median'] = \
                            av.median().item()
                        metrics['train/risk_seeking/w_min'] = av.min().item()
                    metrics['train/risk_seeking/loo_z_min'] = z_min
                elif rs_ess_v > 0:
                    # -- Valid-row ESS variant --------------------------------
                    # beta is solved by bisection against the ESS of the valid
                    # rows only, on standardized valid shaped rewards
                    # z = (s_v - max)/sigma. Valid rows get nv * softmax(beta*z)
                    # (mean 1 over valid rows); invalid rows get
                    # risk_seeking_invalid_replace.
                    scores = batch.batch['token_level_scores']
                    s = scores.sum(-1).float()
                    v = batch.batch['valid_submission_tensor']
                    v = v.sum(-1) if v.dim() == 2 else v
                    vmask = (v.float().reshape(-1) > 0.5)
                    nv = int(vmask.sum().item())
                    inv_r = float(self.config.actor_rollout_ref.actor.get(
                        'risk_seeking_invalid_replace', -10.0))
                    beta_max = float(self.config.actor_rollout_ref.actor.get(
                        'risk_seeking_beta_max', 100.0))
                    w_valid, beta_solved, clamped = None, 0.0, 0
                    sigma, ess_ach, d_valid = 0.0, 1.0, 0.0
                    if nv >= max(2, min_valid):
                        sv = s[vmask]
                        sigma = float(sv.std(unbiased=False).item())
                        # tolerance above float32 noise for a constant vector
                        if sigma > 1e-6:
                            z = (sv - sv.max()) / sigma

                            def _ess_v(b):
                                p = torch.softmax(b * z, dim=0)
                                return float((1.0 / (nv * p.pow(2).sum())).item())
                            if _ess_v(beta_max) > rs_ess_v:
                                beta_solved, clamped = beta_max, 1
                            else:
                                lo, hi = 0.0, beta_max
                                for _ in range(50):
                                    mid = 0.5 * (lo + hi)
                                    if _ess_v(mid) > rs_ess_v:
                                        lo = mid
                                    else:
                                        hi = mid
                                beta_solved = 0.5 * (lo + hi)
                            p = torch.softmax(beta_solved * z, dim=0)
                            w_valid = p * nv
                            ess_ach = float((1.0 / (nv * p.pow(2).sum())).item())
                            qs = torch.tensor([0.1, 0.9], device=sv.device,
                                              dtype=sv.dtype)
                            q = torch.quantile(sv, qs)
                            d_valid = float((q[1] - q[0]).item())
                    if w_valid is None and nv > 0:
                        # too few valid rows or no spread: flat weights of 1
                        w_valid = torch.ones(nv, device=s.device, dtype=s.dtype)
                    applied = torch.full_like(s, inv_r)
                    if nv > 0:
                        applied[vmask] = w_valid.to(applied.dtype)
                    resp_len = batch.batch['responses'].shape[-1]
                    resp_mask = batch.batch['attention_mask'][:, -resp_len:]
                    last_idx = (resp_mask.sum(-1).long() - 1).clamp(min=0)
                    new_scores = torch.zeros_like(scores)
                    new_scores[torch.arange(scores.shape[0],
                                            device=scores.device), last_idx] = \
                        applied.to(scores.dtype)
                    batch.batch['token_level_scores'] = new_scores
                    metrics['train/risk_seeking/beta_solved'] = beta_solved
                    # beta on the raw shaped-reward scale
                    metrics['train/risk_seeking/beta_raw'] = (
                        beta_solved / sigma if sigma > 1e-6 else 0.0)
                    metrics['train/risk_seeking/sigma_valid'] = sigma
                    metrics['train/risk_seeking/ess_valid'] = ess_ach
                    metrics['train/risk_seeking/clamped'] = clamped
                    metrics['train/risk_seeking/n_valid'] = nv
                    metrics['train/risk_seeking/invalid_replace'] = inv_r
                    metrics['train/risk_seeking/w_max'] = applied.max().item()
                    metrics['train/risk_seeking/w_median'] = \
                        applied.median().item()
                    metrics['train/risk_seeking/s_spread'] = \
                        (s.max() - s.min()).item()
                    # p90-p10 of the valid band, the scale beta is solved
                    # against; beta_raw * d_valid is the log weight ratio
                    # between a p90 and a p10 valid sample
                    metrics['train/risk_seeking/d_valid'] = d_valid
                elif rs_ess > 0:
                    # -- Adaptive whole-batch variant --------------------------
                    # Same as the fixed-beta tilt over the whole batch, except
                    # beta is solved per batch by bisection so the ESS fraction
                    # ess(beta) = 1/(B*sum softmax(beta*s)^2) hits rs_ess
                    # (clamped at beta_max).
                    scores = batch.batch['token_level_scores']
                    s = scores.sum(-1).float()
                    n = s.numel()
                    beta_max = float(self.config.actor_rollout_ref.actor.get(
                        'risk_seeking_beta_max', 100.0))

                    def _ess(b):
                        p = torch.softmax(b * s, dim=0)
                        return float((1.0 / (n * p.pow(2).sum())).item())
                    if _ess(beta_max) > rs_ess:
                        beta_solved, clamped = beta_max, 1
                    else:
                        lo, hi = 0.0, beta_max
                        for _ in range(50):
                            mid = 0.5 * (lo + hi)
                            if _ess(mid) > rs_ess:
                                lo = mid
                            else:
                                hi = mid
                        beta_solved, clamped = 0.5 * (lo + hi), 0
                    w = torch.softmax(beta_solved * s, dim=0) * n
                    v = batch.batch['valid_submission_tensor']
                    v = v.sum(-1) if v.dim() == 2 else v
                    valid_mask = (v.float().reshape(-1) > 0.5)
                    # Optionally replace the applied reward of invalid rows with
                    # a flat penalty (the solve still includes them).
                    # ess_achieved describes the pre-replacement weights.
                    rs_inv_replace = self.config.actor_rollout_ref.actor.get(
                        'risk_seeking_invalid_replace', None)
                    applied = w
                    if rs_inv_replace is not None:
                        applied = torch.where(
                            valid_mask, w,
                            torch.full_like(w, float(rs_inv_replace)))
                    resp_len = batch.batch['responses'].shape[-1]
                    resp_mask = batch.batch['attention_mask'][:, -resp_len:]
                    last_idx = (resp_mask.sum(-1).long() - 1).clamp(min=0)
                    new_scores = torch.zeros_like(scores)
                    new_scores[torch.arange(scores.shape[0],
                                            device=scores.device), last_idx] = \
                        applied.to(scores.dtype)
                    batch.batch['token_level_scores'] = new_scores
                    metrics['train/risk_seeking/beta_solved'] = beta_solved
                    metrics['train/risk_seeking/ess_achieved'] = _ess(beta_solved)
                    metrics['train/risk_seeking/clamped'] = clamped
                    metrics['train/risk_seeking/n_valid'] = \
                        int(valid_mask.sum().item())
                    metrics['train/risk_seeking/w_max'] = w.max().item()
                    metrics['train/risk_seeking/w_median'] = w.median().item()
                    metrics['train/risk_seeking/s_spread'] = \
                        (s.max() - s.min()).item()
                    if valid_mask.any():
                        wv = w[valid_mask]
                        pv = wv / wv.sum()
                        metrics['train/risk_seeking/ess_valid'] = \
                            (1.0 / (pv.numel() * pv.pow(2).sum())).item()
                    if rs_inv_replace is not None:
                        metrics['train/risk_seeking/invalid_replace'] = \
                            float(rs_inv_replace)
                elif rs_beta > 0:
                    scores = batch.batch['token_level_scores']
                    s = scores.sum(-1).float()
                    z = rs_beta * s
                    w = torch.exp(z - torch.logsumexp(z, dim=0)) * z.numel()
                    # The scalar reward sits at the last valid response token.
                    resp_len = batch.batch['responses'].shape[-1]
                    resp_mask = batch.batch['attention_mask'][:, -resp_len:]
                    last_idx = (resp_mask.sum(-1).long() - 1).clamp(min=0)
                    new_scores = torch.zeros_like(scores)
                    new_scores[torch.arange(scores.shape[0],
                                            device=scores.device), last_idx] = \
                        w.to(scores.dtype)
                    batch.batch['token_level_scores'] = new_scores
                    metrics['train/risk_seeking/w_max'] = w.max().item()
                    metrics['train/risk_seeking/w_median'] = w.median().item()
                    # Effective-sample-size fraction of the tilt: 1.0 = flat,
                    # 1/B = one sample carries the whole batch (hard argmax).
                    metrics['train/risk_seeking/ess_frac'] = \
                        (w.numel() / w.pow(2).sum()).item()
                    metrics['train/risk_seeking/s_spread'] = \
                        (s.max() - s.min()).item()

            # Update the staged switch on this step's validity.
            self._maybe_switch_to_rr(valid_subm_perc, metrics)

            # KL penalty on rewards; KL-on-loss is applied in dp_actor.update_policy().
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
                # Step and zero the critic's optimizer here, so its gradients
                # are freed before the actor update.
                critic_grad_norm = self.critic_wg.optim_step()
                self.critic_wg.optim_zero_grad()
                append_to_dict(metrics, {'critic/grad_norm': critic_grad_norm})

            # update actor if after critic_warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
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
                # Log the NIW posterior summary alongside train/reward_rate.
                metrics.update({
                    'train/niw/reward_rate_mean': self._niw_last['mean'],
                    'train/niw/reward_rate_p2_5': self._niw_last['lo'],
                    'train/niw/reward_rate_p97_5': self._niw_last['hi'],
                    'train/niw/reward_rate_p25': self._niw_last['p25'],
                    'train/niw/reward_rate_p75': self._niw_last['p75'],
                    'train/niw/reward_rate_max': self._niw_last['max'],
                })
            if self.config.trainer.multi_step:
                # accum_reward/accum_time are only populated in multi_step mode.
                metrics.update(self.get_accum_time_reward_metrics())
            metrics = self.optim_step(metrics)
            self.optim_zero_grad()

            # save_freq <= 0 disables checkpointing.
            if self.config.trainer.save_freq > 0 and \
                    self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

            # collect metrics
            metrics = reduce_metrics(metrics)
            logger.log(data=metrics, step=self.global_steps)

            # Log GPU memory at the end of the step.
            self.actor_rollout_wg.log_memory_snapshot(f'end of step {self.global_steps}')
            self.critic_wg.log_memory_snapshot(f'end of step {self.global_steps}')

            self.global_steps += 1
