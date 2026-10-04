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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import ray
import hydra
# RewardManager is imported lazily below to avoid pulling in the mlebench
# reward stack when only init_start_global_step is needed.
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import shutil
import os
import re


def init_start_global_step(start_global_step, model_path):
    global_step_path = os.path.join(model_path, 'global_step.txt')
    
    if os.path.exists(global_step_path):
        with open(global_step_path, 'r') as f:
            global_step = int(f.read().strip())
        print(f'Found existing global step {global_step} from {global_step_path}. Resuming training from this step.')
        return global_step
    else:
        print(f'No existing global step found at {global_step_path}. Starting training from {start_global_step}.')
        return start_global_step


def init_start_max_reward_rate(max_reward_rate, model_path):
    max_reward_rate_path = os.path.join(model_path, 'max_reward_rate.txt')
    
    if os.path.exists(max_reward_rate_path):
        with open(max_reward_rate_path, 'r') as f:
            max_reward_rate = float(f.read().strip())
        print(f'Found existing max reward rate {max_reward_rate} from {max_reward_rate_path}. Resuming training from this rate.')
        return max_reward_rate
    else:
        print(f'No existing max reward rate found at {max_reward_rate_path}. Starting training from {max_reward_rate}.')
        return max_reward_rate


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    workspace_dir = config.trainer.workspace_dir
    
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    
    os.makedirs(workspace_dir, exist_ok=True)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    info = ray.init(
        _temp_dir=f"/tmp/{os.environ.get('USER', 'ray')}/ray_tmp/{job_id}",
        address="auto",
        runtime_env={
            "working_dir": ".",
            "excludes": [
                ".github/", "cache/", "checkpoints/", "data/", "docker/", "docs/", "examples/",
                "logs/", "outputs/", "patches/", "scripts/", "tests/", "verl.egg-info/",
                "wandb/", "workspace/", "__pycache__/",
            ],
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true', 
                'NCCL_DEBUG': 'WARN',
            },
        },
    )
    print(f"Dashboard: {info.dashboard_url}")
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    local_path = os.path.join(local_path, 'model')
    
    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # prepare the role mapping with ray.remote(), Role is basically a way for indexing
    # Ray actor classes are created here, but resources are not allocated
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    start_global_step = init_start_global_step(
        start_global_step=1,
        model_path=config.actor_rollout_ref.model.path,
    )
    max_reward_rate = init_start_max_reward_rate(
        max_reward_rate=0.0,
        model_path=config.actor_rollout_ref.model.path,
    )
    
    from verl.utils.reward_score.mlebench import RewardManager
    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0, 
        timeout=config.data.timeout,
        workspace_dir=config.trainer.workspace_dir,
        reward_config=config.reward,
        baseline_score=config.reward.baseline_score,
        buffer_size=config.reward.buffer_size,
        num_cpus_per_sample=config.reward.num_cpus_per_sample,
        scalar_fn_name=config.reward.scalar_fn_name,
        code_mem_limit_gb=config.reward.code_mem_limit_gb,
        b_score=config.reward.b_score,
        b_start_step=start_global_step,
    )
    # NOTE: there is no notion of validation set, so cancel it
    val_reward_fn = None
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, 
        mapping=mapping,
    )
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        start_global_step=start_global_step,
        max_reward_rate=max_reward_rate,
    )
    trainer.init_workers()  # allocate resources
    trainer.fit()


if __name__ == '__main__':
    main()
