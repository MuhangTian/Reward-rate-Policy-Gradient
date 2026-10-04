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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
import verl.utils.hdfs_io as hdfs_io

# Optional CUDA allocator history for OOM debugging; a snapshot is dumped
# when update_policy OOMs (see update_actor below). The CUDA check matters
# because the GPU-less Ray driver also imports this module.
if os.environ.get('VERL_MEM_SNAPSHOT_DIR') and torch.cuda.is_available():
    torch.cuda.memory._record_memory_history(max_entries=500000)
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad, init_fsdp_optimizer_state_on_cpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _is_peft_model(module) -> bool:
    """Return True if `module` is a peft PeftModel (False if peft is not installed)."""
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(module, PeftModel)


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                        self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n


    def _build_model_optimizer(
        self,
        local_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        attn_implementation='flash_attention_2',
        ):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(local_path)
        model_path = os.path.join(local_path, 'model')

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(
            model_path, 
            trust_remote_code=trust_remote_code,
        )

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # Meta-tensor init hangs with tied word embeddings. VERL_NO_META_INIT
        # also disables it and loads real weights on every rank (needs
        # model-size CPU RAM per rank at init).
        _use_meta = not actor_model_config.tie_word_embeddings and \
            not os.environ.get('VERL_NO_META_INIT')
        init_context = get_init_weight_context_manager(use_meta_tensor=_use_meta)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
            )
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        # LoRA (actor.lora_rank > 0): wrap in a PeftModel before the FSDP wrap
        # so FSDP shards the adapter params alongside the frozen base.
        lora_rank = int(self.config.actor.get('lora_rank', 0) or 0) if self._is_actor else 0
        if lora_rank > 0:
            from omegaconf import OmegaConf as _OC
            from peft import LoraConfig, get_peft_model
            lora_target = self.config.actor.get('lora_target_modules', None)
            if lora_target is not None and not isinstance(lora_target, str):
                lora_target = _OC.to_container(lora_target)
            lora_config = LoraConfig(
                task_type='CAUSAL_LM',
                r=lora_rank,
                lora_alpha=self.config.actor.get('lora_alpha', 32),
                lora_dropout=self.config.actor.get('lora_dropout', 0.0),
                target_modules=lora_target if lora_target is not None else 'all-linear',
            )
            actor_module = get_peft_model(actor_module, lora_config)
            # keep adapter params in the actor dtype
            actor_module.to(torch_dtype)
            if self.rank == 0:
                actor_module.print_trainable_parameters()

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype, 
            reduce_dtype=reduce_dtype, 
            buffer_dtype=buffer_dtype,
        )

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get('wrap_policy', None),
        )

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # Single rank + LoRA: skip the FSDP wrap. At world_size==1 FSDP shards
        # nothing, and with use_orig_params=True plus a PeftModel the
        # FULL_STATE_DICT gather omits the root unit's frozen params.
        _ws = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        if lora_rank > 0 and _ws == 1 and \
                not os.environ.get('VERL_FORCE_FSDP_SINGLE_RANK'):
            print('[fsdp_workers] world_size=1 + LoRA: skipping the FSDP wrap '
                  '(NO_SHARD would shard nothing and breaks the PEFT '
                  'state_dict gather)', flush=True)
            actor_module_fsdp = actor_module.to(torch.cuda.current_device())
        else:
            # TODO: add transformer policy
            actor_module_fsdp = FSDP(
                actor_module,
                param_init_fn=init_fn,
                # LoRA needs per-param requires_grad to survive the wrap (flat
                # params would fuse frozen base + trainable adapters).
                use_orig_params=lora_rank > 0,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            if lora_rank > 0:
                # adapter-only training: optimize only the trainable params
                optim_params = [p for p in actor_module_fsdp.parameters() if p.requires_grad]
            else:
                optim_params = actor_module_fsdp.parameters()
            actor_optimizer = optim.AdamW(
                optim_params,
                lr=optim_config.lr,
                betas=optim_config.get('betas', (0.9, 0.999)),
                weight_decay=optim_config.get('weight_decay', 1e-2),
            )
            if self._is_offload_optimizer:
                # Pre-allocate the AdamW state on CPU; AdamW otherwise allocates
                # it lazily on the GPU at the first step, before it can be offloaded.
                init_fsdp_optimizer_state_on_cpu(actor_optimizer)
            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer,
                num_warmup_steps=num_warmup_steps,
            )
            
            optim_dir = os.path.join(local_path, 'optim')
            opt_path = os.path.join(optim_dir, 'optimizer.pt')
            sched_path = os.path.join(optim_dir, 'scheduler.pt')
            
            if os.path.exists(opt_path):
                if self.rank == 0:
                    print(f'Resuming optimizer and scheduler from {optim_dir}')
                
                # Load the full optimizer state dict to CPU on all ranks;
                # each rank extracts its own shard from it.
                full_osd = torch.load(opt_path, map_location='cpu')
                
                # Convert the full dict into this rank's sharded optimizer state dict.
                sharded_osd = FSDP.optim_state_dict_to_load(
                    model=actor_module_fsdp,
                    optim=actor_optimizer,
                    optim_state_dict=full_osd
                )
                
                # Load the sharded dict into the optimizer.
                actor_optimizer.load_state_dict(sharded_osd)
                
                # Load the scheduler state (small and not sharded).
                if os.path.exists(sched_path):
                    full_ssd = torch.load(sched_path, map_location='cpu')
                    actor_lr_scheduler.load_state_dict(full_ssd)
                
                if self.rank == 0:
                    print(f'Successfully resumed training from step {actor_lr_scheduler.last_epoch}')
            else:
                if self.rank == 0:
                    print(f'No optimizer checkpoint found at {opt_path}. Initializing fresh training.')
            
            torch.distributed.barrier()
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config


    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        
        assert self.world_size % infer_tp == 0, \
            f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
            
        rollout_device_mesh = init_device_mesh(
            'cuda', 
            mesh_shape=(dp, infer_tp), 
            mesh_dim_names=['dp', 'infer_tp'],
        )

        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            
            log_gpu_memory_usage('Before building vllm rollout', logger=None)

            # `rollout.load_path` lets the vLLM engine load from a different
            # checkpoint than the actor (e.g. a full multimodal config). Only the
            # engine's initial weights are affected; they are overwritten by the
            # actor's weights at the first sync, before any generation.
            rollout_path_cfg = self.config.rollout.get('load_path', None) or self.config.model.path
            local_path = copy_local_path_from_hdfs(rollout_path_cfg)
            rollout_model_path = os.path.join(local_path, 'model')

            rollout = vLLMRollout(
                actor_module=self.actor_module_fsdp,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                model_path=rollout_model_path,
            )
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
                
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params='hf' in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
                
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, \
                self.actor_model_config = self._build_model_optimizer(
                    local_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                    trust_remote_code=self.config.model.get('trust_remote_code', False),
                    attn_implementation=self.config.model.get('attn_implementation', 'flash_attention_2'),
            )
            # get the original unwrapped module (the module itself when the
            # FSDP wrap is skipped)
            self.actor_module = getattr(self.actor_module_fsdp,
                                        '_fsdp_wrapped_module',
                                        self.actor_module_fsdp)

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
                
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                local_path=self.config.model.ref_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                attn_implementation=self.config.model.get('attn_implementation', 'flash_attention_2'),
            )[0]

            if self._is_offload_param:
                offload_fsdp_param_and_grad(
                    module=self.ref_module_fsdp, 
                    offload_grad=self._is_offload_grad,
                )

            OmegaConf.set_struct(self.config.ref, True)
            
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

        torch.cuda.empty_cache()


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        log_gpu_memory_usage('DEBUG_MEM At start of update_actor (before actor load, inherited from critic)', logger=None)
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        # NOTE: the optimizer state is not loaded here. update_policy() only runs
        # backward passes; optimizer.step() happens in optim_step(), which loads
        # and offloads the Adam state itself.

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('DEBUG_MEM Before update policy (after actor load)', logger=None)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            try:
                metrics = self.actor.update_policy(data=data)
            # OOM/CUDA errors inside the C++ backward engine can surface as SystemError
            except (torch.OutOfMemoryError, SystemError, RuntimeError):
                snap_dir = os.environ.get('VERL_MEM_SNAPSHOT_DIR')
                if snap_dir:
                    os.makedirs(snap_dir, exist_ok=True)
                    rank = torch.distributed.get_rank() \
                        if torch.distributed.is_initialized() else 0
                    job = os.environ.get('SLURM_JOB_ID', 'local')
                    torch.cuda.memory._dump_snapshot(os.path.join(
                        snap_dir, f'update_oom_{job}_rank{rank}.pickle'))
                raise

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)

        # optimizer state was not loaded in this call, so nothing to offload
        torch.cuda.empty_cache()
        return output


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optim_zero_grad(self):
        self.actor.actor_optimizer.zero_grad()
    
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optim_step(self):
        # The Adam state lives on CPU between steps; load it for the step.
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())
        log_gpu_memory_usage('Before actor optim_step', logger=logger)
        result = self.actor._optimizer_step()
        log_gpu_memory_usage('After actor optim_step', logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            # Return cached blocks to the driver so vllm's cumem-based wake_up()
            # has free device memory.
            torch.cuda.empty_cache()
            log_gpu_memory_usage('After actor optim_step offload + empty_cache', logger=logger)
        return result


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def log_memory_snapshot(self, tag: str):
        # Diagnostic: log GPU memory usage between training and rollout.
        torch.cuda.empty_cache()
        log_gpu_memory_usage(f'[actor_rollout_ref] {tag}', logger=logger)


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id, 
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['temperature'] = self.config.rollout.temperature
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                output = self.ulysses_sharding_manager.preprocess_data(output)
                old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')
        log_gpu_memory_usage('DEBUG_MEM At start of compute_ref_log_prob (before ref load)', logger=None)

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

        torch.cuda.empty_cache()
        log_gpu_memory_usage('DEBUG_MEM After compute_ref_log_prob offload (should show cleanup)', logger=None)
        return output


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, \
            FullStateDictConfig, FullOptimStateDictConfig
        
        model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

        # With world_size==1 + LoRA the module is not FSDP-wrapped (see
        # init_model); its state_dict is already full and needs no gather.
        from contextlib import nullcontext
        _fsdp = isinstance(self.actor.actor_module, FSDP)
        _ctx = (FSDP.state_dict_type(self.actor.actor_module,
                                     StateDictType.FULL_STATE_DICT,
                                     model_cfg, optim_cfg)
                if _fsdp else nullcontext())
        with _ctx:
            state_dict = self.actor.actor_module.state_dict()
            # optimizer states are large and only needed for exact resume;
            # saved only if checkpoint_save_optimizer is set.
            if self.config.actor.get('checkpoint_save_optimizer', False):
                optim_state_dict = (
                    FSDP.optim_state_dict(self.actor.actor_module,
                                          self.actor_optimizer)
                    if _fsdp else self.actor_optimizer.state_dict())
            else:
                optim_state_dict = None

            if self.actor_lr_scheduler is not None:
                scheduler_state_dict = self.actor_lr_scheduler.state_dict()
            else:
                scheduler_state_dict = None

        if self.rank == 0:
            model_path = os.path.join(local_path, 'model')
            print(f'Saving actor checkpoint to {model_path}')
            os.makedirs(model_path, exist_ok=True)

            # With LoRA, PeftModel.save_pretrained saves only the adapter
            # weights; the base weights are unchanged from model.path.
            if _is_peft_model(self.actor_module):
                print(f'[LoRA] Adapter-only checkpoint: saving adapter weights to '
                      f'{model_path}; base weights stay at the original model.path.')
            self.actor_module.save_pretrained(model_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(model_path)
            
            optim_dir = os.path.join(local_path, 'optim')
            os.makedirs(optim_dir, exist_ok=True)

            if optim_state_dict is not None:
                torch.save(optim_state_dict, os.path.join(optim_dir, 'optimizer.pt'))

            if scheduler_state_dict is not None:
                torch.save(scheduler_state_dict, os.path.join(optim_dir, 'scheduler.pt'))

            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                'cuda',
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=['dp', 'sp'],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)


    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        model_path = os.path.join(local_path, 'model')
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        self.tokenizer = hf_tokenizer(
            model_path, 
            trust_remote_code=config.model.get('trust_remote_code', False),
        )

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                attn_implementation=config.model.get('attn_implementation', 'flash_attention_2'),
                trust_remote_code=trust_remote_code,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(
            critic_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
        )

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get('betas', (0.9, 0.999)),
            weight_decay=config.optim.get('weight_decay', 1e-2),
        )
        if self._is_offload_optimizer:
            # Pre-allocate the AdamW state on CPU; AdamW otherwise allocates
            # it lazily on the GPU at the first step, before it can be offloaded.
            init_fsdp_optimizer_state_on_cpu(critic_optimizer)

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
        
        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')
        
        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=critic_optimizer,
            num_warmup_steps=num_warmup_steps,
        )
        
        optim_dir = os.path.join(local_path, 'optim')
        opt_path = os.path.join(optim_dir, 'optimizer.pt')
        sched_path = os.path.join(optim_dir, 'scheduler.pt')
        
        if os.path.exists(opt_path):
            if self.rank == 0:
                print(f'Resuming optimizer and scheduler from {optim_dir}')
            
            # Load the full optimizer state dict to CPU on all ranks;
            # each rank extracts its own shard from it.
            full_osd = torch.load(opt_path, map_location='cpu')
            
            # Convert the full dict into this rank's sharded optimizer state dict.
            sharded_osd = FSDP.optim_state_dict_to_load(
                model=critic_module,
                optim=critic_optimizer,
                optim_state_dict=full_osd
            )
            
            # Load the sharded dict into the optimizer.
            critic_optimizer.load_state_dict(sharded_osd)
            
            # Load the scheduler state (small and not sharded).
            if os.path.exists(sched_path):
                full_ssd = torch.load(sched_path, map_location='cpu')
                critic_lr_scheduler.load_state_dict(full_ssd)
            
            if self.rank == 0:
                print(f'Successfully resumed training from step {critic_lr_scheduler.last_epoch}')
        else:
            if self.rank == 0:
                print(f'No optimizer checkpoint found at {opt_path}. Initializing fresh training.')
        
        torch.distributed.barrier()

        return critic_module, critic_optimizer, critic_lr_scheduler


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
            
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(
            config=self.config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)

        torch.cuda.empty_cache()


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        micro_batch_size = self.config.forward_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)

        torch.cuda.empty_cache()
        log_gpu_memory_usage('DEBUG_MEM After compute_values offload + empty_cache', logger=None)
        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )
        # NOTE: the optimizer state is not loaded here. update_critic() only runs
        # backward passes; optimizer.step() happens in optim_step(), which loads
        # and offloads the Adam state itself.

        log_gpu_memory_usage('DEBUG_MEM Before update critic (after critic load)', logger=None)

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            metrics = self.critic.update_critic(data=data)

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        log_gpu_memory_usage('DEBUG_MEM After update critic compute (critic peak, before offload)', logger=None)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)

        # optimizer state was not loaded in this call, so nothing to offload
        torch.cuda.empty_cache()
        log_gpu_memory_usage('DEBUG_MEM After update critic offload + empty_cache (final cleanup state)', logger=None)
        output = output.to('cpu')
        return output
    
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optim_zero_grad(self):
        self.critic.critic_optimizer.zero_grad()
    
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optim_step(self):
        # The Adam state lives on CPU between steps; load it for the step.
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())
        log_gpu_memory_usage('Before critic optim_step', logger=logger)
        result = self.critic._optimizer_step()
        log_gpu_memory_usage('After critic optim_step', logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            torch.cuda.empty_cache()
            log_gpu_memory_usage('After critic optim_step offload + empty_cache', logger=logger)
        return result


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def log_memory_snapshot(self, tag: str):
        torch.cuda.empty_cache()
        log_gpu_memory_usage(f'[critic] {tag}', logger=logger)


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.critic_module,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, \
            FullStateDictConfig, FullOptimStateDictConfig
            
        model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.critic.critic_module, StateDictType.FULL_STATE_DICT, 
            model_cfg, optim_cfg,
            ):
            state_dict = self.critic.critic_module.state_dict()
            # optimizer states are saved only if checkpoint_save_optimizer is set
            if self.config.get('checkpoint_save_optimizer', False):
                optim_state_dict = FSDP.optim_state_dict(self.critic.critic_module, self.critic_optimizer)
            else:
                optim_state_dict = None

            if self.critic_lr_scheduler is not None:
                scheduler_state_dict = self.critic_lr_scheduler.state_dict()
            else:
                scheduler_state_dict = None
            
        if self.rank == 0:
            model_path = os.path.join(local_path, 'model')
            print(f'Saving critic checkpoint to {model_path}')
            os.makedirs(model_path, exist_ok=True)
            
            self.critic_module._fsdp_wrapped_module.save_pretrained(model_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(model_path)
            
            optim_dir = os.path.join(local_path, 'optim')
            os.makedirs(optim_dir, exist_ok=True)

            if optim_state_dict is not None:
                torch.save(optim_state_dict, os.path.join(optim_dir, 'optimizer.pt'))

            if scheduler_state_dict is not None:
                torch.save(scheduler_state_dict, os.path.join(optim_dir, 'scheduler.pt'))
                
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
