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

import os
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

_OLD_PINNED_VLLM_VERSIONS = ('0.3.1', '0.4.2', '0.5.4', '0.6.3')


def _peft_model_or_none(fsdp_module):
    """Return the wrapped PeftModel if the module wraps one (LoRA), else None."""
    try:
        from peft import PeftModel
    except ImportError:
        return None
    # with world_size==1 + LoRA the module is not FSDP-wrapped
    wrapped = getattr(fsdp_module, '_fsdp_wrapped_module', fsdp_module)
    return wrapped if isinstance(wrapped, PeftModel) else None


def _strip_fsdp_key(k: str) -> str:
    """Drop FSDP/checkpoint wrapper segments from a parameter name."""
    return k.replace('_fsdp_wrapped_module.', '').replace('_checkpoint_wrapped_module.', '')


def _strip_peft_key(k: str) -> str:
    """Map a merged PeftModel state-dict key to the plain HF name vLLM expects:
    'base_model.model.model.layers.0...q_proj.base_layer.weight'
    -> 'model.layers.0...q_proj.weight'."""
    if k.startswith('base_model.model.'):
        k = k[len('base_model.model.'):]
    return k.replace('.base_layer.', '.')


def _is_fsdp(m) -> bool:
    return isinstance(m, FSDP)


def _maybe_summon(m, **kw):
    """FSDP.summon_full_params on an FSDP module; a no-op context otherwise."""
    import contextlib
    return FSDP.summon_full_params(m, **kw) if _is_fsdp(m) \
        else contextlib.nullcontext()


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        # Modern vllm versions only support the full ('hf') weight sync path,
        # relying on vllm's own load_weights() for TP resharding.
        if vllm_version not in _OLD_PINNED_VLLM_VERSIONS:
            full_params = True
        self.full_params = full_params
        if not _is_fsdp(self.module):
            # unwrapped single-rank module: state_dict() is already full
            pass
        elif full_params:
            # optional CPU offload of the full gather; needed for params
            # larger than 2^31 bytes (e.g. embeddings of very large models)
            _offload_sd = bool(os.environ.get('VERL_SYNC_OFFLOAD_CPU'))
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig(
                                         offload_to_cpu=_offload_sd))
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        # VERL_SYNC_GATHER_BEFORE_WAKE (off by default): gather the weights
        # while the vllm engine is still asleep, then empty_cache and wake.
        # Useful for very large models whose full params cannot co-reside with
        # an awake engine; the default wakes the engine first.
        _gather_first = bool(os.environ.get('VERL_SYNC_GATHER_BEFORE_WAKE'))
        if vllm_version not in _OLD_PINNED_VLLM_VERSIONS and not _gather_first:
            self.inference_engine.wake_up()

        # LoRA sync: merge the adapters into the base weights, ship the merged
        # tensors through the full-weight 'hf' path, then unmerge. merge/unmerge
        # mutate weights in place, so they run under summon_full_params.
        peft_model = _peft_model_or_none(self.module)

        # VERL_SYNC_VIA_SUMMON (off by default): build the sync dict from
        # named_parameters() inside summon_full_params instead of calling
        # state_dict(), which can omit root-unit params at world_size=1.
        # Costs one transient full-model clone.
        if bool(os.environ.get('VERL_SYNC_VIA_SUMMON')):
            log_gpu_memory_usage('Before summon-sync in sharding manager', logger=logger)
            with _maybe_summon(self.module, writeback=True):
                if peft_model is not None:
                    peft_model.merge_adapter()
                params = {
                    _strip_peft_key(_strip_fsdp_key(k)): v.detach().clone()
                    for k, v in self.module.named_parameters()
                    if '.lora_' not in k
                }
                if peft_model is not None:
                    peft_model.unmerge_adapter()
            log_gpu_memory_usage('After summon-sync in sharding manager', logger=logger)
        else:
            if peft_model is not None:
                with _maybe_summon(self.module, writeback=True):
                    peft_model.merge_adapter()

            log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
            params = self.module.state_dict()
            log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
            if peft_model is not None:
                # Drop the adapter tensors (already merged in) and strip the peft
                # naming so the dict looks like a plain HF checkpoint.
                params = {
                    _strip_peft_key(k): v for k, v in params.items() if '.lora_' not in k
                }
        if vllm_version not in _OLD_PINNED_VLLM_VERSIONS and _gather_first:
            # Return cached blocks to the driver before wake_up(): vllm's
            # cumem allocator needs physical memory outside PyTorch's pool.
            torch.cuda.empty_cache()
            self.inference_engine.wake_up()

        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        if peft_model is not None:
            with _maybe_summon(self.module, writeback=True):
                peft_model.unmerge_adapter()
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        self.inference_engine.offload_model_weights()
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            size=vllm_ps.get_tensor_model_parallel_world_size(),
                                            group=vllm_ps.get_tensor_model_parallel_group(),
                                            dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        broadcast_dict_tensor(data.batch,
                              src=vllm_ps.get_tensor_model_parallel_src_rank(),
                              group=vllm_ps.get_tensor_model_parallel_group())
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
