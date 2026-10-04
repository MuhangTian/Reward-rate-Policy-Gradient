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
Tensor-parallel helpers for `FSDPVLLMShardingManager`, backed by vLLM's public
parallel_state functions. With `distributed_executor_backend="external_launcher"`, vLLM
builds its TP groups over the existing `torch.distributed` world, so no patching is needed.
"""
import torch.distributed

from vllm.distributed import parallel_state as _vllm_ps


def initialize_parallel_state(*args, **kwargs) -> None:
    # No-op: LLM(distributed_executor_backend="external_launcher") already
    # initializes vllm's TP groups as part of engine construction.
    pass


def get_tensor_model_parallel_world_size() -> int:
    return _vllm_ps.get_tensor_model_parallel_world_size()


def get_tensor_model_parallel_rank() -> int:
    return _vllm_ps.get_tensor_model_parallel_rank()


def get_tensor_model_parallel_group():
    return _vllm_ps.get_tp_group().device_group


def get_tensor_model_parallel_src_rank() -> int:
    """Global rank of the tensor-parallel group's rank-0 member."""
    return torch.distributed.get_rank() - get_tensor_model_parallel_rank()
