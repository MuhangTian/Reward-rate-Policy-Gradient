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
Thin shim for colocating vLLM inference with FSDP training on recent vLLM releases.

Built on vLLM's public APIs rather than vendored internals:

- `distributed_executor_backend="external_launcher"`: each FSDP rank runs its own vLLM
  worker inside the already-initialized `torch.distributed` world.
- `LLM.apply_model(fn)`: calls the model's `load_weights(...)` with a gathered FSDP
  state dict.
- `LLM.sleep(level=2)` / `LLM.wake_up()`: frees GPU memory between training and rollout.
"""
