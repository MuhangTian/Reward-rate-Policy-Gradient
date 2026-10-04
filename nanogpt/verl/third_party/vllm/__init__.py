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

from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version, InvalidVersion


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)

# Minimum vllm version supported by the `vllm_modern` shim (needs the external_launcher
# backend, LLM.apply_model and LLM.sleep/wake_up).
_MODERN_VLLM_MIN_VERSION = '0.10.0'

if package_version == '0.3.1':
    vllm_version = '0.3.1'
    from .vllm_v_0_3_1.llm import LLM
    from .vllm_v_0_3_1.llm import LLMEngine
    from .vllm_v_0_3_1 import parallel_state
elif package_version == '0.4.2':
    vllm_version = '0.4.2'
    from .vllm_v_0_4_2.llm import LLM
    from .vllm_v_0_4_2.llm import LLMEngine
    from .vllm_v_0_4_2 import parallel_state
elif package_version == '0.5.4':
    vllm_version = '0.5.4'
    from .vllm_v_0_5_4.llm import LLM
    from .vllm_v_0_5_4.llm import LLMEngine
    from .vllm_v_0_5_4 import parallel_state
elif package_version == '0.6.3':
    vllm_version = '0.6.3'
    from .vllm_v_0_6_3.llm import LLM
    from .vllm_v_0_6_3.llm import LLMEngine
    from .vllm_v_0_6_3 import parallel_state
else:
    # Unrecognized version: use the `vllm_modern` shim if vllm is recent enough.
    try:
        _is_modern_enough = Version(package_version) >= Version(_MODERN_VLLM_MIN_VERSION)
    except (InvalidVersion, TypeError):
        _is_modern_enough = False

    if not _is_modern_enough:
        raise ValueError(
            f'vllm version {package_version} not supported. Currently supported versions are '
            f'0.3.1, 0.4.2, 0.5.4, 0.6.3, and >={_MODERN_VLLM_MIN_VERSION}.'
        )

    vllm_version = package_version
    from .vllm_modern.llm import LLM
    from .vllm_modern import parallel_state
    LLMEngine = None  # not used by the vllm_modern shim; callers go through LLM directly
