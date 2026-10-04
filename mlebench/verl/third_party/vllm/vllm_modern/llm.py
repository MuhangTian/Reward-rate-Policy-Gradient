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
Colocated vLLM wrapper for recent (external_launcher-capable) vLLM releases.

Exposes the interface used by `FSDPVLLMShardingManager` / `vLLMRollout`
(`sync_model_weights`, `offload_model_weights`, `generate`, `init_cache_engine`,
`free_cache_engine`).
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

import vllm as _vllm
from vllm import TokensPrompt
from vllm.outputs import RequestOutput

from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer


class LLM:
    """
    Wraps a `vllm.LLM` configured for in-process colocation with an FSDP trainer.
    `model` must be an on-disk checkpoint path; weights are loaded from disk once and
    later updates go through `sync_model_weights` in-process.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        max_model_len: Optional[int] = None,
        load_format: str = "auto",
        **kwargs,
    ) -> None:
        if not isinstance(model, str):
            raise TypeError(
                f"verl's vllm_modern shim requires an on-disk checkpoint path, got {type(model)}. "
                "The old zero-copy live-nn.Module trick isn't supported against vllm's V1 engine; "
                "weight updates after the initial load go through sync_model_weights() in-process."
            )
        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be one of the following: "
                "PreTrainedTokenizer, PreTrainedTokenizerFast, verl.workers.rollout.HybridEngineBaseTokenizer"
            )
        self._tokenizer = tokenizer

        # `load_format` is ignored (weights are always loaded from disk here); kept for
        # call-site compatibility.
        del load_format

        self._llm = _vllm.LLM(
            model=model,
            tokenizer=model,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            skip_tokenizer_init=skip_tokenizer_init,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            # Each Ray actor sees a single GPU, which vllm's custom all-reduce does
            # not support; NCCL is used instead.
            disable_custom_all_reduce=True,
            distributed_executor_backend="external_launcher",
            enable_sleep_mode=True,
            **kwargs,
        )

    def init_cache_engine(self):
        # KV cache allocation is handled by sleep/wake_up; kept for call-site compatibility.
        pass

    def free_cache_engine(self):
        pass

    def wake_up(self) -> None:
        self._llm.wake_up()

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        assert load_format == "hf", (
            f"vllm_modern shim only implements the full/'hf' weight sync path, got load_format={load_format!r}. "
            "It relies on vllm's own load_weights()/weight_loader()s for TP resharding, so it always needs a "
            "full (unsharded) state dict rather than a pre-sharded dtensor."
        )
        weights = dict(actor_weights)
        # With tied embeddings, vllm's model has no separate lm_head parameter.
        if getattr(self._llm.llm_engine.model_config.hf_config, "tie_word_embeddings", False):
            weights.pop("lm_head.weight", None)

        def _load(model):
            # If the vllm model wraps the text model as `language_model` (e.g.
            # Qwen3_5ForConditionalGeneration), prefix the flat causal-LM names to match.
            if hasattr(model, "language_model"):
                weights_list = [(f"language_model.{k}", v) for k, v in weights.items()]
            else:
                weights_list = list(weights.items())
            model.load_weights(weights_list)

        self._llm.apply_model(_load)

    def offload_model_weights(self) -> None:
        self._llm.sleep(level=2)

    def generate(
        self,
        prompts=None,
        sampling_params=None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prompt_token_ids is not None:
            vllm_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
        else:
            vllm_prompts = prompts
        outputs = self._llm.generate(vllm_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        return self._post_process_outputs(outputs)

    # Same as vllm_v_0_6_3/llm.py: returns padded (token_ids, logprobs) tensors.
    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:
            for output in request_output.outputs:
                output_token_ids.append(torch.tensor(output.token_ids))
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprob = []
                    for logprobs_dict, token_id in zip(logprobs_dicts, output.token_ids):
                        logprob.append(logprobs_dict[token_id].logprob)
                    logprobs.append(torch.tensor(logprob))

        pad_token_id = (
            self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        )
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
        return output_token_ids, logprobs
