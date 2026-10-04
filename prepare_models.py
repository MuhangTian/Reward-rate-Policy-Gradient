"""Download Qwen3.5-4B and derive the text-only checkpoint used for training.

    python prepare_models.py <model_dir>

writes
    <model_dir>/Qwen3.5-4B/model        the released checkpoint (critic, vLLM rollout)
    <model_dir>/Qwen3.5-4B-text/model   Qwen3_5ForCausalLM (actor and reference)

The FSDP actor and the vLLM rollout must load the same model class with the
same parameter names. Loading the multimodal checkpoint with
AutoModelForCausalLM resolves to the text-only Qwen3_5ForCausalLM, which is
saved here so both sides load it with standard resolution.
"""
import os
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main(model_dir: str) -> None:
    src = os.path.join(model_dir, "Qwen3.5-4B", "model")
    dst = os.path.join(model_dir, "Qwen3.5-4B-text", "model")
    snapshot_download("Qwen/Qwen3.5-4B", local_dir=src)

    text_cfg = AutoConfig.from_pretrained(src).get_text_config()
    model, info = AutoModelForCausalLM.from_pretrained(
        src, config=text_cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, output_loading_info=True)
    bad = {k: v for k, v in info.items()
           if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if bad:
        sys.exit(f"text-only load does not match the checkpoint: {bad}")
    assert type(model).__name__ == "Qwen3_5ForCausalLM", type(model).__name__
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    print(f"wrote {src} and {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python prepare_models.py <model_dir>")
    main(sys.argv[1])
