"""
Prompt construction for speedrun RL batches.

A batch = batch_size PUCT draws x rollout_n completions. Each draw is an
initial state sampled from the PUCT buffer:
  * <empty>  -> base.txt (no conditioning)
  * any node -> improve.txt with {previous_plan_code} = the node's
                script and {previous_plan_error} = its grade feedback

Templates are validated at import time with string.Formatter, so a stray
brace raises on import.
"""
from __future__ import annotations
import ast
import os
import re
import string
from dataclasses import dataclass

from speedrun.buffer.puct_buffer import PUCTBuffer, Node, EMPTY_ID

_SPEEDRUN_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                                    "speedrun", "prompts")
ALLOWED_FIELDS = {"previous_plan_code", "previous_plan_error"}


def _load_template(name: str, expect_fields: set[str],
                   prompt_dir: str = _SPEEDRUN_PROMPT_DIR) -> str:
    t = open(os.path.join(prompt_dir, name)).read()
    fields = {f for _, f, _, _ in string.Formatter().parse(t) if f}
    if fields != expect_fields:
        raise ValueError(
            f"{name}: format fields {fields} != expected {expect_fields} "
            )
    return t


BASE = _load_template("base.txt", set(), _SPEEDRUN_PROMPT_DIR)
IMPROVE = _load_template("improve.txt", ALLOWED_FIELDS,
                         _SPEEDRUN_PROMPT_DIR)


@dataclass
class GroupPrompt:
    node_id: str
    prompt: str


def chat_template_wrapper(tokenizer):
    """
    Return a prompt -> prompt function that renders the text as one user
    turn and opens the assistant turn with thinking disabled, or None when
    the tokenizer has no chat template.
    """
    if not getattr(tokenizer, "chat_template", None):
        return None

    def wrap(text: str) -> str:
        msgs = [{"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:          # template without an enable_thinking branch
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
    return wrap


def build_group_prompts(buf: PUCTBuffer, n_draws: int,
                        wrap=None) -> list[GroupPrompt]:
    """
    Sample n_draws initial states (one per batch row) and render a prompt
    for each. Records expansions in the buffer (call once per RL step).

    `wrap` (see chat_template_wrapper) renders the text as a chat turn; when
    None the raw string is used.
    """
    finish = wrap if wrap is not None else (lambda prompt: prompt)
    picks = buf.select(n_draws)
    out = []
    for node in picks:
        buf.record_expansion(node.id)
        if node.id == EMPTY_ID:
            out.append(GroupPrompt(node.id, finish(BASE)))
            continue
        code = open(node.code_path).read()
        fb = node.meta.get("feedback")
        if not fb:
            fb = (f"No error detected. val_bpb={node.val_bpb:.5f} "
                  f"(leaderboard record, not yet re-executed locally)"
                  if node.source == "seed" else "result unavailable")
        out.append(GroupPrompt(node.id, finish(IMPROVE.format(
            previous_plan_code=f"```python\n{code}\n```",
            previous_plan_error=fb))))
    return out


# Opening fence of a code block: ``` optionally followed by any language tag
# (python, py, python3, ...) and then a newline.
RE_FENCE_OPEN = re.compile(r"^```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n", re.M)


def extract_script(response: str) -> str | None:
    """
    Pull the train_gpt.py source out of a policy response: the longest fenced
    block (any language tag) that parses as Python, else the longest block.
    """
    blocks: list[str] = []
    for m in RE_FENCE_OPEN.finditer(response):
        j = response.find("\n```", m.end())
        if j < 0:
            continue                     # unterminated fence: truncated
        code = response[m.end():j].strip("\n")
        if code.strip():
            blocks.append(code)
    if not blocks:
        return None
    parseable = []
    for b in blocks:
        try:
            ast.parse(b)
            parseable.append(b)
        except SyntaxError:
            pass
    return max(parseable or blocks, key=len)
