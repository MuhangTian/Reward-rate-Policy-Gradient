"""
Implementer: turn the policy's change specification into a runnable
train_gpt.py.

The policy proposes what to change; the implementer model writes the code.
The model returns SEARCH/REPLACE blocks that are applied deterministically
with exact matching, so code outside the requested change is never altered.

A patched program must parse (ast.parse) and keep the scored stdout log
format; otherwise the error is fed back and the call is retried.
`Implementation.ok` is False only when no valid program was produced after
`max_attempts`; callers record this separately from policy validity.

Backends, selected by SPEEDRUN_IMPL_BACKEND:

  "gateway"  a model-gateway CLI, invoked per call as
                 $GATEWAY_CLI_BIN exec -m <model> -s read-only --ephemeral \
                     --skip-git-repo-check --color never -o <tmpfile> -
             with the prompt on stdin and an empty temp dir as cwd.
             SYSTEM_INSTRUCTION is prepended to the user prompt; no
             temperature control.

  "api"      any OpenAI-compatible chat-completions endpoint:
                 IMPL_API_BASE  the endpoint's base URL (unset = OpenAI)
                 IMPL_API_KEY   (falls back to OPENAI_API_KEY)
             Deterministic (temperature 0).

Unset, the backend is auto-detected: gateway if the CLI and
~/.gateway/cli/auth.json both exist, else api if a key is set.
SPEEDRUN_IMPLEMENTER_MODEL overrides the model on either backend.
"""
from __future__ import annotations

import ast
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MODEL = os.environ.get("SPEEDRUN_IMPLEMENTER_MODEL", "gpt-5.5")
DEFAULT_BACKEND = os.environ.get("SPEEDRUN_IMPL_BACKEND", "")   # ""=auto
DEFAULT_MAX_ATTEMPTS = int(os.environ.get("SPEEDRUN_IMPLEMENTER_RETRIES", 3))
DEFAULT_TIMEOUT_S = float(os.environ.get("SPEEDRUN_IMPLEMENTER_TIMEOUT_S", 240))
# Reasoning control, sent as
#     extra_body={"thinking": {"type": "disabled" | "enabled"}}
# SPEEDRUN_IMPLEMENTER_THINKING: "disabled" (default) | "enabled" | "omit"
# ("omit" sends no field, for endpoints that reject it).
DEFAULT_THINKING = os.environ.get("SPEEDRUN_IMPLEMENTER_THINKING", "disabled")

# One edit. The SEARCH body must appear verbatim in the parent program.
RE_BLOCK = re.compile(
    r"<<<<<<+\s*SEARCH\s*\n(.*?)\n?={5,}\s*\n(.*?)\n?>>>>>>+\s*REPLACE",
    re.S)

SYSTEM_INSTRUCTION = """\
You implement a single requested change to a PyTorch training script, exactly \
and correctly.

The script trains a GPT-style language model on FineWeb to a target \
validation cross-entropy of 3.28 on 8x NVIDIA A100-SXM4-80GB GPUs (Ampere, \
sm_80), launched under torchrun. It is scored on the TOTAL WALL-CLOCK of the \
whole process, from launch to exit, including compile, warmup, data loading \
and validation, so the implementation must actually run and finish. Ampere \
has no FP8 tensor cores and no FlashAttention-3 (Hopper-only): never \
introduce FP8 paths or FA3, they will crash. bf16 / TF32 are the fast paths; \
FlashAttention-2, SDPA, FlexAttention, Triton and torch.compile are available.

You will be given the current script and a CHANGE SPECIFICATION describing \
what to modify. Implement precisely that change. Do not make unrelated \
edits, do not refactor, do not add comments explaining yourself, and do not \
"improve" anything that was not asked for. Implement the specification's \
mechanism FULLY, even when it needs many edit blocks or new functions / \
kernels; do not water a substantive change down into a hyperparameter tweak.

You are responsible for the change being CORRECT: every name you use must be \
defined, imports you rely on must exist in the file or be added, tensor \
shapes and dtypes must be consistent, and the result must be valid Python \
that runs under torchrun with 8 processes.

FIRST decide whether there is anything to implement. Reply with exactly the \
single token NO_SPEC (nothing else) when the CHANGE SPECIFICATION does not \
contain a concrete, implementable change -- for example when it is empty, is \
only a code fence or a heading, only restates the task or the rules (including a \
CHANGE section that merely repeats the instructions about what a CHANGE \
section should contain), is only a diagnosis or a list of ideas with no \
chosen edit, is cut off before it names a specific edit, or is a \
hyperparameter/constant retune with no mechanism behind it. Never invent, guess or "fill in" a change the \
specification did not make: an edit you improvised is worse than no edit, \
because it is scored as if the specification had asked for it.

When the specification does name a concrete change but part of it is \
ambiguous, implement the most reasonable reading of the stated change. Do \
not substitute a smaller or safer change for the one specified.

Reply ONLY with edit blocks in exactly this format (or the single token \
NO_SPEC as described above):

<<<<<<< SEARCH
(lines copied verbatim from the current script)
=======
(the replacement lines)
>>>>>>> REPLACE

Rules for the blocks:
- The SEARCH text must be copied CHARACTER-FOR-CHARACTER from the script, \
including indentation. It must appear exactly once.
- Include enough surrounding lines to make the SEARCH text unique.
- Emit as many blocks as the change needs. Do not emit a block that changes \
nothing.
- To add new code, SEARCH for an existing anchor line and REPLACE it with \
itself plus the new code.
- Output nothing outside the blocks: no prose, no markdown fences, no \
explanation.

THE STDOUT LOG FORMAT IS LOAD-BEARING AND MUST NOT CHANGE. The run is \
validated by parsing these exact lines:
  step:<step>/<steps> train_time:<ms>ms step_avg:<ms>ms
  step:<step>/<steps> val_loss:<loss> train_time:<ms>ms step_avg:<ms>ms
Nothing may be inserted between the fields, and no field may be renamed, \
reordered or reformatted. If the specification asks you to log something \
extra, print it on a SEPARATE line instead of adding it to either line above. \
A run whose validation line does not match is scored as a total failure even \
if it reached the target. (The printed train_time is informational only; the \
score is the harness-measured wall-clock.)

THE VALIDATION CADENCE IS LOAD-BEARING AND MUST NOT CHANGE. Validation must \
run exactly every 125 training steps (step % 125 == 0) and once at the final \
step, printing the val line each time. Do not change `val_loss_every` (it \
must stay 125), do not add, remove or move validation calls, and do not \
change how the validation loss is computed or printed -- even if the \
specification asks for it. In that case implement the rest of the \
specification and leave the cadence untouched. Likewise do not modify the \
data loading of the train / val token streams. Any change to the cadence or \
the streams makes the run invalid.
"""


@dataclass
class Implementation:
    """Result of turning one change specification into code."""
    code: Optional[str]          # the full new program, or None on failure
    ok: bool
    reason: str                  # "ok" | why it failed
    n_blocks: int = 0
    attempts: int = 0
    latency_s: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    notes: list = field(default_factory=list)


# The scored validation line, as the grader's RE_VAL_FULL sees it.
RE_VAL_ORDER = re.compile(r"val_loss:\{\}(.*?)train_time:", re.S)


def _joined_str_text(node: ast.JoinedStr) -> str:
    """Flatten an f-string to literal text with {} standing in for every
    formatted value. Implicitly concatenated f-strings parse to one
    JoinedStr, so this sees the whole logical line.
    """
    out = []
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            out.append(part.value)
        else:
            out.append("{}")
    return "".join(out)


def log_format_intact(code: str) -> tuple[bool, str]:
    """Reject an implementation that broke the scored stdout format, e.g.
    by inserting a field between val_loss and train_time (which would stop
    speedrun.grader.RE_VAL_FULL from matching).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "log_format_broken:unparseable"
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        text = _joined_str_text(node)
        if "val_loss:" not in text:
            continue
        found = True
        m = RE_VAL_ORDER.search(text)
        if m is None:
            return False, "log_format_broken:train_time missing after val_loss"
        between = m.group(1)
        if between.strip():
            return False, f"log_format_broken:inserted {between.strip()[:40]!r}"
    if not found:
        return False, "log_format_broken:no val_loss print found"
    return True, "ok"


def apply_blocks(source: str, reply: str) -> tuple[Optional[str], int, str]:
    """Apply SEARCH/REPLACE blocks to `source`.

    Exact-match only. A SEARCH body that does not appear verbatim, or
    appears more than once, is a failure.
    """
    blocks = RE_BLOCK.findall(reply)
    if not blocks:
        return None, 0, "no_edit_blocks"
    out = source
    for i, (search, replace) in enumerate(blocks):
        if not search.strip():
            return None, len(blocks), f"empty_search_block_{i}"
        n = out.count(search)
        if n == 0:
            return None, len(blocks), f"search_not_found_block_{i}"
        if n > 1:
            return None, len(blocks), f"search_ambiguous_block_{i}_x{n}"
        out = out.replace(search, replace, 1)
    if out == source:
        return None, len(blocks), "no_change"
    return out, len(blocks), "ok"


def _build_prompt(source: str, spec: str, prior_error: Optional[str]) -> str:
    parts = ["# CURRENT SCRIPT\n```python\n", source, "\n```\n\n",
             "# CHANGE SPECIFICATION\n", spec.strip(), "\n"]
    if prior_error:
        parts += ["\n# YOUR PREVIOUS REPLY COULD NOT BE APPLIED\n",
                  prior_error,
                  "\nRe-read the script above and emit corrected edit blocks. "
                  "Copy the SEARCH text character-for-character from the "
                  "script, including indentation.\n"]
    return "".join(parts)


_client_lock = threading.Lock()
_client = None


def _get_client():
    """Return the process-wide OpenAI-compatible client (thread-safe)."""
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            base = os.environ.get("IMPL_API_BASE")
            key = (os.environ.get("IMPL_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
            if not key:
                raise RuntimeError(
                    "no implementer credentials: set IMPL_API_KEY (and "
                    "IMPL_API_BASE for a non-OpenAI endpoint)")
            _client = OpenAI(base_url=base, api_key=key,
                             timeout=DEFAULT_TIMEOUT_S, max_retries=0)
        return _client


def _chat(client, model: str, prompt: str) -> tuple[str, int, int]:
    """One deterministic chat call; returns (text, in_tokens, out_tokens)."""
    kw = {}
    if DEFAULT_THINKING in ("disabled", "enabled"):
        kw["extra_body"] = {"thinking": {"type": DEFAULT_THINKING}}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_INSTRUCTION},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=16384,
        **kw,
    )
    u = getattr(resp, "usage", None)
    return (resp.choices[0].message.content or "",
            int(getattr(u, "prompt_tokens", 0) or 0),
            int(getattr(u, "completion_tokens", 0) or 0))


def _resolve_gateway_bin() -> str | None:
    """Resolution order: $GATEWAY_CLI_BIN, then gatewayx, then gatewaycli."""
    import shutil
    b = os.environ.get("GATEWAY_CLI_BIN")
    if b and os.path.exists(b):
        return b
    return shutil.which("gatewayx") or shutil.which("gatewaycli")


def _gateway_auth_ok() -> bool:
    return os.path.exists(os.path.expanduser("~/.gateway/cli/auth.json"))


def resolve_backend() -> str:
    """Return 'gateway' or 'api'. Explicit SPEEDRUN_IMPL_BACKEND wins;
    otherwise gateway when both the CLI and its auth file exist, else api.
    Raises when neither is configured."""
    want = DEFAULT_BACKEND.strip().lower()
    if want in ("gateway", "api"):
        return want
    if want:
        raise RuntimeError(f"SPEEDRUN_IMPL_BACKEND={want!r} (want gateway|api)")
    if _resolve_gateway_bin() and _gateway_auth_ok():
        return "gateway"
    if os.environ.get("IMPL_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "api"
    raise RuntimeError(
        "no implementer backend configured: install the gateway CLI "
        "(GATEWAY_CLI_BIN + ~/.gateway/cli/auth.json) or set IMPL_API_KEY "
        "(+ IMPL_API_BASE) for an OpenAI-compatible endpoint")


_STRIP_FENCE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*\n|\n?```\s*$")


def _chat_gateway(model: str, prompt: str,
               timeout_s: float) -> tuple[str, int, int]:
    """One gateway CLI call; returns (text, 0, 0) since the CLI reports no
    token usage. SYSTEM_INSTRUCTION is prepended to the user prompt. Runs in
    an empty temp dir under the read-only sandbox."""
    import subprocess
    import tempfile
    binp = _resolve_gateway_bin()
    if not binp:
        raise RuntimeError("gateway CLI not found (GATEWAY_CLI_BIN/gatewayx/gatewaycli)")
    with tempfile.TemporaryDirectory(prefix="gateway_impl_") as td:
        out_path = os.path.join(td, "out.txt")
        proc = subprocess.run(
            [binp, "exec", "-m", model, "-s", "read-only", "--ephemeral",
             "--skip-git-repo-check", "--color", "never", "-o", out_path,
             "-"],
            input=SYSTEM_INSTRUCTION + "\n\n" + prompt,
            capture_output=True, text=True, timeout=timeout_s, cwd=td)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-300:]
            raise RuntimeError(f"gateway exec rc={proc.returncode}: {tail}")
        try:
            text = open(out_path, errors="replace").read()
        except OSError as e:
            raise RuntimeError(f"gateway wrote no output file: {e}")
    # strip a whole-reply markdown fence, if any
    return _STRIP_FENCE.sub("", text).strip(), 0, 0


NO_SPEC_TOKEN = "NO_SPEC"


def _is_no_spec(reply: str) -> bool:
    """True when the implementer declined because the specification names
    no concrete change. Tolerates fencing / trailing punctuation around the
    token, but a reply that also contains edit blocks is not a decline."""
    text = reply.strip().strip("`").strip()
    if RE_BLOCK.search(reply):
        return False
    return text.rstrip(".").strip() == NO_SPEC_TOKEN


def implement(source: str, spec: str, *, model: str = DEFAULT_MODEL,
              max_attempts: int = DEFAULT_MAX_ATTEMPTS,
              timeout_s: float = DEFAULT_TIMEOUT_S,
              client=None) -> Implementation:
    """Apply `spec` to `source`, returning the full new program.

    Never raises: every failure path returns Implementation(ok=False) with a
    reason.
    """
    backend = resolve_backend() if client is None else "api"
    if backend == "api" and client is None:
        client = _get_client()
    t0 = time.monotonic()
    prior_error = None
    notes: list[str] = []
    in_tok = out_tok = 0

    for attempt in range(1, max_attempts + 1):
        try:
            prompt = _build_prompt(source, spec, prior_error)
            if backend == "gateway":
                reply, i_t, o_t = _chat_gateway(model, prompt, timeout_s)
            else:
                reply, i_t, o_t = _chat(client, model, prompt)
            in_tok += i_t
            out_tok += o_t
        except Exception as e:                      # transport / quota / auth
            notes.append(f"a{attempt}:api:{type(e).__name__}")
            prior_error = None
            time.sleep(min(2 ** attempt, 30))
            continue

        # Nothing to implement: a verdict on the specification, returned
        # without retry.
        if _is_no_spec(reply):
            return Implementation(
                code=None, ok=False, reason="no_spec", n_blocks=0,
                attempts=attempt, latency_s=time.monotonic() - t0,
                in_tokens=in_tok, out_tokens=out_tok,
                notes=notes + [f"a{attempt}:no_spec"])

        new_code, n_blocks, why = apply_blocks(source, reply)
        if new_code is None:
            notes.append(f"a{attempt}:{why}")
            prior_error = f"The edit could not be applied: {why}."
            continue

        try:
            ast.parse(new_code)
        except SyntaxError as e:
            notes.append(f"a{attempt}:syntax:L{e.lineno}")
            prior_error = (f"The patched script does not parse: "
                           f"{e.msg} at line {e.lineno}.")
            continue

        intact, why_fmt = log_format_intact(new_code)
        if not intact:
            notes.append(f"a{attempt}:{why_fmt}")
            prior_error = (
                "The patched script no longer prints the scored validation "
                f"line in the required format ({why_fmt}). The line must read "
                "exactly 'step:<step>/<steps> val_loss:<loss> "
                "train_time:<ms>ms step_avg:<ms>ms' with nothing inserted "
                "between the fields. Log any extra value on its own separate "
                "line.")
            continue

        return Implementation(
            code=new_code, ok=True, reason="ok", n_blocks=n_blocks,
            attempts=attempt, latency_s=time.monotonic() - t0,
            in_tokens=in_tok, out_tokens=out_tok, notes=notes)

    return Implementation(
        code=None, ok=False,
        reason=(notes[-1].split(":", 1)[1] if notes else "unknown"),
        attempts=max_attempts, latency_s=time.monotonic() - t0,
        in_tokens=in_tok, out_tokens=out_tok, notes=notes)


def implement_many(items, *, max_workers: int = 8, **kw) -> list:
    """Implement a batch concurrently. `items` is [(source, spec), ...];
    returns Implementations in the same order.
    """
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(lambda it: implement(it[0], it[1], **kw), items))
