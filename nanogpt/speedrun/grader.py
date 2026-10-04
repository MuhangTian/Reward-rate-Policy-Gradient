"""
Grading and reward mapping for modded-nanogpt speedrun attempts
(track_1_short: train to <=3.28 FineWeb val cross-entropy).

The reward measures quality only; speed pressure comes from the shaped
reward s = r - rho*t downstream:

    r = clip((L_REF - best_val_loss) / (L_REF - L_TARGET), 0, 1)
    L_TARGET = 3.28   (the official target; r saturates at 1.0 there)
    L_REF    = 4.20   (weak reference, roughly the llm.c baseline after a
                       few percent of training; a run above it earns 0)

exec_time = total local process wall (compile included) / wall_scale, used
for the reward-rate estimate, not the official leaderboard score. The timed
train ms and the 3.28 crossing are stored as metadata only.

Validity:
    crash | timeout | parse_failure  -> invalid, reward 0
A run that completes and prints a parseable val_loss is valid even if it
never reached 3.28 (partial reward via the map above).

Log-line formats (stdout):
    step:1390/1390 val_loss:3.2775 train_time:79532ms step_avg:57.22ms
    step:1390/1390 train_time:79448ms step_avg:57.16ms      (no val_loss)
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict

L_REF = 4.20
L_TARGET = 3.28
# Credited-margin floor quoted to the policy in the attempt feedback.
# Keep in sync with golf.quality_below_target_floor.
BELOW_TARGET_FLOOR = 3.25
# Enforced validation cadence: a full val line every VAL_EVERY training
# steps (starting at step 0) plus one at the final step.
VAL_EVERY = 125

# full standard validation line: step, val_loss and the timed train ms.
RE_VAL_FULL = re.compile(
    r"step:(\d+)/(\d+) val_loss:([0-9]+\.[0-9]+) "
    r"train_time:([0-9]+(?:\.[0-9]+)?)ms")
# looser: a val_loss with no (or malformed) timing on the line -- still a
# quality readout, but timing_printed stays False for it.
RE_VAL_ANY = re.compile(r"val_loss:([0-9]+\.[0-9]+)")


RE_RANK_PREFIX = re.compile(r"^\[rank\d+\]:\s?", re.M)
# torchrun's epilogue: a per-rank table of exit codes at the end of stderr.
RE_TORCHRUN_EPILOGUE = re.compile(
    r"\n=+\n[^\n]*FAILED\n-+\nFailures:|\nTraceback \(most recent call last\):\n"
    r"(?:[^\n]*\n)*?[^\n]*torch/distributed/(?:run|launcher)")


def useful_stderr(stderr: str, limit: int = 2500) -> str:
    """Extract the part of a failed run's stderr that explains the failure.

    The root-cause traceback belongs to the first rank that died and sits
    near the top of the file; the tail is torchrun's per-rank exit table.
    Returns the first real traceback with [rankN]: prefixes stripped,
    falling back to the tail when there is no traceback.
    """
    if not stderr:
        return ""
    # first traceback that is NOT torchrun's own wrapper frame
    for m in re.finditer(r"^(?:\[rank\d+\]:\s?)?Traceback \(most recent call "
                         r"last\):$", stderr, re.M):
        # take the traceback's full extent (bounded at 200kB)
        block = RE_RANK_PREFIX.sub("", stderr[m.start():m.start() + 200_000])
        # cut at the end of this one traceback
        block = _traceback_only(block)
        if "torch/distributed/launcher/api.py" in block:
            continue                      # torchrun re-raising, not the cause
        return _elide_middle(block.rstrip(), limit)
    return _fallback_error(RE_RANK_PREFIX.sub("", stderr), limit)


def _fallback_error(stderr: str, limit: int) -> str:
    """No traceback header: find the exception line directly.

    A SyntaxError/IndentationError raised while compiling the script prints
    no "Traceback (most recent call last):" header.
    """
    hits = [m for m in RE_EXC_LINE.finditer(stderr)
            if m.start() == 0 or stderr[m.start() - 1] == "\n"]
    # torchrun's ChildFailedError is always last and never the cause
    hits = [m for m in hits
            if not stderr[m.start():].startswith("torch.distributed")] or hits
    if not hits:
        return stderr[-limit:].rstrip()
    m = hits[0]
    end = stderr.find("\n", m.start())
    end = len(stderr) if end < 0 else end
    return stderr[max(0, end - limit):end].rstrip()


RE_CHAIN = re.compile(
    r"^(During handling of the above exception|The above exception was the "
    r"direct cause)")
# re.M: _fallback_error scans a whole file, so '^' must match at every line.
RE_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|"
                         r"Warning|StopIteration|KeyboardInterrupt)\b", re.M)


def _traceback_only(block: str) -> str:
    """Trim to the traceback proper, ending at its exception line.

    Stops at the first line that is not part of a traceback, honouring
    exception chaining.
    """
    out, seen_frame = [], False
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if line.startswith((" ", "\t")) or not line.strip():
            out.append(line)
            seen_frame = seen_frame or line.lstrip().startswith('File "')
            continue
        if line.startswith("Traceback (") or RE_CHAIN.match(line):
            out.append(line)
            continue
        if seen_frame and RE_EXC_LINE.match(line):
            out.append(line)
            # only an explicit chaining marker continues the block; a bare
            # "Traceback (" is a different error (another rank or torchrun)
            nxt = next((l for l in lines[i + 1:i + 4] if l.strip()), "")
            if RE_CHAIN.match(nxt):
                seen_frame = False
                continue
            break
        break                              # unrelated output: stop
    return "\n".join(out).rstrip()


def _elide_middle(block: str, limit: int) -> str:
    """Keep the head and the tail of an over-long traceback: the head has
    the user code frame, the tail names the exception.
    """
    if len(block) <= limit:
        return block
    head, tail = limit // 3, limit - limit // 3
    return (block[:head].rstrip() + "\n    ... "
            + f"{block[head:-tail].count(chr(10))} frames elided ...\n"
            + block[-tail:].lstrip("\n"))


def reward_from_loss(val_loss: float) -> float:
    r = (L_REF - val_loss) / (L_REF - L_TARGET)
    return max(0.0, min(1.0, r))


def parse_run_log(text: str) -> dict:
    """Parse a run's stdout/log text. Returns:
       best_val_loss / final_val_loss  (None if no val_loss printed)
       timing_printed   True iff at least one full standard val line matched
       final_train_ms   timed train ms on the LAST full val line (the final
                        reported time; the official score of a record run)
       final_full_val_loss  val_loss on that same last full line
       target_reached   any val_loss <= L_TARGET
       target_step / target_train_ms  first FULL val line crossing the target
       n_val_lines      count of full val lines
    Numeric-only regexes skip the f-string templates in the source dump at
    the top of the log."""
    full = [(int(m.group(1)), int(m.group(2)), float(m.group(3)),
             float(m.group(4))) for m in RE_VAL_FULL.finditer(text)]
    losses = [float(m.group(1)) for m in RE_VAL_ANY.finditer(text)]
    out = dict(best_val_loss=min(losses) if losses else None,
               final_val_loss=losses[-1] if losses else None,
               timing_printed=bool(full),
               final_train_ms=full[-1][3] if full else None,
               # loss on the same line final_train_ms came from, so loss and
               # time are always read off one complete line
               final_full_val_loss=full[-1][2] if full else None,
               n_val_lines=len(full),
               # (train_time_ms, val_loss) curve from the full val lines,
               # capped in length
               val_trajectory=[[ms, loss] for _s, _t, loss, ms in full[:300]],
               target_reached=bool(losses) and min(losses) <= L_TARGET,
               target_step=None, target_train_ms=None)
    for step, _tot, loss, ms in full:
        if loss <= L_TARGET:
            out["target_step"], out["target_train_ms"] = step, ms
            break
    steps = [step for step, _t, _l, _m in full]
    out["val_steps"] = steps[:300]
    out["val_cadence_ok"] = val_cadence_ok(steps)
    return out


def val_cadence_ok(steps: list) -> bool:
    """True iff the full val lines were printed on the enforced schedule:
    the first at step 0, then exactly every VAL_EVERY steps, with at most
    one trailing off-grid line at the final step. A timed-out run is judged
    on the prefix it printed."""
    if not steps:
        return False
    if steps[0] != 0:
        return False
    for i in range(1, len(steps)):
        gap = steps[i] - steps[i - 1]
        last = (i == len(steps) - 1)
        if gap == VAL_EVERY:
            continue
        if last and 0 < gap < VAL_EVERY:
            continue                        # final-step eval, off-grid
        return False
    return True


@dataclass
class GradeResult:
    valid: bool
    reason: str                    # "ok" or failure class
    reward: float                  # quality map above; 0 for invalid
    val_loss: float | None         # best val_loss of the run
    exec_time: float               # de-scaled TOTAL local wall (floored 0.1)
    official_train_ms: float | None  # final reported timed train ms
    target_reached: bool
    detail: dict

    def to_json(self) -> str:
        d = asdict(self)
        # val_bpb is the generic quality slot read downstream;
        # artifact_bytes is unused here
        d["val_bpb"] = self.val_loss
        d["artifact_bytes"] = None
        return json.dumps(d)


def grade(result_json_path: str, wall_cap: float) -> GradeResult:
    """result_json is written by speedrun.sandbox_profile for one attempt:
       status         "completed" | "crash" | "timeout"
       wall_seconds   de-scaled total local process wall (cap charged in
                      full on timeout)
       best_val_loss / final_train_ms / target_reached / ... (parse_run_log)
       stderr_tail    last lines for the self-improve feedback
    """
    d = json.load(open(result_json_path))
    wall = max(float(d.get("wall_seconds", wall_cap)), 0.1)
    ms = d.get("final_train_ms")
    reached = bool(d.get("target_reached", False))

    def invalid(reason):
        return GradeResult(False, reason, 0.0, d.get("best_val_loss"),
                           wall, ms, reached, d)

    if d.get("status") == "timeout":
        return invalid("timeout")
    if d.get("status") != "completed":
        return invalid("crash")
    loss = d.get("best_val_loss")
    if loss is None:
        return invalid("parse_failure")
    # a completed run that validated on any other schedule is invalid
    if not d.get("val_cadence_ok", True):
        return invalid("val_cadence")
    return GradeResult(True, "ok", reward_from_loss(float(loss)),
                       float(loss), wall, ms, reached, d)


def feedback_string(g: GradeResult, wall_cap: float) -> str:
    """`previous_plan_error` payload for self-improvement prompts."""
    if g.valid:
        ms = (f"{g.official_train_ms:.0f}ms" if g.official_train_ms
              is not None else "NOT PRINTED (standard timing line missing)")
        tgt = "yes" if g.target_reached else "no"
        # report both score components: wall-clock and the loss margin
        # relative to the target (not the saturating reward)
        if g.target_reached:
            margin = (f"{L_TARGET - g.val_loss:.4f} BELOW the 3.28 target "
                      f"(credited down to {BELOW_TARGET_FLOOR:.2f})")
        else:
            margin = (f"{g.val_loss - L_TARGET:.4f} ABOVE the 3.28 target "
                      f"-- the run does not count until it crosses")
        return (f"No error detected. best_val_loss={g.val_loss:.4f}, "
                f"margin {margin}. "
                f"SCORE HAS TWO PARTS, BOTH COUNT: (1) total process "
                f"wall-clock {g.exec_time:.0f}s (cap {wall_cap:.0f}s, lower "
                f"is better) and (2) how far the loss goes below 3.28, worth "
                f"roughly 20s of wall-clock per 0.001. Improving one by "
                f"giving up the other is not progress. printed "
                f"train_time={ms} (informational only).")
    # stderr_error is the extracted traceback; fall back to the raw tail
    err = g.detail.get("stderr_error") or useful_stderr(
        g.detail.get("stderr_tail") or "")
    hint = ""
    if g.reason == "val_cadence":
        hint = (f" The validation schedule was changed: val lines were "
                f"printed at steps {g.detail.get('val_steps')} but the "
                f"harness requires one at step 0, then exactly every "
                f"{VAL_EVERY} steps, plus the final step. The run was "
                f"scored INVALID for that reason alone -- keep "
                f"val_loss_every={VAL_EVERY} and the validation loop exactly "
                f"as in the parent script.")
    return (f"INVALID ({g.reason}). total_wall={g.exec_time:.0f}s"
            f"/{wall_cap:.0f}s best_val_loss={g.val_loss} error: {err}{hint}")
