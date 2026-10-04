"""
GolfPPOTrainer: modded-nanogpt speedrun RL on the fork's RayPPOTrainer.
"""
from __future__ import annotations

import difflib
import json
import os
import time
import uuid

import numpy as np
import torch

from speedrun.reward import (attempt_loss, attempt_valid, reward_constants,
                             rpg_reward, vanilla_reward)
from speedrun.buffer.puct_buffer import PUCTBuffer, Node
from speedrun.exec.netqueue import client_from_env
from speedrun.rl.dataset import (build_group_prompts, extract_script,
                             chat_template_wrapper)

try:  # verl (and its deps) present -> full trainer available.
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    _HAVE_VERL = True
except Exception:  # pure helpers still importable without verl/ray
    RayPPOTrainer = object
    _HAVE_VERL = False


NO_CODE_FEEDBACK = ("INVALID (no_code_block). No terminated ```python fenced "
                    "block found in the response.")


def _golf_cfg(cfg, key, default):
    """cfg may be an OmegaConf node or a plain dict."""
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default)
    except AttributeError:
        v = getattr(cfg, key, default)
    return default if v is None else v


# ---------------- queue side ----------------
def make_attempt_id(step: int, row: int) -> str:
    """Unique attempt id, tagged with the SLURM job id so concurrent runs
    sharing one queue stay distinguishable."""
    job = os.environ.get("SLURM_JOB_ID", "local")
    return f"j{job}_step{step:05d}_r{row:03d}_{uuid.uuid4().hex[:8]}"


NO_SPEC_FEEDBACK = ("INVALID (no_change_spec). The reply did not contain a "
                    "usable CHANGE specification: nothing concrete was "
                    "proposed, so nothing was implemented or run. A reply "
                    "must end with a CHANGE section naming the exact edits.")


class _NoSpec:
    """Sentinel in the `codes` list for a row whose reply named no concrete
    change (blank reply, or the implementer answered NO_SPEC).
    Distinct from None, which means the implementer failed on a real
    specification and is masked out of the loss as an infrastructure fault.
    A NO_SPEC row is the policy's own doing and is scored as invalid."""
    __slots__ = ()

    def __repr__(self):
        return "NO_SPEC"

    def __bool__(self):
        return False


NO_SPEC = _NoSpec()
IMPL_FAILED_FEEDBACK = ("INVALID (implementer_failed). The change "
                        "specification could not be turned into a runnable "
                        "script by the implementer: {why}")


def materialize_specs(specs, parent_codes, *, workers: int = 16) -> tuple[list, dict]:
    """Turn each policy change specification into a full program via the
    implementer (speedrun/implementer.py).

    Returns (codes aligned with specs, stats). A row whose implementation
    failed gets code None, which the caller records as an infrastructure
    failure rather than a policy failure. Blank or empty specifications get
    NO_SPEC.
    """
    from speedrun.implementer import implement_many
    t0 = time.monotonic()
    todo = [(i, p, s) for i, (p, s) in enumerate(zip(parent_codes, specs))
            if p is not None and s and s.strip()]
    codes = [None] * len(specs)
    stats = {"n": len(specs), "attempted": len(todo), "ok": 0,
             "in_tokens": 0, "out_tokens": 0, "reasons": {}, "no_spec": 0}
    # a blank reply is a policy failure: mark it NO_SPEC so it is scored
    for i, (p, s) in enumerate(zip(parent_codes, specs)):
        if p is not None and not (s and s.strip()):
            codes[i] = NO_SPEC
            stats["no_spec"] += 1
    if todo:
        results = implement_many([(p, s) for _, p, s in todo],
                                 max_workers=workers)
        for (i, _, _), r in zip(todo, results):
            stats["in_tokens"] += r.in_tokens
            stats["out_tokens"] += r.out_tokens
            if r.ok:
                codes[i] = r.code
                stats["ok"] += 1
            elif r.reason == "no_spec":
                # no concrete change in the spec: same as a blank reply
                codes[i] = NO_SPEC
                stats["no_spec"] += 1
            else:
                stats["reasons"][r.reason] = stats["reasons"].get(r.reason, 0) + 1
    stats["wall_s"] = time.monotonic() - t0
    return codes, stats


JUDGE_FEEDBACK = (
    "INVALID (pipeline_modified). A reviewer rejected this script before "
    "execution: {reason} The train/validation DATA PIPELINES must not be "
    "modified -- the validation loss must remain the mean cross-entropy "
    "over exactly 10,485,760 unmodified validation tokens, evaluated over "
    "ALL val_steps batches and averaged across ranks, and the printed "
    "train_time must be the genuinely measured elapsed time of the timed "
    "training sections. Batch size, sequence length, attention, "
    "architecture, optimizer and schedule changes are all allowed.")


# Env prefixes that carry experiment-defining knobs. Anything whose name looks
# like a credential is dropped regardless of prefix.
PROVENANCE_PREFIXES = ("GOLF_", "SPEEDRUN_", "SEED_ARCHIVE", "N_GRADERS", "QUALITY_",
                       "ROLLOUT_", "MAX_RESP", "MAX_PROMPT", "OBJECTIVE",
                       "BATCH_SIZE", "VLLM_UTIL", "INFER_MICRO", "NGPU")
_SECRET_MARKS = ("KEY", "TOKEN", "SECRET", "PASS", "B64", "CRED", "AUTH")


def provenance_from_env(environ, repo_root: str | None = None) -> dict:
    """The run's provenance for the wandb config: the checked-out commit and
    every experiment-defining environment knob, credentials excluded.

    The commit is read from SPEEDRUN_COMMIT if set, else `git rev-parse HEAD`
    in `repo_root`, else None."""
    repo_root = repo_root or environ.get("SPEEDRUN_REPO_ROOT") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    commit = environ.get("SPEEDRUN_COMMIT")
    if not commit:
        try:
            import subprocess
            commit = subprocess.check_output(
                ["git", "-C", repo_root, "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=10).strip() or None
        except Exception:  # noqa: BLE001 -- provenance must never fail a run
            commit = None
    knobs = {}
    for k in sorted(environ):
        if not any(k.startswith(p) for p in PROVENANCE_PREFIXES):
            continue
        if any(m in k.upper() for m in _SECRET_MARKS):
            continue
        knobs[k] = str(environ[k])
    return {"commit": commit, "env": knobs}


def enqueue_attempts(responses: list[str], queue_dir: str, step: int,
                     codes: list | None = None,
                     judge_verdicts: list | None = None
                     ) -> tuple[list[str], dict]:
    """Enqueue one attempt per response for the exec worker.

    `codes` (delta pipeline) supplies the already-materialized program for
    each row; None means the row could not be materialized. When omitted,
    the program is extracted from the response itself.

    Returns (attempt_ids aligned with responses, immediate_results):
    rows with no program are not enqueued; they get a synthetic invalid
    done-record so downstream treats them uniformly."""
    pending_dir = os.path.join(queue_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)
    # archive the raw response next to the extracted script
    resp_dir = os.path.join(
        os.environ.get("GOLF_ARCHIVE_DIR", queue_dir), "responses")
    try:
        os.makedirs(resp_dir, exist_ok=True)
    except OSError:
        resp_dir = None
    attempt_ids, immediate = [], {}
    for i, resp in enumerate(responses):
        aid = make_attempt_id(step, i)
        attempt_ids.append(aid)
        if resp_dir:                      # never fail the step over logging
            try:
                with open(os.path.join(resp_dir, aid + ".txt"), "w") as f:
                    f.write(resp)
            except OSError:
                pass
        if codes is not None:
            code = codes[i]
            # No concrete change proposed: a policy failure, scored invalid
            # and kept in the loss. See _NoSpec.
            if code is NO_SPEC:
                immediate[aid] = {
                    "attempt_id": aid, "valid": False,
                    "reason": "no_change_spec", "infra_fault": False,
                    "reward": 0.0, "val_bpb": None, "exec_time": 0.1,
                    "artifact_bytes": None,
                    "feedback": NO_SPEC_FEEDBACK,
                }
                continue
            # An implementer failure is an infrastructure fault: infra_fault
            # masks the row out of the loss and the batch statistics.
            if code is None:
                immediate[aid] = {
                    "attempt_id": aid, "valid": False,
                    "reason": "implementer_failed", "infra_fault": True,
                    "reward": 0.0, "val_bpb": None, "exec_time": 0.1,
                    "artifact_bytes": None,
                    "feedback": IMPL_FAILED_FEEDBACK.format(why="see run log"),
                }
                continue
        else:
            code = extract_script(resp)
            if code is None:
                immediate[aid] = {
                    "attempt_id": aid, "valid": False,
                    "reason": "no_code_block",
                    "reward": 0.0, "val_bpb": None, "exec_time": 0.1,
                    "artifact_bytes": None, "feedback": NO_CODE_FEEDBACK,
                }
                continue
        # Judge gate (GOLF_JUDGE=1): a script the pipeline judge rejected is
        # invalidated before execution. This is a policy failure: it is scored
        # as invalid and the judge's reason is returned in the feedback.
        jv = judge_verdicts[i] if judge_verdicts is not None else None
        if jv is not None and not jv.ok:
            immediate[aid] = {
                "attempt_id": aid, "valid": False,
                "reason": "pipeline_modified",
                "reward": 0.0, "val_bpb": None, "exec_time": 0.1,
                "artifact_bytes": None,
                "judge": {"verdict": jv.verdict, "reason": jv.reason},
                "feedback": JUDGE_FEEDBACK.format(reason=jv.reason.rstrip(".") + "."),
            }
            continue
        attempt = {"attempt_id": aid, "code": code,
                   "enqueued_at": time.time()}
        net = client_from_env()
        if net is not None:
            net.enqueue(attempt)          # HTTP transport (no shared FS)
        else:
            tmp = os.path.join(pending_dir, aid + ".json.tmp")
            with open(tmp, "w") as f:
                json.dump(attempt, f)
            os.rename(tmp, os.path.join(pending_dir, aid + ".json"))
    return attempt_ids, immediate


WORKER_BEACON_PREFIX = "worker_alive_"
# Consecutive polls an attempt must be absent from pending/, running/ and
# done/ before it is declared lost. At poll_s=5 this is a ~60s window, sized
# for cross-node filesystem visibility delays on a shared filesystem.
VANISH_STRIKES = 12


def workers_alive(queue_dir: str, max_age_s: float = 180.0) -> int:
    """How many exec workers have touched their liveness beacon recently
    (speedrun/exec/worker.py:start_liveness_beacon, 30s period).
    Returns 0 when no worker is alive or none has ever written a beacon.
    """
    net = client_from_env()
    if net is not None:
        try:
            info = net.workers_alive(max_age_s)
        except Exception:
            return 0                      # unreachable server == no workers
        return sum(1 for age in info.get("workers", {}).values()
                   if age <= max_age_s)
    now = time.time()
    n = 0
    try:
        names = os.listdir(queue_dir)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(WORKER_BEACON_PREFIX):
            continue
        try:
            if now - os.path.getmtime(os.path.join(queue_dir, name)) <= max_age_s:
                n += 1
        except OSError:
            pass
    return n


def _infra_fault(aid: str, reason: str, exec_time: float) -> dict:
    """Done-record for an attempt the infrastructure failed to grade (no live
    worker / worker died mid-attempt), shaped like a worker done-file.

    `infra_fault` marks the row non-gradable: apply_group_results excludes it
    from the group statistics and gives it advantage 0 (the `gradable` mask).
    Faults are counted in train/golf/n_infra_fault.
    """
    return {"attempt_id": aid, "valid": False, "reason": reason,
            "reward": 0.0, "val_bpb": None, "exec_time": exec_time,
            "artifact_bytes": None, "infra_fault": True,
            "feedback": f"INVALID ({reason}). Not graded: exec-worker fault."}


def collect_done(queue_dir: str, attempt_ids: list[str], immediate: dict,
                 poll_s: float = 5.0, log_every_s: float = 60.0,
                 claim_timeout_s: float = None,
                 run_timeout_s: float = None) -> dict:
    """Block until every attempt has a result, with a bounded wait.

    Two separate bounds distinguish "no worker" from "slow attempt":
      * still in pending/ (never claimed) for claim_timeout_s  -> no live
        worker. Default 300s: a live worker claims within POLL_S=5s.
      * claimed into running/ but no done-file within run_timeout_s -> the
        worker died mid-attempt. Default 3x GOLF_WALL_SECONDS, since the
        worker's own wall bounds a healthy attempt.
    Env overrides: GOLF_CLAIM_TIMEOUT_S, GOLF_RUN_TIMEOUT_S.
    """
    if claim_timeout_s is None:
        claim_timeout_s = float(os.environ.get("GOLF_CLAIM_TIMEOUT_S", 300))
    wall = float(os.environ.get("GOLF_WALL_SECONDS", 1500))
    if run_timeout_s is None:
        run_timeout_s = float(os.environ.get("GOLF_RUN_TIMEOUT_S", 3 * wall))

    done_dir = os.path.join(queue_dir, "done")
    pend_dir = os.path.join(queue_dir, "pending")
    run_dir = os.path.join(queue_dir, "running")
    results = dict(immediate)
    waiting = [a for a in attempt_ids if a not in results]
    # Arm the GPU heartbeat for the duration of this wait (speedrun/tools/
    # gpu_heartbeat.py polls the sentinel); removed in the finally below so
    # it never contends with generation or the update.
    _hb = os.path.join(queue_dir,
                       f"heartbeat_on_{os.environ.get('SLURM_JOB_ID', 'local')}")
    try:
        open(_hb, "w").close()
    except OSError:
        pass
    try:
        return _collect_done_loop(
            attempt_ids, results, waiting, done_dir, pend_dir, run_dir,
            poll_s, log_every_s, claim_timeout_s, run_timeout_s, wall)
    finally:
        try:
            os.remove(_hb)
        except OSError:
            pass


def _collect_done_loop(attempt_ids, results, waiting, done_dir, pend_dir,
                       run_dir, poll_s, log_every_s, claim_timeout_s,
                       run_timeout_s, wall) -> dict:
    t0 = last_log = time.monotonic()
    claimed_at: dict[str, float] = {}     # aid -> when we first saw running/
    faults = 0
    qdir = os.path.dirname(done_dir)
    vanish_strikes: dict[str, int] = {}   # aid -> consecutive absent polls
    no_worker_since = None                # monotonic ts of the last beacon
    n_alive = -1
    # Liveness is judged by queue progress: any new done-file (for this batch
    # or any other) or a live beacon counts as progress.
    def _done_count() -> int:
        try:
            return len(os.listdir(done_dir))
        except OSError:
            return -1
    net = client_from_env()
    snap_states: dict = {}
    snap_results: dict = {}
    last_progress = t0
    last_done_n = -1 if net is not None else _done_count()
    while waiting:
        now = time.monotonic()
        alive = workers_alive(qdir)
        if alive != n_alive:
            print(f"[golf] exec workers alive: {alive}", flush=True)
            n_alive = alive
        if net is not None:
            try:
                snap = net.poll(list(waiting))
            except Exception as e:
                # transient server errors do not fault attempts; the
                # claim/run timeouts still bound a dead server
                print(f"[golf] netqueue poll error (will retry): {e}",
                      flush=True)
                time.sleep(poll_s)
                continue
            snap_states = snap.get("states", {})
            snap_results = snap.get("results", {})
            n_done = snap.get("done_count", -1)
        else:
            n_done = _done_count()
        if n_done != last_done_n:
            last_done_n = n_done
            last_progress = now
        if alive:
            last_progress = now
        no_worker_since = None if alive else (no_worker_since or now)
        for aid in list(waiting):
            if net is not None:
                # server state is atomic; the VANISH_STRIKES debounce still
                # covers a server restart mid-poll
                s = snap_states.get(aid, "absent")
                if s == "done":
                    results[aid] = snap_results[aid]
                    waiting.remove(aid)
                    vanish_strikes.pop(aid, None)
                elif s == "running":
                    claimed_at.setdefault(aid, now)
                    if now - claimed_at[aid] > run_timeout_s:
                        print(f"[golf] FAULT {aid}: claimed but no result "
                              f"after {now - claimed_at[aid]:.0f}s "
                              f"(> {run_timeout_s:.0f}s) -- worker died "
                              f"mid-attempt", flush=True)
                        results[aid] = _infra_fault(aid, "worker_died", wall)
                        waiting.remove(aid)
                        faults += 1
                elif s == "absent" and aid not in claimed_at:
                    vanish_strikes[aid] = vanish_strikes.get(aid, 0) + 1
                    if vanish_strikes[aid] >= VANISH_STRIKES:
                        print(f"[golf] FAULT {aid}: absent from the queue "
                              f"server on {VANISH_STRIKES} consecutive polls",
                              flush=True)
                        results[aid] = _infra_fault(aid, "attempt_lost", wall)
                        waiting.remove(aid)
                        faults += 1
                elif s == "pending" and aid not in claimed_at \
                        and now - t0 > claim_timeout_s \
                        and now - last_progress > claim_timeout_s:
                    print(f"[golf] FAULT {aid}: unclaimed after "
                          f"{now - t0:.0f}s and the queue has not moved for "
                          f"{now - last_progress:.0f}s "
                          f"(workers_alive={alive})", flush=True)
                    results[aid] = _infra_fault(aid, "worker_absent", wall)
                    waiting.remove(aid)
                    faults += 1
                else:
                    vanish_strikes.pop(aid, None)
                continue
            p = os.path.join(done_dir, aid + ".json")
            if os.path.exists(p):
                results[aid] = json.load(open(p))
                waiting.remove(aid)
                continue
            # not done yet: classify by which queue dir holds it
            if os.path.exists(os.path.join(run_dir, aid + ".json")):
                claimed_at.setdefault(aid, now)
                if now - claimed_at[aid] > run_timeout_s:
                    print(f"[golf] FAULT {aid}: claimed but no result after "
                          f"{now - claimed_at[aid]:.0f}s (> {run_timeout_s:.0f}s) "
                          f"-- worker died mid-attempt", flush=True)
                    results[aid] = _infra_fault(aid, "worker_died", wall)
                    waiting.remove(aid)
                    faults += 1
            elif aid not in claimed_at and \
                    not os.path.exists(os.path.join(pend_dir, aid + ".json")):
                # Absent from all three dirs, which can also happen briefly
                # while the worker renames between them. Re-read done/ first
                # and require the absence to persist for several polls.
                p_done = os.path.join(done_dir, aid + ".json")
                if os.path.exists(p_done):
                    results[aid] = json.load(open(p_done))
                    waiting.remove(aid)
                    vanish_strikes.pop(aid, None)
                    continue
                vanish_strikes[aid] = vanish_strikes.get(aid, 0) + 1
                if vanish_strikes[aid] >= VANISH_STRIKES:
                    print(f"[golf] FAULT {aid}: absent from pending/running/"
                          f"done on {VANISH_STRIKES} consecutive polls",
                          flush=True)
                    results[aid] = _infra_fault(aid, "attempt_lost", wall)
                    waiting.remove(aid)
                    faults += 1
            elif aid not in claimed_at and now - t0 > claim_timeout_s \
                    and now - last_progress > claim_timeout_s:
                # Unclaimed for long enough and the whole queue has been
                # frozen for that stretch (no beacon and no new done-file).
                # A draining queue keeps refreshing last_progress.
                print(f"[golf] FAULT {aid}: unclaimed after {now - t0:.0f}s "
                      f"and the queue has not moved for "
                      f"{now - last_progress:.0f}s "
                      f"(workers_alive={alive})", flush=True)
                results[aid] = _infra_fault(aid, "worker_absent", wall)
                waiting.remove(aid)
                faults += 1
            else:
                # seen somewhere it belongs: any earlier absence was a race
                vanish_strikes.pop(aid, None)
        if not waiting:
            break
        now = time.monotonic()
        if now - last_log >= log_every_s:
            n_claimed = sum(1 for a in waiting if a in claimed_at)
            print(f"[golf] waiting on {len(waiting)}/{len(attempt_ids)} "
                  f"attempts ({n_claimed} claimed, "
                  f"{len(waiting) - n_claimed} queued) "
                  f"({now - t0:.0f}s elapsed)", flush=True)
            last_log = now
        time.sleep(poll_s)
    if faults:
        print(f"[golf] *** {faults}/{len(attempt_ids)} attempts were NOT "
              f"graded due to exec-worker faults; this step's reward signal "
              f"is contaminated ***", flush=True)
    return results


# ---------------- scoring ----------------
def apply_group_results(batch: dict, done_results: dict, cfg) -> dict:
    """Turn per-attempt grade results into token_level_scores + buffer nodes.

    batch: responses (B, L), attention_mask (B, P+L), attempt_ids and
        group_ids (each row's parent node id), all in row order.
    done_results: attempt_id -> graded result from the exec worker.
    cfg: the golf config; reward_mode "quality" is the reward-rate arm,
        "time_to_target" the vanilla arm.

    Returns the raw reward at each row's last response token. The reward-rate
    arm's time charge is applied in fit(): a valid row pays rho * delta and an
    invalid row rho * delta_max (penalty_time). rho is fitted on rho_reward,
    which is max(0, L_ref - L) without the threshold bonuses.
    """
    attempt_ids = batch["attempt_ids"]
    group_ids = batch["group_ids"]
    B = len(attempt_ids)
    g = reward_constants(cfg)
    rpg = str(_golf_cfg(cfg, "reward_mode", "quality")) == "quality"

    reward = torch.zeros(B)
    rho_reward = torch.zeros(B)
    exec_time = torch.full((B,), 0.1)
    valid = torch.zeros(B, dtype=torch.bool)
    # False only for rows the exec worker never returned a verdict for
    gradable = torch.ones(B, dtype=torch.bool)
    losses = []
    for i, aid in enumerate(attempt_ids):
        d = done_results[aid]
        loss, ok = attempt_loss(d), attempt_valid(d)
        delta = max(float(d.get("exec_time", 0.1)), 0.1)
        reward[i] = (rpg_reward(loss, ok, g) if rpg
                     else vanilla_reward(loss, ok, delta, g))
        rho_reward[i] = (max(0.0, g["L_ref"] - loss)
                         if loss is not None and ok else g["r_invalid"])
        exec_time[i] = delta
        valid[i] = ok
        gradable[i] = not bool(d.get("infra_fault", False))
        losses.append(loss)

    penalty_time = torch.where(valid, exec_time,
                               torch.full_like(exec_time, g["delta_max"]))

    resp_len = batch["responses"].shape[-1]
    resp_mask = batch["attention_mask"][:, -resp_len:]
    last_idx = (resp_mask.sum(-1).long() - 1).clamp(min=0)
    scores = torch.zeros(B, resp_len, dtype=torch.float32)
    scores[torch.arange(B), last_idx] = reward

    nodes = []
    for i, aid in enumerate(attempt_ids):
        d = done_results[aid]
        feedback = d.get("feedback", "")
        if bool(valid[i]) and not d.get("valid", False):
            # graded "INVALID (timeout)" by the worker, but valid at cutoff
            feedback = (f"SCORED AT CUTOFF: hit the {g['delta_max']:.0f}s wall cap "
                        f"before the schedule finished; best val_loss within "
                        f"budget = {d.get('val_bpb')}. A shorter schedule "
                        f"that finishes inside the cap avoids the cutoff.")
        nodes.append({
            "id": aid,
            "parent": group_ids[i],
            "reward": float(reward[i]),
            "quality_reward": float(d.get("reward", 0.0)),
            "val_bpb": losses[i],
            "target_train_ms": (d.get("detail") or {}).get("target_train_ms"),
            "exec_time": float(exec_time[i]),
            "valid": bool(valid[i]),
            "reason": d.get("reason", "ok" if valid[i] else "unknown"),
            "feedback": feedback,
            "infra_fault": bool(d.get("infra_fault", False)),
        })

    crossed = [n for n in nodes if n["valid"] and n["val_bpb"] is not None
               and n["val_bpb"] <= g["L_star"]]
    n_gradable = int(gradable.sum())
    metrics = {
        "train/golf/n_valid": int(valid.sum()),
        "train/golf/valid_share": float(valid.float().mean()),
        "train/golf/n_target_reached": len(crossed),
        "train/golf/reward_mean": float(reward.mean()),
        "train/golf/reward_min": float(reward.min()),
        "train/golf/reward_max": float(reward.max()),
        "train/golf/rho_reward_mean": float(rho_reward.mean()),
        "train/golf/n_infra_fault": B - n_gradable,
        "train/golf/gradable_share": n_gradable / max(B, 1),
        "train/golf/exec_time_mean": float(exec_time.mean()),
    }
    if crossed:
        metrics["train/golf/best_exec_time_crossed"] = float(
            min(n["exec_time"] for n in crossed))
    return dict(token_level_scores=scores, reward=reward, rho_reward=rho_reward,
                exec_time=exec_time, penalty_time=penalty_time,
                valid=valid, gradable=gradable, nodes=nodes, metrics=metrics)


def sequence_metrics(batch, max_response_length: int) -> dict:
    """Response/prompt length metrics for one step's rollouts."""
    resp = batch.batch["responses"]
    resp_len = resp.shape[-1]
    attn = batch.batch["attention_mask"]
    resp_mask = attn[:, -resp_len:]
    rlen = resp_mask.sum(-1).float()
    plen = attn[:, :-resp_len].sum(-1).float()
    return {
        "response_length/mean": float(rlen.mean()),
        "response_length/max": float(rlen.max()),
        "response_length/min": float(rlen.min()),
        # share of rollouts truncated at the response cap
        "response_length/clip_ratio": float(
            (rlen >= max_response_length).float().mean()),
        "response_length/n_empty": int((rlen == 0).sum()),
        "prompt_length/mean": float(plen.mean()),
        "prompt_length/max": float(plen.max()),
    }


def add_attempts_to_buffer(buf: PUCTBuffer, nodes: list[dict],
                           scripts: dict, scripts_dir: str) -> None:
    """Persist each attempt's script and insert it as a buffer Node
    (parent = the group's source node). Unparseable attempts get a
    placeholder file so a later PUCT selection cannot crash on a missing
    code_path.

    Rows the infrastructure never graded are skipped.
    """
    os.makedirs(scripts_dir, exist_ok=True)
    for n in nodes:
        if n.get("infra_fault"):
            continue
        path = os.path.join(scripts_dir, n["id"] + ".py")
        code = scripts.get(n["id"])
        with open(path, "w") as f:
            f.write(code if code is not None
                    else "# (no code block extracted from the response)\n")
        # Edit size against the parent (changed lines of a zero-context
        # unified diff), the quantity the buffer's explore_edit_frac quota
        # keys on. None when there is no code or no readable parent.
        delta_lines = None
        parent = buf.nodes.get(n["parent"]) if n.get("parent") else None
        if code is not None and parent is not None and parent.code_path:
            try:
                with open(parent.code_path) as pf:
                    parent_code = pf.read()
                delta_lines = sum(
                    1 for l in difflib.unified_diff(
                        parent_code.splitlines(), code.splitlines(),
                        n=0, lineterm="")
                    if l[:1] in "+-" and not l.startswith(("+++", "---")))
            except OSError:
                delta_lines = None
        buf.add(Node(
            id=n["id"], parent=n["parent"], code_path=path, source="attempt",
            val_bpb=n["val_bpb"],
            # quality reward, not the mode reward: the buffer applies
            # reward_mode itself (objective_reward)
            reward=float(n.get("quality_reward", n["reward"])),
            exec_time=n["exec_time"], valid=n["valid"],
            meta={"feedback": n["feedback"], "reason": n["reason"],
                  "target_train_ms": n.get("target_train_ms"),
                  "delta_lines": delta_lines},
        ))


# ---------------- rho-hat ----------------
class GolfRhoHat:
    """Reward-rate estimate rho-hat for the relative reward r - rho*t.

    Each batch row is a slot that accumulates its own reward and time:
        accum_reward[i] += reward_i if valid else 0
        accum_time[i]   += exec_time_i
    Every step the online NIW estimator (log-time, forgetting factor 0.3) is
    updated on the accumulated (reward, time) pairs, and rho-hat is the 95th
    percentile of the posterior-predictive batch rates, clamped at 0.
    Invalid rows are charged the same rate (invalid_rate).
    """

    def __init__(self, forgetting: float = 0.3):
        from verl.utils.reward_rate_niw import OnlineNIWRewardRate
        self.niw = OnlineNIWRewardRate(
            forgetting_factor=forgetting, log_time=True,
            pool_samples=False, nu_cap=None)
        self.last = {}
        self.rho = 0.0
        self.invalid_rate = 0.0
        self.accum_reward = None
        self.accum_time = None

    def update(self, reward: torch.Tensor, exec_time: torch.Tensor,
               valid: torch.Tensor) -> float:
        r = reward.detach().cpu().float()
        t = exec_time.detach().cpu().float()
        v = valid.detach().cpu().float()
        if self.accum_reward is None:
            self.accum_reward = torch.zeros(len(r))
            self.accum_time = torch.zeros(len(r))
        self.accum_reward += (r * v).clamp(min=0.0)
        self.accum_time += t
        self.last = self.niw.update(self.accum_reward.numpy(),
                                    self.accum_time.numpy())
        self.rho = max(0.0, float(self.last.get("p95", 0.0)))
        self.invalid_rate = self.rho
        return self.rho


# ---------------- trainer ----------------
class GolfPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer with the reward step replaced by the golf exec queue.

    Reuses the worker plumbing (init_workers, optim_step,
    generate_sequences/update_actor RPCs); replaces the dataloader with
    per-step PUCT-buffer prompts and fit() with the golf loop. reward_fn is
    unused (grading happens in the external exec worker)."""

    def _create_dataloader(self):
        # Prompts are rebuilt from the PUCT buffer every step; there is no
        # parquet dataset. Buffer path defaults to <queue_dir>/buffer.jsonl.
        gcfg = self.config.golf
        self._queue_dir = gcfg.get("queue_dir", None) or os.environ["GOLF_QUEUE_DIR"]
        buffer_path = gcfg.get("buffer_path", None) or os.path.join(
            self._queue_dir, "buffer.jsonl")
        # The buffer ranks parents on this run's objective, so it is built
        # from the same golf.* keys the trainer reads.
        self.puct_buffer = PUCTBuffer(
            buffer_path,
            reward_mode=str(gcfg.get("reward_mode", "quality")),
            allow_empty=bool(gcfg.get("allow_empty_state", False)),
            time_miss_s=float(gcfg.get("timeout", 1500.0)),
            quality_miss_loss=float(gcfg.get("quality_miss_loss", 10.0)),
            quality_cross_bonus=float(gcfg.get("quality_cross_bonus", 0.0)),
            quality_target_loss=float(gcfg.get("quality_target_loss", 3.28)),
            quality_below_target_weight=float(
                gcfg.get("quality_below_target_weight", 0.0)),
            quality_below_target_floor=float(
                gcfg.get("quality_below_target_floor", 3.25)),
            time_below_target_weight=float(
                gcfg.get("time_below_target_weight", 0.0)),
            explore_edit_frac=float(gcfg.get("explore_edit_frac", 0.0)),
            explore_edit_min_lines=int(gcfg.get("explore_edit_min_lines", 20)),
            # reproducible tie-break, see PUCTBuffer._tie
            tie_break_seed=int(gcfg.get("tie_break_seed", 0)),
            # shaped value of an invalid node, as in the reward
            invalid_floor=(
                float(gcfg.get("quality_invalid_reward", -10.0))
                if str(gcfg.get("reward_mode", "quality")) == "quality"
                else -2.0 * float(gcfg.get("timeout", 1500.0))),
        )
        self._scripts_dir = os.path.join(self._queue_dir, "scripts")
        print(f"[golf] queue={self._queue_dir} buffer={buffer_path} "
              f"reward_mode={self.puct_buffer.reward_mode} "
              f"allow_empty={self.puct_buffer.allow_empty} "
              f"stats={self.puct_buffer.stats()}", flush=True)
        print(f"Total training steps: {self.config.trainer.total_training_steps}",
              flush=True)

    def _build_gen_batch(self, group_prompts):
        """Tokenize the group prompts like RLHFDataset (left pad,
        truncation='error')."""
        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask
        from verl import DataProto

        ids, masks = [], []
        for gp in group_prompts:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=gp.prompt,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="error",
            )
            ids.append(input_ids[0])
            masks.append(attention_mask[0])
        input_ids = torch.stack(ids)
        attention_mask = torch.stack(masks)
        position_ids = compute_position_id_with_mask(attention_mask)
        batch = DataProto.from_single_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "group_node_id": np.array([gp.node_id for gp in group_prompts],
                                      dtype=object),
        })
        return batch

    def _save_checkpoint(self):
        # Actor weights + global step. The buffer is an append-only jsonl,
        # durable on every add.
        actor_local_path = os.path.join(
            self.config.trainer.default_local_dir, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_local_path, None)
        with open(os.path.join(actor_local_path, "global_step.txt"), "w") as f:
            f.write(str(self.global_steps))

    def fit(self):
        from verl.utils.tracking import Tracking
        from verl.trainer.ppo.ray_trainer import reduce_metrics, append_to_dict
        from omegaconf import OmegaConf

        # attach the commit and env-only knobs under config.provenance
        wandb_config = OmegaConf.to_container(self.config, resolve=True)
        wandb_config["provenance"] = provenance_from_env(os.environ)
        print(f"[golf] provenance: {wandb_config['provenance']}", flush=True)
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=wandb_config,
        )

        gcfg = self.config.golf
        # batching is configured by batch_size and rollout_n only
        for dead in ("n_groups", "group_size", "norm_group_size"):
            assert gcfg.get(dead, None) is None, (
                f"golf.{dead} is not supported; use golf.batch_size "
                f"(PUCT draws per step) and golf.rollout_n (completions per "
                f"draw). Normalisation is always batch-level.")
        n_draws = int(gcfg.get("batch_size", 32))
        rollout_n = int(gcfg.get("rollout_n", 1))
        # Delta pipeline: the policy writes a change specification and an
        # implementer turns it into the program. Off => the policy writes the
        # whole program itself.
        use_delta = bool(gcfg.get("use_delta_specs", False))
        impl_workers = int(gcfg.get("implementer_workers", 16))
        batch_size = n_draws * rollout_n
        use_rr = bool(gcfg.get("use_reward_rate", True))
        assert self.config.actor_rollout_ref.rollout.n == rollout_n, (
            "rollout.n must equal golf.rollout_n (main_golf sets this)")
        # the buffer must use the same objective as the trainer
        assert self.puct_buffer.reward_mode == \
            str(gcfg.get("reward_mode", "quality")), \
            "PUCT buffer reward_mode disagrees with golf.reward_mode"
        print(f"[golf] arm: use_reward_rate={use_rr} "
              f"reward_mode={gcfg.get('reward_mode', 'quality')} "
              f"batch={batch_size} ({n_draws} PUCT draws x {rollout_n} "
              f"rollouts, batch-level normalisation) "
              f"delta_specs={use_delta}", flush=True)
        for sub in ("pending", "done"):
            os.makedirs(os.path.join(self._queue_dir, sub), exist_ok=True)
        rho_hat = GolfRhoHat()

        self.global_steps = self.start_global_step
        while self.global_steps <= self.config.trainer.total_training_steps:
            metrics = {}
            t_step = time.monotonic()
            print(f"GLOBAL_STEP={self.global_steps}", flush=True)

            group_prompts = build_group_prompts(
                self.puct_buffer, n_draws,
                wrap=chat_template_wrapper(self.tokenizer))
            batch = self._build_gen_batch(group_prompts)
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"])

            # rollout.n == rollout_n, so this returns n_draws*rollout_n rows
            t_gen = time.monotonic()
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            metrics["timing_s/gen"] = time.monotonic() - t_gen

            batch.non_tensor_batch["uid"] = np.array(
                [f"step{self.global_steps}_{uuid.uuid4().hex}"
                 for _ in range(len(batch.batch))], dtype=object)
            batch = batch.repeat(repeat_times=rollout_n, interleave=True)
            batch = batch.union(gen_batch_output)
            # no _balance_batch: apply_group_results relies on row order

            responses = batch.batch["responses"]
            responses_text = self.tokenizer.batch_decode(
                responses, skip_special_tokens=True)

            # ---- delta pipeline -------------------------------------------
            # Each row's parent program plus its change specification go to
            # the implementer, which returns the full patched script.
            if use_delta:
                parent_codes = []
                for nid in batch.non_tensor_batch["group_node_id"]:
                    node = self.puct_buffer.nodes.get(nid)
                    try:
                        parent_codes.append(
                            open(node.code_path).read()
                            if node is not None and node.code_path else None)
                    except OSError:
                        parent_codes.append(None)
                t_impl = time.monotonic()
                codes_list, impl_stats = materialize_specs(
                    responses_text, parent_codes, workers=impl_workers)
                metrics["timing_s/implementer"] = time.monotonic() - t_impl
                metrics["train/impl/ok"] = impl_stats["ok"]
                metrics["train/impl/attempted"] = impl_stats["attempted"]
                metrics["train/impl/ok_share"] = (
                    impl_stats["ok"] / max(impl_stats["attempted"], 1))
                metrics["train/impl/in_tokens"] = impl_stats["in_tokens"]
                metrics["train/impl/out_tokens"] = impl_stats["out_tokens"]
                # blank reply or implementer NO_SPEC; scored invalid
                metrics["train/impl/no_spec"] = impl_stats["no_spec"]
                metrics["train/impl/no_spec_share"] = (
                    impl_stats["no_spec"] / max(impl_stats["n"], 1))
                print(f"[golf] implementer: {impl_stats['ok']}/"
                      f"{impl_stats['attempted']} ok in "
                      f"{impl_stats['wall_s']:.0f}s "
                      f"(in={impl_stats['in_tokens']} "
                      f"out={impl_stats['out_tokens']}) "
                      f"no_spec={impl_stats['no_spec']} "
                      f"reasons={impl_stats['reasons']}", flush=True)
            else:
                codes_list = None

            # ---- pipeline judge (GOLF_JUDGE=1) ---------------------------
            # LLM-as-a-judge on the official "do not modify the train/val
            # data pipelines" rule, run trainer-side between the implementer
            # and the queue. Rejected scripts are never executed.
            judge_verdicts = None
            if os.environ.get("GOLF_JUDGE", "0") == "1" and codes_list:
                from speedrun.judge import judge_codes
                t_j = time.monotonic()
                judge_verdicts = judge_codes(
                    codes_list,
                    workers=int(os.environ.get("GOLF_JUDGE_WORKERS", "8")))
                n_j = sum(1 for v in judge_verdicts if v is not None)
                n_bad = sum(1 for v in judge_verdicts
                            if v is not None and not v.ok)
                n_err = sum(1 for v in judge_verdicts
                            if v is not None and v.verdict == "error")
                metrics["timing_s/judge"] = time.monotonic() - t_j
                metrics["train/judge/judged"] = n_j
                metrics["train/judge/rejected"] = n_bad
                metrics["train/judge/rejected_share"] = n_bad / max(n_j, 1)
                metrics["train/judge/errors"] = n_err
                print(f"[golf] judge: {n_bad}/{n_j} rejected "
                      f"({n_err} transport errors, fail-open) in "
                      f"{metrics['timing_s/judge']:.0f}s", flush=True)

            attempt_ids, immediate = enqueue_attempts(
                responses_text, self._queue_dir, self.global_steps,
                codes=codes_list, judge_verdicts=judge_verdicts)
            if use_delta:
                scripts = {aid: (c if isinstance(c, str) else None)
                           for aid, c in zip(attempt_ids, codes_list)}
            else:
                scripts = {aid: extract_script(txt)
                           for aid, txt in zip(attempt_ids, responses_text)}
            t_wait = time.monotonic()
            results = collect_done(self._queue_dir, attempt_ids, immediate)
            metrics["train/golf/grade_wait_s"] = time.monotonic() - t_wait

            group_ids = list(batch.non_tensor_batch["group_node_id"])
            # number of distinct parents drawn this step
            metrics["train/golf/n_distinct_parents"] = len(set(group_ids))
            out = apply_group_results(
                {"responses": responses,
                 "attention_mask": batch.batch["attention_mask"],
                 "attempt_ids": attempt_ids,
                 "group_ids": group_ids},
                results,
                # the whole golf config; _golf_cfg reads keys with defaults
                gcfg,
            )

            # rho: update on this step's grades, then the surrogate charges
            # this step's batch at the updated rate (see GolfRhoHat)
            rho = rho_hat.update(out["rho_reward"], out["exec_time"],
                                 out["valid"])
            # charge time in the PUCT key only when the arm charges for it
            self.puct_buffer.set_rho(rho if use_rr else 0.0)
            metrics["train/golf/rho"] = rho if use_rr else 0.0
            for k in ("mean", "lo", "hi", "p95", "max"):
                if k in rho_hat.last:
                    metrics[f"train/niw/reward_rate_{k}"] = float(rho_hat.last[k])

            batch.batch["token_level_scores"] = out["token_level_scores"]
            resp_len_now = responses.shape[-1]
            rmask = batch.batch["attention_mask"][:, -resp_len_now:]
            lastc = (rmask.sum(-1).long() - 1).clamp(min=0)
            rows = torch.arange(len(attempt_ids))

            # ---- reward-rate surrogate (rr arm only) ----------------------
            #   r' = r - rho * delta         (valid rows)
            #   r' = r - rho * delta_max     (invalid rows)
            # The vanilla arm applies nothing here: rho is still estimated
            # and logged, never charged.
            if use_rr:
                from verl.trainer.ppo.ray_trainer import (
                    compute_reward_rate_surrogate)
                # penalty_time at the last response token, in the 2D
                # token-level shape the surrogate expects
                pt2d = torch.zeros_like(batch.batch["token_level_scores"])
                pt2d[rows, lastc] = out["penalty_time"]
                v2d = out["valid"].float()
                batch.batch["token_level_scores"] = \
                    compute_reward_rate_surrogate(
                        exec_time_tensor=pt2d,
                        reward_tensor=batch.batch["token_level_scores"],
                        reward_rate=rho,
                        valid_submission_tensor=v2d,
                        penalty_coef=1.0,
                        penalize_invalid_time=True,
                        invalid_reward_rate=rho_hat.invalid_rate,
                    )
            shaped_per_sample = batch.batch["token_level_scores"].sum(-1)
            metrics["train/golf/shaped_mean"] = float(shaped_per_sample.mean())
            metrics["train/golf/shaped_min"] = float(shaped_per_sample.min())
            metrics["train/golf/shaped_max"] = float(shaped_per_sample.max())
            metrics.update(out["metrics"])
            metrics.update(sequence_metrics(
                batch, int(self.config.data.max_response_length)))

            add_attempts_to_buffer(self.puct_buffer, out["nodes"], scripts,
                                   self._scripts_dir)
            for k, v in self.puct_buffer.stats().items():
                if v is not None:
                    metrics[f"train/golf/buffer_{k}"] = v

            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1).tolist()
            batch.meta_info["avg_time"] = 1  # as in RayPPOTrainer.fit()

            resp_len = responses.shape[-1]
            resp_mask = batch.batch["attention_mask"][:, -resp_len:]
            # dp_critic.update_critic selects "exec_time_tensor", so supply
            # it: per-sample exec seconds at the last response token.
            et = torch.zeros_like(resp_mask, dtype=torch.float32)
            et[rows, lastc] = out["exec_time"].to(et.dtype)
            batch.batch["exec_time_tensor"] = et

            # ---- KL penalty on rewards (apply_kl_penalty) ----------------
            # kld = KL(pi_old, pi_ref) per token; token_level_rewards =
            # token_level_scores - kl_coef * kld, kl_coef fixed at
            # algorithm.kl_ctrl.kl_coef. kl-on-loss (actor.use_kl_loss) is off.
            from verl.trainer.ppo.ray_trainer import (apply_kl_penalty,
                                                      compute_advantage)
            t_ref = time.monotonic()
            batch = batch.union(
                self.ref_policy_wg.compute_ref_log_prob(batch))
            metrics["timing_s/ref_log_prob"] = time.monotonic() - t_ref
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl, kl_penalty="kl")
            metrics.update(kl_metrics)

            # ---- GAE with the learned critic -----------------------------
            # values are computed before update_critic (pre-update baseline);
            # compute_gae_advantage_return whitens the advantages over the
            # response mask, the only normalisation in this loop.
            t_val = time.monotonic()
            batch = batch.union(self.critic_wg.compute_values(batch))
            metrics["timing_s/values"] = time.monotonic() - t_val
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
            )
            # Ungraded rows (infra faults) contribute nothing:
            #   advantage 0      -> no actor gradient from this row
            #   return == value  -> no critic error from this row
            ng = ~out["gradable"].to(batch.batch["advantages"].device)
            if bool(ng.any()):
                batch.batch["advantages"][ng] = 0.0
                # values are bf16 while returns are fp32; match dtypes
                batch.batch["returns"][ng] = \
                    batch.batch["values"][ng].to(batch.batch["returns"].dtype)

            # Multi-epoch PPO: repeat the update+step sequence on the same
            # batch. The optimizer step stays inside this loop because the
            # optimizers are stepped from the driver.
            ppo_epochs = int(
                self.config.actor_rollout_ref.actor.get("ppo_epochs", 1) or 1)
            for _ppo_epoch in range(ppo_epochs):
                t_crit = time.monotonic()
                critic_output = self.critic_wg.update_critic(batch)
                metrics["timing_s/update_critic"] = \
                    metrics.get("timing_s/update_critic", 0.0) \
                    + (time.monotonic() - t_crit)
                append_to_dict(
                    metrics,
                    reduce_metrics(critic_output.meta_info["metrics"]))
                append_to_dict(
                    metrics, {"critic/grad_norm": self.critic_wg.optim_step()})
                self.critic_wg.optim_zero_grad()

                t_update = time.monotonic()
                actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics["timing_s/update_actor"] = \
                    metrics.get("timing_s/update_actor", 0.0) \
                    + (time.monotonic() - t_update)
                append_to_dict(metrics,
                               reduce_metrics(actor_output.meta_info["metrics"]))
                metrics = self.optim_step(metrics)
                self.optim_zero_grad()

            if self.config.trainer.save_freq > 0 and \
                    self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

            # step timing and throughput
            metrics["timing_s/step_total"] = time.monotonic() - t_step
            n_tok = int(batch.batch["attention_mask"].sum())
            metrics["perf/total_tokens"] = n_tok
            metrics["perf/tokens_per_s"] = n_tok / max(
                metrics["timing_s/step_total"], 1e-6)

            metrics = reduce_metrics(metrics)
            # commit=True so each wandb row is flushed immediately
            wb = logger.logger.get("wandb")
            if wb is not None:
                # Per-sample step table: one row per batch sample with the
                # delta spec, the materialized program, the grader feedback,
                # wall time, val loss and the (train_time_ms, val_loss)
                # trajectory. Logged uncommitted right before the metrics row
                # so both land in the same step.
                try:
                    import wandb as _wandb
                    cols = ["uid", "parent_id", "qwen_delta_spec",
                            "implementer_code", "error", "exec_time_s",
                            "val_loss", "val_trajectory_ms_loss"]
                    tbl = _wandb.Table(columns=cols)
                    for _i, _aid in enumerate(attempt_ids):
                        _r = results.get(_aid) or {}
                        _det = _r.get("detail") or {}
                        tbl.add_data(
                            _aid,
                            str(group_ids[_i]),
                            responses_text[_i],
                            scripts.get(_aid) or "",
                            _r.get("feedback") or "",
                            float(_r.get("exec_time") or 0.0),
                            _r.get("val_bpb"),
                            json.dumps(_det.get("val_trajectory") or []),
                        )
                    wb.log(data={"table": tbl}, step=self.global_steps,
                           commit=False)
                except Exception as _e:  # noqa: BLE001
                    print(f"[golf] step table logging failed: {_e}",
                          flush=True)
                wb.log(data=metrics, step=self.global_steps, commit=True)
            for _name, _inst in logger.logger.items():
                if _name != "wandb":
                    _inst.log(data=metrics, step=self.global_steps)
            self.global_steps += 1
