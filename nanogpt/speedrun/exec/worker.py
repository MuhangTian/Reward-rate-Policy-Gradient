"""Serial exec worker: claims attempts from a filesystem queue and runs them
one at a time on this node's GPUs (the official 8-GPU topology is enforced by
giving each attempt a whole node, not by running a single worker).

Queue protocol (no external services; atomic on POSIX rename):
    $GOLF_QUEUE_DIR/pending/<attempt_id>.json   <- trainer enqueues
    $GOLF_QUEUE_DIR/running/<attempt_id>.json   <- worker claims via rename
    $GOLF_QUEUE_DIR/done/<attempt_id>.json      <- worker writes grade result

Attempt file: {"attempt_id": ..., "code": ..., "enqueued_at": ...}
Done file:    GradeResult JSON + {"attempt_id": ...}

GPU hygiene between attempts: after each run the worker kills any process
of ours still holding the exec GPUs and polls nvidia-smi until memory is
clean; a GPU that stays dirty for >GOLF_DIRTY_LIMIT_S (default 300) exits 3,
which scripts/run.sh treats as "restart me".

Run:  bash scripts/run.sh grader
"""
from __future__ import annotations
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from speedrun.grader import grade, feedback_string   # noqa: E402
from speedrun.sandbox_profile import run_attempt_speedrun  # noqa: E402
from speedrun.exec.netqueue import client_from_env    # noqa: E402

POLL_S = 5
DIRTY_LIMIT_S = float(os.environ.get('GOLF_DIRTY_LIMIT_S', 300))
DIRTY_MB = 300
DIRTY_GRACE_S = float(os.environ.get('GOLF_DIRTY_GRACE_S', 120))


def env(name: str, default=None, cast=str):
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise SystemExit(f"missing env {name} (see scripts/run.sh)")
        return default
    return cast(v)


def gpus_clean(gpu_ids: str) -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    want = set(gpu_ids.split(","))
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        if idx in want and int(mem) > DIRTY_MB:
            return False
    return True


BEACON_PREFIX = "worker_alive_"
BEACON_PERIOD_S = 30.0


def start_liveness_beacon(qdir: str, me: str) -> str:
    """
    Touch $GOLF_QUEUE_DIR/worker_alive_<job> every 30s from a daemon
    thread, so a trainer can tell "no worker exists" from "a worker is busy".
    """
    import threading
    net = client_from_env()
    if net is not None:
        # HTTP transport: same 30s cadence, POSTed to the queue server
        # instead of touching a file.
        def beat_net():
            while True:
                try:
                    net.beacon(me)
                except Exception:
                    pass
                time.sleep(BEACON_PERIOD_S)

        try:
            net.beacon(me)
        except Exception:
            pass
        threading.Thread(target=beat_net, daemon=True).start()
        return f"<netqueue beacon {me}>"
    path = os.path.join(qdir, BEACON_PREFIX + me)

    def beat():
        while True:
            try:
                with open(path, "w") as f:
                    f.write(str(time.time()))
            except OSError:
                pass
            time.sleep(BEACON_PERIOD_S)

    with open(path, "w") as f:
        f.write(str(time.time()))
    threading.Thread(target=beat, daemon=True).start()
    return path


def start_idle_heartbeat(python: str):
    env = dict(os.environ, GPU_HEARTBEAT_ALWAYS="1")
    try:
        return subprocess.Popen(
            [python, "-m", "speedrun.tools.gpu_heartbeat"], env=env,
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[worker] could not start idle heartbeat: {e}", flush=True)
        return None


def stop_idle_heartbeat(proc) -> None:
    """
    Kill the idle heartbeat and wait for it, so its CUDA context is gone
    before the attempt starts and before gpus_clean() checks.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass


def live_slurm_jobs() -> set[str] | None:
    """
    Job ids of this user's queued/running SLURM jobs, or None if squeue is
    unavailable (callers then fall back to a time bound).
    """
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    # array/step suffixes: 123_4 and 123.batch both belong to job 123
    return {line.strip().split("_")[0].split(".")[0]
            for line in out.stdout.splitlines() if line.strip()}


def requeue_orphans(qdir: str, me: str, stale_after_s: float) -> None:
    """
    Move attempts left in running/ by a worker that no longer exists back
    to pending.
    """
    live = live_slurm_jobs()
    now = time.time()
    for path in glob.glob(os.path.join(qdir, "running", "*.json")):
        aid = os.path.basename(path)[:-5]
        try:
            owner = (json.load(open(path)) or {}).get("owner_job")
        except (OSError, ValueError):
            owner = None
        age = now - os.path.getmtime(path) if os.path.exists(path) else 0.0
        if owner and owner != me and live is not None and owner in live:
            print(f"[worker] leaving {aid} to live worker job {owner}",
                  flush=True)
            continue
        if owner is None and live is None and age < stale_after_s:
            print(f"[worker] leaving untagged {aid} (age {age:.0f}s < "
                  f"{stale_after_s:.0f}s, squeue unavailable)", flush=True)
            continue
        try:
            os.rename(path, os.path.join(qdir, "pending", aid + ".json"))
            print(f"[worker] requeued orphan {aid} (owner={owner}, "
                  f"age={age:.0f}s)", flush=True)
        except OSError:
            pass


def dirty_report(gpu_ids: str) -> str:
    parts = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
        want = set(gpu_ids.split(","))
        dirty = [(i, m) for i, m in
                 (tuple(x.strip() for x in l.split(","))
                  for l in out.strip().splitlines())
                 if i in want and int(m) > DIRTY_MB]
        parts.append("dirty=" + ",".join(f"gpu{i}:{m}MiB" for i, m in dirty))
    except Exception as e:
        parts.append(f"nvidia-smi query failed: {e}")
    try:
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        parts.append("holders=[" + (procs.replace("\n", " | ") or "none") + "]")
    except Exception as e:
        parts.append(f"compute-apps query failed: {e}")
    return "  ".join(parts)


def _own_pids() -> set[int]:
    """This process and every ancestor; these are never killed."""
    keep, pid = set(), os.getpid()
    for _ in range(64):
        if pid <= 1:
            break
        keep.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return keep


_CUDA_INIT_OOM = re.compile(
    r"(out of memory|OutOfMemoryError)", re.I)
_CUDA_INIT_FRAME = re.compile(
    r"torch\.empty\(1,\s*device=|_cuda_setDevice|torch\.cuda\.set_device|"
    r"init_process_group", re.I)


def _oom_at_cuda_init(feedback: str | None) -> bool:
    """
    True when the attempt ran out of GPU memory while setting up CUDA,
    before its own code ran.
    """
    if not feedback:
        return False
    if not _CUDA_INIT_OOM.search(feedback):
        return False
    return bool(_CUDA_INIT_FRAME.search(feedback))


def _proc_gpu_holders() -> set[int]:
    """
    PIDs in THIS pid namespace holding an /dev/nvidia* handle.
    """
    holders = set()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return holders
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        fddir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fddir):
                try:
                    target = os.readlink(os.path.join(fddir, fd))
                except OSError:
                    continue
                if target.startswith("/dev/nvidia"):
                    holders.add(pid)
                    break
        except OSError:
            continue        # gone, or not ours to inspect
    return holders


def reap_gpu_holders(gpu_ids: str) -> list[int]:
    """
    SIGKILL whatever still holds our GPUs. Returns the pids killed.
    """
    keep = _own_pids()
    out = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        print(f"[worker] holder query failed: {e}", flush=True)
    # Union of both sources: nvidia-smi sees host pids, /proc covers our own
    # pid namespace when nvidia-smi's process table is empty.
    cands = set()
    for line in out.strip().splitlines():
        try:
            cands.add(int(line.strip()))
        except ValueError:
            continue
    smi_n = len(cands)
    cands |= _proc_gpu_holders()
    print(f"[worker] holders: nvidia-smi={smi_n} /proc={len(cands) - smi_n} "
          f"total={len(cands)}", flush=True)
    killed = []
    for pid in sorted(cands):
        if pid in keep or pid <= 1:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass                       # already gone, or not ours to kill
    return killed


def wait_clean(gpu_ids: str) -> bool:
    """
    Wait for the cards to drain; escalate to killing holders before giving
    up. Returns False only if the GPUs are STILL dirty after the kill.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < DIRTY_LIMIT_S:
        if gpus_clean(gpu_ids):
            return True
        time.sleep(5)
    # Passive window expired: say who is holding the cards, then take them.
    print(f"[worker] {dirty_report(gpu_ids)}", flush=True)
    killed = reap_gpu_holders(gpu_ids)
    print(f"[worker] reaped {len(killed)} GPU holder(s): {killed or 'none'}",
          flush=True)
    # Even if nothing was killed, the allocation may still drain during the
    # grace window.
    if not killed:
        print("[worker] no killable holder found; waiting out the grace "
              "window before declaring the cards dirty", flush=True)
    t1 = time.monotonic()
    while time.monotonic() - t1 < DIRTY_GRACE_S:
        if gpus_clean(gpu_ids):
            print(f"[worker] GPUs clean {time.monotonic() - t1:.0f}s after "
                  "reaping holders; continuing", flush=True)
            return True
        time.sleep(5)
    print(f"[worker] still dirty after reaping: {dirty_report(gpu_ids)}",
          flush=True)
    return False


def main():
    qdir = env("GOLF_QUEUE_DIR")
    for sub in ("pending", "running", "done"):
        os.makedirs(os.path.join(qdir, sub), exist_ok=True)
    workdir = os.path.join(env("GOLF_QUEUE_DIR"), "sandbox")
    data_path = env("GOLF_ATTEMPT_DATA_PATH")       # the fineweb10B shards
    n_gpus = env("GOLF_EXEC_GPUS", 8, int)
    train_wall = env("GOLF_WALL_SECONDS", 1500, float)
    wall_scale = env("GOLF_WALL_SCALE", 1.0, float)
    gpu_ids = os.environ.get("GOLF_EXEC_GPU_IDS",
                             ",".join(str(i) for i in range(n_gpus)))
    att_python = env("GOLF_ATTEMPT_PYTHON", sys.executable)
    print(f"[worker] queue={qdir} gpus={gpu_ids} "
          f"wall={train_wall}x{wall_scale}", flush=True)

    me = os.environ.get("SLURM_JOB_ID", "local")
    net = client_from_env()
    if net is not None:
        import socket
        me = f"{me}_{socket.gethostname()}"
    beacon = start_liveness_beacon(qdir, me)
    print(f"[worker] liveness beacon {beacon}", flush=True)
    # Requeue attempts orphaned by a worker that died mid-attempt. The network
    # queue does not need this: the server's claim lease requeues orphans.
    if net is None:
        requeue_orphans(qdir, me, stale_after_s=2 * train_wall)

    hb = None
    import atexit
    atexit.register(lambda: stop_idle_heartbeat(hb))

    while True:
        # FIFO by enqueue time.
        if net is not None:
            # HTTP claim: the server pops FIFO under one lock and holds a
            # lease; no rename race, no stamping needed.
            try:
                task = net.claim(me)
            except Exception as e:
                print(f"[worker] netqueue claim error (will retry): {e}",
                      flush=True)
                task = None
            if task is None:
                if hb is None or hb.poll() is not None:
                    hb = start_idle_heartbeat(att_python)
                time.sleep(POLL_S)
                continue
            stop_idle_heartbeat(hb)
            hb = None
            aid = task["attempt_id"]
            dst = None
        else:
            pend = [p for _, p in sorted(
                ((os.path.getmtime(p), p)
                 for p in glob.glob(os.path.join(qdir, "pending", "*.json"))))]
            if not pend:
                # nothing to grade: keep the GPUs busy with the idle heartbeat
                if hb is None or hb.poll() is not None:
                    hb = start_idle_heartbeat(att_python)
                time.sleep(POLL_S)
                continue
            # work to do: stop the heartbeat first so the timed attempt gets
            # the GPUs to itself and gpus_clean() sees a clean card
            stop_idle_heartbeat(hb)
            hb = None
            src = pend[0]
            aid = os.path.basename(src)[:-5]
            dst = os.path.join(qdir, "running", aid + ".json")
            try:
                os.rename(src, dst)               # atomic claim
            except OSError:
                continue                          # someone else took it
            task = json.load(open(dst))
            # Stamp the claim so a peer starting up can tell a live attempt
            # from an orphan (requeue_orphans).
            try:
                with open(dst, "w") as f:
                    json.dump({**task, "owner_job": me,
                               "claimed_at": time.time()}, f)
            except OSError:
                pass

        # Pre-flight: never grade on a GPU that is already occupied. Exiting
        # without publishing lets the claim lease requeue the attempt.
        if not gpus_clean(gpu_ids):
            print(f"[worker] GPUs dirty BEFORE {aid}: {dirty_report(gpu_ids)}",
                  flush=True)
            if not wait_clean(gpu_ids):
                print(f"[worker] still dirty pre-flight; exiting 3 WITHOUT "
                      f"grading {aid} -- its lease will requeue it elsewhere",
                      flush=True)
                sys.exit(3)
            print("[worker] cards recovered; proceeding", flush=True)
        print(f"[worker] running {aid} (job {me})", flush=True)
        t0 = time.monotonic()
        try:
            rpath = run_attempt_speedrun(
                task["code"], aid, workdir, data_path=data_path,
                n_gpus=n_gpus, train_wall=train_wall,
                wall_scale=wall_scale, gpu_ids=gpu_ids, python=att_python)
            g = grade(rpath, wall_cap=train_wall)
        except Exception as e:  # worker-side fault, not the attempt's
            g = None
            err = f"worker_error: {type(e).__name__}: {e}"
            print(f"[worker] ERROR {aid}: {err}", flush=True)
        out = {"attempt_id": aid,
               "worker_wall": time.monotonic() - t0}
        if g is None:
            out.update(valid=False, reason="worker_error", reward=0.0,
                       val_bpb=None, exec_time=train_wall,
                       artifact_bytes=None,
                       feedback=err)
        else:
            out.update(json.loads(g.to_json()))
            out["feedback"] = feedback_string(g, train_wall)
            # An OOM during CUDA init means the GPU was already full: mark it
            # as an infrastructure fault rather than charging the policy.
            if not out.get("valid") and _oom_at_cuda_init(out.get("feedback")):
                out["infra_fault"] = True
                print(f"[worker] {aid}: OOM during CUDA init -- marking "
                      "infra_fault (pod state, not the attempt)", flush=True)
        if net is not None:
            try:
                net.publish(aid, out)
            except Exception as e:
                # the claim lease will requeue the attempt
                print(f"[worker] netqueue publish FAILED for {aid}: {e}",
                      flush=True)
        else:
            tmp = os.path.join(qdir, "done", aid + ".json.tmp")
            with open(tmp, "w") as f:
                json.dump(out, f)
            os.rename(tmp, os.path.join(qdir, "done", aid + ".json"))
            os.remove(dst)
        print(f"[worker] done {aid}: valid={out['valid']} "
              f"reward={out['reward']:.4f} bpb={out.get('val_bpb')}",
              flush=True)
        if not wait_clean(gpu_ids):
            print("[worker] GPUs dirty after cleanup window; exiting 3",
                  flush=True)
            sys.exit(3)


if __name__ == "__main__":
    main()
