"""
Sandboxed execution of one modded-nanogpt speedrun attempt
(run_attempt_speedrun, called by speedrun/exec/worker.py).

  * invocation: torchrun with a static loopback rendezvous on train_gpt.py.
  * data: <attempt_cwd>/data/fineweb10B links to the data root and
    DATA_PATH=<attempt_cwd>, covering both cwd-relative and DATA_PATH-based
    record scripts.
  * wall: one external SIGKILL at `train_wall` local seconds.
  * exec_time: total local process wall (compile included) / wall_scale,
    used for the reward-rate estimate, not the official score.
  * GPU memory capped by speedrun/exec/_memcap/sitecustomize.py; optional
    netns isolation.

Output: result JSON consumed by speedrun.grader.grade().
"""
from __future__ import annotations
import glob
import json
import os
import re
import signal
import shutil
import socket
import subprocess
import threading
import time

from speedrun.grader import parse_run_log

_MEMCAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "exec", "_memcap")
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _have_unshare() -> bool:
    try:
        return subprocess.run(["unshare", "-rn", "true"], timeout=10,
                              capture_output=True).returncode == 0
    except Exception:
        return False

# A harness output line ("step:5100/5100 val_loss:..."). Requiring digits on
# both sides of the slash keeps the source dump at the top of the log (which
# contains the f-string template) from matching.
_STEP_LINE = re.compile(r"step:\d+/\d+")


def _startup_heartbeat_deadline(train_wall: float) -> float:
    """How long the startup heartbeat may run before it is killed anyway."""
    return min(float(train_wall), float(
        os.environ.get("GOLF_STARTUP_HEARTBEAT_CAP_S", 900)))


def _kill_hb(hb) -> None:
    """Kill the heartbeat and wait for it, so its CUDA context is released
    before timed training and before the worker's GPU-clean check."""
    if hb is None or hb.poll() is not None:
        return
    try:
        hb.terminate()
        hb.wait(timeout=20)
    except Exception:
        try:
            hb.kill()
            hb.wait(timeout=10)
        except Exception:
            pass


def _start_startup_heartbeat(adir: str, gpu_ids, python: str,
                             train_wall: float):
    """Keep the GPUs busy during an attempt's untimed startup.

    Killed at the first harness `step:N/M` line; everything before that is
    setup, which the harness timer (training_time_ms) excludes.
    GOLF_STARTUP_HEARTBEAT=0 disables this.
    """
    if os.environ.get("GOLF_STARTUP_HEARTBEAT", "1") in ("0", "", "false"):
        return None
    env = dict(os.environ, GPU_HEARTBEAT_ALWAYS="1")
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    try:
        hb = subprocess.Popen(
            [python, "-m", "speedrun.tools.gpu_heartbeat"], env=env,
            cwd=_REPO_ROOT, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[sandbox:speedrun] startup heartbeat failed: {e}", flush=True)
        return None

    cap = _startup_heartbeat_deadline(train_wall)

    def _watch():
        end = time.monotonic() + cap
        while time.monotonic() < end and hb.poll() is None:
            for p in glob.glob(os.path.join(adir, "logs", "*.txt")) + \
                    [os.path.join(adir, "stdout.log")]:
                try:
                    with open(p, "r", errors="ignore") as f:
                        if _STEP_LINE.search(f.read()):
                            _kill_hb(hb)
                            return
                except OSError:
                    pass
            time.sleep(2.0)
        _kill_hb(hb)

    threading.Thread(target=_watch, daemon=True).start()
    return hb


def _plant_data(adir: str, data_path: str) -> None:
    """Make the shards visible to the attempt at cwd-relative data/fineweb10B."""
    os.symlink(data_path, os.path.join(adir, "data", "fineweb10B"))


def prepare_attempt_dir(adir: str, code: str, data_path: str) -> None:
    """Create a fresh attempt dir with the script and the data layout the
    record scripts expect (cwd-relative data/fineweb10B)."""
    if os.path.exists(adir):
        shutil.rmtree(adir)
    os.makedirs(os.path.join(adir, "data"))
    # modded-nanogpt scripts log to logs/<uuid>.txt (some to logs/final/),
    # and the upstream repo ships that directory.
    os.makedirs(os.path.join(adir, "logs", "final"))
    # some record scripts save checkpoints here
    os.makedirs(os.path.join(adir, "verify_ckpt"))
    with open(os.path.join(adir, "train_gpt.py"), "w") as f:
        f.write(code)
    _plant_data(adir, data_path)


def run_attempt_speedrun(code: str, attempt_id: str, workdir: str, *,
                         data_path: str, n_gpus: int, train_wall: float,
                         wall_scale: float = 1.0, gpu_ids: str | None = None,
                         no_net: bool = True, python: str = "python3") -> str:
    """Run one speedrun attempt; returns path to the result JSON."""
    adir = os.path.join(workdir, attempt_id)
    prepare_attempt_dir(adir, code, data_path)

    # GOLF_SANDBOX_NETNS=0 skips the netns; the NCCL interface default
    # depends on it.
    want_netns = no_net and \
        os.environ.get("GOLF_SANDBOX_NETNS", "1") != "0"

    env = dict(os.environ)
    env.update(
        RUN_ID=attempt_id,
        DATA_PATH=adir,               # $DATA_PATH/data/fineweb10B resolves
        OMP_NUM_THREADS="8",
    )
    # NCCL/gloo interface: "lo" inside the netns (the only interface there);
    # outside it, inherit the host's NCCL env. GOLF_NCCL_IFNAME overrides;
    # "" = inherit.
    _ifname = os.environ.get("GOLF_NCCL_IFNAME",
                             "lo" if want_netns else "")
    if _ifname:
        env["NCCL_SOCKET_IFNAME"] = _ifname
        env["GLOO_SOCKET_IFNAME"] = _ifname
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # cap each GPU at the official 80 GB budget (no-op on 80 GB cards).
    mem_gb = float(os.environ.get("GOLF_GPU_MEM_GB", "80"))
    env["GOLF_GPU_MEM_BYTES"] = str(int(mem_gb * (1 << 30)))
    env["PYTHONPATH"] = _MEMCAP_DIR + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Inside a container the netns is usually off: `unshare -r` nests a user
    # namespace that breaks CUDA/NCCL IPC.
    # Master port: a fixed port is safe only inside the private netns;
    # otherwise probe a free port per attempt.
    if want_netns:
        master_port = 29631
    else:
        with socket.socket() as _s:
            _s.bind(("127.0.0.1", 0))
            master_port = _s.getsockname()[1]
    cmd = [python, "-m", "torch.distributed.run", "--nnodes=1",
           "--node-rank=0", "--master-addr=127.0.0.1",
           f"--master-port={master_port}",
           f"--nproc_per_node={n_gpus}", "train_gpt.py"]
    if want_netns and _have_unshare():
        inner = "ip link set lo up 2>/dev/null; exec \"$@\""
        cmd = ["unshare", "-rn", "sh", "-c", inner, "--"] + cmd
    elif no_net:
        print(f"[sandbox:speedrun] WARNING: netns disabled or unshare "
              f"unavailable; {attempt_id} runs WITHOUT network isolation")

    out_path = os.path.join(adir, "stdout.log")
    err_path = os.path.join(adir, "stderr.log")
    deadline = float(train_wall)              # local cap, no scaling
    t0 = time.monotonic()
    rc = None
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        proc = subprocess.Popen(cmd, cwd=adir, env=env, stdout=fo,
                                stderr=fe, start_new_session=True)
        # load the cards through the untimed startup; its watcher kills it at
        # the first `step:N/M` line, before any timed training
        hb = _start_startup_heartbeat(adir, gpu_ids, python, train_wall)
        try:
            rc = proc.wait(timeout=deadline)
            status = "completed" if rc == 0 else "crash"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status = "timeout"
        finally:
            # ensure the heartbeat is gone after a crash/timeout before the
            # first step line
            _kill_hb(hb)
        # Always reap the process group: a rank stuck in NCCL teardown can
        # outlive the torchrun launcher and hold GPU memory. The attempt runs
        # in its own session, so this kills only its processes.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass                      # already gone: the normal case
    wall_total = time.monotonic() - t0

    stdout = open(out_path, errors="replace").read()
    stderr_full = open(err_path, errors="replace").read()
    stderr_tail = stderr_full[-4000:]
    # the root cause is usually the first traceback, not the tail
    from speedrun.grader import useful_stderr
    stderr_error = useful_stderr(stderr_full)
    parsed = parse_run_log(stdout)

    # timeout charges the full cap; t_official = t_local / wall_scale. This
    # is resource occupancy for the rate estimate, not the leaderboard time
    # (parsed["final_train_ms"]).
    wall_local = deadline if status == "timeout" else wall_total
    wall_seconds = wall_local / wall_scale if wall_scale > 0 else wall_local

    result = dict(attempt_id=attempt_id, status=status,
                  wall_seconds=wall_seconds, wall_total_local=wall_total,
                  stderr_tail=stderr_tail, stderr_error=stderr_error,
                  returncode=rc if status != "timeout" else None,
                  **parsed)
    rpath = os.path.join(adir, "result.json")
    with open(rpath, "w") as f:
        json.dump(result, f)
    return rpath
