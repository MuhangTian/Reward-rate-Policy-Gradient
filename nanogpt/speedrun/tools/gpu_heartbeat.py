"""
Keep GPUs busy while a job waits (e.g. the trainer blocking on the exec
worker, or an idle worker), so clusters that reap GPU-idle jobs leave it alone.
"""
from __future__ import annotations
import os
import sys
import time


def main() -> int:
    qdir = os.environ.get("GOLF_QUEUE_DIR")
    if not qdir:
        print("[heartbeat] no GOLF_QUEUE_DIR; exiting", flush=True)
        return 0
    job = os.environ.get("SLURM_JOB_ID", "local")
    sentinel = os.path.join(qdir, f"heartbeat_on_{job}")
    # GPU_HEARTBEAT_ALWAYS: spin unconditionally; the parent controls the
    # heartbeat by process lifetime (used by the exec worker, which kills it
    # before each timed attempt to free both the SMs and the CUDA context).
    # Otherwise spin only while the sentinel file exists.
    always = os.environ.get("GPU_HEARTBEAT_ALWAYS", "") not in ("", "0")
    dim = int(os.environ.get("GPU_HEARTBEAT_DIM", 4096))
    # Near-saturation duty cycle so sampled utilization reads ~100%; the
    # remaining fraction keeps the process responsive to signals.
    duty = float(os.environ.get("GPU_HEARTBEAT_DUTY", 0.97))
    period = float(os.environ.get("GPU_HEARTBEAT_PERIOD_S", 0.5))

    import torch
    if not torch.cuda.is_available():
        print("[heartbeat] no CUDA; exiting", flush=True)
        return 0
    # Spin on every visible GPU, not just cuda:0.
    n_dev = torch.cuda.device_count()
    mats = []
    for i in range(n_dev):
        d = torch.device(f"cuda:{i}")
        # two dim x dim fp32 matrices per device (~128MB at dim=4096)
        mats.append((torch.randn(dim, dim, device=d),
                     torch.randn(dim, dim, device=d)))
    print(f"[heartbeat] armed: {'ALWAYS (parent-controlled)' if always else sentinel}"
          f" dim={dim} duty={duty} devices={n_dev}", flush=True)

    on = False
    while True:
        if not always and not os.path.exists(sentinel):
            if on:
                print("[heartbeat] idle (sentinel gone)", flush=True)
                on = False
            time.sleep(2.0)
            continue
        if not on:
            print("[heartbeat] spinning (trainer is waiting on the graders)",
                  flush=True)
            on = True
        # burn `duty` of each period, then yield the SMs back
        t_end = time.monotonic() + period * duty
        while time.monotonic() < t_end:
            for _ in range(20):
                for a, b in mats:      # every device, every pass
                    a.matmul(b)
        for i in range(n_dev):
            torch.cuda.synchronize(i)
        time.sleep(period * (1.0 - duty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
