"""
Fold locally measured seed results back into a seed archive.

Seed records initially carry numbers from their leaderboard logs. This
script re-executes every seed through the ordinary exec worker and replaces
reward, exec_time and feedback with local measurements, so seeds and
generated attempts are scored on the same hardware and scale.

Seeds that fail locally are kept but marked valid=False, which excludes them
from PUCT selection (PUCTBuffer.candidates).

Usage:
    # 1. enqueue: one attempt per seed, id "rescore_<record>"
    python speedrun/rescore_seed_archive.py enqueue --archive DIR [--priority low]
    # 2. once the done-files exist, rewrite the archive
    python speedrun/rescore_seed_archive.py apply --archive DIR --out DIR2
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from speedrun.grader import reward_from_loss                    # noqa: E402

PREFIX = "rescore_"


def _seed_records(buffer_path: str):
    for line in open(buffer_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("_event") == "expand" or r.get("source") != "seed":
            continue
        yield r


def cmd_enqueue(args):
    from speedrun.exec.netqueue import client_from_env
    net = client_from_env()
    # Low priority: pending/ is ordered by mtime, so a future timestamp puts
    # these behind every trainer attempt. The network queue takes the same
    # value as priority_ts.
    stamp = time.time() + 86_400 if args.priority == "low" else time.time()
    n = 0
    if net is not None:
        for r in _seed_records(os.path.join(args.archive, "buffer.jsonl")):
            aid = PREFIX + r["meta"]["record"]
            net.enqueue({"attempt_id": aid,
                         "code": open(r["code_path"]).read(),
                         "enqueued_at": stamp}, priority_ts=stamp)
            n += 1  # server-side enqueue is idempotent on attempt_id
        print(f"enqueued {n} seed re-executions ({args.priority} priority, "
              f"netqueue)")
        return
    qdir = args.queue or os.environ["GOLF_QUEUE_DIR"]
    for r in _seed_records(os.path.join(args.archive, "buffer.jsonl")):
        aid = PREFIX + r["meta"]["record"]
        dst = os.path.join(qdir, "pending", aid + ".json")
        if os.path.exists(dst) or os.path.exists(
                os.path.join(qdir, "done", aid + ".json")):
            continue
        tmp = dst + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"attempt_id": aid, "code": open(r["code_path"]).read(),
                       "enqueued_at": stamp}, f)
        os.rename(tmp, dst)
        os.utime(dst, (stamp, stamp))
        n += 1
    print(f"enqueued {n} seed re-executions ({args.priority} priority)")


def cmd_apply(args):
    from speedrun.exec.netqueue import client_from_env
    net = client_from_env()
    net_results = {}
    if net is not None:
        seeds = list(_seed_records(os.path.join(args.archive, "buffer.jsonl")))
        ids = [PREFIX + r["meta"]["record"] for r in seeds]
        net_results = net.poll(ids).get("results", {})
    else:
        qdir = args.queue or os.environ["GOLF_QUEUE_DIR"]
    out_buf = os.path.join(args.out, "buffer.jsonl")
    if os.path.exists(out_buf):
        raise SystemExit(f"{out_buf} exists; refusing to overwrite")
    os.makedirs(args.out, exist_ok=True)

    rows, missing, kept, dropped = [], 0, 0, 0
    for r in _seed_records(os.path.join(args.archive, "buffer.jsonl")):
        rec = r["meta"]["record"]
        if net is not None:
            d = net_results.get(PREFIX + rec)
            if d is None:
                missing += 1
                if args.require_all:
                    continue
                rows.append(r)      # leave the leaderboard numbers in place
                continue
        else:
            dpath = os.path.join(qdir, "done", PREFIX + rec + ".json")
            if not os.path.exists(dpath):
                missing += 1
                if args.require_all:
                    continue
                rows.append(r)      # leave the leaderboard numbers in place
                continue
            d = json.load(open(dpath))
        det = d.get("detail") or {}
        valid = bool(d.get("valid", False))
        loss = det.get("best_val_loss")
        r = dict(r)
        r["valid"] = valid
        r["exec_time"] = max(float(d.get("exec_time", 0.1)), 0.1)
        r["val_bpb"] = loss
        r["reward"] = reward_from_loss(float(loss)) if loss is not None else 0.0
        m = dict(r["meta"])
        m["target_train_ms"] = det.get("target_train_ms")
        m["local_final_train_ms"] = det.get("final_train_ms")
        m["rescored"] = True
        m["exec_time_estimated"] = False
        m["local_reason"] = d.get("reason")
        # feedback shown to the policy describes the local result
        m["feedback"] = d.get("feedback", "")
        r["meta"] = m
        rows.append(r)
        kept += valid
        dropped += (not valid)

    if args.drop_invalid:
        rows = [r for r in rows if r.get("valid")]
    with open(out_buf, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in rows if r.get("valid")]
    crossed = [r for r in ok if (r["meta"] or {}).get("target_train_ms")]
    print(f"wrote {out_buf}: {len(rows)} seeds "
          f"({len(ok)} run locally, {len(rows) - len(ok)} do not, "
          f"{missing} never re-executed)")
    print(f"  reach the 3.28 target locally: {len(crossed)}")
    if crossed:
        cr = sorted(crossed,
                    key=lambda r: r["meta"]["target_train_ms"])
        print(f"  fastest local crossing: "
              f"{cr[0]['meta']['target_train_ms']/1000:.1f}s "
              f"({cr[0]['meta']['record']})")
        for r in cr[:12]:
            # never print an un-re-executed seed's leaderboard number as if it
            # were a local measurement
            tag = "local " if r["meta"].get("rescored") else "OFFICIAL(not run)"
            print(f"    {r['meta']['target_train_ms']/1000:8.1f}s {tag} "
                  f"wall={r['exec_time']:6.1f}s  {r['meta']['record']}")
    if not ok:
        print("  *** no seed runs locally: the buffer would have no "
              "selectable root and PUCTBuffer.select() will raise ***")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enqueue")
    e.add_argument("--archive", required=True)
    e.add_argument("--queue", default=None)
    e.add_argument("--priority", choices=("low", "normal"), default="low")
    e.set_defaults(func=cmd_enqueue)
    a = sub.add_parser("apply")
    a.add_argument("--archive", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--queue", default=None)
    a.add_argument("--drop-invalid", action="store_true",
                   help="omit locally-failing seeds entirely instead of "
                        "keeping them as unselectable records")
    a.add_argument("--require-all", action="store_true",
                   help="skip seeds with no done-file rather than keeping "
                        "their leaderboard numbers")
    a.set_defaults(func=cmd_apply)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
