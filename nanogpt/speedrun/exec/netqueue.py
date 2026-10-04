"""HTTP transport for the trainer <-> grader attempt queue.

The file-based queue (pending/ running/ done/ + os.rename) requires a shared
POSIX filesystem between the trainer and every grader. When the trainer and
the graders run on separate machines without one, the queue speaks HTTP
instead: the trainer runs `python -m speedrun.exec.netqueue serve` and every
side sets

    GOLF_QUEUE_URL=http://<trainer-host>:<port>

Semantics match the file queue:
  * FIFO by (priority_ts, seq): priority_ts is the enqueue wall time, except
    rescore's --priority low, which passes now+86400.
  * Claim is atomic (one lock) and records the worker and claim time.
  * A claim older than lease_s (default 2 x GOLF_WALL_SECONDS) goes back to
    the front of pending with its original priority_ts.
  * done is written before the claim is cleared.
  * Worker liveness: POST /beacon + GET /workers_alive (180 s freshness).

State is in-memory only; the trainer and graders restart together.
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8377
BEACON_FRESH_S = 180.0


def _log(msg: str) -> None:
    print(f"[netqueue] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------

class QueueState:
    """All queue state behind one lock. Methods are the whole protocol."""

    def __init__(self, lease_s: float):
        self.lock = threading.Lock()
        self.lease_s = float(lease_s)
        self._heap: list[tuple[float, int, str]] = []  # (priority_ts, seq, aid)
        self._pending: dict[str, dict] = {}            # aid -> attempt payload
        self._running: dict[str, dict] = {}            # aid -> {attempt, worker_id, claimed_at, priority_ts, seq}
        self._done: dict[str, dict] = {}               # aid -> result payload
        self._beacons: dict[str, float] = {}           # worker_id -> last ts
        self._seq = itertools.count()

    def enqueue(self, attempt: dict, priority_ts: float | None) -> None:
        aid = attempt["attempt_id"]
        ts = float(priority_ts) if priority_ts is not None else time.time()
        with self.lock:
            if aid in self._pending or aid in self._running or aid in self._done:
                return  # idempotent re-enqueue (rescore skips dupes this way)
            self._pending[aid] = attempt
            heapq.heappush(self._heap, (ts, next(self._seq), aid))

    def claim(self, worker_id: str) -> dict | None:
        with self.lock:
            self._reap_locked()
            while self._heap:
                ts, seq, aid = heapq.heappop(self._heap)
                attempt = self._pending.pop(aid, None)
                if attempt is None:
                    continue  # was cancelled or already claimed via requeue
                self._running[aid] = {
                    "attempt": attempt, "worker_id": worker_id,
                    "claimed_at": time.time(), "priority_ts": ts, "seq": seq,
                }
                return attempt
            return None

    def publish(self, aid: str, result: dict) -> None:
        with self.lock:
            self._done[aid] = result       # done first, like the file queue
            self._running.pop(aid, None)
            self._pending.pop(aid, None)   # result for a reaped claim

    def beacon(self, worker_id: str) -> None:
        with self.lock:
            self._beacons[worker_id] = time.time()

    def poll(self, ids: list[str]) -> dict:
        with self.lock:
            self._reap_locked()
            states, results = {}, {}
            for aid in ids:
                if aid in self._done:
                    states[aid] = "done"
                    results[aid] = self._done[aid]
                elif aid in self._running:
                    states[aid] = "running"
                elif aid in self._pending:
                    states[aid] = "pending"
                else:
                    states[aid] = "absent"
            return {"states": states, "results": results,
                    "done_count": len(self._done),
                    "pending_count": len(self._pending),
                    "running_count": len(self._running)}

    def workers_alive(self, max_age_s: float) -> dict:
        now = time.time()
        with self.lock:
            workers = {w: now - ts for w, ts in self._beacons.items()}
        fresh = {w: age for w, age in workers.items() if age <= max_age_s}
        return {"alive": bool(fresh), "workers": workers}

    def _reap_locked(self) -> None:
        """Return expired claims to the front of pending (original priority)."""
        now = time.time()
        expired = [aid for aid, c in self._running.items()
                   if now - c["claimed_at"] > self.lease_s]
        for aid in expired:
            c = self._running.pop(aid)
            self._pending[aid] = c["attempt"]
            heapq.heappush(self._heap, (c["priority_ts"], c["seq"], aid))
            _log(f"lease expired ({self.lease_s:.0f}s) -> requeued {aid} "
                 f"(was on {c['worker_id']})")


def _make_handler(state: QueueState):
    class Handler(BaseHTTPRequestHandler):
        # ThreadingHTTPServer gives each connection its own thread.
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # default logs every request
            pass

        def _reply(self, code: int, payload: dict | None = None):
            body = json.dumps(payload or {}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            path, _, query = self.path.partition("?")
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/health":
                self._reply(200, {"ok": True})
            elif path == "/workers_alive":
                max_age = float(params.get("max_age", BEACON_FRESH_S))
                self._reply(200, state.workers_alive(max_age))
            else:
                self._reply(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError):
                self._reply(400, {"error": "bad json"})
                return
            if self.path == "/enqueue":
                state.enqueue(body["attempt"], body.get("priority_ts"))
                self._reply(200)
            elif self.path == "/claim":
                attempt = state.claim(body.get("worker_id", "unknown"))
                if attempt is None:
                    self._reply(204)
                else:
                    self._reply(200, {"attempt": attempt})
            elif self.path == "/publish":
                state.publish(body["attempt_id"], body["result"])
                self._reply(200)
            elif self.path == "/beacon":
                state.beacon(body.get("worker_id", "unknown"))
                self._reply(200)
            elif self.path == "/poll":
                self._reply(200, state.poll(body.get("ids", [])))
            else:
                self._reply(404, {"error": "unknown path"})

    return Handler


class _V6Server(ThreadingHTTPServer):
    # Bind "::" so IPv6-only hosts work too; an AF_INET6 bind is dual-stack on
    # Linux (bindv6only=0), so IPv4 clients keep working.
    address_family = socket.AF_INET6


def serve(port: int, lease_s: float) -> None:
    state = QueueState(lease_s=lease_s)
    try:
        httpd = _V6Server(("::", port), _make_handler(state))
    except OSError as e:
        _log(f"IPv6 bind failed ({e}); falling back to IPv4")
        httpd = ThreadingHTTPServer(("", port), _make_handler(state))
    _log(f"serving on :{port} lease={lease_s:.0f}s")
    httpd.serve_forever()


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------

class QueueClient:
    """
    Blocking JSON-over-HTTP client with bounded retries.
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0,
                 retries: int = 5, backoff_s: float = 2.0):
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s

    def _request(self, path: str, payload: dict | None = None) -> tuple[int, dict]:
        last: Exception | None = None
        for i in range(self.retries):
            try:
                if payload is None:
                    req = urllib.request.Request(self.base + path)
                else:
                    req = urllib.request.Request(
                        self.base + path, data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    body = r.read()
                    return r.status, (json.loads(body) if body else {})
            except urllib.error.HTTPError as e:
                # HTTP errors are not transient; raise immediately
                raise
            except Exception as e:  # URLError, timeout, ConnectionReset...
                last = e
                time.sleep(self.backoff_s * (2 ** i))
        raise ConnectionError(f"netqueue {path} failed after "
                              f"{self.retries} tries: {last}")

    # -- protocol ----------------------------------------------------------
    def enqueue(self, attempt: dict, priority_ts: float | None = None) -> None:
        self._request("/enqueue",
                      {"attempt": attempt, "priority_ts": priority_ts})

    def claim(self, worker_id: str) -> dict | None:
        code, body = self._request("/claim", {"worker_id": worker_id})
        return body.get("attempt") if code == 200 else None

    def publish(self, attempt_id: str, result: dict) -> None:
        self._request("/publish",
                      {"attempt_id": attempt_id, "result": result})

    def beacon(self, worker_id: str) -> None:
        self._request("/beacon", {"worker_id": worker_id})

    def poll(self, ids: list[str]) -> dict:
        _, body = self._request("/poll", {"ids": list(ids)})
        return body

    def workers_alive(self, max_age_s: float = BEACON_FRESH_S) -> dict:
        _, body = self._request(f"/workers_alive?max_age={max_age_s}")
        return body

    def health(self) -> bool:
        try:
            code, _ = self._request("/health")
            return code == 200
        except Exception:
            return False


def client_from_env() -> QueueClient | None:
    """GOLF_QUEUE_URL set -> network queue client, else None (file queue)."""
    url = os.environ.get("GOLF_QUEUE_URL", "").strip()
    return QueueClient(url) if url else None


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the queue server (trainer pod)")
    s.add_argument("--port", type=int,
                   default=int(os.environ.get("GOLF_QUEUE_PORT", DEFAULT_PORT)))
    s.add_argument("--lease-s", type=float, default=None,
                   help="claim lease; default 2 x GOLF_WALL_SECONDS")
    w = sub.add_parser("wait", help="block until the server answers /health")
    w.add_argument("--url", default=os.environ.get("GOLF_QUEUE_URL", ""))
    w.add_argument("--timeout-s", type=float, default=300.0)
    args = ap.parse_args()

    if args.cmd == "serve":
        lease = args.lease_s
        if lease is None:
            wall = float(os.environ.get("GOLF_WALL_SECONDS", 1500))
            lease = 2.0 * wall
        serve(args.port, lease)
    elif args.cmd == "wait":
        if not args.url:
            raise SystemExit("wait: need --url or GOLF_QUEUE_URL")
        c = QueueClient(args.url, timeout_s=5.0, retries=1)
        deadline = time.time() + args.timeout_s
        while time.time() < deadline:
            if c.health():
                _log(f"server at {args.url} is up")
                return
            time.sleep(3)
        raise SystemExit(f"wait: {args.url} not healthy "
                         f"after {args.timeout_s:.0f}s")


if __name__ == "__main__":
    main()
