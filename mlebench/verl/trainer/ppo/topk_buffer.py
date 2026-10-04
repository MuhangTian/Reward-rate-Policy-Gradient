"""
Top-k reuse buffer for the mle-bench self-improve prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class BufferNode:
    id: str
    code: str           # shown as {previous_plan_code}
    error: str          # shown as {previous_plan_error}
    raw_reward: float   # quality reward only (no bonus or penalties)
    exec_time: float    # penalty-relevant seconds (exec_time_tensor row sum)
    step: int           # global step the solution was generated at
    n_selected: int = 0


class TopKBuffer:

    def __init__(self, use_reward_rate: bool, capacity: int, rho: float = 0.0):
        assert capacity >= 1, capacity
        self.use_reward_rate = bool(use_reward_rate)
        self.capacity = int(capacity)
        self.rho = 0.0
        self.set_rho(rho)
        self.nodes: dict[str, BufferNode] = {}
        self.total_selected = 0

    # ---------------- key ----------------
    def set_rho(self, rho: float):
        """Set the price of time in the ranking key (always 0 when not using
        the reward rate, so the key is the raw reward)."""
        self.rho = max(float(rho), 0.0) if self.use_reward_rate else 0.0

    def key(self, n: BufferNode) -> float:
        return n.raw_reward - self.rho * n.exec_time

    def best_key(self):
        """Best key in the buffer under the current rho, or None when empty."""
        if not self.nodes:
            return None
        return max(self.key(n) for n in self.nodes.values())

    def best_raw_reward(self):
        if not self.nodes:
            return None
        return max(n.raw_reward for n in self.nodes.values())

    # ---------------- ops ----------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def add(self, node: BufferNode):
        if node.id in self.nodes:
            print(f'[TopK buffer] WARNING: duplicate node id {node.id}, skipping.')
            return
        self.nodes[node.id] = node

    def trim(self):
        """Keep only the top `capacity` nodes by the current key (ties broken
        by id)."""
        if len(self.nodes) <= self.capacity:
            return
        ranked = sorted(self.nodes.values(), key=lambda n: (-self.key(n), n.id))
        self.nodes = {n.id: n for n in ranked[:self.capacity]}

    def select(self, k: int) -> list[BufferNode]:
        """The k best nodes by the current key, cycling from the top when
        fewer than k exist, so a k-row batch is always filled."""
        if not self.nodes:
            raise RuntimeError('TopKBuffer.select on an empty buffer -- the '
                               'caller must fall back to base prompts.')
        ranked = sorted(self.nodes.values(), key=lambda n: (-self.key(n), n.id))
        out = [ranked[i % len(ranked)] for i in range(k)]
        for n in out:
            n.n_selected += 1
        self.total_selected += len(out)
        return out

    # ---------------- persistence / logging ----------------
    def state_dict(self) -> dict:
        return {
            'nodes': [asdict(n) for n in self.nodes.values()],
            'total_selected': self.total_selected,
            'use_reward_rate': self.use_reward_rate,
            'capacity': self.capacity,
            'rho': self.rho,
        }

    def load_state_dict(self, sd: dict):
        self.nodes = {rec['id']: BufferNode(**rec) for rec in sd['nodes']}
        self.total_selected = int(sd.get('total_selected', 0))

    def stats(self) -> dict:
        if not self.nodes:
            return {'n_nodes': 0, 'best_raw': 0.0, 'best_key': 0.0,
                    'total_selected': self.total_selected}
        return {
            'n_nodes': len(self.nodes),
            'best_raw': max(n.raw_reward for n in self.nodes.values()),
            'best_key': self.best_key(),
            'total_selected': self.total_selected,
        }
