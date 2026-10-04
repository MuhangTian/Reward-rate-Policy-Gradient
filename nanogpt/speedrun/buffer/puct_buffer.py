"""Top-k reuse buffer for the modded-nanogpt speedrun RL trainer.

Every graded attempt (and every leaderboard seed) is a node:
    {id, parent, code_path, source, val_bpb, reward, exec_time,
     valid, n_expansions, meta}

Selection is plain top-k: select(k) returns the k valid nodes with the
highest shaped value, repeating from the top if fewer than k exist.

The shaped value is the training objective: the arm's reward
(speedrun.reward, recomputed from val_bpb, valid and exec_time so seeds
and attempts share one scale) minus rho * exec_time, where rho is 0 unless
the run charges for time. One seed archive therefore serves both arms.

The empty state <empty> is only selectable when allow_empty=True; its shaped
value is the miss value, so it is only chosen when fewer than k real nodes
exist. Invalid nodes are recorded but never selected.

Persistence: append-only jsonl (one node per line; expansion events are
logged and folded on load). One trainer owns one buffer file; concurrent
writers are not supported.
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from speedrun.reward import rpg_reward, vanilla_reward

EMPTY_ID = "<empty>"


@dataclass
class Node:
    id: str
    parent: Optional[str]
    code_path: str          # "" for <empty>
    source: str             # "seed" | "attempt" | "empty"
    val_bpb: Optional[float]
    reward: float           # quality map in [0,1]; kept for reporting only
    exec_time: float        # seconds of training wall used (0.1 floor)
    valid: bool
    n_expansions: int = 0
    meta: dict = field(default_factory=dict)


class PUCTBuffer:
    """Top-k parent buffer (see module docstring)."""

    def __init__(self, path: str,
                 rho: float = 0.0,
                 reward_mode: str = "quality", allow_empty: bool = True,
                 invalid_floor: float = -1500.0,
                 time_miss_s: float = 1500.0,
                 quality_miss_loss: float = 10.0,
                 quality_cross_bonus: float = 0.0,
                 quality_target_loss: float = 3.28,
                 quality_below_target_weight: float = 0.0,
                 quality_below_target_floor: float = 3.25,
                 time_below_target_weight: float = 0.0,
                 explore_edit_frac: float = 0.0,
                 explore_edit_min_lines: int = 20,
                 tie_break_seed: int = 0):
        """
        rho: price of time in the shaped value, updated per step via
        set_rho(); 0 for time_to_target.
        time_miss_s (delta_max), quality_* and time_below_target_weight: the
        reward constants of speedrun.reward; must match the golf config.
        invalid_floor: shaped value of invalid nodes, on the scale of the
        reward mode.
        """
        assert reward_mode in ("quality", "time_to_target"), reward_mode
        assert 0.0 <= explore_edit_frac <= 1.0, explore_edit_frac
        # Edit-size exploration quota: explore_edit_frac reserves that share
        # of the k parent slots for the best-shaped valid nodes whose edit
        # against their parent is at least explore_edit_min_lines
        # (meta.delta_lines). The objective is unchanged. 0.0 = plain top-k.
        self.explore_edit_frac = float(explore_edit_frac)
        self.explore_edit_min_lines = int(explore_edit_min_lines)
        self.tie_break_seed = int(tie_break_seed)
        self.last_select_explore = 0
        self.last_select_delta_mean = None
        self.path = path
        self.rho = rho
        self.reward_mode = reward_mode
        self.allow_empty = allow_empty
        self.invalid_floor = invalid_floor
        self.time_miss_s = time_miss_s
        self.quality_miss_loss = quality_miss_loss
        self.quality_cross_bonus = quality_cross_bonus
        self.quality_target_loss = quality_target_loss
        self.quality_below_target_weight = quality_below_target_weight
        self.quality_below_target_floor = quality_below_target_floor
        self.time_below_target_weight = time_below_target_weight
        self.nodes: dict[str, Node] = {}
        self.total_expansions = 0
        self._ensure_empty()
        if os.path.exists(path):
            self._load()

    # ---------------- persistence ----------------
    def _ensure_empty(self):
        if EMPTY_ID not in self.nodes:
            self.nodes[EMPTY_ID] = Node(
                id=EMPTY_ID, parent=None, code_path="", source="empty",
                val_bpb=None, reward=0.0, exec_time=0.1, valid=False)

    def _load(self):
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("_event") == "expand":
                    nid = rec["id"]
                    if nid in self.nodes:
                        self.nodes[nid].n_expansions += 1
                    self.total_expansions += 1
                else:
                    rec.pop("_event", None)
                    node = Node(**rec)
                    self.nodes[node.id] = node
        self._ensure_empty()

    def _append(self, rec: dict):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------- core ops ----------------
    def set_rho(self, rho: float):
        self.rho = max(float(rho), 0.0)

    def objective_reward(self, n: Node) -> float:
        """The reward the trainer assigns (speedrun.reward), recomputed from
        node fields so seeds and graded attempts share one scale."""
        g = dict(delta_max=self.time_miss_s,
                 L_star=self.quality_target_loss,
                 window=self.quality_target_loss - self.quality_below_target_floor,
                 L_ref=self.quality_miss_loss,
                 r_invalid=self.invalid_floor,
                 bonus=self.quality_cross_bonus,
                 w_rpg=self.quality_below_target_weight,
                 w_vanilla=self.time_below_target_weight)
        loss = None if n.val_bpb is None else float(n.val_bpb)
        if self.reward_mode == "quality":
            return rpg_reward(loss, n.valid, g)
        return vanilla_reward(loss, n.valid, float(n.exec_time), g)

    def shaped(self, n: Node) -> float:
        if n.id == EMPTY_ID:
            # "start fresh" is worth the miss value, so it is never preferred
            # over a program that crossed
            return -abs(self.time_miss_s)
        if not n.valid:
            return self.invalid_floor
        return self.objective_reward(n) - self.rho * n.exec_time

    def candidates(self) -> list[Node]:
        """Nodes that may be selected as a parent: valid nodes, plus the
        empty node when allow_empty."""
        out = [n for n in self.nodes.values()
               if n.id != EMPTY_ID and n.valid]
        if self.allow_empty:
            out.append(self.nodes[EMPTY_ID])
        return out

    def add(self, node: Node):
        assert node.id not in self.nodes, f"duplicate node id {node.id}"
        self.nodes[node.id] = node
        self._append({"_event": "add", **asdict(node)})

    def _tie(self, n: Node) -> bytes:
        """
        Tie-break key among nodes with equal shaped value: a digest of the
        node id seeded by tie_break_seed, so the order is reproducible but
        uncorrelated with step or batch row (attempt ids sort by both).
        """
        return hashlib.blake2b(
            f"{self.tie_break_seed}:{n.id}".encode(), digest_size=8).digest()

    @staticmethod
    def edit_lines(n: Node):
        """Changed lines of this node's script against its parent's
        (meta.delta_lines); None when not recorded (e.g. seeds)."""
        v = (n.meta or {}).get("delta_lines")
        return None if v is None else int(v)

    def select(self, k: int = 1) -> list[Node]:
        """
        Parents for the next batch: the k best valid nodes by shaped value
        (ties broken by _tie), cycling from the top when there are fewer
        than k candidates.

        With explore_edit_frac > 0, round(k * frac) slots are filled first
        from the same ranking restricted to nodes whose edit against their
        parent is >= explore_edit_min_lines (cycling if there are fewer),
        and the rest from the unrestricted ranking. A node can appear in
        both pools.
        """
        cand = self.candidates()
        if not cand:
            raise RuntimeError(
                f"{self.path}: no selectable node "
                f"(allow_empty={self.allow_empty}, "
                f"{len(self.nodes)} nodes, "
                f"{sum(1 for n in self.nodes.values() if n.valid)} valid). "
                "Seed it from speedrun/seed_archive_rescored_1500s.")
        ranked = sorted(cand, key=lambda n: (-self.shaped(n), self._tie(n)))
        q = int(round(k * self.explore_edit_frac)) if self.explore_edit_frac > 0 else 0
        big = [n for n in ranked
               if (self.edit_lines(n) or 0) >= self.explore_edit_min_lines
               and n.id != EMPTY_ID]
        if q == 0 or not big:
            picks = [ranked[i % len(ranked)] for i in range(k)]
            self.last_select_explore = 0
        else:
            explore = [big[i % len(big)] for i in range(q)]
            rest = [ranked[i % len(ranked)] for i in range(k - q)]
            picks = explore + rest
            self.last_select_explore = q
        deltas = [self.edit_lines(n) for n in picks]
        deltas = [d for d in deltas if d is not None]
        self.last_select_delta_mean = (sum(deltas) / len(deltas)
                                       if deltas else None)
        return picks

    def record_expansion(self, node_id: str):
        """Record that a node was drawn as a parent (bookkeeping only; does
        not affect selection)."""
        n = self.nodes[node_id]
        n.n_expansions += 1
        self.total_expansions += 1
        self._append({"_event": "expand", "id": node_id})

    def best(self) -> Optional[Node]:
        valid = [n for n in self.nodes.values() if n.valid]
        return max(valid, key=self.objective_reward) if valid else None

    def stats(self) -> dict:
        valid = [n for n in self.nodes.values() if n.valid]
        ms = [(n.meta or {}).get("target_train_ms") for n in valid]
        ms = [float(m) for m in ms if m is not None]
        return dict(
            n_nodes=len(self.nodes), n_valid=len(valid),
            n_candidates=len(self.candidates()),
            total_expansions=self.total_expansions,
            best_bpb=min((n.val_bpb for n in valid if n.val_bpb is not None),
                         default=None),
            # fastest target crossing anywhere in the buffer, seeds included
            best_target_ms=min(ms, default=None),
            best_reward=max((self.objective_reward(n) for n in valid),
                            default=None),
            # edit-size exploration (select): slots given to the >= min-lines
            # pool on the last draw, and the mean edit size of the parents
            # drawn (None until attempts carry meta.delta_lines)
            n_explore_parents=self.last_select_explore,
            parent_delta_lines_mean=self.last_select_delta_mean,
            n_valid_big_edits=sum(
                1 for n in valid
                if (self.edit_lines(n) or 0) >= self.explore_edit_min_lines))
