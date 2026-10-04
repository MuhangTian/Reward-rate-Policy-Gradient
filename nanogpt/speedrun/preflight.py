"""
Pre-flight for the speedrun trainer: build the config from the exact
argument list the trainer will get and check key settings before launch.
scripts/run.sh calls it with that list:

    python -m speedrun.preflight <key=value> ...
"""
from __future__ import annotations

import os
import sys

from speedrun.rl.main_golf import build_config


def check(args: list[str]) -> int:
    # OmegaConf.from_dotlist reads a Hydra-style '+key=value' as a key named
    # "+key", leaving the real key at its default.
    plus = [a for a in args if a.startswith("+")]
    if plus:
        print(f"  FAIL hydra '+' prefix (inert under from_dotlist): {plus}")
        return 1

    cfg = build_config(args)
    g, a, c = cfg.golf, cfg.actor_rollout_ref, cfg.critic
    objective = "rr" if g.use_reward_rate else "vanilla"
    fails = []

    def want(label, got, expect):
        ok = got == expect
        print(f"  {'ok  ' if ok else 'FAIL'} {label:38s} {got!r}"
              + ("" if ok else f"  (expected {expect!r})"))
        if not ok:
            fails.append(label)

    print(f"[preflight] objective={objective}  ({len(args)} overrides)")
    want("algorithm.adv_estimator", cfg.algorithm.adv_estimator, "gae")
    want("actor attn_implementation", a.model.get("attn_implementation"), "sdpa")
    want("critic attn_implementation", c.model.get("attn_implementation"), "sdpa")
    want("rollout.n == golf.rollout_n", a.rollout.n, g.rollout_n)
    want("golf.use_delta_specs", g.use_delta_specs, True)

    # critic must be buildable by AutoModelForTokenClassification: the
    # text-only tower is not registered for it
    from transformers import AutoConfig
    ccfg = AutoConfig.from_pretrained(c.model.path + "/model")
    want("critic model_type", ccfg.model_type, "qwen3_5")

    # KL on rewards via a ref policy; no golf.tilt keys
    want("algorithm.kl_ctrl.kl_coef", float(cfg.algorithm.kl_ctrl.kl_coef),
         0.001)
    want("algorithm.gamma", float(cfg.algorithm.gamma), 1.0)
    want("algorithm.lam", float(cfg.algorithm.lam), 1.0)
    # the ref policy must be the actor's base model
    want("ref_path == actor base", a.model.get("ref_path"), str(a.model.path))
    want("critic.cliprange_value", float(c.cliprange_value), 0.5)
    # response-only logits must be enabled on both the actor and the ref,
    # otherwise long prompts materialize seq x vocab fp32 logits
    want("actor.logits_to_keep_response_only",
         bool(a.actor.get("logits_to_keep_response_only", False)), True)
    want("ref.logits_to_keep_response_only",
         bool(a.ref.get("logits_to_keep_response_only", False)), True)
    ok = "tilt" not in g
    print(f"  {'ok  ' if ok else 'FAIL'} {'no golf.tilt keys':38s} {not ok!r}"
          + ("" if ok else "  (not a setting of this trainer)"))
    if not ok:
        fails.append("golf.tilt resurrected")

    # reward scale must match the condition
    if objective == "vanilla":
        want("golf.reward_mode", g.reward_mode, "time_to_target")
    else:
        want("golf.reward_mode", g.reward_mode, "quality")

        # rho = E[r]/E[t] is clamped at 0, so a reward scale with negative
        # values would make the reward-rate arm charge nothing for time.
        from speedrun.rl.verl_golf import GolfRhoHat
        import numpy as _np
        import torch as _t
        _e, _rng = GolfRhoHat(), _np.random.default_rng(0)
        for _ in range(5):
            _r = _t.tensor(float(g.quality_miss_loss) - (3.2 + 0.15 * _rng.random(64)),
                           dtype=_t.float32)
            _rho = _e.update(_r, _t.tensor(200 + 400 * _rng.random(64),
                                           dtype=_t.float32),
                             _t.ones(64, dtype=_t.bool))
        ok = _rho > 1e-4
        print(f"  {'ok  ' if ok else 'FAIL'} {'rho > 0 on the rr scale':38s} "
              f"{_rho:.6g}" + ("" if ok else "  (rho pinned at the clamp -> "
                               "reward-rate charges nothing for time)"))
        if not ok:
            fails.append("rho pinned at 0")

    # micro-batch sizes are floor-divided by world size in fsdp_workers, so
    # any value below n_gpus becomes 0 and raises ZeroDivisionError
    ng = cfg.trainer.n_gpus_per_node
    for label, val in (
            ("actor.ppo_micro_batch_size", a.actor.ppo_micro_batch_size),
            ("rollout.log_prob_micro_batch_size",
             a.rollout.log_prob_micro_batch_size),
            ("ref.log_prob_micro_batch_size",
             a.ref.log_prob_micro_batch_size),
            ("critic.ppo_micro_batch_size", c.ppo_micro_batch_size),
            ("critic.forward_micro_batch_size", c.forward_micro_batch_size)):
        per_rank = int(val) // ng
        ok = per_rank >= 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label:38s} {int(val)}"
              f" -> {per_rank}/rank"
              + ("" if ok else f"  (< 1 after //{ng}: ZeroDivisionError)"))
        if not ok:
            fails.append(label)

    # save_freq > 0 makes the checkpoint path live: an unwritable one would
    # crash the run at its first save instead of now.
    if int(cfg.trainer.save_freq) > 0:
        d = str(cfg.trainer.default_local_dir)
        p = os.path.abspath(d)
        while not os.path.exists(p):
            p = os.path.dirname(p)
        ok = os.access(p, os.W_OK)
        print(f"  {'ok  ' if ok else 'FAIL'} {'checkpoint dir (save_freq>0)':38s} "
              f"{d}" + ("" if ok else "  (not writable)"))
        if not ok:
            fails.append("default_local_dir")

    # batch must be chunkable across ranks
    bs = g.batch_size * g.rollout_n
    ok = bs % cfg.trainer.n_gpus_per_node == 0
    print(f"  {'ok  ' if ok else 'FAIL'} batch {bs} divisible by "
          f"n_gpus {cfg.trainer.n_gpus_per_node}")
    if not ok:
        fails.append("batch/n_gpus")

    print(f"[preflight] {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(check(sys.argv[1:]))
