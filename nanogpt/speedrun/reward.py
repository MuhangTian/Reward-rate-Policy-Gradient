"""Rewards of the two NanoGPT arms, as defined in the paper.

L is the validation loss on an attempt's last complete validation line,
L* = 3.28 the target, delta the attempt's wall-clock time and delta_max the
timeout. The depth below (and, for vanilla, above) the target is credited
over the window L* - L_floor = 0.03.
"""


def attempt_loss(d):
    """Validation loss L of a graded attempt: the loss on its last complete
    validation line."""
    det = d.get("detail") or {}
    for v in (det.get("final_full_val_loss"), det.get("final_val_loss"),
              d.get("val_loss"), d.get("val_bpb"), det.get("best_val_loss")):
        if v is not None:
            return float(v)
    return None


def attempt_valid(d) -> bool:
    """A finished run is valid as graded. A run killed at the timeout is
    also valid when it printed real validation lines at the required
    cadence; it is scored on the last line it reached."""
    if bool(d.get("valid", False)):
        return True
    det = d.get("detail") or {}
    return (d.get("reason") == "timeout"
            and bool(det.get("timing_printed"))
            and det.get("best_val_loss") is not None
            and bool(det.get("val_cadence_ok", True)))


def rpg_reward(loss, valid, g) -> float:
    """Reward of the reward-rate arm, before the time charge rho * delta:

        max(0, L_ref - L) + B + w * min(L* - L, L* - L_floor)   L <= L*
        max(0, L_ref - L)                                       L >  L*
        r_invalid                                               invalid
    """
    if loss is None or not valid:
        return g["r_invalid"]
    r = max(0.0, g["L_ref"] - loss)
    if loss <= g["L_star"]:
        r += g["bonus"] + g["w_rpg"] * min(g["L_star"] - loss, g["window"])
    return r


def vanilla_reward(loss, valid, delta, g) -> float:
    """Reward of the vanilla arm (the time to reach the target):

        -delta    + w * min(L* - L, L* - L_floor)               L <= L*
        -delta_max - w * min(L - L*, L* - L_floor)              valid, L > L*
        -2 * delta_max                                          invalid
    """
    if not valid:
        return -2.0 * g["delta_max"]
    if loss is not None and loss <= g["L_star"]:
        return -delta + g["w_vanilla"] * min(g["L_star"] - loss, g["window"])
    gap = g["window"] if loss is None else min(loss - g["L_star"], g["window"])
    return -g["delta_max"] - g["w_vanilla"] * gap


def reward_constants(cfg) -> dict:
    """The reward constants of both arms, read from the golf config."""
    get = cfg.get
    L_star = float(get("quality_target_loss", 3.28))
    return dict(
        delta_max=float(get("timeout", 1500.0)),
        L_star=L_star,
        window=L_star - float(get("quality_below_target_floor", 3.25)),
        L_ref=float(get("quality_miss_loss", 10.0)),
        r_invalid=float(get("quality_invalid_reward", -10.0)),
        bonus=float(get("quality_cross_bonus", 5.0)),
        w_rpg=float(get("quality_below_target_weight", 100.0)),
        w_vanilla=float(get("time_below_target_weight", 20000.0)),
    )
