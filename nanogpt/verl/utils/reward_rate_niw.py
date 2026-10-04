"""
Online normal-inverse-Wishart (NIW) reward-rate estimator.
"""
import numpy as np

EPS = 1e-8


class OnlineNIWRewardRate:
    """
    Stateful, one-step-at-a-time NIW log-time reward-rate estimator.
    """

    def __init__(
        self,
        n_groups: int = 1000,
        seed: int = 0,
        kappa0: float = 1e-3,
        nu0: float = 3.0,
        psi0_scale: float = 1e-3,
        ci: float = 0.95,
        forgetting_factor: float = 0.3,
        log_time: bool = True,
        pool_samples: bool = False,
        nu_cap: float = None,
    ):
        self.d = 2
        if nu_cap is not None and nu_cap <= self.d + 1:
            raise ValueError(
                f"nu_cap={nu_cap} must be > d+1={self.d + 1} (the posterior "
                f"dof floor is d+2; a cap at or below d+1 would make the "
                f"predictive undefined or leave it with no finite variance)")
        self.n_groups = n_groups
        self.kappa0 = kappa0
        self.nu0 = nu0
        self.psi0_scale = psi0_scale
        self.ci = ci
        self.forgetting_factor = forgetting_factor
        self.log_time = log_time
        self.pool_samples = pool_samples
        self.nu_cap = nu_cap

        self.rng = np.random.default_rng(seed)
        # Carried posterior state, initialised to the fixed weak prior; each
        # update() overwrites these with that step's posterior.
        self.kappa_prev = float(kappa0)
        self.nu_prev = float(nu0)
        self.mu_prev = np.zeros(self.d)
        self.Psi_prev = psi0_scale * np.eye(self.d)
        self.last = None

    def state_dict(self) -> dict:
        return {
            "kappa_prev": float(self.kappa_prev),
            "nu_prev": float(self.nu_prev),
            "mu_prev": np.asarray(self.mu_prev, dtype=np.float64).tolist(),
            "Psi_prev": np.asarray(self.Psi_prev, dtype=np.float64).tolist(),
            "last": self.last,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.kappa_prev = float(sd.get("kappa_prev", self.kappa_prev))
        self.nu_prev = float(sd.get("nu_prev", self.nu_prev))
        if sd.get("mu_prev") is not None:
            self.mu_prev = np.asarray(sd["mu_prev"], dtype=np.float64)
        if sd.get("Psi_prev") is not None:
            self.Psi_prev = np.asarray(sd["Psi_prev"], dtype=np.float64)
        self.last = sd.get("last", self.last)
        if sd.get("rng_state") is not None:
            self.rng.bit_generator.state = sd["rng_state"]

    def update(self, accum_reward, accum_time, valid_mask=None) -> dict:
        """
        One NIW step. `accum_reward`/`accum_time` are per-slot vectors of the same
        length; if `valid_mask` is given, only slots with a truthy mask enter the fit.
        """
        d = self.d
        accum_reward = np.asarray(accum_reward, dtype=np.float64).ravel()
        accum_time = np.asarray(accum_time, dtype=np.float64).ravel()
        if valid_mask is not None:
            m = np.asarray(valid_mask).ravel().astype(bool)
            accum_reward = accum_reward[m]
            accum_time = accum_time[m]

        n = accum_reward.shape[0]
        if n < 2:
            # Too few valid slots to fit this step -- don't corrupt the
            # carried posterior; hold the last estimate.
            if self.last is None:
                self.last = {"mean": 0.0, "lo": 0.0, "hi": 0.0, "p25": 0.0,
                             "p75": 0.0, "p95": 0.0, "max": 0.0}
            return self.last

        # accum_time is positive; the EPS floor guards against log(0).
        time_model = np.log(np.maximum(accum_time, EPS)) if self.log_time else accum_time
        data = np.stack([accum_reward, time_model], axis=1)  # (n, 2)
        xbar = data.mean(axis=0)
        centered = data - xbar
        S = centered.T @ centered

        lam = self.forgetting_factor
        kappa_prior = lam * self.kappa_prev
        nu_prior = max(lam * self.nu_prev, d + 2.0)
        Psi_prior = lam * self.Psi_prev
        mu_prior = self.mu_prev

        kappa_n = kappa_prior + n
        nu_n = nu_prior + n
        if self.nu_cap is not None:
            # Clip the posterior dof before it is carried to the next step.
            nu_n = min(nu_n, self.nu_cap)
        mu_n = (kappa_prior * mu_prior + n * xbar) / kappa_n
        diff = xbar - mu_prior
        Psi_n = Psi_prior + S + (kappa_prior * n / kappa_n) * np.outer(diff, diff)

        self.kappa_prev, self.nu_prev, self.mu_prev, self.Psi_prev = (
            kappa_n, nu_n, mu_n, Psi_n,
        )

        df_pred = nu_n - d + 1.0
        scale_matrix = Psi_n * (kappa_n + 1.0) / (kappa_n * df_pred)

        # Synthetic "batch" size = the number of slots that entered the fit
        # this step (n = batch_size when nothing is excluded), so the grouped
        # ratio-of-means is taken over the same sample size as a real batch.
        z = self.rng.multivariate_normal(np.zeros(d), scale_matrix, size=(self.n_groups, n))
        u = self.rng.chisquare(df_pred, size=(self.n_groups, n))
        samples = mu_n[None, None, :] + z / np.sqrt(u / df_pred)[..., None]

        X_samp, Y_samp = samples[..., 0], samples[..., 1]
        if self.log_time:
            Y_samp = np.exp(Y_samp)
        if self.pool_samples:
            ratio = X_samp.ravel() / (Y_samp.ravel() + EPS)
        else:
            ratio = X_samp.mean(axis=1) / (Y_samp.mean(axis=1) + EPS)

        alpha = 1.0 - self.ci
        lo, hi = np.percentile(ratio, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p25, p75, p95 = np.percentile(ratio, [25.0, 75.0, 95.0])
        self.last = {
            "mean": float(ratio.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "p25": float(p25),
            "p75": float(p75),
            "p95": float(p95),
            # max over the n_groups synthetic-batch ratio-of-means: the most
            # optimistic whole-batch reward rate the fitted model produces.
            "max": float(ratio.max()),
        }
        return self.last
