"""
Contextual continuous-time bandit environments.

See more discussion in the paper's appendix.
"""
import numpy as np
from functools import lru_cache

FAMILIES = ["E1_indep", "E2_poscorr", "E3_negcorr", "E4_lognorm"]
CORR = {"E1_indep": 0.0, "E2_poscorr": 0.8, "E3_negcorr": -0.8, "E4_lognorm": 0.5}
DELTA_FLOOR = 0.5
SKIP_TIME = 0.35          # delta_0: declining still costs overhead
R_SIG, D_SIG_FRAC, LOG_SIG = 0.20, 0.15, 0.5
N_CTX = 4

_BASE = [(0.50, 1.0), (0.57, 1.2), (0.90, 3.0), (0.21, 0.6)]     # MARGINAL
_BASE_LO, _BASE_HI = 0.42, 0.18                                   # MARGINAL filler rates
_COUPLING = [(1.80, 2.00), (1.00, 1.00), (1.60, 3.0), (0.20, 0.6)]
_FILL_HI, _FILL_LO = 1.35, 0.42


def _filler_scales(n):
    if n <= 0:
        return np.zeros(0)
    k = (np.arange(n) + 0.5) / n
    return _FILL_HI - (_FILL_HI - _FILL_LO) * k


def context_scale(x, n_ctx=N_CTX):
    """
    Reward multipliers for each context type
    """
    if x == 0:
        return 1.50
    if x == 1:
        return None
    if x == 2:
        return 1.00
    if x == 3:
        return 0.32
    return float(_filler_scales(max(n_ctx - 4, 1))[x - 4])


# ---------------------------------------------------------------------------
_FILL_D_LO, _FILL_D_HI = 0.70, 2.00      # continuous filler duration range


def _instance_rng(K, n_ctx):
    """
    Keep a deterministic construction for the parameters of the distributions, 
    so that stochasticity comes from sampling only.
    
    We do this because the scale of regret would change if the true parameters change across trials.
    """
    import hashlib
    key = f"perm-v1|{K}|{n_ctx}".encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


@lru_cache(maxsize=None)
def _instance(K, n_ctx):
    rng = _instance_rng(K, n_ctx)
    mu_r = np.zeros((n_ctx, K))
    mu_d = np.zeros((n_ctx, K))
    
    for x in range(n_ctx):
        mu_r[x], mu_d[x] = context_profile(x, K, n_ctx, rng)
        
    return mu_r, mu_d


def context_profile(x, K, n_ctx=N_CTX, rng=None):
    """(mu_r, mu_delta) for the K arms of context x."""
    mu_r = np.zeros(K)
    mu_d = np.zeros(K)
    
    sc = context_scale(x, n_ctx)
    
    if sc is None:                                   # COUPLING
        base, lo, hi = _COUPLING, _BASE_LO, _BASE_HI
    else:                                            # a pure reward rescaling
        base = [(r * sc, d) for r, d in _BASE]
        lo, hi = _BASE_LO * sc, _BASE_HI * sc
        
    for i, (r, d) in enumerate(base[:min(K, 4)]):
        mu_r[i], mu_d[i] = r, d
        
    if K > 4:
        rates = np.linspace(lo, hi, K - 4)
        for j, rate in enumerate(rates):
            d = rng.uniform(_FILL_D_LO, _FILL_D_HI)
            mu_r[4 + j], mu_d[4 + j] = rate * d, d

    return mu_r, mu_d


class CtxEnv:
    """
    The environment class
    """

    def __init__(self, family, K, seed, n_ctx=N_CTX):
        assert family in FAMILIES
        self.family, self.K, self.n_ctx = family, K, n_ctx
        
        # seed for the trial's stochasticity, not the instance construction
        self.rng = np.random.default_rng(seed)
        self.p = np.full(n_ctx, 1.0 / n_ctx)  # distribution for the contexts
        
        # context and arm permutation on every trial.
        mr, md = _instance(K, n_ctx)
        pr = np.random.default_rng([int(seed), 0xC0FFEE])
        
        self.cperm = pr.permutation(n_ctx)
        self.aperm = pr.permutation(K)
        self.mu_r = mr[self.cperm][:, self.aperm].copy()
        self.mu_d = md[self.cperm][:, self.aperm].copy()
        self.SKIP = K

    def draw_context(self):
        return int(self.rng.choice(self.n_ctx, p=self.p))

    def _draw(self, x, a, rng, n):
        """
        Draw from p(r, d | x, a) for n samples.  The SKIP action is deterministic.
        """
        if a == self.SKIP:
            return np.zeros(n), np.full(n, SKIP_TIME)
        
        mr, md = self.mu_r[x, a], self.mu_d[x, a]
        rho = CORR[self.family]
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho * rho) * z2
        
        if self.family == "E4_lognorm":
            r = np.exp(np.log(max(mr, 1e-6)) - LOG_SIG**2 / 2 + LOG_SIG * z1)
            d = np.maximum(np.exp(np.log(md) - LOG_SIG**2 / 2 + LOG_SIG * z2), DELTA_FLOOR)
        else:
            r = np.maximum(mr + R_SIG * z1, 0.0)
            d = np.maximum(md + D_SIG_FRAC * md * z2, DELTA_FLOOR)
        return r, d

    def step(self, x, a):
        r, d = self._draw(x, a, self.rng, 1)
        return float(r[0]), float(d[0])


# ---------------------------------------------------------------------------
_CLOSED_FORM = {"E1_indep", "E2_poscorr", "E3_negcorr", "E4_lognorm"}


def _exact_means(family, K, n_ctx):
    """Closed-form E[r], E[delta] for every (x, a), or None if unsupported.

    The Gaussian copula correlates r with delta but does not change either
    MARGINAL mean, so the two factor and no simulation is needed.

      E1-E3   r = max(m + s Z, 0),  d = max(m_d + phi m_d Z, DELTA_FLOOR)
              rectified normal:  E[max(X,c)] = c F(a) + m(1-F(a)) + s f(a),
              a = (c - m)/s, with F, f the standard normal cdf and pdf.
      E4      For r, we have E[r] = m exactly.
              d is a FLOORED lognormal with mu = log(m_d) - sig^2/2:
              E[max(Y,c)] = c F(b) + m_d (1 - F(b - sig)),  b = (log c - mu)/sig.
      SKIP    E[r] = 0, E[delta] = SKIP_TIME, both exact by definition.
    """
    if family not in _CLOSED_FORM:
        return None
    from scipy.stats import norm
    
    mr, md = _instance(K, n_ctx)
    ER = np.zeros((n_ctx, K + 1))
    ED = np.zeros((n_ctx, K + 1))
    
    if family == "E4_lognorm":
        sig = LOG_SIG
        ER[:, :K] = np.maximum(mr, 1e-6)
        mu = np.log(md) - sig * sig / 2.0
        b = (np.log(DELTA_FLOOR) - mu) / sig
        ED[:, :K] = DELTA_FLOOR * norm.cdf(b) + md * (1.0 - norm.cdf(b - sig))
    else:
        a = -mr / R_SIG
        ER[:, :K] = mr * (1.0 - norm.cdf(a)) + R_SIG * norm.pdf(a)
        sd = D_SIG_FRAC * md
        b = (DELTA_FLOOR - md) / sd
        ED[:, :K] = (DELTA_FLOOR * norm.cdf(b) + md * (1.0 - norm.cdf(b))
                     + sd * norm.pdf(b))
    ER[:, K] = 0.0
    ED[:, K] = SKIP_TIME
    return ER, ED


def permute_gap(GAP, env):
    """
    GAP is computed in CANONICAL arm/context order; a trial sees a permuted
    version, so need to permute gap as well so pseudo-regret stays meaningful
    """
    cols = np.concatenate([env.aperm, [GAP.shape[1] - 1]])   # SKIP stays last
    return GAP[env.cperm][:, cols]


def ctx_ground_truth(family, K, n_per=200_000, n_ctx=N_CTX, exact=True):
    """
    rho* as the root of
       F(rho) = sum_x p(x) max_a ( E[r_a(x)] - rho E[d_a(x)] ) = 0
    with the skip action included, using bisection search.

    exact=True uses the closed-form means (default; deterministic)
    exact=False uses Monte Carlo approximations.
    """
    e = CtxEnv(family, K, 0, n_ctx)
    got = _exact_means(family, K, n_ctx) if exact else None
    if got is not None:
        ER, ED = got
    else:
        rng = np.random.default_rng(9)
        ER = np.zeros((n_ctx, K + 1))
        ED = np.zeros((n_ctx, K + 1))
        
        for x in range(n_ctx):
            for a in range(K + 1):
                r, d = e._draw(x, a, rng, n_per)
                ER[x, a], ED[x, a] = r.mean(), d.mean()

    def F(rho):
        return float(e.p @ (ER - rho * ED).max(1))

    # bisection search for the root of F(rho) = 0, which is rho*
    lo, hi = 0.0, 50.0
    assert F(lo) > 0 > F(hi), (F(lo), F(hi))
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if F(mid) > 0 else (lo, mid)
    rho = (lo + hi) / 2
    
    # optimal action, relative value gap, and rate greedy policy
    best = (ER - rho * ED).argmax(1)
    GAP = (ER - rho * ED).max(1)[:, None] - (ER - rho * ED)   # >= 0, per (x,a)
    rate_greedy = np.where(ED[:, :K] > 0, ER[:, :K] / ED[:, :K], -np.inf).argmax(1)
    
    return dict(ER=ER, ED=ED, p=e.p, rho_star=rho, best=best, GAP=GAP,
                rate_greedy=rate_greedy, skip=K, F_at_root=F(rho),
                r_min=float(ER.min()), r_max=float(ER.max()),
                d_min=float(ED.min()), d_max=float(ED.max()))
