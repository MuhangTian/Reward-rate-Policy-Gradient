"""
  ctx_ucb_theory / ctx_ucb_tuned          (C-UCB, theory / C-UCB, tuned)
    C-UCB with exact enumeration of the policy class.
  ctx_ucb_theory_samp / ctx_ucb_tuned_samp      (C-UCB, theory / C-UCB, tuned)
    C-UCB with a Monte Carlo approximation over M sampled policies.
  ctx_niw_pg_acc                                              (NPG-NIW)
  ctx_niw_pg_qvan_acc                                         (SPG-NIW)
"""
import sys
import os
import itertools

import numpy as np

import importlib.util
_NIW_PATH = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))),
    "mlebench", "verl", "utils", "reward_rate_niw.py")
_spec = importlib.util.spec_from_file_location("reward_rate_niw", _NIW_PATH)
_niw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_niw)
OnlineNIWRewardRate = _niw.OnlineNIWRewardRate


# ----------------------------------------------------------------- utilities
class Slots:
    """B cumulative (reward, time) accumulators, in round-robin fashion."""
    def __init__(self, B):
        self.B = B
        self.r = np.zeros(B)
        self.d = np.zeros(B)
        self.n = np.zeros(B)
        self._i = 0
        
    def add(self, r, d):
        i = self._i
        self.r[i] += r
        self.d[i] += d
        self.n[i] += 1
        self._i = (i + 1) % self.B
        
    def filled(self):
        return self.n > 0


# ----------------------------------------------------------------- algorithms
class CtxSMDPUCB:
    """
    Associative continuous UCB (C-UCB) of Gyorgy et al. (IJCAI-07).
    rho1, rho2 are the paper's c1 and a0+a1 for the theory baseline, and
    tuned constants for the tuned baseline.
    """
    _CHUNK = 1 << 22        # 4MB

    def __init__(self, K, n_ctx, c1, c2, seed=0):
        self.K, self.X, self.A = K, n_ctx, K + 1
        self.rho1, self.rho2 = float(c1), float(c2)
        self.sum_r = np.zeros((n_ctx, self.A))
        self.sum_d = np.zeros((n_ctx, self.A))
        self.cnt = np.zeros((n_ctx, self.A))
        self.t = 1
        self.n_policies = self.A ** self.X
        self.logU = self.X * np.log(self.A)
        self._h = self.X // 2
        self._ready = False                      # every (x,a) cell visited?
        lo = self.A ** (self.X - self._h)
        self._step = max(1, self._CHUNK // lo)  # chunking for memory efficiency
        n = min(self._step, self.A ** self._h) * lo
        self._bR = np.empty(n, np.float32)
        self._bD = np.empty(n, np.float32)
        self._bT = np.empty(n, np.float32)

    def _L(self):
        return np.log(max(self.t, 2)) + 0.5 * np.logaddexp(self.logU, 0.0)

    @staticmethod
    def _outer(M):
        """Flat array of sum_x M[x, u(x)] over every u"""
        out = np.zeros(1)
        for row in M:
            out = (out[:, None] + row).ravel()
        return out

    def _halves(self):
        h = self._h
        return ((self._outer(self.sum_r[:h]), self._outer(self.sum_d[:h]),
                 self._outer(self.cnt[:h])),
                (self._outer(self.sum_r[h:]).astype(np.float32),
                 self._outer(self.sum_d[h:]).astype(np.float32),
                 self._outer(self.cnt[h:]).astype(np.float32)))

    def _lambda(self):
        kappa = 2.0 * self.rho1 * self._L()
        (Rh, Dh, Th), (Rl, Dl, Tl) = self._halves()
        if not self._ready:
            if (self.cnt > 0).all():
                self._ready = True
            else:
                return self._lambda_masked(kappa, Rh, Dh, Th, Rl, Dl, Tl)
            
        m = Rl.size
        step = self._step
        best = -np.inf
        
        for i in range(0, Rh.size, step):
            k = min(step, Rh.size - i) * m
            R = self._bR[:k].reshape(-1, m); D = self._bD[:k].reshape(-1, m)
            T = self._bT[:k].reshape(-1, m)
            np.add(Rh[i:i + step, None].astype(np.float32), Rl, out=R)
            np.add(Dh[i:i + step, None].astype(np.float32), Dl, out=D)
            np.add(Th[i:i + step, None].astype(np.float32), Tl, out=T)
            np.divide(R, D, out=R)                    # ratio
            np.divide(np.float32(kappa), T, out=T)
            np.sqrt(T, out=T)
            np.subtract(R, T, out=R)                  # ratio - c(T_u)
            v = float(R.max())
            if v > best:
                best = v
        return max(best, 1e-9)

    def _lambda_masked(self, kappa, Rh, Dh, Th, Rl, Dl, Tl):
        """Exact but allocation-heavy; used only until every cell is visited,
        while some policies still have D_u = 0 or T_u = 0."""
        m = Rl.size
        step = self._step
        best = -np.inf
        
        for i in range(0, Rh.size, step):
            D = Dh[i:i + step, None] + Dl
            T = Th[i:i + step, None] + Tl
            ok = (D > 0) & (T > 0)
            if not ok.any():
                continue
            R = Rh[i:i + step, None] + Rl
            J = np.where(ok, R / np.where(ok, D, 1.0)
                         - np.sqrt(kappa / np.where(ok, T, 1.0)), -np.inf)
            v = float(J.max())
            if v > best:
                best = v
        return max(best, 1e-9)

    def act(self, x):
        if np.any(self.cnt[x] == 0):
            return int(np.argmin(self.cnt[x]))
        L = self._L()
        chat = self.rho2 * np.sqrt(L / self.cnt[x])
        rbar = self.sum_r[x] / self.cnt[x]
        dbar = self.sum_d[x] / self.cnt[x]
        return int(np.argmax(rbar - dbar * self._lambda() + chat))

    def update(self, x, a, r, d):
        self.cnt[x, a] += 1
        self.sum_r[x, a] += r
        self.sum_d[x, a] += d
        self.t += 1


class CtxSoftmaxPG:
    """
    Softmax policy with policy gradient updates
    """
    B_SLOTS = 128
    NIW_EVERY = 8

    def __init__(
        self, K, n_ctx, lr=0.2, seed=0, nomax=False,
        grad_mode="npg_full_rp", sample_mode="cum", sm_par=None,
        ):
        self.K, self.X, self.A = K, n_ctx, K + 1
        self.lr = float(lr)
        self.theta = np.zeros((n_ctx, self.A))
        self.rng = np.random.default_rng(seed)
        self.rho = 0.0
        self.t = 0
        self.nomax = bool(nomax)        # True skips the max-shift in policy()
        
        # grad_mode:
        #   npg_full_rp   theta[x] += lr * (Qhat - pi.Qhat)        NATURAL
        #   q_vanilla_rp  theta[x] += lr * pi (*) (Qhat - pi.Qhat) VANILLA
        self.grad_mode = grad_mode
        self.n_overflow = 0
        
        # per-arm buffer of (sum_r, sum_d, n)
        self.q_n = np.zeros((n_ctx, self.A))
        self.qr = np.zeros((n_ctx, self.A))
        self.qd = np.zeros((n_ctx, self.A))
        
        # sample_mode: WHICH samples the NIW estimator is fitted on.  Neither
        # mode touches the learner, the gradient or the acting policy.
        #   cum   cumulative slots over the whole history
        #   gage  greedy-gated, and a slot restarts once older than sm_par
        #         steps, so the pool holds accumulators of bounded AGE
        self.sample_mode = sample_mode
        self.sm_par = sm_par
        self._slot_birth = np.zeros(self.B_SLOTS)
        self.niw = OnlineNIWRewardRate(forgetting_factor=0.3, log_time=True,
                                       pool_samples=False, seed=seed)
        self.slots = Slots(self.B_SLOTS)

    def _feed_managed(self, x, a, r, d):
        """Sample retention for sample_mode='gage'.  Only the greedy action of
        the CURRENT policy feeds the estimate, and a slot restarts once it is
        older than sm_par steps.  Sets self.rho at the NIW_EVERY cadence."""
        # ties go to the lowest index, which matters only before any gradient
        # step has run
        if a != int(np.argmax(self.theta[x])):
            return
        i = self.slots._i
        stale = (self.slots.n[i] > 0
                 and self.t - self._slot_birth[i] > int(self.sm_par or 4000))
        if stale:
            self.slots.r[i] = 0.0
            self.slots.d[i] = 0.0
            self.slots.n[i] = 0.0
            self._slot_birth[i] = self.t
            
        self.slots.add(r, d)
        
        if self.t % self.NIW_EVERY != 0:
            return
        
        m = self.slots.filled()
        if m.sum() < 2:
            return
        
        est = self.niw.update(self.slots.r[m], self.slots.d[m])
        self.rho = max(0.0, float(est["p95"]))

    def policy(self, x):
        # for numerical stability
        z = self.theta[x] if self.nomax else self.theta[x] - self.theta[x].max()
        
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.exp(z)
            tot = p.sum()
            
        if not np.isfinite(tot) or tot <= 0.0:      # exp() overflowed
            self.n_overflow += 1
            m = (z == z.max())
            return m / m.sum()                      # argmax-degenerate fallback
        
        return p / tot

    def act(self, x):
        return int(self.rng.choice(self.A, p=self.policy(x)))

    def update(self, x, a, r, d):
        self.t += 1
        
        if self.sample_mode != "cum":
            self._feed_managed(x, a, r, d)
        else:
            self.slots.add(r, d)
            m = self.slots.filled()
            if m.sum() >= 2 and self.t % self.NIW_EVERY == 0:
                est = self.niw.update(self.slots.r[m], self.slots.d[m])
                self.rho = max(0.0, float(est["p95"]))
                
        p = self.policy(x)
        self.qr[x, a] += r
        self.qd[x, a] += d
        self.q_n[x, a] += 1.0
        n = np.maximum(self.q_n[x], 1)
        Q = np.where(self.q_n[x] > 0,
                     (self.qr[x] - self.rho * self.qd[x]) / n, 0.0)
        
        if self.grad_mode == "q_vanilla_rp":
            self.theta[x] += self.lr * (p * (Q - float(p @ Q)))
        else:                                       # npg_full_rp
            self.theta[x] += self.lr * (Q - float(p @ Q))


class CtxSMDPUCBSampled(CtxSMDPUCB):
    """
    C-UCB approximated by Monte Carlo over M sampled policies.  The original C-UCB
    paper enumerates every policy, which is infeasible for large |X| and |A|.
    """
    M_DEFAULT = 2048

    def __init__(self, K, n_ctx, c1, c2, seed=0, n_pol=None, every=1):
        self.K, self.X, self.A = K, n_ctx, K + 1
        self.rho1, self.rho2 = float(c1), float(c2)
        
        self.sum_r = np.zeros((n_ctx, self.A))
        self.sum_d = np.zeros((n_ctx, self.A))
        self.cnt = np.zeros((n_ctx, self.A))
        self.t = 1
        
        self.n_policies = float("inf")
        self.logU = self.X * np.log(self.A)      # act()'s radius: full class
        self._ready = True
        self.M = int(n_pol or self.M_DEFAULT)

        self.every = int(every)
        self._lam_cache = 1e-9
        self.srng = np.random.default_rng([int(seed), 90210])
        self.logU_s = np.log(max(self.M, 2))

    def _L_s(self):
        return np.log(max(self.t, 2)) + 0.5 * np.logaddexp(self.logU_s, 0.0)

    def _lambda(self):
        if self.every > 1 and (self.t % self.every) and self._lam_cache > 0:
            return self._lam_cache
        kappa = 2.0 * self.rho1 * self._L_s()
        seen = self.cnt > 0                      # (X, A) visited mask
        R = np.zeros(self.M)
        D = np.zeros(self.M)
        T = np.zeros(self.M)
        
        for x in range(self.X):
            idx = np.flatnonzero(seen[x])
            if idx.size == 0:
                continue
            pick = idx[self.srng.integers(0, idx.size, self.M)]
            R += self.sum_r[x, pick]
            D += self.sum_d[x, pick]
            T += self.cnt[x, pick]
            
        ok = (D > 0) & (T > 0)
        if not ok.any():
            return 1e-9
        J = np.where(ok, R / np.where(ok, D, 1.0)
                     - np.sqrt(kappa / np.where(ok, T, 1.0)), -np.inf)
        self._lam_cache = max(float(J.max()), 1e-9)
        return self._lam_cache


def make_ctx_agent(name, K, n_ctx, gt, seed, hp=None):
    """
    Initialize the algorithms
    
    ctx_ucb_theory / ctx_ucb_tuned            C-UCB, exact enumeration (|X|=4)
    ctx_ucb_theory_samp / ctx_ucb_tuned_samp  C-UCB, M sampled policies (|X|>4)
    ctx_niw_pg_acc                            NPG-NIW
    ctx_niw_pg_qvan_acc                       SPG-NIW (vanilla geometry)
    """
    hp = hp or {}   # hyperparams
    if name in ("ctx_ucb_theory", "ctx_ucb_tuned"):
        return CtxSMDPUCB(K, n_ctx, hp["c1"], hp["c2"], seed)
    if name in ("ctx_ucb_theory_samp", "ctx_ucb_tuned_samp"):
        return CtxSMDPUCBSampled(K, n_ctx, hp["c1"], hp["c2"], seed,
                                 n_pol=hp.get("n_pol"), every=hp.get("every", 1))
    if name == "ctx_niw_pg_acc":
        return CtxSoftmaxPG(K, n_ctx, lr=hp["lr"], seed=seed,
                            grad_mode="npg_full_rp",
                            sample_mode=hp.get("sm", "cum"),
                            sm_par=hp.get("smp"))
    if name == "ctx_niw_pg_qvan_acc":
        return CtxSoftmaxPG(K, n_ctx, lr=hp["lr"], seed=seed,
                            grad_mode="q_vanilla_rp",
                            sample_mode=hp.get("sm", "cum"),
                            sm_par=hp.get("smp"))
    raise ValueError(name)
