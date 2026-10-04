"""
Big-context sweep: K in {70,100,130,160} x |X| in {8,16,32,64}.
small-context sweep is in run_small_context.py.
"""
import sys, os, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctx_envs import CtxEnv, ctx_ground_truth, permute_gap
from ctx_agents import make_ctx_agent

FAMILY, K, NCTX, NCPU = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
M = 2048          # policies sampled from U for the C-UCB index
EVERY = 1         # recompute lambda every step
N = 30_000        # steps per trial
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("BIG_RES", os.path.join(_HERE, "results_large_context")),
                   f"big_{FAMILY}_K{K}_X{NCTX}.npy")

TUNE = list(range(900, 930))          # 30 tuning seeds, as |X|=4 uses
EVAL = list(range(100))               # 100 eval trials
CHK = np.unique(np.round(np.logspace(np.log10(50), np.log10(N), 60)).astype(int))
C1_GRID = np.logspace(-2, 2, 5)
# wider than the small-context c2 grid (adds 0, 1e-6, 1e-5)
C2_GRID = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
LR_GRID = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
A_GRID = [60., 125., 250., 500., 1000., 2000., 4000.]   # accumulator age bound

AGENTS = ["ctx_ucb_theory_samp", "ctx_ucb_tuned_samp",
          "ctx_niw_pg_acc", "ctx_niw_pg_qvan_acc"]

# closed-form means (deterministic)
GT =ctx_ground_truth(FAMILY, K, n_ctx=NCTX)
RHO, GAP = GT["rho_star"], GT["GAP"]
C1 = 2 * max((GT["r_max"] - GT["r_min"])**2 / GT["d_min"]**2,
             GT["r_max"]**2 * (GT["d_max"] - GT["d_min"])**2 / GT["d_min"]**4)
A0 = np.sqrt(8 * max((GT["r_max"] - GT["r_min"])**2,
                     GT["r_max"]**2 * (GT["d_max"] - GT["d_min"])**2 / GT["d_min"]**2))
A1 = np.sqrt(2 * GT["d_max"]**2 * C1)


def grid(agent):
    if agent == "ctx_ucb_tuned_samp":
        return [{"c1": float(a), "c2": float(b), "n_pol": M, "every": EVERY}
                for a, b in itertools.product(C1_GRID, C2_GRID)]
    if agent in ("ctx_niw_pg_acc", "ctx_niw_pg_qvan_acc"):
        return [{"lr": float(l), "sm": "gage", "smp": float(a)}
                for l, a in itertools.product(LR_GRID, A_GRID)]
    return [None]                       # C-UCB theory takes its constants from
                                        # the environment: nothing to tune


def run_trial(args):
    name, hp, tr = args
    env = CtxEnv(FAMILY, K, 10_000 + tr, NCTX)
    gap = permute_gap(GAP, env)   # GAP is canonical; this trial is permuted
    ag = make_ctx_agent(name, K, NCTX, GT, 30_000 + tr, hp)
    cum_r = cum_d = pr = 0.0
    ci = 0
    R = np.empty(len(CHK))
    P = np.empty(len(CHK))
    
    for k in range(N):
        x = env.draw_context()
        a = ag.act(x)
        r, d = env.step(x, a)
        ag.update(x, a, r, d)
        cum_r += r
        cum_d += d
        pr += gap[x, a]
        
        if ci < len(CHK) and k + 1 == CHK[ci]:
            R[ci] = RHO * cum_d - cum_r
            P[ci] = pr
            ci += 1
            
    return name, hp, tr, R, P


if __name__ == "__main__":
    t0 = time.time()
    pool = Pool(NCPU)
    
    print(f"{FAMILY} K={K} |X|={NCTX}  rho*={RHO:.4f}  M={M}\n"
          f"  |U| = (K+1)^|X| = {K+1}^{NCTX}, sampled at {M} policies", flush=True)

    jobs = [(a, hp, t) for a in AGENTS for hp in grid(a) if hp is not None
            for t in TUNE]
    
    print(f"tuning: {len(jobs)} trials", flush=True)
    
    rows = []
    by = {a: {} for a in AGENTS}
    n = 0
    
    for name, hp, tr, R, P in pool.imap_unordered(run_trial, jobs, chunksize=1):
        rows.append(dict(agent=name, hp=hp, seed=tr, final_regret=float(R[-1])))
        by[name].setdefault(tuple(sorted(hp.items())), []).append(float(R[-1]))
        n += 1
        if n % 200 == 0:
            el = time.time() - t0
            print(f"  tune {n}/{len(jobs)} [{el:.0f}s eta {el*(len(jobs)-n)/n:.0f}s]",
                  flush=True)

    sel, diag = {}, {}
    for a in AGENTS:
        if not by[a]:
            sel[a] = ({"c1": float(C1), "c2": float(A0 + A1), "n_pol": M,
                       "every": EVERY} if a == "ctx_ucb_theory_samp" else {})
            continue
        stats = {k: dict(median=float(np.median(v)), mean=float(np.mean(v)),
                         sem=float(np.std(v, ddof=1) / np.sqrt(len(v))), n=len(v))
                 for k, v in by[a].items()}
        sel[a] = dict(min(stats, key=lambda k: stats[k]["median"]))
        diag[a] = {str(k): v for k, v in stats.items()}
    print(f"selected: {sel}  [{time.time()-t0:.0f}s]", flush=True)

    jobs = [(a, sel[a], t) for a in AGENTS for t in EVAL]
    
    print(f"evaluation: {len(jobs)} trials", flush=True)
    
    ev = {a: dict(R=np.zeros((len(EVAL), len(CHK))),
                  P=np.zeros((len(EVAL), len(CHK)))) for a in AGENTS}
    
    for name, hp, tr, R, P in pool.imap_unordered(run_trial, jobs, chunksize=1):
        ev[name]["R"][tr] = R
        ev[name]["P"][tr] = P
        
    pool.close()
    pool.join()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, dict(family=FAMILY, K=K, n_ctx=NCTX, N=N, chk=CHK.tolist(),
                      agents=AGENTS, rho_star=RHO, M=M, lam_every=EVERY,
                      tune_seeds=TUNE, eval_seeds=EVAL, selected=sel,
                      selection_diagnostics=diag, tuning=rows,
                      c1_grid=C1_GRID.tolist(), c2_grid=C2_GRID.tolist(),
                      lr_grid=LR_GRID.tolist(), acc_a_grid=A_GRID,
                      c1=C1, c2_theory=A0 + A1,
                      selection="median-of-30-tuning-seeds", ground_truth="closed-form",
                      eval={a: {k: v.tolist() for k, v in d.items()}
                            for a, d in ev.items()}), allow_pickle=True)
    
    for a in AGENTS:
        f = ev[a]["R"][:, -1]
        print(f"  {a:22s} {f.mean():10,.1f} +- {f.std(ddof=1)/np.sqrt(len(f)):7,.1f}"
              f"   sel={sel[a]}", flush=True)
        
    print(f"saved {OUT} [{time.time()-t0:.0f}s]", flush=True)
