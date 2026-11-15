# -*- coding: utf-8 -*-
"""Identification analysis for n = 8-10 (steps X8, X9, X10): exact Bayes decoder over all 4^n - 1 Paulis with the
Walsh-Hadamard evaluation of bign.py (front-ends with +-1 response tables), per-site visibility kappa fitted on the
calibration quarter (every 4th circuit), accuracy curves of every Pauli on the remaining circuits (one random shot per
circuit), and a bootstrap over Paulis of their mean (kappa kept at its point estimate).
Usage: python hw_bign.py --step X8 [--B 1000 --trials 40]"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hw_analyze as H
from fast_site import tables
from bign import Walsh

ap = argparse.ArgumentParser()
ap.add_argument("--step", default="X8"); ap.add_argument("--mode", default="origin")
ap.add_argument("--B", type=int, default=1000); ap.add_argument("--trials", type=int, default=40, help="episodes per Pauli and number of copies")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--cal_every", type=int, default=4, help="every k-th circuit of a Pauli calibrates kappa, the rest evaluate")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
man, raw, _ = H.load(a.step, a.mode)
n = max(r["n"] for r in man["records"]); d = 2 ** n
bound = (d + 1) * (0.9 * np.log(d * d - 1) - np.log(2))
thetas = {k: np.array(v, float) for k, v in (man.get("thetas_extra") or {}).items()}
q_all, _ = H.pools(man, raw, None)              # circuits in repetition order
q_cal = {k: v[::a.cal_every] for k, v in q_all.items()}
q_ev = {k: [c for j, c in enumerate(v) if j % a.cal_every] for k, v in q_all.items()}
shift = np.array([2 * (n - 1 - i) for i in range(n)]); pw4 = 4 ** np.arange(n - 1, -1, -1)
grid = np.unique(np.round(np.logspace(np.log10(40), np.log10(4000), 28) / 2).astype(int) * 2)


def digits(arr):
    return (np.asarray(arr)[:, None] >> shift[None, :]) & 3


def kappa_per_pauli(g, circs, P):
    Pv = np.array(P); dg = digits(np.concatenate(circs))
    prod = np.prod(g[Pv[None, :].repeat(len(dg), 0), dg], axis=1)
    s_sq = (g ** 2).mean(1); lam = prod.mean() / float(np.prod(s_sq[Pv]))
    return lam ** (1 / int((Pv > 0).sum())) if lam > 0 else np.nan


def curve(W, kq, items, trials):
    acc = np.zeros(len(grid)); cnt = np.zeros(len(grid)); kw = kq ** np.arange(n + 1)
    for _ in range(trials):
        P, circs = items[rng.integers(len(items))]; tgt = int(np.array(P) @ pw4)
        K = len(circs); order = rng.permutation(K)
        shots = np.array([circs[o][rng.integers(len(circs[o]))] for o in order]); Q = W.Q_index(digits(shots))
        for j, m in enumerate(grid):
            k = m // 2
            if k > K:
                break
            L = W.loglik(Q[:k], kw); L[0] = -np.inf
            best = np.flatnonzero(L == L.max()); acc[j] += (tgt in best) / len(best); cnt[j] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


res = dict(step=a.step, n=n, bound=bound, grid=grid.tolist(), policies={},
           method="per-Pauli accuracy curves at the point estimate of kappa; Pauli bootstrap of their mean")
for pol, th in thetas.items():
    g = tables(th)[2]; W = Walsh(n, g)
    ev = {P: v for (p, P), v in q_ev.items() if p == pol}; cal = {P: v for (p, P), v in q_cal.items() if p == pol}
    keys = sorted(ev)
    kap = {P: kappa_per_pauli(g, cal[P], P) for P in keys}
    kq = float(np.nanmedian([kap[k] for k in keys]))
    C = np.array([curve(W, kq, [(k, ev[k])], a.trials) for k in keys])        # (16, grid)
    point = np.nanmean(C, 0)
    print(f"  {pol}: kappa {kq:.3f}  copies90 {H.interp(point, grid):.0f}  (bound {bound:.0f})", flush=True)
    print("    curve " + " ".join(f"{m}:{x:.2f}" for m, x in zip(grid, point) if np.isfinite(x)), flush=True)
    boots = [point] + [np.nanmean(C[rng.integers(len(keys), size=len(keys))], 0) for _ in range(a.B)]
    vals = [H.interp(c, grid) for c in boots]
    per = {"".join(map(str, k)): dict(w=int(sum(1 for x in k if x)), copies90=H.interp(C[i], grid), kappa=kap[k],
                                      curve=C[i].tolist()) for i, k in enumerate(keys)}
    v = np.array(vals, float); bt = v[1:]; ok = bt[np.isfinite(bt)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan)
    below = float(np.mean(np.where(np.isfinite(bt), bt, np.inf) < bound))
    print(f"  {pol}: copies90 {v[0]:.0f} [{lo:.0f}, {hi:.0f}]  = {v[0] / bound:.3f} x bound; resamples below bound {below:.2f}",
          flush=True)
    jm = int(np.max(np.flatnonzero(np.isfinite(point)))); M = int(grid[jm])
    accs = np.array([c[jm] for c in boots], float); ceil = min(1.0, (M / (d + 1) + np.log(2)) / np.log(d * d - 1))
    alo, ahi = np.nanpercentile(accs[1:], [2.5, 97.5])
    print(f"  {pol}: accuracy at {M} copies {accs[0]:.2f} [{alo:.2f}, {ahi:.2f}]; single-copy ceiling {ceil:.2f}", flush=True)
    res["policies"][pol] = dict(kappa=kq, point=v[0], ci=[lo, hi], below=below, boot=vals, curve=point.tolist(),
                                per_pauli=per, acc_at=dict(copies=M, acc=accs[0], ci=[alo, ahi], single_copy_ceiling=ceil),
                                frac_not_reached=float(np.mean(~np.isfinite(bt))))
out = os.path.join(HERE, "runs", f"{a.step}_bign_{a.mode}.json")
json.dump(res, open(out, "w"), indent=1, default=float); print("->", out)
