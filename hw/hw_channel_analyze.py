# -*- coding: utf-8 -*-
"""Analysis of the channel task on the chip (steps CH5, CH6): uses of the hidden-Pauli channel needed for 90 %
identification. For input W and measurement V the outcome distribution of one use is
  p(beta | P) = prod_i A(beta_i) * [1 + kappa^w prod_i g(P_i, beta_i)],  g = B / A  (hw_train.channel_tables),
with kappa the per-site visibility fitted on the calibration half. Exact Bayes over all 4^n - 1 Paulis on the
evaluation half (one random shot per circuit), Pauli bootstrap. Usage: python hw_channel_analyze.py --step CH5 --B 100"""
import os, sys, json, argparse, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hw_analyze as H
from hw_train import channel_tables

ap = argparse.ArgumentParser()
ap.add_argument("--step", default="CH5"); ap.add_argument("--mode", default="origin")
ap.add_argument("--B", type=int, default=100); ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
man, raw, _ = H.load(a.step, a.mode)
n = max(r["n"] for r in man["records"]); d = 2 ** n
bound = (d + 1) * (0.9 * np.log(d * d - 1) - np.log(2))            # uses, any ancilla-free non-adaptive strategy
Pall = np.array(list(itertools.product(range(4), repeat=n)))[1:]; N = len(Pall); w = (Pall > 0).sum(1)
index = {tuple(p): i for i, p in enumerate(Pall.tolist())}
shift = np.array([2 * (n - 1 - i) for i in range(n)])
grid = np.unique(np.round(np.logspace(np.log10(4), np.log10(400), 36)).astype(int))
q_cal, _ = H.pools(man, raw, "cal"); q_ev, _ = H.pools(man, raw, "eval")


def digits(arr):
    return (np.asarray(arr)[:, None] >> shift[None, :]) & 3


def gtab(th):
    A, B = channel_tables(th[:13], th[13:]); return B / np.clip(A, 1e-12, None)[None, :], A


def kappa(g, A, cal_items):
    vals = []
    for P, circs in cal_items:
        Pv = np.array(P); dg = digits(np.concatenate(circs))
        prod = np.prod(g[Pv[None, :].repeat(len(dg), 0), dg], axis=1)
        # E_P[prod g] = kappa^w * sum_b prod_i A(b_i) g(P_i, b_i)^2 (ideal second moment)
        s_sq = (A[None, :] * g ** 2).sum(1)
        lam = prod.mean() / float(np.prod(s_sq[Pv]))
        if lam > 0:
            vals.append(lam ** (1 / int((Pv > 0).sum())))
    return float(np.median(vals))


def curve(g, amp, items, trials):
    acc = np.zeros(len(grid)); cnt = np.zeros(len(grid))
    for _ in range(trials):
        P, circs = items[rng.integers(len(items))]; t = index[P]
        K = min(grid[-1], len(circs)); order = rng.permutation(len(circs))[:K]
        dg = digits(np.array([circs[o][rng.integers(len(circs[o]))] for o in order]))
        V = np.ones((N, K))
        for i in range(n):
            V *= g[Pall[:, i]][:, dg[:, i]]
        C = np.cumsum(np.log(np.clip(1 + amp[:, None] * V, 1e-12, None)), axis=1)
        for j, m in enumerate(grid):
            if m > K:
                break
            L = C[:, m - 1]; best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best); cnt[j] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


res = dict(step=a.step, n=n, bound=bound, grid=grid.tolist(), unit="channel uses", policies={})
for pol, th in (man.get("thetas_extra") or {}).items():
    th = np.array(th, float); g, A = gtab(th)
    ev = {P: v for (p, P), v in q_ev.items() if p == pol}; cal = {P: v for (p, P), v in q_cal.items() if p == pol}
    keys = sorted(ev); vals = []; c0 = None
    for b in range(a.B + 1):
        ks = keys if b == 0 else [keys[i] for i in rng.integers(len(keys), size=len(keys))]
        kq = kappa(g, A, [(k, cal[k]) for k in ks]); c = curve(g, kq ** w, [(k, ev[k]) for k in ks], a.trials)
        vals.append(H.interp(c, grid))
        if b == 0:
            c0, k0 = c, kq
    v = np.array(vals, float); bt = v[1:]; ok = bt[np.isfinite(bt)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan)
    below = float(np.mean(np.where(np.isfinite(bt), bt, np.inf) < bound))
    print(f"  {pol}: kappa {k0:.3f}  uses90 {v[0]:.0f} [{lo:.0f}, {hi:.0f}]  bound {bound:.0f}  ({v[0] / bound:.2f} x)  "
          f"resamples below bound {below:.2f}", flush=True)
    res["policies"][pol] = dict(kappa=k0, point=v[0], ci=[lo, hi], below=below, boot=vals, curve=c0.tolist())
out = os.path.join(HERE, "runs", f"{a.step}_channel_{a.mode}.json")
json.dump(res, open(out, "w"), indent=1, default=float); print("->", out)
