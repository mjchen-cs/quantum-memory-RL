# -*- coding: utf-8 -*-
"""Bootstrap for n >= 7 (no 4^n x 4^n tables): Paulis resampled with replacement; for each resample, kappa_q from the
calibration half and an accuracy curve from the evaluation half, with each episode's likelihoods computed in one
batch over its shots (factorized over sites). Usage: python bootstrap_large.py --step S7 --B 50 --trials 200"""
import os, sys, json, argparse, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hw_analyze as H
from fast_site import tables, bell_theta

ap = argparse.ArgumentParser()
ap.add_argument("--step", default="S7"); ap.add_argument("--mode", default="origin")
ap.add_argument("--B", type=int, default=50); ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--decoders", default="bayes,score"); ap.add_argument("--policies", default=""); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
man, raw, _ = H.load(a.step, a.mode)
n = max(r["n"] for r in man["records"])
thetas = dict(bell=bell_theta(), **{k: np.array(v) for k, v in (man.get("thetas_extra") or {}).items()})
q_cal, _ = H.pools(man, raw, "cal"); q_ev, _ = H.pools(man, raw, "eval")
Pall = np.array(list(itertools.product(range(4), repeat=n)))[1:]; N = len(Pall); w = (Pall > 0).sum(1)
index = {tuple(p): i for i, p in enumerate(Pall.tolist())}
grid = np.unique((np.round(np.logspace(0.3, 3.3, 30)).astype(int) // 2) * 2)
d = 2 ** n; bound = (d + 1) * (0.9 * np.log(d * d - 1) - np.log(2))
shift = np.array([2 * (n - 1 - i) for i in range(n)])


def digits(arr):
    return (np.asarray(arr)[:, None] >> shift[None, :]) & 3


def kappa(g, cal_items):
    s_sq = (g ** 2).mean(1); vals = []
    for P, circs in cal_items:
        Pv = np.array(P); dg = digits(np.concatenate(circs))
        prod = np.prod(g[Pv[None, :].repeat(len(dg), 0), dg], axis=1)
        lam = prod.mean() / float(np.prod(s_sq[Pv]))
        if lam > 0:
            vals.append(lam ** (1 / int((Pv > 0).sum())))
    return float(np.median(vals))


def curve(g, amp, items, dec, trials):
    acc = np.zeros(len(grid)); cnt = np.zeros(len(grid)); kmax = grid[-1] // 2
    for _ in range(trials):
        P, circs = items[rng.integers(len(items))]; t = index[P]
        K = min(kmax, len(circs)); order = rng.permutation(len(circs))[:K]
        shots = np.array([circs[o][rng.integers(len(circs[o]))] for o in order]); dg = digits(shots)   # (K, n)
        V = np.ones((N, K), dtype=np.float32)
        for i in range(n):
            V *= g[Pall[:, i]][:, dg[:, i]].astype(np.float32)
        X = np.log(np.clip(1 + amp[:, None] * V, 1e-12, None)) if dec == "bayes" else V
        C = np.cumsum(X, axis=1)
        for j, m in enumerate(grid):
            k = m // 2
            if k > K:
                break
            L = C[:, k - 1]; best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best); cnt[j] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


res = {}
for pol, th in thetas.items():
    if a.policies and pol not in a.policies.split(","):
        continue
    g = tables(th)[2]
    ev = {P: v for (p, P), v in q_ev.items() if p == pol}; cal = {P: v for (p, P), v in q_cal.items() if p == pol}
    keys = sorted(ev)
    for dec in a.decoders.split(","):
        vals = []
        for b in range(a.B + 1):
            idx = range(len(keys)) if b == 0 else rng.integers(len(keys), size=len(keys))
            ks = [keys[i] for i in idx]
            kq = kappa(g, [(k, cal[k]) for k in ks]); amp = kq ** w
            vals.append(H.interp(curve(g, amp, [(k, ev[k]) for k in ks], dec, a.trials), grid))
        res[f"{pol}_{dec}"] = vals
        v = np.array(vals, float); boot = v[1:]; ok = boot[np.isfinite(boot)]
        lo, hi = np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan)
        below = float(np.mean(np.where(np.isfinite(boot), boot, np.inf) < bound))
        print(f"  {pol}_{dec}: copies90 {v[0]:.0f}  95% CI [{lo:.0f}, {hi:.0f}]  not reached {np.mean(~np.isfinite(boot)):.2f}"
              f"  below bound {bound:.0f}: {below:.2f}", flush=True)
json.dump(dict(step=a.step, n=n, B=a.B, trials=a.trials, bound=bound, results=res),
          open(os.path.join(HERE, "runs", f"{a.step}_bootstrap_large_{a.mode}_{a.policies or 'all'}_{a.decoders.replace(',', '-')}.json"), "w"))
