# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals for 'copies needed for 90 % identification' (no QPU time).
Resampling unit: Paulis (with replacement), and within each drawn Pauli its circuits (with replacement), separately
for the calibration half (-> kappa_q) and the evaluation half (-> accuracy curve). Classical: Paulis and circuits.
Usage:  python bootstrap_ci.py --step S5 --mode origin --B 200
        python bootstrap_ci.py --step S3P --mode origin --B 200        (classical baseline from S3_origin_sm)
Writes runs/<step>_bootstrap_<mode>.json and prints point estimate, 95 % percentile interval and
P(copies90 < single-copy lower bound)."""
import os
import json
import time
import argparse
import numpy as np
from hw_analyze import load, load_merged, pools, ab_table, quantum_table, interp, HERE
from fast_site import SiteModel, bell_theta

ap = argparse.ArgumentParser()
ap.add_argument("--step", required=True)
ap.add_argument("--mode", default="origin")
ap.add_argument("--B", type=int, default=200)
ap.add_argument("--trials", type=int, default=200, help="simulated episodes per curve")
ap.add_argument("--eps", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--only", default="learned,bell,classical,onchip,onchip20,onchip22", help="comma-separated subset of learned,bell,classical")
ap.add_argument("--level", choices=["pauli", "both"], default="both",
                help="pauli: resample Paulis only; both: also circuits within a Pauli (conservative: a duplicated "
                     "circuit repeats one product-state preparation, which lowers the information per copy)")
a = ap.parse_args()
eps = a.eps
rng = np.random.default_rng(a.seed)

man, raw, _ = load_merged(a.step, a.mode)
# same kappa_c as hw_analyze.py: S2 value for S3P (classical data from the S3 run), otherwise S2_summary_<mode>
# if it exists, otherwise estimated from this step's own classical circuits
from hw_analyze import kappa_c_from
s2f = os.path.join(HERE, "runs", "S2_summary_origin_sm.json" if a.step == "S3P" else f"S2_summary_{a.mode}.json")
kc = json.load(open(s2f))["kappa_c"] if os.path.exists(s2f) else None

n = max(r["n"] for r in man["records"])
thetas = dict(learned=np.array(man["theta"]), bell=bell_theta())
if man.get("theta_onchip"):
    thetas["onchip"] = np.array(man["theta_onchip"])
for k_, v_ in (man.get("thetas_extra") or {}).items():
    thetas[k_] = np.array(v_)
q_cal, c_cal = pools(man, raw, "cal")
q_ev, c_ev = pools(man, raw, "eval")
if kc is None:
    kc = kappa_c_from(n, c_cal, eps) if c_cal else 1.0
print(f"kappa_c used by the classical decoder: {kc:.3f}", flush=True)
Pall = SiteModel(n).P; supp = Pall > 0; N = len(Pall); w_all = supp.sum(1)
index = {tuple(p): i for i, p in enumerate(Pall.tolist())}
d = 2 ** n
bound = (d + 1) * (0.9 * np.log(d * d - 1) - np.log(2)) / eps ** 2
grid_q = np.unique((np.round(np.logspace(0.3, 3.0, 40)).astype(int) // 2) * 2)
grid_c = np.unique(np.round(np.logspace(0.3, 3.8, 40)).astype(int))


def kappa(theta, cal_items):
    S, prod = ab_table(theta, n)
    pid = SiteModel(n, eps, twirl=True).probs(theta)
    vals = []
    for P, circs in cal_items:
        arr = np.concatenate(circs); i = index[P]; w = sum(1 for x in P if x)
        lam = prod[i, arr].mean() / float((pid[i] * prod[i]).sum() / pid[i].sum())
        if lam > 0:
            vals.append(lam ** (1 / w))
    return float(np.median(vals))


def curve_q(items, lp, prodAB, decoder, trials):
    acc = np.zeros(len(grid_q)); cnt = np.zeros(len(grid_q))
    for _ in range(trials):
        P, circs = items[rng.integers(len(items))]; t = index[P]
        order = rng.permutation(len(circs)); L = np.zeros(N); used = 0
        for j, m in enumerate(grid_q):
            k = m // 2
            if k > len(circs):
                break
            bs = np.array([circs[order[i]][rng.integers(len(circs[order[i]]))] for i in range(used, k)], dtype=int)
            used = k
            if len(bs):
                L += (lp[:, bs] if decoder == "bayes" else prodAB[:, bs]).sum(1)
            best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best); cnt[j] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def curve_c(items, decoder, trials):
    vis = eps * kc ** w_all
    acc = np.zeros(len(grid_c)); cnt = np.zeros(len(grid_c))
    for _ in range(trials):
        P, circs = items[rng.integers(len(items))]; t = index[P]
        order = rng.permutation(len(circs)); L = np.zeros(N); used = 0
        for j, m in enumerate(grid_c):
            if m > 2 * len(circs):
                break
            while used < m:
                sa, sb, ba, bb = circs[order[used // 2]]
                s_, b_ = (sa, ba) if used % 2 == 0 else (sb, bb)
                row = b_[rng.integers(len(b_))]; s_ = np.array(s_)
                compat = np.all((Pall == 0) | (Pall == s_[None, :]), axis=1)
                sign = np.prod(np.where(supp, row[None, :], 1), axis=1)
                L += np.log(np.clip(1 + compat * vis * sign, 1e-12, None)) if decoder == "bayes" else compat * sign
                used += 1
            best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best); cnt[j] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def resample(d_, keys):
    """Paulis with replacement, then circuits with replacement within each Pauli."""
    out = []
    for i in rng.integers(len(keys), size=len(keys)):
        P = keys[i]; c = d_[P]
        out.append((P, [c[j] for j in rng.integers(len(c), size=len(c))]))
    return out


res = {}
t0 = time.time()
ONLY = set(a.only.split(","))
_ri = np.random.default_rng(a.seed + 1000)                           # Pauli resamples shared by all policies (paired
IDX = [None] + [_ri.integers(10 ** 6, size=64) for _ in range(a.B)]   # comparisons when they measured the same Paulis)
for pol in sorted({p for p, _ in q_ev} & ONLY):
    ev = {P: v for (p, P), v in q_ev.items() if p == pol}
    cal = {P: v for (p, P), v in q_cal.items() if p == pol}
    keys = sorted(ev)
    _, prodAB = ab_table(thetas[pol], n)
    for dec in os.environ.get("BS_DECODERS", "bayes,score").split(","):
        vals = []
        for b in range(a.B + 1):                                     # b = 0: original data
            if b == 0:
                ev_i, cal_i = list(ev.items()), list(cal.items())
            else:
                idx = IDX[b][:len(keys)] % len(keys)
                if a.level == "pauli":
                    ev_i = [(keys[i], ev[keys[i]]) for i in idx]; cal_i = [(keys[i], cal[keys[i]]) for i in idx]
                else:
                    ev_i = [(keys[i], [ev[keys[i]][j] for j in rng.integers(len(ev[keys[i]]), size=len(ev[keys[i]]))]) for i in idx]
                    cal_i = [(keys[i], [cal[keys[i]][j] for j in rng.integers(len(cal[keys[i]]), size=len(cal[keys[i]]))]) for i in idx]
            kq = kappa(thetas[pol], cal_i)
            _, pQ = quantum_table(thetas[pol], n, eps, kq)
            vals.append(interp(curve_q(ev_i, np.log(pQ), prodAB, dec, a.trials), grid_q))
        res[f"{pol}_{dec}"] = vals
        print(f"  {pol} {dec} done ({time.time() - t0:.0f} s)", flush=True)
if c_ev and "classical" in ONLY:
    keys = sorted(c_ev)
    for dec in ("bayes",):
        vals = []
        for b in range(min(a.B, 100) + 1):
            if b == 0:
                items = list(c_ev.items())
            elif a.level == "pauli":
                items = [(keys[i], c_ev[keys[i]]) for i in rng.integers(len(keys), size=len(keys))]
            else:
                items = resample(c_ev, keys)
            vals.append(interp(curve_c(items, dec, max(50, a.trials // 2)), grid_c))
        res[f"classical_{dec}"] = vals
        print(f"  classical {dec} done ({time.time() - t0:.0f} s)", flush=True)

out = dict(step=a.step, mode=a.mode, n=n, B=a.B, level=a.level, trials=a.trials, single_copy_lower_bound_ideal=bound, results={})
print(f"\n{a.step} (n={n}), bootstrap B={a.B} ({a.level}); lower bound for any non-adaptive single-copy strategy (ideal hardware): {bound:.0f}")
for k, v in res.items():
    v = np.array(v, float); boot = v[1:]; ok = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    below = float(np.mean(np.where(np.isfinite(boot), boot, np.inf) < bound))
    out["results"][k] = dict(point=float(v[0]), ci95=[float(lo), float(hi)], frac_not_reached=float(np.mean(~np.isfinite(boot))),
                             p_below_bound=below, reps=[float(x) for x in v])
    print(f"  {k:16s} copies90 {v[0]:7.0f}   95% CI [{lo:.0f}, {hi:.0f}]   fraction of resamples below the bound: {below:.3f}")
json.dump(out, open(os.path.join(HERE, "runs", f"{a.step}_bootstrap_{a.level}_{a.mode}" + ("" if ONLY >= {"learned", "bell", "classical", "onchip"} else "_" + "_".join(sorted(ONLY))) + ".json"), "w"), indent=1)
