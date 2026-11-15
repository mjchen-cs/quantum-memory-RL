# -*- coding: utf-8 -*-
"""
Analysis of the "classical shortcut" check on WK_C180 (steps SC3, SC4; task WITHOUT twirl: pairs of copies of rho_P).
Front-ends: shortcut20 / shortcut22 (trained on the chip without twirl, n = 2, best by reward) and Bell (native gates).

Decoder (one per front-end, Bayes over all N = 4^n - 1 Paulis): with the site tables g[k, sigma, beta] of V(theta)
(k = copy A, copy B, both), p(beta | P) = (1 + a^w prod g_A + a^w prod g_B + b^w prod g_AB) / D, where a and b are
per-site visibilities of the single-copy and two-copy terms (w = weight of P), fitted by maximum likelihood on the
calibration half (odd repetitions). Evaluation on the other half: episodes of distinct circuits (one preparation each,
one shot each), accuracy of the maximum-posterior Pauli vs copies (2 per circuit); copies for 90 % if reached.
Also: the local noise model of hw_noise.py (calibration snapshot, the qubits of this step) evaluated on the same 16
Paulis, i.e. the model behind the ideal/noisy extrapolation to larger n, compared with the measured curve.
Usage:  python hw_shortcut.py --step SC3        python hw_shortcut.py --step SC4
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hw_analyze as H                                    # noqa: E402
from fast_site import tables                              # noqa: E402


def site_prods(th, Pall, n):
    g = tables(th); D = 4 ** n
    B = np.array(list(itertools.product(range(4), repeat=n)))      # outcome digits, site 0 most significant
    out = []
    for k in range(3):
        pr = np.ones((len(Pall), D))
        for i in range(n):
            pr *= g[k][Pall[:, i]][:, B[:, i]]
        out.append(pr)
    return out


def table(prods, w, a, b):
    pA, pB, pAB = prods
    p = 1 + (a ** w)[:, None] * (pA + pB) + (b ** w)[:, None] * pAB
    p = np.clip(p, 1e-9, None)
    return p / p.sum(1, keepdims=True)


def curve(lp, index, items, grid_pairs, trials, rng):
    """items: [(P, [shot arrays per circuit])]. Accuracy of argmax posterior vs pairs (distinct circuits only)."""
    acc = np.zeros(len(grid_pairs)); cnt = np.zeros(len(grid_pairs))
    for t in range(trials):
        P, circs = items[rng.integers(len(items))]; tr = index[P]
        K = len(circs); order = rng.permutation(K)
        obs = np.array([circs[i][rng.integers(len(circs[i]))] for i in order])
        L = np.cumsum(lp[:, obs], axis=1)
        for j, m in enumerate(grid_pairs):
            if m > K:
                break
            col = L[:, m - 1]; best = np.flatnonzero(col == col.max())
            acc[j] += (tr in best) / len(best); cnt[j] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True); ap.add_argument("--mode", default="origin")
    ap.add_argument("--trials", type=int, default=400); ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    man, raw, _ = H.load(a.step, a.mode)
    n = man["records"][0]["n"]; d = 2 ** n
    Pall = np.array(list(itertools.product(range(4), repeat=n)))[1:]; N = len(Pall); w = (Pall > 0).sum(1)
    index = {tuple(p): i for i, p in enumerate(Pall)}
    q_cal, _ = H.pools(man, raw, "cal"); q_ev, _ = H.pools(man, raw, "eval")
    thetas = {k: np.array(v) for k, v in man["thetas_extra"].items()}
    out = dict(step=a.step, n=n, twirl=False, agents={})
    # noise model of the extrapolation (same qubits, calibration of this step)
    model = None
    chip = os.path.join(HERE, "runs", "chip.json")
    if os.path.exists(chip):
        from hw_noise import load_chip, noisy_quantum_probs
        _, s1, f2 = load_chip(chip)
        sites = man["sites"][:n]
        for s in sites:
            s.setdefault("f_cz", f2.get((s["a"], s["b"]), f2.get((s["b"], s["a"]), 0.98)))
        model = (noisy_quantum_probs, s1, sites)
    print(f"{a.step} (n = {n}), task without twirl; Bayes decoder over {N} Paulis; calibration {man.get('chip_fetched')}")
    for pol in sorted({p for p, _ in q_ev}):
        prods = site_prods(thetas[pol], Pall, n)
        cal = [(P, c) for (p_, P), c in q_cal.items() if p_ == pol]
        ev = [(P, c) for (p_, P), c in q_ev.items() if p_ == pol]
        # ML fit of (a, b) on all calibration shots
        obs = [(index[P], np.concatenate(c)) for P, c in cal]
        best = (-np.inf, None)
        for aa in np.arange(0.50, 1.001, 0.02):
            for bb in np.arange(0.50, 1.001, 0.02):
                lp = np.log(table(prods, w, aa, bb))
                ll = sum(lp[i, o].sum() for i, o in obs)
                if ll > best[0]:
                    best = (ll, (aa, bb))
        aa, bb = best[1]
        lp = np.log(table(prods, w, aa, bb))
        K = min(len(c) for _, c in ev)
        grid = np.unique(np.round(np.logspace(0, np.log10(K), 30)).astype(int))
        acc = curve(lp, index, ev, grid, a.trials, rng)
        c90 = H.interp(acc, 2 * grid)
        boots = []
        for _ in range(a.B):                                         # Pauli bootstrap (decoder fixed)
            items = [ev[i] for i in rng.integers(len(ev), size=len(ev))]
            boots.append(curve(lp, index, items, grid, max(100, a.trials // 4), rng))
        boots = np.array(boots)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        r = dict(a=float(aa), b=float(bb), circuits_eval=int(K), copies=(2 * grid).tolist(), acc=acc.tolist(),
                 acc_lo=lo.tolist(), acc_hi=hi.tolist(), copies90=c90, acc_at_max=float(acc[~np.isnan(acc)][-1]),
                 max_copies=int(2 * grid[~np.isnan(acc)][-1]))
        if model is not None:
            f, s1, sites = model
            _, pm = f(n, thetas[pol], sites, s1, twirl=False)
            lpm = np.log(pm); Pset = [P for P, _ in ev]
            ga = []
            for _ in range(a.trials):                                 # model accuracy on the same Paulis, same copies
                P = Pset[rng.integers(len(Pset))]; tr = index[P]; cdf = np.cumsum(pm[tr])
                o = np.searchsorted(cdf, rng.random(grid[-1])).clip(0, 4 ** n - 1)
                L = np.cumsum(lpm[:, o], axis=1); row = []
                for m in grid:
                    col = L[:, m - 1]; bst = np.flatnonzero(col == col.max()); row.append((tr in bst) / len(bst))
                ga.append(row)
            r["model_acc"] = np.mean(ga, 0).tolist()
        out["agents"][pol] = r
        txt = f"{c90:.0f}" if np.isfinite(c90) else f"not reached within {r['max_copies']}"
        print(f"[{pol:12s}] fitted per-site visibility: single-copy a = {aa:.2f}, two-copy b = {bb:.2f}  | copies for 90%: {txt}"
              f"  | accuracy at {r['max_copies']} copies: {r['acc_at_max']:.2f} [{lo[~np.isnan(acc)][-1]:.2f}, {hi[~np.isnan(acc)][-1]:.2f}]"
              + (f"  (noise model: {r['model_acc'][int(np.flatnonzero(~np.isnan(acc))[-1])]:.2f})" if model else ""), flush=True)
    json.dump(out, open(os.path.join(HERE, "runs", f"{a.step}_shortcut_{a.mode}.json"), "w"), indent=1)
    print(f"saved runs/{a.step}_shortcut_{a.mode}.json")


if __name__ == "__main__":
    main()
