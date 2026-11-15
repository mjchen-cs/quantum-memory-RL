# -*- coding: utf-8 -*-
"""
S0 rehearsal: qubit selection from the WK_C180 calibration snapshot and a local noise model that predicts how many
copies the learned quantum agent and the Bayes-optimal classical-shadow agent need on the chip.

Noise model (all local, so outcome probabilities still factorize over sites):
  * preparation: each prepared single-qubit eigenstate keeps a fraction a_q of its Bloch vector
    (three single-qubit gates at the calibrated gate fidelity, plus dephasing exp(-t/T2) over the circuit);
  * site circuit V: two-qubit depolarizing channel with average fidelity F = f_CZ^2 * g_a^6 * g_b^6
    (CRX compiles into two CZ and single-qubit gates);
  * readout: symmetric bit flip with probability e_q = 1 - readout fidelity, or e_q = 0 for ideal mitigation.
Classical shadows: per-qubit visibility v_q = a_q * g_q * (1 - 2 e_q).
"""
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fast_site import SiteModel, tables      # noqa: E402

T_CIRC_US = 0.3


def load_chip(path):
    d = json.load(open(path))
    s1 = {}
    for x in d["single_qubits"]:
        q = int(x["qubit"])
        if q in set(d["available_qubits"]) and (x.get("gate_fidelity") or 0) > 0 and (x.get("readout_fidelity") or 0) > 0:
            s1[q] = dict(g=float(x["gate_fidelity"]), r=float(x["readout_fidelity"]), t2=float(x.get("t2") or np.inf))
    f2 = {}
    for x in d["double_qubits"]:
        a, b = map(int, x["qubits"])
        if a in s1 and b in s1 and (x.get("fidelity") or 0) > 0:
            f2[(a, b)] = float(x["fidelity"])
    return d, s1, f2


def qubit_params(q, s1, dephase=True):
    g, r, t2 = s1[q]["g"], s1[q]["r"], s1[q]["t2"]
    a = (1 - 2 * (1 - g)) ** 3 * (np.exp(-T_CIRC_US / t2) if dephase else 1.0)
    return dict(a=a, g=g, e=1 - r)


def select_sites(n, s1, f2, dephase=True):
    """Greedy choice of n disjoint couplers (one per site) maximizing the predicted two-copy visibility."""
    cand = []
    for (a, b), f in f2.items():
        pa, pb = qubit_params(a, s1, dephase), qubit_params(b, s1, dephase)
        F = f ** 2 * pa["g"] ** 6 * pb["g"] ** 6
        score = (1 - 4 / 3 * (1 - F)) * pa["a"] * pb["a"] * (1 - 2 * pa["e"]) * (1 - 2 * pb["e"])
        cand.append((score, a, b, f))
    cand.sort(reverse=True)
    used, sites = set(), []
    for score, a, b, f in cand:
        if a in used or b in used:
            continue
        sites.append(dict(a=a, b=b, f_cz=f, score=score)); used |= {a, b}
        if len(sites) == n:
            break
    return sites


def noisy_quantum_probs(n, theta, sites, s1, eps=1.0, twirl=True, mitigated=False, dephase=True):
    S = SiteModel(n, eps, twirl=twirl)
    g = tables(theta)                                         # g[k, sigma, beta], k = A, B, AB
    G = []                                                    # per site noisy tables
    for st in sites:
        pa, pb = qubit_params(st["a"], s1, dephase), qubit_params(st["b"], s1, dephase)
        F = st["f_cz"] ** 2 * pa["g"] ** 6 * pb["g"] ** 6
        p = 4 / 3 * (1 - F)
        ea, eb = (0.0, 0.0) if mitigated else (pa["e"], pb["e"])
        Ma = np.array([[1 - ea, ea], [ea, 1 - ea]]); Mb = np.array([[1 - eb, eb], [eb, 1 - eb]])
        M = np.kron(Ma, Mb)
        shrink = (pa["a"], pb["a"], pa["a"] * pb["a"])
        t = np.ones((3, 4, 4))
        for k in range(3):
            for sgm in range(1, 4):
                t[k, sgm] = M @ ((1 - p) * shrink[k] * g[k, sgm])
        G.append(t)
    prod = np.ones((3, S.N, S.D))
    for i in range(n):
        prod *= G[i][:, S.P[:, i]][:, :, S.B[:, i]]
    if twirl:
        p = (1 + eps ** 2 * prod[2]) / S.D
    else:
        p = (1 + eps * prod[0] + eps * prod[1] + eps ** 2 * prod[2]) / S.D
    p = np.clip(p, 1e-15, None)
    return S, p / p.sum(1, keepdims=True)


def copies90_table(p, trials, rng, grid):
    """Pairs needed for 90% identification with the Bayes decoder of the given outcome table."""
    N, D = p.shape
    lp, cdf = np.log(p), np.cumsum(p, 1)
    acc = np.zeros(len(grid))
    for _ in range(trials):
        t = rng.integers(N); L = np.zeros(N); used = 0
        for j, m in enumerate(grid):
            bs = np.searchsorted(cdf[t], rng.random(m - used)).clip(0, D - 1); used = m
            L += lp[:, bs].sum(1)
            best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best)
    return interp90(acc / trials, grid)


def interp90(acc, grid):
    idx = np.flatnonzero(acc >= 0.9)
    if len(idx) == 0:
        return float("nan")
    j = idx[0]
    if j == 0:
        return float(grid[0])
    return float(np.exp(np.log(grid[j - 1]) + (0.9 - acc[j - 1]) / (acc[j] - acc[j - 1]) * (np.log(grid[j]) - np.log(grid[j - 1]))))


def classical_copies90(n, sites, s1, eps, trials, rng, grid, mitigated=False, dephase=True):
    """Bayes-optimal random-Pauli shadows on the same 2n qubits: copy A on the a-qubits, copy B on the b-qubits."""
    P = SiteModel(n).P
    N = len(P); supp = P > 0
    vis = []
    for side in ("a", "b"):
        v = []
        for st in sites:
            pq = qubit_params(st[side], s1, dephase)
            e = 0.0 if mitigated else pq["e"]
            v.append(pq["a"] * pq["g"] * (1 - 2 * e))
        vis.append(np.array(v))
    VP = [np.prod(np.where(supp, vv[None, :], 1.0), axis=1) for vv in vis]        # (N,) per copy
    acc = np.zeros(len(grid))
    for _ in range(trials):
        t = rng.integers(N); L = np.zeros(N); used = 0
        for j, m in enumerate(grid):
            while used < m:
                side = used % 2                       # alternate copies A and B
                b = rng.integers(1, 4, size=n)
                compat_true = np.all((P[t] == 0) | (P[t] == b))
                # sample outcome bits: parity over support has bias eps*VP[t] if compatible
                bits = rng.integers(0, 2, size=n)
                sp = np.nonzero(P[t])[0]
                if compat_true:
                    want_even = rng.random() < (1 + eps * VP[side][t]) / 2
                    if (bits[sp].sum() % 2 == 0) != want_even:
                        bits[sp[0]] ^= 1
                compat = np.all((P == 0) | (P == b[None, :]), axis=1)
                par = (supp & (bits[None, :] == 1)).sum(1) % 2
                L += np.log(1 + eps * compat * VP[side] * (1 - 2 * par))
                used += 1
            best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best)
    return interp90(acc / trials, grid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", default="runs/chip.json")
    ap.add_argument("--agent", default="agent_n2.json")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--out", default="rehearsal.json")
    a = ap.parse_args()
    d, s1, f2 = load_chip(a.chip)
    theta = np.array(json.load(open(a.agent))["theta"])
    rng = np.random.default_rng(0)
    grid = np.unique(np.round(np.logspace(0, 3.6, 40)).astype(int))
    res = dict(chip_fetched=d["fetched"], eps=a.eps, rows=[])
    print(f"chip snapshot {d['fetched']}: {len(s1)} usable qubits, {len(f2)} usable couplers", flush=True)
    for n in (2, 3, 4, 5):
        sites = select_sites(n, s1, f2)
        row = dict(n=n, sites=sites)
        for label, mit, deph in (("ideal", None, None), ("raw readout", False, True), ("mitigated", True, True)):
            if label == "ideal":
                S = SiteModel(n, a.eps, twirl=True); p = S.probs(theta); p = p / p.sum(1, keepdims=True)
                qc = copies90_table(p, 300, rng, grid)
                cc = classical_copies90(n, [dict(a=s["a"], b=s["b"]) for s in sites],
                                        {q: dict(g=1.0, r=1.0, t2=np.inf) for q in s1}, a.eps, 150, rng, grid)
            else:
                S, p = noisy_quantum_probs(n, theta, sites, s1, a.eps, mitigated=mit, dephase=deph)
                qc = copies90_table(p, 300, rng, grid)
                cc = classical_copies90(n, sites, s1, a.eps, 150, rng, grid, mitigated=mit, dephase=deph)
            row[label] = dict(quantum_copies90=2 * qc, classical_copies90=cc)
            print(f"n={n} [{label:11s}] quantum (learned, twirl) copies90 = {2*qc:7.1f}   classical shadows = {cc:7.1f}", flush=True)
        res["rows"].append(row)
        json.dump(res, open(a.out, "w"), indent=1)
    print("selected sites (n=5):", [(s["a"], s["b"], round(s["f_cz"], 4)) for s in res["rows"][-1]["sites"]])
