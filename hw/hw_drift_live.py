# -*- coding: utf-8 -*-
"""
Drift-tracking task STREAMED on WK_C180 (step DR5; replaces the replay of recorded data in hw_drift.py).

Hidden Pauli P_t uniform on ALL N = 4^n - 1 Paulis, redrawn with probability q per round (sequences stored in the
manifest); each round every agent receives k = 4 copies = 2 fresh circuits of P_t, measured in time order on the chip.
Agents: Bell and the pruned chip-trained front-end (two-copy, native gates) and the classical random-basis agent.
Every agent runs the exact Bayesian forward filter of the hidden-Markov model over all N Paulis (knowing q) and
outputs the posterior mode; reward 1{a_t = P_t}; regret R_T = sum_t (1 - reward).
Likelihood calibration by cross-fitting: for run r, kappa is estimated from the circuits of all OTHER runs.
One shot per circuit; the regret is averaged over --draws random choices of that shot.
Reference: Theorem 1 lower bound on the regret of every non-adaptive single-copy agent (ideal hardware),
lb = (T - l0)(1 - (1 - q)^l0) / 2 with l0 = floor((d + 1)(ln N / 2 - ln 2) / (k eps^2)).
Usage:  python hw_drift_live.py            (after  python hw_run.py --step DR5 --mode origin)
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hw_analyze as H                                    # noqa: E402
from fast_site import SiteModel                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="DR5"); ap.add_argument("--mode", default="origin")
    ap.add_argument("--draws", type=int, default=10); ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    man, raw, _ = H.load(a.step, a.mode)
    dr = man["drift"]; T, k = dr["T"], dr["k"]; n = 5; d = 2 ** n
    Pall = SiteModel(n).P; N = len(Pall); supp = Pall > 0; w = supp.sum(1)
    allP = [tuple(p) for p in Pall.tolist()]
    thetas = {kk: np.array(v) for kk, v in man["thetas_extra"].items()}
    # data[(agent, qi, run)][t] = list of circuit data (quantum: array of site outcome indices; classical: tuple)
    data = {}
    qs = dr["q"]
    for r in man["records"]:
        if r["key"] not in raw:
            continue
        m = r["meta"]; qi = qs.index(m["q"]); cnt = {int(x): v for x, v in raw[r["key"]]["counts"].items() if v > 0}
        if r["kind"] == "quantum":
            ag = m["policy"]
            item = np.array([H.to_site_index(i, n) for i, v in cnt.items() for _ in range(v)])
        else:
            ag = "classical"
            rows = 1 - 2 * np.array([[(i >> (2 * n - 1 - qq)) & 1 for qq in range(2 * n)] for i, v in cnt.items() for _ in range(v)])
            item = (tuple(m["setting_a"]), tuple(m["setting_b"]), rows[:, :n], rows[:, n:])
        data.setdefault((ag, qi, m["run"]), {}).setdefault(m["t"], []).append((tuple(r["P"]), item))
    runs = sorted({(qi, rn) for _, qi, rn in data})
    agents = sorted({ag for ag, _, _ in data}, key=lambda x: (x == "classical", x))
    l0 = int(np.floor((d + 1) * (0.5 * np.log(N) - np.log(2)) / (k * a.eps ** 2)))
    out = dict(step=a.step, n=n, T=T, k=k, q=qs, runs=len(runs), l0=l0, results={})
    print(f"{a.step}: n={n}, N={N}, T={T}, k={k}, {len(runs)} runs; Theorem 1 horizon l0 = {l0} rounds", flush=True)
    rng = np.random.default_rng(a.seed)
    for ag in agents:
        res = {}
        for (qi, rn) in runs:
            # cross-fitted calibration from all other runs
            cal = {}
            for (ag2, qj, rm), byt in data.items():
                if ag2 != ag or (qj, rm) == (qi, rn):
                    continue
                for lst in byt.values():
                    for P, item in lst:
                        cal.setdefault(P, []).append(item)
            if ag == "classical":
                kc = H.kappa_c_from(n, cal, a.eps); vis = a.eps * kc ** w; kap = kc
            else:
                kap = H.kappa_from(thetas[ag], n, a.eps, cal)
                _, pQ = H.quantum_table(thetas[ag], n, a.eps, kap); lQ = np.log(pQ)
            q = qs[qi]; byt = data[(ag, qi, rn)]; regs = []
            for _ in range(a.draws):
                logpost = np.full(N, -np.log(N)); reg = 0.0; curve = []
                for t in range(T):
                    if t > 0:
                        post = np.exp(logpost - logpost.max()); post /= post.sum()
                        logpost = np.log((1 - q) * post + q / N)
                    Ptrue = None
                    for P, item in byt.get(t, []):
                        Ptrue = P
                        if ag == "classical":
                            sa, sb, ba, bb = item; j = rng.integers(len(ba))
                            for s_, row in ((sa, ba[j]), (sb, bb[j])):
                                s_ = np.array(s_)
                                compat = np.all((Pall == 0) | (Pall == s_[None, :]), axis=1)
                                sign = np.prod(np.where(supp, row[None, :], 1), axis=1)
                                logpost += np.log(np.clip(1 + compat * vis * sign, 1e-12, None))
                        else:
                            logpost += lQ[:, item[rng.integers(len(item))]]
                    if Ptrue is None:
                        Ptrue = allP[man["drift"]["sequences"][f"{qi}_{rn}"][t]]
                    best = np.flatnonzero(logpost == logpost.max())
                    reg += 1 - (index_of(Ptrue, n) in best) / len(best)
                    curve.append(reg)
                regs.append(curve)
            res[f"{qi}_{rn}"] = dict(kappa=float(kap), curve=np.mean(regs, 0)[::5].tolist(), final=float(np.mean(regs, 0)[-1]))
        out["results"][ag] = res
    for qi, q in enumerate(qs):
        lb = 0.5 * (T - l0) * (1 - (1 - q) ** l0)
        line = []
        for ag in agents:
            f = np.array([v["final"] for key, v in out["results"][ag].items() if key.startswith(f"{qi}_")])
            line.append(f"{ag} {f.mean():.0f}+-{f.std(ddof=1) / np.sqrt(len(f)):.0f}")
            out["results"][ag][f"q{q}_mean"] = float(f.mean()); out["results"][ag][f"q{q}_sem"] = float(f.std(ddof=1) / np.sqrt(len(f)))
        out[f"lower_bound_q{q}"] = lb
        print(f"q={q}: regret over T={T} (mean+-sem over runs): " + ", ".join(line)
              + f" | Thm 1 bound, any non-adaptive single-copy agent: {lb:.0f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "runs", f"{a.step}_drift_{a.mode}.json"), "w"), indent=1)
    print(f"saved runs/{a.step}_drift_{a.mode}.json")


def index_of(P, n):
    i = 0
    for x in P:
        i = 4 * i + int(x)
    return i - 1                                             # Pall excludes the identity (index 0 of the product)


if __name__ == "__main__":
    main()
