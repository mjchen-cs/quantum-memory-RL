# -*- coding: utf-8 -*-
"""
Drift-tracking task (Definition 1 of qdrift_theory) played on RECORDED WK_C180 data: no new QPU time.

Hidden Pauli P_t: P_1 uniform on the evaluation set S, and in every later round, with probability q, a redraw
(uniform on S). In every round the agent receives k = 4 copies of rho_{P_t}:
  * quantum agents (Bell / learned front-end): 2 pairs = 2 recorded quantum circuits of P_t (evaluation half, one shot
    each, every circuit used at most once per run);
  * classical-shadow agent: 4 single copies = 2 recorded classical circuits of P_t (random Pauli bases), one shot each.
Both agents use the exact Bayesian forward filter of the hidden-Markov model over ALL N non-identity Paulis (they do not
know S), with the hardware-calibrated likelihoods of hw_analyze (kappa_q from the calibration half; kappa_c of the run),
and output the posterior mode. Reward 1{a_t = P_t}; regret R_T = sum_t (1 - reward).

The evaluation set S is the 8 Paulis on which S5C measured the classical baseline and S5 measured both quantum policies
(n = 5), so both agents face exactly the same hidden process (common random numbers for P_t across agents).
The theory lower bound of Theorem 1 for every non-adaptive single-copy agent is printed for reference; it assumes P
uniform on all N Paulis, while here P is uniform on S, which can only make the task easier for every agent.

Usage:  python hw_drift.py --T 1000 --runs 40 --q 0.005,0.01,0.02,0.05
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hw_analyze as H                                  # noqa: E402
from fast_site import SiteModel, bell_theta              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum", default="S5", help="step with the quantum circuits")
    ap.add_argument("--classical", default="S5C", help="step with the classical circuits (same Paulis)")
    ap.add_argument("--mode", default="origin")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--q", default="0.005,0.01,0.02,0.05")
    ap.add_argument("--k", type=int, default=4, help="copies per round (even)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--eps", type=float, default=1.0)
    a = ap.parse_args()

    man, raw, _ = H.load_merged(a.classical, a.mode)            # S5C merged with S5: quantum + classical records
    n = max(r["n"] for r in man["records"])
    q_cal, _ = H.pools(man, raw, "cal")
    q_ev, c_all = H.pools(man, raw, "eval")
    S = sorted(c_all)                                           # evaluation set = Paulis with classical data
    Pall = SiteModel(n).P; N = len(Pall); supp = Pall > 0; w = supp.sum(1)
    index = {tuple(p): i for i, p in enumerate(Pall.tolist())}
    kc = H.kappa_c_from(n, c_all, a.eps)
    vis_c = a.eps * kc ** w
    thetas = dict(learned=np.array(man["theta"]), bell=bell_theta())
    agents = {}
    for pol, th in thetas.items():
        cal = {P: v for (p, P), v in q_cal.items() if p == pol}
        ev = {P: v for (p, P), v in q_ev.items() if p == pol and P in set(S)}
        if len(ev) < len(S):
            continue
        kq = H.kappa_from(th, n, a.eps, cal)
        _, pQ = H.quantum_table(th, n, a.eps, kq)
        agents[pol] = dict(kind="quantum", lik=pQ, pool=ev, kappa=kq)        # pQ[hypothesis, outcome]
    agents["classical"] = dict(kind="classical", pool={P: c_all[P] for P in S}, kappa=kc)
    pairs = a.k // 2
    d = 2 ** n
    l0 = int(np.floor((d + 1) * (0.5 * np.log(N) - np.log(2)) / (a.k * a.eps ** 2)))
    print(f"n={n}, N={N}, evaluation set |S|={len(S)}, k={a.k}; kappa_c {kc:.3f}; "
          + ", ".join(f"kappa_q[{p}] {agents[p]['kappa']:.3f}" for p in agents if agents[p]['kind'] == 'quantum'), flush=True)
    print("pool sizes (circuits per Pauli): " + ", ".join(f"{p} {min(len(v) for v in agents[p]['pool'].values())}" for p in agents),
          flush=True)

    out = dict(n=n, k=a.k, T=a.T, runs=a.runs, S=[list(P) for P in S], kappa_c=kc,
               kappa_q={p: agents[p]["kappa"] for p in agents if agents[p]["kind"] == "quantum"}, results={})
    for q in [float(x) for x in a.q.split(",")]:
        res = {p: [] for p in agents}; reuse = {p: 0 for p in agents}
        for run in range(a.runs):
            rng = np.random.default_rng([a.seed, run, int(q * 1e6)])
            Pt = []; cur = S[rng.integers(len(S))]
            for t in range(a.T):
                if t > 0 and rng.random() < q:
                    cur = S[rng.integers(len(S))]
                Pt.append(cur)
            for p, ag in agents.items():
                r_ = np.random.default_rng([a.seed, run, int(q * 1e6), {"learned": 1, "bell": 2, "classical": 3}.get(p, 4)])
                order = {P: list(r_.permutation(len(v))) for P, v in ag["pool"].items()}
                logpost = np.full(N, -np.log(N)); regret = 0.0; curve = []
                for t in range(a.T):
                    if t > 0:                                       # HMM prediction step (stay or uniform redraw)
                        post = np.exp(logpost - logpost.max()); post /= post.sum()
                        post = (1 - q) * post + q / N
                        logpost = np.log(post)
                    P = Pt[t]
                    for _ in range(pairs):                         # one recorded circuit per pair / per two copies
                        if not order[P]:
                            order[P] = list(r_.permutation(len(ag["pool"][P]))); reuse[p] += 1
                        circ = ag["pool"][P][order[P].pop()]
                        if ag["kind"] == "quantum":
                            b = circ[r_.integers(len(circ))]
                            logpost += np.log(ag["lik"][:, b])
                        else:
                            sa, sb, ba, bb = circ
                            j = r_.integers(len(ba))
                            for s_, row in ((sa, ba[j]), (sb, bb[j])):
                                s_ = np.array(s_)
                                compat = np.all((Pall == 0) | (Pall == s_[None, :]), axis=1)
                                sign = np.prod(np.where(supp, row[None, :], 1), axis=1)
                                logpost += np.log(np.clip(1 + compat * vis_c * sign, 1e-12, None))
                    best = np.flatnonzero(logpost == logpost.max())
                    regret += 1 - (index[P] in best) / len(best)
                    curve.append(regret)
                res[p].append(curve)
        out["results"][str(q)] = {}
        line = []
        for p in agents:
            R = np.array(res[p]); fin = R[:, -1]
            out["results"][str(q)][p] = dict(final_mean=float(fin.mean()), final_sem=float(fin.std(ddof=1) / np.sqrt(len(fin))),
                                             mean_curve=R.mean(0)[::10].tolist(), pool_reuses=reuse[p])
            line.append(f"{p} {fin.mean():.0f}+-{fin.std(ddof=1) / np.sqrt(len(fin)):.0f}")
        lb = 0.5 * (a.T - l0) * (1 - (1 - q) ** l0)
        out["results"][str(q)]["lower_bound_nonadaptive"] = lb
        print(f"q={q}: regret over T={a.T} (mean+-sem, {a.runs} runs): " + ", ".join(line)
              + f" | Thm 1 lower bound, any non-adaptive single-copy agent (P uniform on all N): {lb:.0f}", flush=True)
    f = os.path.join(HERE, "runs", f"drift_{a.quantum}_{a.classical}_{a.mode}.json")
    json.dump(out, open(f, "w"), indent=1)
    print("saved", f)


if __name__ == "__main__":
    main()
