# -*- coding: utf-8 -*-
"""
Task 2 analysis (detection of a hidden signal) on recorded WK_C180 data (steps DT5/DT6/DT7 quantum, DTC5 classical).

Agents (likelihoods calibrated on the calibration half of the signal circuits, evaluated on the other half):
  * quantum front-ends (Bell, task-1 agent trained on the chip, detection agents trained on the chip): per pair the
    log-likelihood ratio of hypothesis Q against I/d is log(1 + kappa^w(Q) prod_i g_AB[Q_i, beta_i]);
  * classical single-copy agent: both copies of a pair measured in independent random Pauli bases; for hypothesis Q
    the pair LLR is log(1 + kappa_c^(2w) c_A(Q) c_B(Q) s_A(Q) s_B(Q)) (c = basis compatible with Q, s = product of the
    outcomes on the support of Q): with the shared twirl sign only the two-copy correlation carries information.
Fixed-sample test: mixture likelihood ratio over all N hypotheses, H1 iff log mean_Q exp(LLR_Q) > 0; accuracy =
(1 - P_FA + 1 - P_miss)/2 versus the number of copies; copies for 90 % compared with Theorem 3 (1.6 (d+1)).
Quickest detection: GLR-CUSUM, C_Q <- max(0, C_Q + llr_Q), alarm when max_Q C_Q > h; h set so that the mean run length
on null data is ARL0 pairs; reported: mean detection delay (copies) when the signal is present from the start.
Usage:  python hw_detect.py --step DT5            python hw_detect.py --step DTC5
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hw_analyze as H                                    # noqa: E402
from fast_site import tables, bell_theta                  # noqa: E402


def order(circs, K, rng):
    """K distinct circuits in random order, one random shot of each. Every circuit is a separate preparation; the
    shots of one circuit repeat the SAME prepared product state, so they are not independent copies and are never
    combined in one episode (v4; v3 reused shots once the circuits ran out, which inflated large-M accuracies)."""
    return [(int(i), int(rng.integers(64))) for i in rng.permutation(len(circs))[:K]]


def lse_mean(L):
    m = L.max(0); return m + np.log(np.mean(np.exp(L - m), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True); ap.add_argument("--mode", default="origin")
    ap.add_argument("--trials", type=int, default=300); ap.add_argument("--arl0", type=int, default=500)
    ap.add_argument("--seq_runs", type=int, default=100); ap.add_argument("--kappa_c", type=float, default=0.936,
                    help="classical per-qubit visibility if too few both-compatible pairs to estimate it (S5C value)"); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    man, raw, _ = H.load(a.step, a.mode)
    n = max(r["n"] for r in man["records"]); d = 2 ** n
    Pall = np.array(list(itertools.product(range(4), repeat=n)))[1:]; N = len(Pall); w = (Pall > 0).sum(1)
    supp = Pall > 0
    shift = np.array([2 * (n - 1 - i) for i in range(n)])
    bound = float(np.ceil(1.6 * (d + 1)))
    ap_grid_max = 4.5 if a.step.startswith("DTC") else 3.7
    grid = np.unique((np.round(np.logspace(0.3, ap_grid_max, 46)).astype(int) // 2) * 2)
    out = dict(step=a.step, n=n, bound_theorem3=bound, grid=grid.tolist(), agents={})
    q_cal, c_cal = H.pools(man, raw, "cal"); q_ev, c_ev = H.pools(man, raw, "eval")
    q_all, c_all = H.pools(man, raw, None)
    zero = tuple([0] * n)

    agents = {}
    if q_ev:
        thetas = dict(bell=bell_theta(), **{k: np.array(v) for k, v in (man.get("thetas_extra") or {}).items()})
        for pol in sorted({p for p, _ in q_ev}):
            g = tables(thetas[pol])[2]; s_sq = (g ** 2).mean(1); vals = []
            for (p_, P), circs in q_cal.items():
                if p_ != pol or P == zero:
                    continue
                Pv = np.array(P); dg = (np.concatenate(circs)[:, None] >> shift[None, :]) & 3
                prod = np.prod(g[Pv[None, :].repeat(len(dg), 0), dg], axis=1)
                lam = prod.mean() / float(np.prod(s_sq[Pv]))
                if lam > 0:
                    vals.append(lam ** (1 / int((Pv > 0).sum())))
            kq = float(np.median(vals)); amp = kq ** w

            def llr(shots, g=g, amp=amp):                      # (K,) outcome indices -> (N, K) pair LLRs
                dg = (np.asarray(shots)[:, None] >> shift[None, :]) & 3
                V = np.ones((N, len(shots)))
                for i in range(n):
                    V *= g[Pall[:, i]][:, dg[:, i]]
                return np.log(np.clip(1 + amp[:, None] * V, 1e-12, None))
            sig = {P: v for (p_, P), v in q_ev.items() if p_ == pol and P != zero}
            null = q_all[(pol, zero)]
            sig_all = {P: v for (p_, P), v in q_all.items() if p_ == pol and P != zero}
            agents[pol] = dict(kappa=kq, sig=sig, sig_all=sig_all, null=null, llr=llr,
                               draw=lambda circs, idx: np.array([int(circs[i][j % len(circs[i])]) for i, j in idx]))
    if c_all:
        sig_cal = {P: v for P, v in c_cal.items() if P != zero}
        vals = []
        for P, circs in sig_cal.items():                       # kappa_c from pairs whose two bases both fit P
            sp = [i for i in range(n) if P[i]]; prods = []
            for sa, sb, ba, bb in circs[: len(circs) // 2]:
                if all(sa[i] == P[i] and sb[i] == P[i] for i in sp):
                    prods += list(np.prod(ba[:, sp], 1) * np.prod(bb[:, sp], 1))
            if prods and np.mean(prods) > 0:
                vals.append(np.mean(prods) ** (1 / (2 * len(sp))))
        n_used = len(vals)
        kc = float(np.median(vals)) if n_used >= 3 else a.kappa_c       # both bases fit P with prob. 3^(-2w): rare
        vis2 = kc ** (2 * w)
        held = {P: v[len(v) // 2:] for P, v in c_all.items() if P != zero}

        def llr_c(items, vis2=vis2):                          # vectorised over hypotheses (N) and pairs (K)
            SA = np.array([it[0] for it in items]); SB = np.array([it[1] for it in items])
            RA = np.array([it[2] for it in items]); RB = np.array([it[3] for it in items])
            ok = np.ones((N, len(items)), bool); sg = np.ones((N, len(items)))
            for i in range(n):
                Pi = Pall[:, i][:, None]; on = supp[:, i][:, None]
                ok &= (Pi == 0) | ((Pi == SA[None, :, i]) & (Pi == SB[None, :, i]))
                sg *= np.where(on, (RA[:, i] * RB[:, i])[None, :], 1)
            return np.log(np.clip(1 + vis2[:, None] * ok * sg, 1e-12, None))

        def draw_c(circs, idx):
            out_ = []
            for i, j in idx:
                sa, sb, ba, bb = circs[i]; out_.append((sa, sb, ba[j % len(ba)], bb[j % len(bb)]))
            return out_
        agents["classical"] = dict(kappa=kc, sig=held, sig_all={P: v for P, v in c_all.items() if P != zero},
                                   null=c_all[zero], llr=llr_c, draw=draw_c)

    for name, ag in agents.items():
        Ps = sorted(ag["sig"]); kmax = grid[-1] // 2
        acc = np.zeros(len(grid)); cnt = np.zeros(len(grid)); S = np.full((a.trials, len(grid)), np.nan); Y = np.arange(a.trials) % 2
        for t in range(a.trials):
            y = t % 2
            circs = ag["sig"][Ps[rng.integers(len(Ps))]] if y else ag["null"]
            K = min(kmax, len(circs))
            idx = order(circs, K, rng)
            C = np.cumsum(ag["llr"](ag["draw"](circs, idx)), axis=1)
            for j, M in enumerate(grid):
                k = M // 2
                if k > K:
                    cnt[j:] = np.nan; break
                S[t, j] = lse_mean(C[:, k - 1:k])[0]
                acc[j] += float((S[t, j] > 0) == y); cnt[j] += 1
        accu = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        c90 = H.interp(accu, grid)
        # threshold calibrated by 2-fold cross-validation over trials (honest), and oracle threshold (optimistic, for reference)
        acc_or = np.full(len(grid), np.nan); acc_cv = np.full(len(grid), np.nan)
        fold = rng.permutation(a.trials) % 2

        def best_t(v, y):
            th = np.r_[-np.inf, np.unique(v)]
            b = [0.5 * ((v[y == 1] > t_).mean() + (v[y == 0] <= t_).mean()) for t_ in th]
            return th[int(np.argmax(b))], max(b)

        def bal(v, y, t_):
            return 0.5 * ((v[y == 1] > t_).mean() + (v[y == 0] <= t_).mean())
        for j in range(len(grid)):
            v = S[:, j]
            if np.isnan(v).any():
                continue
            acc_or[j] = best_t(v, Y)[1]
            r = []
            for f in (0, 1):
                t_ = best_t(v[fold == f], Y[fold == f])[0]; r.append(bal(v[fold != f], Y[fold != f], t_))
            acc_cv[j] = np.mean(r)
        c90_cv = H.interp(acc_cv, grid)
        c90_or = H.interp(acc_or, grid)
        # quickest detection: GLR-CUSUM on pairs
        def run(circs, h, maxlen):
            """GLR-CUSUM over distinct circuits (no circuit used twice); returns (run length, censored)."""
            Cq = np.zeros(N); t = 0; perm = rng.permutation(len(circs)); maxlen = min(maxlen, len(circs))
            while t < maxlen:
                blk = min(64, maxlen - t)
                idx = [(int(i), int(rng.integers(64))) for i in perm[t:t + blk]]
                L = ag["llr"](ag["draw"](circs, idx))
                for k in range(blk):
                    Cq = np.maximum(0, Cq + L[:, k]); t += 1
                    if Cq.max() > h:
                        return t, False
            return maxlen, True
        lo, hi = 0.5, 40.0
        for _ in range(12):                                     # threshold for mean run length ARL0 on null data
            h = 0.5 * (lo + hi)
            arl = np.mean([run(ag["null"], h, 4 * a.arl0)[0] for _ in range(max(20, a.seq_runs // 3))])
            lo, hi = (h, hi) if arl < a.arl0 else (lo, h)
        h = 0.5 * (lo + hi)
        pool = ag.get("sig_all", ag["sig"])                      # CUSUM needs no calibration split: all signal circuits
        dc = [run(pool[Ps[rng.integers(len(Ps))]], h, 20 * a.arl0) for _ in range(a.seq_runs)]
        delays = [x for x, _ in dc]; cens = float(np.mean([c for _, c in dc]))
        out["agents"][name] = dict(kappa=ag["kappa"], acc=accu.tolist(), copies90=c90, acc_oracle=acc_or.tolist(), copies90_oracle=c90_or, acc_cv=acc_cv.tolist(), copies90_cv=c90_cv, cusum_h=h,
                                   delay_pairs_mean=float(np.mean(delays)), delay_pairs_sem=float(np.std(delays) / np.sqrt(len(delays))),
                                   delay_censored_frac=cens, n_signal_circuits_eval=int(np.median([len(v) for v in ag["sig"].values()])),
                                   n_null_circuits=len(ag["null"]))
        print(f"[{name:9s}] kappa {ag['kappa']:.3f}  fixed-sample copies for 90%: {c90:.0f} (CV threshold {c90_cv:.0f}, oracle {c90_or:.0f})  | quickest detection at "
              f"ARL0 = {a.arl0} pairs: mean delay {2 * np.mean(delays):.0f} copies"
              f"{f' ({100 * cens:.0f}% of runs reached the end of the data: lower bound)' if cens else ''}", flush=True)
    print(f"Theorem 3: any adaptive single-copy agent needs >= {bound:.0f} copies for 90% (ideal hardware)")
    json.dump(out, open(os.path.join(HERE, "runs", f"{a.step}_detect_{a.mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
