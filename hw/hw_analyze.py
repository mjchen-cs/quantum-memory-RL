# -*- coding: utf-8 -*-
"""
Analysis of the first hardware batch.

  python hw_analyze.py --step S1 [--mode origin]   timing per job (fill in the billed amount by hand)
  python hw_analyze.py --step S2                    visibilities on the chip and a prediction for n = 3, 4
  python hw_analyze.py --step S3 / S4               accuracy versus number of copies, quantum versus classical

Decoders (both strategies get the same treatment):
  * calibrated Bayes: likelihood of every Pauli hypothesis with the ideal model shrunk by visibilities measured in S2;
  * model-free score: quantum, sum over pairs of prod_i g_AB[P_i, b_i] (the decoder family of stage 3);
                      classical, sum over compatible copies of the product of +-1 outcomes on the support.
"""
import argparse
import itertools
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fast_site import SiteModel, tables      # noqa: E402


def load(step, mode):
    d = os.path.join(HERE, "runs", f"{step}_{mode}")
    man = json.load(open(os.path.join(d, "manifest.json")))
    raw = json.load(open(os.path.join(d, "raw.json")))
    jobs = json.load(open(os.path.join(d, "jobs.json")))
    return man, raw, jobs


MERGE = {"S6X": "S6", "S5C": "S5"}      # extension steps are analysed together with the step they extend


def load_merged(step, mode):
    """Records and counts of a step, plus those of the step it extends (S6X -> S6, S5C -> S5) and, for S3P, the
    classical-shadow circuits of S3 (server-mapped run)."""
    man, raw, jobs = load(step, mode)
    extra = []
    if step in MERGE:
        bm, braw, _ = load(MERGE[step], mode); extra.append((bm["records"], braw))
    if step == "S3P":
        cm, craw, _ = load("S3", "origin_sm")
        extra.append(([r for r in cm["records"] if r["kind"] == "classical"], craw))
    for recs, r_ in extra:
        recs = [r for r in recs if r["key"] in r_]
        man["records"] += recs; raw.update({r["key"]: r_[r["key"]] for r in recs})
    return man, raw, jobs


def site_digits(idx, n):
    """logical outcome index (qubit 0 = MSB; A = 0..n-1, B = n..2n-1) -> per-site digits beta_i = 2 a_i + b_i"""
    bits = [(idx >> (2 * n - 1 - q)) & 1 for q in range(2 * n)]
    return [2 * bits[i] + bits[n + i] for i in range(n)]


def ab_table(theta, n):
    """prod_i g_AB[P_i, beta_i] for all non-identity P (rows) and outcomes (columns, SiteModel order)."""
    S = SiteModel(n)
    g = tables(theta)[2]
    prod = np.ones((S.N, S.D))
    for i in range(n):
        prod *= g[S.P[:, i]][:, S.B[:, i]]
    return S, prod


def to_site_index(idx, n):
    dig = site_digits(idx, n)
    k = 0
    for x in dig:
        k = k * 4 + x
    return k


def pools(man, raw, half=None):
    """Per circuit (= one independent preparation).
    quantum:   {(policy, P): [array of site-order outcome indices, one per shot] per circuit}
    classical: {P: [(setting_a, setting_b, bits_a (shots x n, +-1), bits_b)] per circuit}
    half = 'cal' keeps odd quantum repetitions, 'eval' even ones, None all (classical circuits always kept)."""
    q, c = {}, {}
    for r in man["records"]:
        k = r["key"]
        if k not in raw:
            continue
        rep = r.get("meta", {}).get("rep")
        if r["kind"] == "quantum" and half is not None and rep is not None and (rep % 2 == 1) != (half == "cal"):
            continue
        n = r["n"]; P = tuple(r["P"])
        cnt = {int(x): v for x, v in raw[k]["counts"].items() if v > 0}
        if r["kind"] == "quantum":
            pol = r.get("meta", {}).get("policy", "learned")
            shots = [to_site_index(idx, n) for idx, v in cnt.items() for _ in range(v)]
            if shots:
                q.setdefault((pol, P), []).append(np.array(shots))
        else:
            rows = [[(idx >> (2 * n - 1 - qq)) & 1 for qq in range(2 * n)] for idx, v in cnt.items() for _ in range(v)]
            if rows:
                b = 1 - 2 * np.array(rows)
                c.setdefault(P, []).append((tuple(r["meta"]["setting_a"]), tuple(r["meta"]["setting_b"]), b[:, :n], b[:, n:]))
    return q, c


# ------------------------------------------------------------------ S1
def analyze_S1(man, raw, jobs):
    print("job, circuits x shots, wall time from submission to collection (s)")
    for jid, j in jobs.items():
        dt = (j.get("finished_unix") or 0) - (j.get("submitted_unix") or 0)
        print(f"  {jid}: {len(j['keys'])} x {j['shots']}  {dt:.1f}s  info={j.get('info')}")
    print("Write down the account balance before and after S1 to obtain the billed seconds per job type:\n"
          "billed = c0 * circuits + c1 * shots / 1000; the single-circuit jobs give c1, the 25-circuit job gives c0.")


# ------------------------------------------------------------------ S2
def analyze_S2(man, raw, eps, mode="origin"):
    theta = np.array(man["theta"]); n = 2
    S, prod = ab_table(theta, n)
    pq, pc0 = pools(man, raw)
    pq = {P: np.concatenate(v) for (pol, P), v in pq.items()}
    pc = {P: [(s_a, tuple(row)) for s_a, s_b, ba, bb in v for row in ba] + [(s_b, tuple(row)) for s_a, s_b, ba, bb in v for row in bb]
          for P, v in pc0.items()}
    p_ideal = SiteModel(n, eps, twirl=True).probs(theta)
    rows = []
    for P, arr in pq.items():
        a = S.P.tolist().index(list(P))
        ideal = float((p_ideal[a] * prod[a]).sum() / p_ideal[a].sum())
        emp = float(prod[a, arr].mean())
        w = sum(1 for x in P if x)
        lam = emp / ideal
        cs = pc.get(P, [])
        supp = [i for i in range(n) if P[i]]
        single = np.mean([np.prod([b[i] for i in supp]) for s, b in cs]) / eps if cs else np.nan
        rows.append(dict(P="".join("IXYZ"[x] for x in P), weight=w, two_copy_visibility=lam,
                         per_site=lam ** (1 / w), single_copy_visibility=single, per_qubit=abs(single) ** (1 / w)))
        print(f"  {rows[-1]['P']}: two-copy visibility {lam:.3f} (per site {lam ** (1 / w):.3f}); "
              f"single-copy {single:.3f} (per qubit {abs(single) ** (1 / w):.3f})")
    kq = float(np.median([r["per_site"] for r in rows])); kc = float(np.median([r["per_qubit"] for r in rows]))
    print(f"median per-site two-copy visibility kappa_q = {kq:.3f}; per-qubit single-copy visibility kappa_c = {kc:.3f}")
    rng = np.random.default_rng(0)
    pred = {}
    for nn in (3, 4):
        qc = predict_quantum(theta, nn, eps, kq, rng)
        cc = predict_classical(nn, eps, kc, rng)
        pred[nn] = dict(quantum=qc, classical=cc)
        print(f"  prediction n={nn}: copies for 90% identification  quantum {qc:.0f}   classical {cc:.0f}   ratio {cc / qc:.1f}")
    ok = pred[4]["classical"] / pred[4]["quantum"] > 3
    print("GATE:", "PASS - proceed to S3" if ok else "FAIL - stop and review (better qubits, shorter circuit)")
    out = dict(rows=rows, kappa_q=kq, kappa_c=kc, prediction=pred, gate_pass=bool(ok))
    json.dump(out, open(os.path.join(HERE, "runs", f"S2_summary_{mode}.json"), "w"), indent=1)
    return out


def quantum_table(theta, n, eps, kq):
    S, prod = ab_table(theta, n)
    w = (S.P > 0).sum(1)
    p = (1 + eps ** 2 * (kq ** w)[:, None] * prod) / S.D
    p = np.clip(p, 1e-12, None)
    return S, p / p.sum(1, keepdims=True)


def predict_quantum(theta, n, eps, kq, rng, trials=200):
    S, p = quantum_table(theta, n, eps, kq)
    lp, cdf = np.log(p), np.cumsum(p, 1)
    grid = np.unique(np.round(np.logspace(0, 3.3, 30)).astype(int))
    acc = np.zeros(len(grid))
    for _ in range(trials):
        t = rng.integers(S.N); L = np.zeros(S.N); used = 0
        for j, m in enumerate(grid):
            L += lp[:, np.searchsorted(cdf[t], rng.random(m - used)).clip(0, S.D - 1)].sum(1); used = m
            best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best)
    return 2 * interp(acc / trials, grid)


def predict_classical(n, eps, kc, rng, trials=100):
    P = SiteModel(n).P; supp = P > 0; w = supp.sum(1); N = len(P)
    vis = eps * kc ** w
    grid = np.unique(np.round(np.logspace(0, 4, 36)).astype(int))
    acc = np.zeros(len(grid))
    for _ in range(trials):
        t = rng.integers(N); L = np.zeros(N); used = 0
        for j, m in enumerate(grid):
            while used < m:
                b = rng.integers(1, 4, size=n); bits = rng.integers(0, 2, size=n); sp = np.nonzero(P[t])[0]
                if np.all((P[t] == 0) | (P[t] == b)):
                    if (bits[sp].sum() % 2 == 0) != (rng.random() < (1 + vis[t]) / 2):
                        bits[sp[0]] ^= 1
                compat = np.all((P == 0) | (P == b[None, :]), axis=1)
                par = (supp & (bits[None, :] == 1)).sum(1) % 2
                L += np.log(np.clip(1 + compat * vis * (1 - 2 * par), 1e-12, None)); used += 1
            best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best)
    return interp(acc / trials, grid)


def interp(acc, grid, target=0.9):
    idx = np.flatnonzero(acc >= target)
    if len(idx) == 0:
        return float("nan")
    j = idx[0]
    if j == 0:
        return float(grid[0])
    return float(np.exp(np.log(grid[j - 1]) + (target - acc[j - 1]) / (acc[j] - acc[j - 1]) * (np.log(grid[j]) - np.log(grid[j - 1]))))


# ------------------------------------------------------------------ S3 / S4 / S5
def kappa_from(theta, n, eps, pq_cal):
    """Per-site two-copy visibility from the calibration half: median over Paulis of (empirical/ideal)^(1/weight)."""
    S, prod = ab_table(theta, n)
    pid = SiteModel(n, eps, twirl=True).probs(theta)
    index = {tuple(p): i for i, p in enumerate(S.P.tolist())}
    vals = []
    for P, circs in pq_cal.items():
        arr = np.concatenate(circs)
        a = index[P]; w = sum(1 for x in P if x)
        ideal = float((pid[a] * prod[a]).sum() / pid[a].sum())
        lam = prod[a, arr].mean() / ideal
        if lam > 0:
            vals.append(lam ** (1 / w))
    return float(np.median(vals))


def kappa_c_from(n, pc_cal, eps):
    vals = []
    for P, circs in pc_cal.items():
        arr = [(sa, tuple(r)) for sa, sb, ba, bb in circs for r in ba] + [(sb, tuple(r)) for sa, sb, ba, bb in circs for r in bb]
        supp = [i for i in range(n) if P[i]]
        comp = [np.prod([b[i] for i in supp]) for s, b in arr if all(s[i] == P[i] for i in supp)]
        if comp and np.mean(comp) > 0:
            vals.append((np.mean(comp) / eps) ** (1 / len(supp)))
    return float(np.median(vals))


def site_prod_vec(g, Pall, b_idx, n):
    """prod_i g[P_i, beta_i] for ALL hypotheses P (rows of Pall) and ONE outcome index b (site 0 = most significant
    base-4 digit). Factorized over sites, so it never builds the N x 4^n table (needed for n >= 7)."""
    dig = [(b_idx >> (2 * (n - 1 - i))) & 3 for i in range(n)]
    G = g[:, dig]                                         # (4, n)
    return np.prod(G[Pall, np.arange(n)], axis=1)


def analyze_core_lazy(step, man, raw, eps, trials=200, mode="origin"):
    """Quantum-only analysis for large n (n >= 7: 4^n outcomes x (4^n - 1) hypotheses do not fit in memory).
    Same estimators as analyze_core: per-site kappa_q from the calibration half (median over Paulis of
    (empirical / ideal)^(1/weight)), Bayes decoder log(1 + kappa^w prod) and score decoder prod, episodes of m copies
    = m/2 pairs, one shot from each of m/2 different evaluation circuits."""
    import itertools
    n = max(r["n"] for r in man["records"])
    from fast_site import bell_theta
    thetas = dict(learned=np.array(man["theta"]), bell=bell_theta())
    for k_, v_ in (man.get("thetas_extra") or {}).items():
        thetas[k_] = np.array(v_)
    q_cal, _ = pools(man, raw, "cal")
    q_ev, _ = pools(man, raw, "eval")
    policies = sorted({pol for pol, _ in q_ev})
    Pall = np.array(list(itertools.product(range(4), repeat=n)))[1:]; N = len(Pall)
    w = (Pall > 0).sum(1)
    index = {tuple(p): i for i, p in enumerate(Pall.tolist())}
    rng = np.random.default_rng(1)
    grid_q = np.unique((np.round(np.logspace(0.3, 3.3, 30)).astype(int) // 2) * 2)
    out = dict(step=step, n=n, kappa_q={}, grid_q=grid_q.tolist(), results={}, n_paulis={})
    for pol in policies:
        g = tables(thetas[pol])[2]
        s_sq = (g ** 2).mean(1)                           # ideal E[g^2] per site Pauli (twirled, eps = 1)
        vals = []
        for (p_, P), circs in q_cal.items():
            if p_ != pol:
                continue
            arr = np.concatenate(circs); Pv = np.array(P)
            prod = np.array([np.prod([g[Pv[i], (b >> (2 * (n - 1 - i))) & 3] for i in range(n)]) for b in arr])
            ideal = float(np.prod([s_sq[x] for x in Pv])) * eps ** 2
            lam = prod.mean() / ideal; wt = int((Pv > 0).sum())
            if lam > 0:
                vals.append(lam ** (1 / wt))
        kq = float(np.median(vals)); out["kappa_q"][pol] = kq
        amp = (eps ** 2) * kq ** w
        pq = {P: v for (p_, P), v in q_ev.items() if p_ == pol}
        Pset = sorted(pq); out["n_paulis"][pol] = len(Pset)
        for dec in ("bayes", "score"):
            acc = np.zeros(len(grid_q)); cnt = np.zeros(len(grid_q))
            for _ in range(trials):
                P = Pset[rng.integers(len(Pset))]; t = index[P]; circs = pq[P]
                order = rng.permutation(len(circs)); L = np.zeros(N); used = 0
                for j, m in enumerate(grid_q):
                    k = m // 2
                    if k > len(circs):
                        break
                    for i in range(used, k):
                        c = circs[order[i]]; b = int(c[rng.integers(len(c))])
                        v = site_prod_vec(g, Pall, b, n)
                        L += np.log(np.clip(1 + amp * v, 1e-12, None)) if dec == "bayes" else v
                    used = k
                    best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best); cnt[j] += 1
            a_ = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            out["results"][f"{pol}_{dec}"] = dict(acc=a_.tolist(), copies90=interp(a_, grid_q))
            print(f"[{pol:7s} {dec:5s}] kappa_q {kq:.3f}  copies for 90%: {interp(a_, grid_q):.0f} over {len(Pset)} Paulis",
                  flush=True)
    d = 2 ** n; Nn = d * d - 1
    out["single_copy_lower_bound_ideal"] = (d + 1) * (0.9 * np.log(Nn) - np.log(2)) / eps ** 2
    print(f"lower bound for any NON-ADAPTIVE single-copy strategy on ideal hardware: {out['single_copy_lower_bound_ideal']:.0f} copies")
    json.dump(out, open(os.path.join(HERE, "runs", f"{step}_summary_{mode}.json"), "w"), indent=1)
    return out


def analyze_core(step, man, raw, eps, s2, trials=200, mode="origin"):
    n = max(r["n"] for r in man["records"])
    if n >= 7:
        return analyze_core_lazy(step, man, raw, eps, trials=trials, mode=mode)
    from fast_site import bell_theta
    thetas = dict(learned=np.array(man["theta"]), bell=bell_theta())
    if man.get("theta_onchip"):
        thetas["onchip"] = np.array(man["theta_onchip"])
    for k_, v_ in (man.get("thetas_extra") or {}).items():
        thetas[k_] = np.array(v_)
    q_cal, c_cal = pools(man, raw, "cal")
    q_ev, c_ev = pools(man, raw, "eval")
    policies = sorted({pol for pol, _ in q_ev})
    kc = (s2 or {}).get("kappa_c") or (kappa_c_from(n, c_cal, eps) if c_cal else 1.0)   # independent S2 value if available
    Pall = SiteModel(n).P; supp = Pall > 0; w = supp.sum(1); N = len(Pall)
    vis = eps * kc ** w
    index = {tuple(p): i for i, p in enumerate(Pall.tolist())}
    rng = np.random.default_rng(1)
    grid_q = np.unique((np.round(np.logspace(0.3, 3.0, 25)).astype(int) // 2) * 2)
    grid_c = np.unique(np.round(np.logspace(0.3, 3.8, 32)).astype(int))
    Pc = sorted(c_ev)
    out = dict(step=step, n=n, kappa_c=kc, kappa_q={}, grid_q=grid_q.tolist(), grid_c=grid_c.tolist(), results={},
               n_paulis=dict(classical=len(Pc)))

    def run_q(pq, Pset, lp, prodAB, decoder):
        """Episodes of m copies = m/2 pairs, each pair from a different circuit (independent preparations)."""
        acc = np.zeros(len(grid_q)); cnt = np.zeros(len(grid_q))
        for _ in range(trials):
            P = Pset[rng.integers(len(Pset))]; t = index[P]; circs = pq[P]
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

    def run_c(Pset, decoder):
        """Episodes of m copies, two independent copies (A and B) per circuit, one shot per circuit."""
        acc = np.zeros(len(grid_c)); cnt = np.zeros(len(grid_c)); T = max(50, trials // 2)
        for _ in range(T):
            P = Pset[rng.integers(len(Pset))]; t = index[P]; circs = c_ev[P]
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
                    L += (np.log(np.clip(1 + compat * vis * sign, 1e-12, None)) if decoder == "bayes" else compat * sign)
                    used += 1
                best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best); cnt[j] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    for pol in policies:
        pq_ev = {P: v for (p_, P), v in q_ev.items() if p_ == pol}
        pq_cal = {P: v for (p_, P), v in q_cal.items() if p_ == pol}
        kq = kappa_from(thetas[pol], n, eps, pq_cal)
        out["kappa_q"][pol] = kq
        S, pQ = quantum_table(thetas[pol], n, eps, kq)
        _, prodAB = ab_table(thetas[pol], n)
        lp = np.log(pQ)
        Pq = sorted(pq_ev); Pq_same = [P for P in Pc if P in pq_ev]
        out["n_paulis"][pol] = len(Pq)
        for dec in ("bayes", "score"):
            aq = run_q(pq_ev, Pq, lp, prodAB, dec)
            aqs = run_q(pq_ev, Pq_same, lp, prodAB, dec) if Pq_same else np.full(len(grid_q), np.nan)
            out["results"][f"{pol}_{dec}"] = dict(acc=aq.tolist(), acc_same_set=aqs.tolist(),
                                                  copies90=interp(aq, grid_q), copies90_same_set=interp(aqs, grid_q))
            print(f"[{pol:7s} {dec:5s}] kappa_q {kq:.3f}  copies for 90%: {interp(aq, grid_q):.0f} over {len(Pq)} Paulis"
                  f" ({interp(aqs, grid_q):.0f} on the classical set)", flush=True)
    for dec in (("bayes", "score") if Pc else ()):
        ac = run_c(Pc, dec)
        out["results"][f"classical_{dec}"] = dict(acc=ac.tolist(), copies90=interp(ac, grid_c))
        print(f"[classical {dec:5s}] kappa_c {kc:.3f}  copies for 90%: {interp(ac, grid_c):.0f} over {len(Pc)} Paulis", flush=True)
    d = 2 ** n; Nn = d * d - 1
    out["single_copy_lower_bound_ideal"] = (d + 1) * (0.9 * np.log(Nn) - np.log(2)) / eps ** 2
    print(f"lower bound for any NON-ADAPTIVE single-copy strategy on ideal hardware: {out['single_copy_lower_bound_ideal']:.0f} copies")
    json.dump(out, open(os.path.join(HERE, "runs", f"{step}_summary_{mode}.json"), "w"), indent=1)


def analyze_D1(man, raw, eps, mode="origin"):
    """Two-copy visibility per Pauli and per site, for each logical layout."""
    theta = np.array(man["theta"]); n = 4
    S, prod = ab_table(theta, n)
    pid = SiteModel(n, eps, twirl=True).probs(theta)
    index = {tuple(p): i for i, p in enumerate(S.P.tolist())}
    data = {}
    for r in man["records"]:
        if r["key"] not in raw:
            continue
        cnt = {int(x): v for x, v in raw[r["key"]]["counts"].items() if v > 0}
        arr = [to_site_index(i, n) for i, v in cnt.items() for _ in range(v)]
        data.setdefault((r["meta"]["layout"], tuple(r["P"])), []).extend(arr)
    out = {}
    for layout in ("block", "interleave"):
        site = {i: [] for i in range(n)}; full = []
        for (lay, P), arr in data.items():
            if lay != layout:
                continue
            a_ = index[P]
            lam = prod[a_, np.array(arr)].mean() / ((pid[a_] * prod[a_]).sum() / pid[a_].sum())
            w = sum(1 for x in P if x)
            if w == 1:
                site[[i for i in range(n) if P[i]][0]].append(lam)
            else:
                full.append(lam ** (1 / w))
        out[layout] = dict(per_site={i: float(np.mean(v)) for i, v in site.items()}, full_weight_per_site=float(np.median(full)))
        print(f"{layout:10s} per-site visibility (weight-one Paulis): "
              + "  ".join(f"site{i} {np.mean(v):.3f}" for i, v in site.items())
              + f"   | weight-4 Paulis, per site: {np.median(full):.3f}")
    json.dump(out, open(os.path.join(HERE, "runs", f"D1_summary_{mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["S1", "S2", "S3", "S4", "S5", "S6", "D1", "S3P", "S6X", "S5C", "R4", "R5", "T4", "T5", "S5R", "S6R", "S7", "PR5", "PR6", "PR7"], required=True)
    ap.add_argument("--mode", default="origin", help="origin, origin_sm (cloud-mapped run) or dryrun")
    ap.add_argument("--eps", type=float, default=1.0)
    a = ap.parse_args()
    man, raw, jobs = load(a.step, a.mode)
    if a.step == "D1":
        analyze_D1(man, raw, a.eps, a.mode)
    elif a.step == "S1":
        analyze_S1(man, raw, jobs)
    elif a.step == "S2":
        analyze_S2(man, raw, a.eps, a.mode)
    elif a.step == "S3P":
        # classical baseline: the classical-shadow circuits of S3 (same Paulis; single-qubit gates only)
        man, raw, jobs = load_merged(a.step, a.mode)
        f = os.path.join(HERE, "runs", "S2_summary_origin_sm.json")
        analyze_core("S3P", man, raw, a.eps, json.load(open(f)) if os.path.exists(f) else None, mode=a.mode)
    else:
        if a.step in MERGE:
            man, raw, jobs = load_merged(a.step, a.mode)
        f = os.path.join(HERE, "runs", f"S2_summary_{a.mode}.json")
        analyze_core(a.step, man, raw, a.eps, json.load(open(f)) if os.path.exists(f) else None, mode=a.mode)
