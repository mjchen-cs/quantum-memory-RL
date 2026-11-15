# -*- coding: utf-8 -*-
"""Stage 2 experiments: training a lightweight parametrized quantum policy for the drift-tracking task.

E1  trainability: success rate of gradient training vs n, for the unconstrained ansatz ('full', every qubit pair)
    and the permutation-equivariant ansatz ('equiv', parameters shared across qubit positions), with the
    two-copy-visibility objective; the information objective is reported for comparison.
E2  structured pruning (DQCP-style L1 on entangler strengths) on a 2-layer equivariant ansatz.
E3  zero-shot size transfer: the best equivariant circuit trained at n=2 is evaluated at n=2..5
    (identification copies and drift regret with the exact Bayesian filter for the learned measurement).
"""
import argparse
import json
import math
import time
import numpy as np
import torch
from learn_policy import Model, bell_info, accuracy_curve, m90

EPS = 0.9


def fit(n, variant, layers, seed, steps, objective="vis", lam=0.0, lr=0.05, mask=None, init=None):
    m = Model(n, EPS, layers, variant, seed, math.pi)
    if init is not None:
        with torch.no_grad():
            m.rot.copy_(init["rot"]); m.phi.copy_(init["phi"]); m.mask = init["mask"].clone()
    if mask is not None:
        m.mask = mask.clone()
    opt = torch.optim.Adam([m.rot, m.phi], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        if objective == "vis":
            loss = -m.two_copy_visibility()
        else:
            loss = -m.info()
        loss = loss + lam * m.strengths().sum()
        loss.backward()
        m.phi.grad *= m.mask
        opt.step()
    with torch.no_grad():
        vis = float(m.two_copy_visibility())
    return m, vis


def drift_regret(p, k, q, T, trials, rng):
    """Exact HMM Bayesian filter for a fixed measurement with outcome table p (N x D); k copies -> k//2 pairs."""
    p = p / p.sum(1, keepdims=True)
    N, D = p.shape
    logp = np.log(p)
    cdf = np.cumsum(p, 1)
    regs = []
    for _ in range(trials):
        lp = np.full(N, -np.log(N)); t = rng.integers(N); reg = 0.0
        for s in range(T):
            if s > 0 and rng.random() < q:
                t = rng.integers(N)
            if s > 0:
                post = np.exp(lp - lp.max()); post /= post.sum(); lp = np.log((1 - q) * post + q / N)
            bs = np.searchsorted(cdf[t], rng.random(k // 2)).clip(0, D - 1)
            lp = lp + logp[:, bs].sum(1)
            best = np.flatnonzero(lp == lp.max())
            reg += 1 - (t in best) / len(best)
        regs.append(reg)
    return float(np.mean(regs)), float(np.std(regs, ddof=1) / np.sqrt(trials))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["E1", "E2", "E3"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = []
    if a.exp == "E1":
        plan = [(2, "full", 10, 300), (2, "equiv", 10, 300), (3, "full", 10, 300), (3, "equiv", 10, 300),
                (4, "equiv", 10, 150), (4, "full", 5, 150)]
        for n, var, R, steps in plan:
            for sd in range(R):
                m, vis = fit(n, var, 1, sd, steps)
                res.append(dict(n=n, variant=var, objective="vis", seed=sd, steps=steps, vis=vis))
                print(n, var, "vis seed", sd, round(vis, 4), f"{time.time()-t0:.0f}s", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
        for n, var in ((2, "equiv"), (3, "equiv")):          # information objective for comparison
            for sd in range(10):
                m, vis = fit(n, var, 1, sd, 300, objective="info")
                res.append(dict(n=n, variant=var, objective="info", seed=sd, steps=300, vis=vis))
                print(n, var, "info-objective seed", sd, round(vis, 4), f"{time.time()-t0:.0f}s", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
    elif a.exp == "E2":
        for sd in range(8):
            m, vis = fit(3, "equiv", 2, sd, 400, lam=0.01)
            s = m.strengths().detach().numpy()
            mask = torch.tensor((s > 0.1).astype(float))
            m2, vis2 = fit(3, "equiv", 2, sd, 200, mask=mask,
                           init=dict(rot=m.rot.detach(), phi=m.phi.detach(), mask=mask))
            kept = m2.kept_pairs(0.0)
            res.append(dict(seed=sd, vis_sparse=vis, vis_final=vis2, strengths=s.tolist(),
                            kept_orbits=[[int(l), int(k)] for l, k in zip(*np.nonzero(mask.numpy()))],
                            n_two_qubit=len(kept), n_cross_same=sum(1 for k in kept if k["pair"][1] - k["pair"][0] == 3),
                            candidates=2 * 15))
            print("E2 seed", sd, "vis", round(vis2, 4), "two-qubit gates", len(kept), "orbits", res[-1]["kept_orbits"],
                  f"{time.time()-t0:.0f}s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    else:
        best = None
        for sd in range(10):
            m, vis = fit(2, "equiv", 1, sd, 300)
            if best is None or vis > best[1]:
                best = (m, vis, sd)
        m2, vis2, sd2 = best
        print("best n=2 seed", sd2, "vis", vis2, flush=True)
        params = dict(rot=m2.rot.detach(), phi=m2.phi.detach(), mask=m2.mask)
        torch.save(params, a.out.replace(".json", "_params.pt"))
        rng = np.random.default_rng(5)
        for n in (2, 3, 4, 5):
            m = Model(n, EPS, 1, "equiv", 0, math.pi)
            with torch.no_grad():
                m.rot.copy_(params["rot"]); m.phi.copy_(params["phi"])
                m.mask = params["mask"] * m.mask
                p = m.probs()
                vis = float(m.two_copy_visibility())
                I = float(m.info(p))
            grid = np.unique(np.round(np.logspace(0, 2.6, 26)).astype(int))
            acc = accuracy_curve(p, grid, 300, rng)
            pn = p.numpy()
            reg, se = drift_regret(pn, 4, 0.01, 2000, 40, rng)
            res.append(dict(n=n, vis=vis, info=I, info_bell=bell_info(n, EPS), pairs90=m90(acc, grid),
                            copies90=2 * m90(acc, grid), regret=reg, se=se, two_qubit_gates=len(m.kept_pairs(0.1)),
                            trained_at=2, train_seed=int(sd2)))
            print("transfer n", n, "vis", round(vis, 4), "info", round(I, 4), "bell", round(bell_info(n, EPS), 4),
                  "copies90", 2 * m90(acc, grid), "regret", round(reg, 1), f"{time.time()-t0:.0f}s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
