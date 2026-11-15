# -*- coding: utf-8 -*-
"""
Stage 3: end-to-end, reward-only training of a lightweight quantum agent for the drift-tracking task.

Agent (everything is shared across qubit positions, so the parameter count does not depend on n):
  * quantum front-end: the site-local equivariant circuit of fast_site.py (13 angles), measured in the
    computational basis;
  * decoder: score(a | b) = sum_k c_k prod_i h_k[a_i, b_i] (K tables of size 4x4 shared across sites), summed over the
    m pairs of an episode; action a ~ softmax(score). For K = 3 this family contains the Bayes decoder of the Bell
    agent and of product (single-copy-type) measurements.
Learning signal: reward 1{a = P} only. The agent never sees rho_P, P or any likelihood.
  * decoder: REINFORCE with running baseline and entropy bonus (autograd);
  * front-end: antithetic evolution strategies with common random numbers (zeroth order, hardware compatible).
"""
import argparse
import json
import time
import numpy as np
import torch
from fast_site import SiteModel, NPAR

torch.set_default_dtype(torch.float64)


class Decoder(torch.nn.Module):
    def __init__(self, n, K=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.K = n, K
        self.h = torch.nn.Parameter(1 + 0.05 * torch.randn(K, 4, 4, generator=g))
        self.c = torch.nn.Parameter(0.05 * torch.randn(K, generator=g))

    def scores(self, b):
        """b: (E, m, n) outcome digits -> (E, N) scores over non-identity Paulis."""
        E, m, n = b.shape
        v = self.h[:, :, b.reshape(-1, n)]                     # (K, 4, E*m, n): h_k[sigma, b_i]
        v = v.permute(2, 0, 3, 1)                             # (E*m, K, n, 4)
        out = v[:, :, 0, :]
        for i in range(1, n):
            out = (out[..., :, None] * v[:, :, i, None, :]).reshape(E * m, self.K, -1)
        s = (self.c[None, :, None] * out).sum(1)[:, 1:]        # drop identity string
        return s.reshape(E, m, -1).sum(1)


def sample(cdf, Bdig, P, U):
    D = cdf.shape[1]
    idx = np.empty(U.shape, dtype=np.int64)
    for e in range(U.shape[0]):
        idx[e] = np.searchsorted(cdf[P[e]], U[e]).clip(0, D - 1)
    return Bdig[idx]                                          # (E, m, n)


def act(scores, Ua=None, greedy=False):
    if greedy:
        return scores.argmax(1).numpy()
    pr = torch.softmax(scores, 1).detach().numpy()
    return (pr.cumsum(1) < Ua[:, None]).sum(1).clip(0, pr.shape[1] - 1)


def cdf_of(S, th):
    p = S.probs(th); p = p / p.sum(1, keepdims=True)
    return np.cumsum(p, 1)


def run(n, m, iters, seed, twirl=False, init=None, E=256, Kdir=6, sigma=0.2, lr_th=0.05, lr_dec=0.03, ent=0.003, log=50, tag=""):
    rng = np.random.default_rng(seed)
    S = SiteModel(n, twirl=twirl)
    th = rng.uniform(-np.pi, np.pi, NPAR)
    dec = Decoder(n, seed=seed)
    if init is not None:                       # warm start from a smaller system (all parameters are n-independent)
        th = np.array(init["theta"], float)
        with torch.no_grad():
            dec.h.copy_(torch.tensor(init["h"])); dec.c.copy_(torch.tensor(init["c"]))
    opt = torch.optim.Adam(dec.parameters(), lr=lr_dec)
    mT = np.zeros(NPAR); vT = np.zeros(NPAR); base = 1.0 / S.N
    hist = []; t0 = time.time()
    for it in range(1, iters + 1):
        # decoder: REINFORCE
        cdf = cdf_of(S, th)
        P = rng.integers(S.N, size=E)
        b = torch.tensor(sample(cdf, S.B, P, rng.random((E, m))))
        sc = dec.scores(b)
        a = act(sc, rng.random(E))
        r = (a == P).astype(float)
        logp = torch.log_softmax(sc, 1)
        pr = logp.exp()
        adv = torch.tensor(r - base)
        loss = -(adv * logp[torch.arange(E), torch.tensor(a)]).mean() - ent * (-(pr * logp).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        base = 0.95 * base + 0.05 * r.mean()
        # front-end: antithetic ES with common random numbers
        P = rng.integers(S.N, size=E); Uo = rng.random((E, m)); Ua = rng.random(E)
        g = np.zeros(NPAR)
        with torch.no_grad():
            for _ in range(Kdir):
                d = rng.normal(0, sigma, NPAR)
                rp = (act(dec.scores(torch.tensor(sample(cdf_of(S, th + d), S.B, P, Uo))), Ua) == P).mean()
                rm = (act(dec.scores(torch.tensor(sample(cdf_of(S, th - d), S.B, P, Uo))), Ua) == P).mean()
                g += (rp - rm) / (2 * sigma ** 2) * d
        g /= Kdir
        mT = 0.9 * mT + 0.1 * g; vT = 0.999 * vT + 0.001 * g ** 2
        th = th + lr_th * (mT / (1 - 0.9 ** it)) / (np.sqrt(vT / (1 - 0.999 ** it)) + 1e-8)
        if it % log == 0 or it == iters:
            with torch.no_grad():
                Pe = rng.integers(S.N, size=1000)
                be = torch.tensor(sample(cdf_of(S, th), S.B, Pe, rng.random((1000, m))))
                acc = float((act(dec.scores(be), greedy=True) == Pe).mean())
            vis = S.visibility(th)
            hist.append(dict(it=it, reward=float(r.mean()), acc=acc, vis=vis))
            print(f"{tag} it {it:5d} reward {r.mean():.3f} greedy-acc {acc:.3f} vis {vis:.3f} {time.time()-t0:.0f}s", flush=True)
    return th, dec, hist


def bayes_accuracy(S, th, m, trials, rng):
    """Accuracy of the Bayes-optimal decoder for a given front-end (reference, uses the model)."""
    p = S.probs(th); p = p / p.sum(1, keepdims=True); lp = np.log(p); cdf = np.cumsum(p, 1); ok = 0.0
    for _ in range(trials):
        t = rng.integers(S.N); bs = np.searchsorted(cdf[t], rng.random(m)).clip(0, S.D - 1)
        L = lp[:, bs].sum(1); best = np.flatnonzero(L == L.max()); ok += (t in best) / len(best)
    return ok / trials


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--twirl", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from fast_site import bell_theta
    S = SiteModel(a.n)
    ref = dict(bell=bayes_accuracy(S, bell_theta(), a.m, 2000, np.random.default_rng(1)))
    print("Bayes-optimal Bell accuracy with", a.m, "pairs:", ref["bell"], flush=True)
    res = []
    for sd in [int(s) for s in a.seeds.split(",")]:
        th, dec, hist = run(a.n, a.m, a.iters, sd, twirl=bool(a.twirl), tag=f"[n={a.n} tw{a.twirl} s{sd}]")
        res.append(dict(n=a.n, m=a.m, seed=sd, twirl=a.twirl, theta=th.tolist(),
                        h=dec.h.detach().tolist(), c=dec.c.detach().tolist(), hist=hist, final=hist[-1],
                        bayes_acc_learned_frontend=bayes_accuracy(S, th, a.m, 2000, np.random.default_rng(2))))
        json.dump(dict(ref=ref, runs=res), open(a.out, "w"), indent=1)
