# -*- coding: utf-8 -*-
"""
Numerical checks of Theorems 1-2 (quantum drift-tracking task).

Task: hidden non-identity n-qubit Pauli P_t; each round the agent receives k copies of
rho_P = (I + eps P) / 2^n and must name P_t (reward 1 if correct). P_t is kept with prob. 1-q
and redrawn uniformly with prob. q.

Agents (both use the exact Bayesian filter for their own measurement data, i.e. they are optimal
given their measurement):
  * classical  : measures every copy separately in a uniformly random product Pauli basis
                 (classical shadows); non-adaptive single-copy measurements as in Theorem 1.
  * quantum    : measures pairs of copies in the Bell basis (two-copy measurement, Theorem 2).

Outcome models (exact):
  random-Pauli basis b in {X,Y,Z}^n, outcome s in {+-1}^n:
      Pr(s) = (1 + eps * [P compatible with b] * prod_{i in supp P} s_i) / 2^n
  Bell measurement on rho (x) rho, outcome (x,z) in {+-1}^{2n}:
      Pr(x,z) = (1 + eps^2 * chi_P(x,z)) / 4^n,  chi_P = prod_i c(P_i), c(I)=1, c(X)=x_i, c(Z)=z_i, c(Y)=-x_i z_i
"""
import argparse
import json
import time
import numpy as np


def paulis(n):
    """All non-identity Paulis as integer arrays (N, n), 0=I,1=X,2=Y,3=Z."""
    g = np.array(np.meshgrid(*[np.arange(4)] * n, indexing="ij")).reshape(n, -1).T
    return g[1:]


# ------------------------------------------------------------------ single-copy random-Pauli measurement
class ClassicalShadowAgent:
    def __init__(self, n, eps, rng):
        self.n, self.eps, self.rng = n, eps, rng
        self.P = paulis(n)                       # (N, n)
        self.supp = (self.P > 0)                 # (N, n)

    def sample(self, true_idx, R):
        """R independent copies of rho_{P[true_idx]}; returns bases (R,n) in 1..3 and outcome bits (R,n) in 0/1."""
        n, P = self.n, self.P[true_idx]
        b = self.rng.integers(1, 4, size=(R, n))
        bits = self.rng.integers(0, 2, size=(R, n))
        compat = np.all((P == 0) | (P == b), axis=1)
        supp = np.nonzero(P)[0]
        # enforce parity distribution on compatible copies: Pr(parity even) = (1+eps)/2
        par = bits[:, supp].sum(1) % 2
        want_even = self.rng.random(R) < (1 + self.eps) / 2
        flip = compat & ((par == 0) != want_even)
        bits[flip, supp[0]] ^= 1
        return b, bits

    def loglik(self, b, bits):
        """Log-likelihood (up to a constant) of each hypothesis for a batch of copies. Returns (N,)."""
        compat = np.all((self.P[None] == 0) | (self.P[None] == b[:, None, :]), axis=2)   # (R, N)
        par = (self.supp[None] & (bits[:, None, :] == 1)).sum(2) % 2                     # (R, N)
        val = 1.0 + self.eps * compat * (1 - 2 * par)
        return np.log(val).sum(0)


# ------------------------------------------------------------------ two-copy Bell measurement
class BellAgent:
    def __init__(self, n, eps, rng):
        self.n, self.eps, self.rng = n, eps, rng
        self.P = paulis(n)
        self.hasX = np.isin(self.P, (1, 2))     # chi depends on x_i
        self.hasZ = np.isin(self.P, (2, 3))     # chi depends on z_i
        self.nY = (self.P == 2).sum(1) % 2       # sign (-1)^{#Y}

    def chi(self, x, z):
        """x,z bits (R,n) with value 1 meaning eigenvalue -1. Returns chi_Q in {+-1}, shape (R,N)."""
        e = (self.hasX[None] & (x[:, None, :] == 1)).sum(2) + (self.hasZ[None] & (z[:, None, :] == 1)).sum(2)
        return 1 - 2 * ((e + self.nY[None]) % 2)

    def sample(self, true_idx, R):
        n = self.n
        x = self.rng.integers(0, 2, size=(R, n))
        z = self.rng.integers(0, 2, size=(R, n))
        c = self.chi(x, z)[:, true_idx]
        want_plus = self.rng.random(R) < (1 + self.eps ** 2) / 2
        flip = (c == 1) != want_plus
        # flip one bit that chi_P depends on
        Pt = self.P[true_idx]
        i = int(np.nonzero(Pt)[0][0])
        if Pt[i] in (1, 2):
            x[flip, i] ^= 1
        else:
            z[flip, i] ^= 1
        return x, z

    def loglik(self, x, z):
        return np.log(1.0 + self.eps ** 2 * self.chi(x, z)).sum(0)


# ------------------------------------------------------------------ experiment 1: static identification
def copies_to_accuracy(kind, n, eps, trials, checkpoints, seed):
    rng = np.random.default_rng(seed)
    A = ClassicalShadowAgent(n, eps, rng) if kind == "classical" else BellAgent(n, eps, rng)
    N = len(A.P)
    acc = np.zeros(len(checkpoints))
    step = 1 if kind == "classical" else 2           # copies consumed per measurement
    for tr in range(trials):
        t = rng.integers(N)
        L = np.zeros(N)
        used = 0
        for j, m in enumerate(checkpoints):
            R = (m - used) // step
            while R > 0:                                  # chunks keep memory bounded
                r = min(R, max(1, 2_000_000 // (N * A.n)))
                L += A.loglik(*A.sample(t, r))
                used += r * step
                R -= r
            best = np.flatnonzero(L == L.max())
            acc[j] += (t in best) / len(best)
    return acc / trials


def m_at(acc, checkpoints, target=0.9):
    idx = np.flatnonzero(acc >= target)
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return float(checkpoints[0])
    # linear interpolation in log-copies
    a0, a1 = acc[j - 1], acc[j]
    c0, c1 = np.log(checkpoints[j - 1]), np.log(checkpoints[j])
    return float(np.exp(c0 + (target - a0) / (a1 - a0) * (c1 - c0)))


# ------------------------------------------------------------------ experiment 2: drifting task, regret
def run_drift(kind, n, eps, k, q, T, trials, seed):
    rng = np.random.default_rng(seed)
    A = ClassicalShadowAgent(n, eps, rng) if kind == "classical" else BellAgent(n, eps, rng)
    N = len(A.P)
    regrets = []
    for tr in range(trials):
        logpost = np.full(N, -np.log(N))
        p = rng.integers(N)
        reg = 0.0
        for t in range(T):
            if t > 0 and rng.random() < q:
                p = rng.integers(N)
            if t > 0:                                   # HMM prediction step
                post = np.exp(logpost - logpost.max()); post /= post.sum()
                post = (1 - q) * post + q / N
                logpost = np.log(post)
            R = k if kind == "classical" else k // 2
            if R > 0:
                logpost = logpost + A.loglik(*A.sample(p, R))
            best = np.flatnonzero(logpost == logpost.max())
            reg += 1.0 - (p in best) / len(best)
        regrets.append(reg)
    return float(np.mean(regrets)), float(np.std(regrets, ddof=1) / np.sqrt(trials))


# ------------------------------------------------------------------ theoretical bounds
def lower_bound_copies(n, eps, delta=0.1):
    """Theorem 1 / Lemma 2: any non-adaptive single-copy strategy needs at least this many copies."""
    d = 2 ** n; N = d * d - 1
    return max(0.0, (d + 1) * ((1 - delta) * np.log(N) - np.log(2)) / eps ** 2)


def upper_bound_copies(n, eps, delta=0.1):
    """Lemma 3: Bell sampling with this many copies identifies P with prob >= 1-delta."""
    return 2 * np.ceil(8 * (n * np.log(4) + np.log(1 / delta)) / eps ** 4)


def regret_lower(n, eps, k, q, T):
    d = 2 ** n; N = d * d - 1
    l0 = np.floor((d + 1) * (np.log(N) / 2 - np.log(2)) / (k * eps ** 2))
    if l0 < 1 or T < 2 * l0:
        return 0.0
    return 0.5 * (T - l0) * (1 - (1 - q) ** l0)


def regret_upper(n, eps, k, q, T):
    h = k // 2
    W = np.ceil(8 * (n * np.log(4) + np.log(1 / q)) / (h * eps ** 4))
    return min(T, W + q * T * (W + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["static", "drift"], required=True)
    ap.add_argument("--eps", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    t0 = time.time()
    if a.exp == "static":
        cps = np.unique(np.round(np.logspace(0.3, 4.3, 41)).astype(int))
        cps = cps[cps % 2 == 0] if True else cps
        for kind, ns, trials, cap in (("quantum", range(1, 8), 300, 600), ("classical", range(1, 6), 200, 20000)):
            for n in ns:
                acc = copies_to_accuracy(kind, n, a.eps, trials, cps[cps <= cap], seed=100 * n + (kind == "quantum"))
                m90 = m_at(acc, cps[cps <= cap])
                res.append(dict(kind=kind, n=n, eps=a.eps, m90=m90, trials=trials,
                                lb=lower_bound_copies(n, a.eps), ub=upper_bound_copies(n, a.eps),
                                checkpoints=cps[cps <= cap].tolist(), acc=acc.tolist()))
                print(kind, n, "m90=", m90, "LB=", round(lower_bound_copies(n, a.eps), 1),
                      "UB=", upper_bound_copies(n, a.eps), f"{time.time() - t0:.0f}s", flush=True)
    else:
        k, q, T = 4, 0.01, 2000
        for n in (2, 3, 4, 5):
            for kind in ("quantum", "classical"):
                trials = 20 if kind == "classical" else 40
                m, se = run_drift(kind, n, a.eps, k, q, T, trials, seed=7 + n)
                res.append(dict(kind=kind, n=n, eps=a.eps, k=k, q=q, T=T, regret=m, se=se,
                                lb=regret_lower(n, a.eps, k, q, T), ub=regret_upper(n, a.eps, k, q, T)))
                print(kind, n, "regret=", round(m, 1), "+-", round(se, 1), "LB(any single-copy)=",
                      round(regret_lower(n, a.eps, k, q, T), 1), "UB(Bell)=", round(regret_upper(n, a.eps, k, q, T), 1),
                      f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
