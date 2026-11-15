# -*- coding: utf-8 -*-
"""Identification at large n (n >= 8) with a Clifford two-copy front-end whose response table g3 has entries +-1.
For such a front-end the site outcome beta fixes a single-qubit Pauli q_beta and a sign pattern e_beta with
  g3[sigma, beta] = chi(sigma, q_beta) * e_beta(sigma),  e_beta(sigma) = (-1)^[sigma != I] or 1,
so prod_i g3[P_i, beta_i] = chi(P, Q) * (-1)^{|supp P cap E|}. When e is the same for every outcome (Bell-equivalent
front-ends), the pair log-likelihood of every hypothesis depends only on whether P commutes with Q and on w(P), and the
number of commuting pairs is one Walsh-Hadamard transform of the histogram of the Q's. Exact Bayes over all 4^n - 1
hypotheses in O(n 4^n) per evaluation."""
import numpy as np

CHI = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])   # chi(sigma, q), I X Y Z
XZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])                                   # (x, z) bits of I X Y Z


def decompose(g, tol=0.01):
    """g: (4, 4) response table g3[sigma, beta]. Returns q[beta], e (common sign: -1 means (-1)^[sigma != I])."""
    s = np.sign(np.where(np.abs(g) < tol, 0, g))
    if np.any(np.abs(np.abs(g) - 1) > tol):
        raise ValueError("response table not +-1: Walsh decoder not applicable")
    q, e = [], []
    for b in range(4):
        col = s[:, b]
        hit = [(qq, ee) for qq in range(4) for ee in (1, -1)
               if np.all(CHI[:, qq] * np.array([1, ee, ee, ee]) == col)]
        if not hit:
            raise ValueError("column is not a character pattern")
        q.append(hit[0][0]); e.append(hit[0][1])
    if len(set(e)) != 1:
        raise ValueError("outcome-dependent e: not supported")
    return np.array(q), e[0]


def fwht(a):
    a = a.astype(np.float32).copy(); h = 1; N = len(a)
    while h < N:
        v = a.reshape(-1, 2, h); x = v[:, 0].copy(); v[:, 0] += v[:, 1]; v[:, 1] = x - v[:, 1]; h *= 2
    return a


class Walsh:
    def __init__(self, n, g):
        self.n = n; self.q, self.e = decompose(g)
        # Pauli index (base 4, site 0 most significant) <-> bit index: bits (x_0..x_{n-1}, z_0..z_{n-1})
        P = np.array(np.unravel_index(np.arange(4 ** n), [4] * n)).T          # (4^n, n)
        x, z = XZ[P][..., 0], XZ[P][..., 1]
        pw = 1 << np.arange(n - 1, -1, -1)
        self.xi, self.zi = x @ pw, z @ pw                                       # x-part, z-part as integers
        self.bit_index = (self.xi << n) | self.zi                               # index of Q in the Walsh domain
        # chi(P, Q) = (-1)^{x_P . z_Q + z_P . x_Q}: evaluate the transform at (z_P, x_P)
        self.eval_index = (self.zi << n) | self.xi
        self.w = (P > 0).sum(1)

    def Q_index(self, beta):
        """beta: (K, n) site digits -> base-4 index of the Pauli Q_k"""
        qd = self.q[beta]; pw = 4 ** np.arange(self.n - 1, -1, -1)
        return qd @ pw

    def commute_counts(self, Qidx):
        h = np.bincount(self.bit_index[Qidx], minlength=4 ** self.n)
        W = fwht(h)                                    # sum_Q h(Q) (-1)^{<b, Q>} at bit vector b
        return (len(Qidx) + W[self.eval_index]) / 2     # number of k with chi(P, Q_k) = +1, per P (base-4 order)

    def loglik(self, Qidx, kappa_w):
        """kappa_w: (n+1,) amplitude per weight. Returns log-likelihood (up to a constant) of all 4^n Paulis."""
        c = self.commute_counts(Qidx); K = len(Qidx)
        key = tuple(np.round(kappa_w, 12))
        if getattr(self, "_key", None) != key:                         # per-hypothesis log factors, cached per kappa
            a = np.clip(kappa_w[self.w] * (float(self.e) ** self.w), -0.999999, 0.999999)
            self._la, self._lb = np.log1p(a).astype(np.float32), np.log1p(-a).astype(np.float32); self._key = key
        return c * (self._la - self._lb) + K * self._lb


def sample_pairs(W, P, kappa, K, rng):
    """Synthetic twirled pairs at per-site visibility kappa: prod g = +1 w.p. (1 + kappa^w)/2... returns beta (K, n).
    Outcome distribution: p(beta) = (1 + kappa^w prod_i g[P_i, beta_i]) / 4^n."""
    n = len(P); w = int((np.array(P) > 0).sum()); amp = kappa ** w
    out = []
    while len(out) < K:
        b = rng.integers(4, size=(4 * K, n))
        sgn = np.prod(CHI[np.array(P)[None, :], W.q[b]] * np.where(np.array(P)[None, :] > 0, W.e, 1), axis=1)
        keep = rng.random(4 * K) < (1 + amp * sgn) / 2
        out.extend(b[keep].tolist())
    return np.array(out[:K])
