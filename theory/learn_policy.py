# -*- coding: utf-8 -*-
"""
Stage 2: can a trained, compressed parametrized quantum policy learn the Bell-type processing of Theorem 2?

Policy front-end: a parametrized circuit U(theta) on the 2n qubits of two copies (copy A = qubits 0..n-1,
copy B = qubits n..2n-1), followed by a computational-basis measurement. Each layer applies Euler rotations
RZ.RY.RZ on every qubit and a controlled-RX(phi_e) on every candidate pair e. The decision is made by the exact
Bayesian filter for the learned measurement model (model-based agent).

Training objective: the per-pair mutual information I_theta(P; B) between the hidden Pauli and the outcome,
which upper-bounds the information the agent gains per decision step (Fano), minus an L1 penalty on the
entangler strengths |sin(phi_e/2)| (DQCP-style structured pruning). After training, entanglers below a threshold
are removed and the circuit is fine-tuned.

Ansatz variants
  full    : candidate entanglers on all qubit pairs (within and across copies)
  local   : entanglers only within each copy (product across copies = single-copy measurement class,
            Lemma 1 caps the information at 2 eps^2/(d+1) per pair)

Exact computation: rho_P (x) rho_P = (I + eps P.I + eps I.P + eps^2 P.P)/D, so
  p_P(b) = (1 + eps f_{P.I}(b) + eps f_{I.P}(b) + eps^2 f_{P.P}(b)) / D,  f_O(b) = <b|U O U^dag|b>.
"""
import argparse
import itertools
import json
import math
import time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
CD = torch.complex128


# ------------------------------------------------------------------ Pauli strings on 2n qubits
def pauli_strings(n):
    return np.array(list(itertools.product(range(4), repeat=n)))[1:]          # non-identity, 0=I 1=X 2=Y 3=Z


def pauli_action(s, nq):
    """Pauli string s (len nq) -> (xmask, zmask, nY); P|j> = i^nY (-1)^{pop(j&z)} |j^x>, qubit 0 = MSB."""
    x = z = 0
    for k, p in enumerate(s):
        bit = 1 << (nq - 1 - k)
        if p in (1, 2):
            x |= bit
        if p in (2, 3):
            z |= bit
    return x, z, int(np.sum(np.asarray(s) == 2))


def popcount(a):
    return np.array([bin(v).count("1") for v in a])


class Model:
    def __init__(self, n, eps, layers, variant, seed, phi_init=math.pi, rot_init="uniform"):
        self.n, self.eps, self.L, self.variant = n, eps, layers, variant
        self.nq = 2 * n
        self.D = 2 ** self.nq
        torch.manual_seed(seed)
        P = pauli_strings(n)
        self.P = P
        self.N = len(P)
        I = np.zeros_like(P)
        # operator strings on 2n qubits: P.I, I.P, P.P
        js = np.arange(self.D)
        self.ops = []
        for blk in (np.hstack([P, I]), np.hstack([I, P]), np.hstack([P, P])):
            perm = np.zeros((self.N, self.D), dtype=np.int64)
            ph = np.zeros((self.N, self.D), dtype=np.complex128)
            for a, s in enumerate(blk):
                x, z, ny = pauli_action(s, self.nq)
                perm[a] = js ^ x
                ph[a] = (1j) ** ny * (-1.0) ** popcount(js & z)
            self.ops.append((torch.tensor(perm), torch.tensor(ph)))
        pairs = list(itertools.combinations(range(self.nq), 2))
        if variant == "local":
            pairs = [(a, b) for a, b in pairs if (a < n) == (b < n)]
        self.pairs = pairs
        # orbits of qubit pairs under simultaneous permutation of positions in both copies
        def orbit(pr):
            a, b = pr
            if a < n and b >= n:
                return 0 if b - n == a else 1          # same-position cross / different-position cross
            return 2 if a < n else 3                    # within copy A / within copy B
        self.orbit = torch.tensor([orbit(p) for p in pairs])
        self.equiv = variant == "equiv"
        nrot = 2 if variant == "equiv" else self.nq
        nphi = 4 if variant == "equiv" else len(pairs)
        self.rot = torch.nn.Parameter(torch.empty(layers + 1, nrot, 3).uniform_(-math.pi, math.pi)
                                       if rot_init == "uniform" else 0.3 * torch.randn(layers + 1, nrot, 3))
        self.phi = torch.nn.Parameter(torch.empty(layers, nphi).uniform_(-phi_init, phi_init))
        self.mask = torch.ones(layers, nphi)
        if variant == "equiv":                          # orbits absent for this n are masked
            present = torch.zeros(4)
            present[torch.unique(self.orbit)] = 1
            self.mask = self.mask * present

    # -------------------------------------------------------------- circuit
    def _apply1(self, T, q, g):
        T = torch.movedim(T, q, 0)
        sh = T.shape
        T = (g @ T.reshape(2, -1)).reshape(sh)
        return torch.movedim(T, 0, q)

    @staticmethod
    def _euler(a, b, c):
        def rz(t):
            return torch.stack([torch.stack([torch.exp(-0.5j * t), torch.zeros((), dtype=CD)]),
                                torch.stack([torch.zeros((), dtype=CD), torch.exp(0.5j * t)])])
        cb, sb = torch.cos(b / 2).to(CD), torch.sin(b / 2).to(CD)
        ry = torch.stack([torch.stack([cb, -sb]), torch.stack([sb, cb])])
        return rz(c) @ ry @ rz(a)

    def _crx(self, T, c, t, phi):
        T = torch.movedim(T, (c, t), (0, 1))
        cp, sp = torch.cos(phi / 2).to(CD), torch.sin(phi / 2).to(CD)
        g = torch.stack([torch.stack([cp, -1j * sp]), torch.stack([-1j * sp, cp])])
        T1 = (g @ T[1].reshape(2, -1)).reshape(T[1].shape)
        T = torch.stack([T[0], T1])
        return torch.movedim(T, (0, 1), (c, t))

    def unitary(self):
        nq = self.nq
        T = torch.eye(self.D, dtype=CD).reshape([2] * nq + [self.D])
        for l in range(self.L + 1):
            for q in range(nq):
                r = self.rot[l, (0 if q < self.n else 1)] if self.equiv else self.rot[l, q]
                T = self._apply1(T, q, self._euler(r[0], r[1], r[2]))
            if l < self.L:
                for e, (a, b) in enumerate(self.pairs):
                    k = int(self.orbit[e]) if self.equiv else e
                    if self.mask[l, k] > 0:
                        T = self._crx(T, a, b, self.phi[l, k])
        return T.reshape(self.D, self.D)                 # U[b, j] = <b|U|j>

    def fterms(self, U=None, which=(0, 1, 2), chunk=None):
        U = self.unitary() if U is None else U
        Uc = U.conj()
        chunk = chunk or max(1, int(2e7 // (self.D * self.D)))
        f = []
        for w in which:
            perm, ph = self.ops[w]
            parts = []
            for s0 in range(0, self.N, chunk):
                Ug = U[:, perm[s0:s0 + chunk]]
                parts.append(torch.einsum("bnj,nj,bj->nb", Ug, ph[s0:s0 + chunk], Uc).real)
            f.append(torch.cat(parts, 0))
        return f

    def two_copy_visibility(self, f=None):
        """Mean over Paulis and outcomes of <b|U (P x P) U^dag|b>^2: equals 1 iff every P x P is diagonal in the
        measurement basis (Bell-type), i.e. every tr(P rho)^2 can be read out from a single pair."""
        f = self.fterms(which=(2,))[0] if f is None else f
        return (f ** 2).mean()

    def probs(self, U=None):
        f = self.fterms(U)
        e = self.eps
        p = (1 + e * f[0] + e * f[1] + e * e * f[2]) / self.D       # (N, D)
        return p.clamp_min(1e-15)

    def info(self, p=None):
        p = self.probs() if p is None else p
        pbar = p.mean(0)
        H = -(pbar * pbar.log()).sum()
        Hc = -(p * p.log()).sum(1).mean()
        return H - Hc

    def union_bound(self, m, p=None):
        """log of the union (Bhattacharyya) bound on the ML identification error with m i.i.d. pairs:
        Pr[err] <= (1/N) sum_P sum_{Q!=P} BC(p_P, p_Q)^m."""
        p = self.probs() if p is None else p
        S = p.sqrt()
        BC = (S @ S.T).clamp(1e-300, 1.0)
        logBC = m * BC.log()
        logBC = logBC - torch.diag(torch.full((self.N,), float("inf")))
        return torch.logsumexp(logBC.reshape(-1), 0) - math.log(self.N)

    def softmin_chi2(self, beta, p=None):
        """Soft minimum over Paulis of chi^2(p_P || uniform): every hidden direction must be visible."""
        p = self.probs() if p is None else p
        chi2 = self.D * ((p - 1.0 / self.D) ** 2).sum(1)
        return -(torch.logsumexp(-beta * chi2, 0) - math.log(self.N)) / beta

    def strengths(self):
        return torch.sin(self.phi / 2).abs() * self.mask

    def n_entanglers(self, thr=0.0):
        s = self.strengths()
        return int((s > thr).sum())

    def kept_pairs(self, thr):
        s = self.strengths().detach().numpy()
        out = []
        for l in range(self.L):
            for e, pr in enumerate(self.pairs):
                k = int(self.orbit[e]) if self.equiv else e
                if s[l, k] > thr:
                    out.append(dict(layer=l, pair=pr, cross=(pr[0] < self.n) != (pr[1] < self.n), strength=float(s[l, k])))
        return out


def bell_info(n, eps):
    """Exact per-pair mutual information of the Bell measurement, Pr(x,z)=(1+eps^2 chi_P(x,z))/4^n."""
    from qdrift_sim import BellAgent
    B = BellAgent(n, eps, np.random.default_rng(0))
    xs = np.array([[(v >> i) & 1 for i in range(n)] for v in range(2 ** n)])
    X = np.repeat(xs, 2 ** n, 0); Z = np.tile(xs, (2 ** n, 1))
    p = (1 + eps ** 2 * B.chi(X, Z).T) / 4 ** n
    pb = p.mean(0)
    return float(-(pb * np.log(pb)).sum() + (p * np.log(p)).sum(1).mean())


def train(model, steps, lr, lam, log_every=50, tag="", objective="info", m_obj=10):
    params = [model.rot, model.phi]
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    t0 = time.time()
    for s in range(steps):
        opt.zero_grad()
        p = model.probs()
        I = model.info(p)
        if objective == "info":
            main = -I
        elif objective == "union":
            main = model.union_bound(m_obj, p)
        elif objective == "vis":
            main = -model.two_copy_visibility()
        else:
            main = -model.softmin_chi2(float(m_obj), p)
        loss = main + lam * model.strengths().sum()
        loss.backward()
        with torch.no_grad():
            model.phi.grad *= model.mask
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            hist.append((s, float(I.detach()), model.n_entanglers(0.1)))
            print(f"{tag} step {s:4d}  I={float(I):.5f}  entanglers>0.1: {model.n_entanglers(0.1)}  {time.time()-t0:.0f}s", flush=True)
    return hist


# ------------------------------------------------------------------ Monte-Carlo identification accuracy (Bayes decoder)
def accuracy_curve(p, pairs_grid, trials, rng):
    p = p.detach().numpy()
    p = p / p.sum(1, keepdims=True)
    N, D = p.shape
    logp = np.log(p)
    acc = np.zeros(len(pairs_grid))
    cdf = np.cumsum(p, 1)
    for _ in range(trials):
        t = rng.integers(N)
        L = np.zeros(N)
        used = 0
        for j, m in enumerate(pairs_grid):
            r = m - used
            if r > 0:
                u = rng.random(r)
                bs = np.searchsorted(cdf[t], u).clip(0, D - 1)
                L += logp[:, bs].sum(1)
                used = m
            best = np.flatnonzero(L == L.max())
            acc[j] += (t in best) / len(best)
    return acc / trials


def m90(acc, grid):
    idx = np.flatnonzero(acc >= 0.9)
    if len(idx) == 0:
        return float("nan")
    j = idx[0]
    if j == 0:
        return float(grid[0])
    a0, a1 = acc[j - 1], acc[j]
    return float(np.exp(np.log(grid[j - 1]) + (0.9 - a0) / (a1 - a0) * (np.log(grid[j]) - np.log(grid[j - 1]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.9)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--variant", default="full", choices=["full", "local", "equiv"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--finetune", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.002)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--phi_init", type=float, default=math.pi)
    ap.add_argument("--objective", default="info", choices=["info", "union", "chi2", "vis"])
    ap.add_argument("--m_obj", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = Model(a.n, a.eps, a.layers, a.variant, a.seed, a.phi_init)
    tag = f"[n={a.n} {a.variant} s{a.seed}]"
    cand = len(m.kept_pairs(-1.0))
    h1 = train(m, a.steps, a.lr, a.lam, tag=tag + " sparse", objective=a.objective, m_obj=a.m_obj)
    kept = m.kept_pairs(a.thr)
    with torch.no_grad():
        m.mask = (m.strengths() > a.thr).double()
    h2 = train(m, a.finetune, a.lr / 2, 0.0, tag=tag + " finetune", objective=a.objective, m_obj=a.m_obj)
    with torch.no_grad():
        p = m.probs()
        I_final = float(m.info(p))
    d = 2 ** a.n
    res = dict(objective=a.objective, m_obj=a.m_obj, n=a.n, eps=a.eps, layers=a.layers, variant=a.variant, seed=a.seed, candidates=cand,
               kept=m.kept_pairs(0.0), n_kept=len(m.kept_pairs(0.0)),
               n_cross=sum(k["cross"] for k in m.kept_pairs(0.0)),
               info=I_final, info_bell=bell_info(a.n, a.eps), info_single_copy_cap=2 * a.eps ** 2 / (d + 1),
               hist_sparse=h1, hist_finetune=h2)
    rng = np.random.default_rng(1000 + a.seed)
    grid = np.unique(np.round(np.logspace(0, 3.3, 30)).astype(int))
    acc = accuracy_curve(p, grid, 300, rng)
    res.update(grid=grid.tolist(), acc=acc.tolist(), pairs90=m90(acc, grid))
    torch.save(dict(rot=m.rot.detach(), phi=m.phi.detach(), mask=m.mask, pairs=m.pairs), a.out.replace(".json", ".pt"))
    json.dump(res, open(a.out, "w"), indent=1)
    print(tag, "RESULT info", round(I_final, 5), "bell", round(res["info_bell"], 5), "single-copy cap",
          round(res["info_single_copy_cap"], 5), "kept", res["n_kept"], "cross", res["n_cross"],
          "pairs90", res["pairs90"], flush=True)
