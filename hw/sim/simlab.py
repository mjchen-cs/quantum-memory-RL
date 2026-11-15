# -*- coding: utf-8 -*-
"""Simulation lab (n = 2, four qubits) for reward-only training with different front-ends, mirroring hw_train.py:
antithetic ES (K = 2, sigma = 0.25, Adam-normalized, lr 0.08) on the reward, REINFORCE decoder, decoder restarts every
20 iterations, annealing after the 10-iteration reward exceeds 0.7, front-end restarts (patience 40, below 0.5).
Per iteration and front-end: 15 Paulis x 8 circuits x 20 shots, episodes of m = 8 pairs (one shot from each of 8
distinct circuits), 2,000 episodes; common random numbers across the 5 front-ends of an iteration.
Noise: every pair outcome is replaced by a uniform one with probability lam (two-copy visibility 1 - lam).
Decoder: general tabular score(a) = sum_j W[a, b_j] over the 16 outcomes of a pair (contains the product decoder).

Front-ends ('kind'):
  site13       paper's agent: the same 13-angle V(theta) (Euler, CRX, Euler) on both sites
  su4_shared   general two-qubit unitary (KAK form, 15 angles), the same on both sites
  su4_free     independent general two-qubit unitaries on the two sites (30 angles)
  hea4         pairing-agnostic hardware-efficient circuit on all four qubits (A0, A1, B0, B1): 5 layers of RY, RZ on
               every qubit with CZ on the ring A0-A1-B0-B1-A0 between layers (40 angles); it is not told which qubits
               hold corresponding parts of the two copies
  product6     single-copy agent: Euler rotations on each qubit, no two-qubit gate (6 angles)
  channel26    channel task: input W(theta_in) on (system, ancilla) |00>, hidden-Pauli channel on the system qubits,
               then V(theta_out) (13 + 13 angles)
"""
import itertools
import numpy as np

I2 = np.eye(2, dtype=complex); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]).astype(complex)
PA = [I2, X, Y, Z]
CHI = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
PAULIS = [p for p in itertools.product(range(4), repeat=2)][1:]           # 15, site 0 first


def rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
def ry(t): return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], complex)
def euler(a, b, c): return rz(c) @ ry(b) @ rz(a)


def V13(t):
    pre = np.kron(euler(*t[0:3]), euler(*t[3:6]))
    crx = np.eye(4, dtype=complex); crx[2:, 2:] = np.cos(t[6] / 2) * I2 - 1j * np.sin(t[6] / 2) * X
    return np.kron(euler(*t[7:10]), euler(*t[10:13])) @ crx @ pre


def su4(t):
    H = t[6] * np.kron(X, X) + t[7] * np.kron(Y, Y) + t[8] * np.kron(Z, Z)
    w, v = np.linalg.eigh(H); A = v @ np.diag(np.exp(1j * w)) @ v.conj().T
    return np.kron(euler(*t[9:12]), euler(*t[12:15])) @ A @ np.kron(euler(*t[0:3]), euler(*t[3:6]))


def g3(U):
    g = np.zeros((4, 4))
    for s in range(4):
        g[s] = np.real(np.diag(U @ np.kron(PA[s], PA[s]) @ U.conj().T))
    return g


def site_probs(gs):
    """gs: list of two (4, 4) tables -> (15, 16) outcome distributions of the twirled pair (I + P x P) / 16."""
    out = np.zeros((15, 16))
    for k, P in enumerate(PAULIS):
        prod = np.outer(gs[0][P[0]], gs[1][P[1]]).reshape(16)
        out[k] = (1 + prod) / 16
    return out


# 4-qubit generic circuit, qubit order (A0, A1, B0, B1)
def kron_all(ms):
    out = np.array([[1]], complex)
    for m in ms:
        out = np.kron(out, m)
    return out


CZ_RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
def cz_diag(a, b):
    d = np.ones(16, complex)
    for i in range(16):
        if (i >> (3 - a)) & 1 and (i >> (3 - b)) & 1:
            d[i] = -1
    return d
CZD = np.prod([cz_diag(a, b) for a, b in CZ_RING], axis=0)


def hea_U(t, L=4):
    U = np.eye(16, dtype=complex); k = 0
    for l in range(L + 1):
        U = kron_all([rz(t[k + 4 + q]) @ ry(t[k + q]) for q in range(4)]) @ U; k += 8
        if l < L:
            U = CZD[:, None] * U
    return U


PAIR = [None] * 15
for k, P in enumerate(PAULIS):
    PP = kron_all([PA[P[0]], PA[P[1]]])
    PAIR[k] = (np.eye(16) + np.kron(PP, PP)) / 16           # (A0 A1) x (B0 B1)


def hea_probs(t):
    U = hea_U(t)
    return np.array([np.real(np.diag(U @ PAIR[k] @ U.conj().T)) for k in range(15)])


def channel_tables(tin, tout):
    """Site (system s, ancilla a): input W|00>, Pauli r on s, measurement V. Returns A[b] and B[sigma, b]."""
    W = V13(tin); V = V13(tout); psi = W[:, 0]; w = np.outer(psi, psi.conj())
    rho = [np.kron(PA[r], I2) @ w @ np.kron(PA[r], I2) for r in range(4)]
    A = np.real(np.diag(V @ (sum(rho) / 4) @ V.conj().T))
    B = np.array([np.real(np.diag(V @ (sum(CHI[s, r] * rho[r] for r in range(4)) / 4) @ V.conj().T)) for s in range(4)])
    return A, B


def channel_probs(t):
    A, B = channel_tables(t[:13], t[13:])
    out = np.zeros((15, 16))
    for k, P in enumerate(PAULIS):
        out[k] = np.outer(A, A).reshape(16) + np.outer(B[P[0]], B[P[1]]).reshape(16)
    return np.clip(out, 0, None) / np.clip(out, 0, None).sum(1, keepdims=True)


def dagger13(t):
    """Angles of V(theta)^dagger in the same 13-angle form."""
    t = np.asarray(t, float)
    return np.array([-t[9], -t[8], -t[7], -t[12], -t[11], -t[10], -t[6], -t[2], -t[1], -t[0], -t[5], -t[4], -t[3]])


V_FIXED = None      # measurement used by 'channel_in13' (set by the caller)
NPAR = dict(site13=13, channel_echo13=13, channel_in13=13, su4_shared=15, su4_free=30, hea4=40, product6=6, channel26=26, product18=18)
NOUT = dict(product18=144)          # (setting_1, setting_2, b_1, b_2) for the randomized single-copy agent


def product18_probs(t):
    """Randomized single-copy agent: three product site settings (Euler on A and on B, 6 angles each); every site picks
    one of them uniformly at random in every circuit, and the decoder sees the settings. Outcome index
    ((s1 * 3 + s2) * 16 + b1 * 4 + b2)."""
    gs = [g3(np.kron(euler(*t[6 * k:6 * k + 3]), euler(*t[6 * k + 3:6 * k + 6]))) for k in range(3)]
    out = np.zeros((15, 144))
    for k, P in enumerate(PAULIS):
        for s1 in range(3):
            for s2 in range(3):
                prod = np.outer(gs[s1][P[0]], gs[s2][P[1]]).reshape(16)
                out[k, (s1 * 3 + s2) * 16:(s1 * 3 + s2 + 1) * 16] = (1 + prod) / 16 / 9
    return out


def probs(kind, t):
    if kind == "site13":
        g = g3(V13(t)); return site_probs([g, g])
    if kind == "su4_shared":
        g = g3(su4(t)); return site_probs([g, g])
    if kind == "su4_free":
        return site_probs([g3(su4(t[:15])), g3(su4(t[15:]))])
    if kind == "hea4":
        return hea_probs(t)
    if kind == "product6":
        U = np.kron(euler(*t[0:3]), euler(*t[3:6])); g = g3(U); return site_probs([g, g])
    if kind == "channel26":
        return channel_probs(t)
    if kind == "product18":
        return product18_probs(t)
    if kind == "channel_echo13":
        return channel_probs(np.r_[t, dagger13(t)])
    if kind == "channel_in13":
        return channel_probs(np.r_[t, V_FIXED])
    raise ValueError(kind)


def info_bits(p):
    """Mutual information I(P; b) (bits) of one pair, uniform prior over the 15 Paulis."""
    p = np.clip(p, 1e-15, None); pb = p.mean(0)
    return float((p * np.log2(p / pb[None, :])).sum(1).mean())


# ------------------------------------------------------------------ training
class Dec:
    def __init__(self, rng, nout=16):
        self.W = 0.05 * rng.standard_normal((15, nout)); self.m = np.zeros_like(self.W); self.v = np.zeros_like(self.W); self.t = 0

    def scores(self, b):                 # b: (E, m) outcome indices -> (E, 15)
        return self.W[:, b].sum(2).T

    def step(self, b, tgt, rng, lr=0.05, ent=0.003, base=0.0):
        s = self.scores(b); s -= s.max(1, keepdims=True); pr = np.exp(s); pr /= pr.sum(1, keepdims=True)
        a = (pr.cumsum(1) < rng.random(len(pr))[:, None]).sum(1).clip(0, 14); r = (a == tgt).astype(float)
        # gradient of -(r - base) log pi(a) - ent H
        gs = -(r - base)[:, None] * (np.eye(15)[a] - pr)
        logp = np.log(np.clip(pr, 1e-15, None)); H = -(pr * logp).sum(1)
        gH = -pr * (logp + H[:, None])                               # dH/ds
        gs -= ent * gH
        gs /= len(b)
        G = np.zeros_like(self.W)
        for j in range(b.shape[1]):
            np.add.at(G.T, b[:, j], gs)
        self.t += 1; self.m = 0.9 * self.m + 0.1 * G; self.v = 0.999 * self.v + 0.001 * G ** 2
        self.W -= lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)
        return r.mean()


def act_reward(dec, b, tgt, u):
    s = dec.scores(b); s -= s.max(1, keepdims=True); pr = np.exp(s); pr /= pr.sum(1, keepdims=True)
    a = (pr.cumsum(1) < u[:, None]).sum(1).clip(0, 14); return float((a == tgt).mean())


def sample_data(p, U):
    """p: (15, 16), U: (15, 8, 20) uniforms -> outcome indices (15, 8, 20) by inverse CDF (common random numbers)."""
    c = np.cumsum(p, 1); c[:, -1] = 1.0
    return np.stack([np.searchsorted(c[k], U[k]) for k in range(15)]).clip(0, p.shape[1] - 1)


def episodes(D, rng, E=2000, m=8):
    tgt = rng.integers(15, size=E); sh = rng.integers(20, size=(E, m))
    circ = np.argsort(rng.random((E, 8)), 1)[:, :m]
    return D[tgt[:, None], circ, sh], tgt


def train(kind, seed, iters=250, lam=0.2256, sigma=0.25, lr=0.08, K=2, anneal=0.7, restart_every=20,
          fe_patience=40, fe_below=0.5, fe_gain=0.05, init=None, log_every=0):
    rng = np.random.default_rng(seed); npar = NPAR[kind]
    th = rng.uniform(-np.pi, np.pi, npar) if init is None else np.array(init, float)
    nout = NOUT.get(kind, 16); dec = Dec(rng, nout); mT = np.zeros(npar); vT = np.zeros(npar); base = 1 / 15
    hist = []; anneal_from = None; fr = dict(start=1, best=-1.0, best_it=1); restarts = []
    for it in range(1, iters + 1):
        U = rng.random((15, 8, 20)); dirs = [rng.normal(0, sigma, npar) for _ in range(K)]
        fronts = [th] + [th + s * d for d in dirs for s in (1, -1)]
        D = []
        for f in fronts:
            p0 = probs(kind, f)
            if kind == "product18":      # noise acts on the outcome digits, not on the (classical) settings
                p0 = p0.reshape(15, 9, 16); p = ((1 - lam) * p0 + lam * p0.sum(2, keepdims=True) / 16).reshape(15, 144)
            else:
                p = (1 - lam) * p0 + lam / 16
            D.append(sample_data(p, U))
        se = int(rng.integers(2 ** 31)); rew = []
        for Df in D:
            r_ = np.random.default_rng(se); b, tgt = episodes(Df, r_); rew.append(act_reward(dec, b, tgt, r_.random(len(b))))
        g = sum((rew[1 + 2 * k] - rew[2 + 2 * k]) / (2 * sigma ** 2) * d for k, d in enumerate(dirs)) / K
        mT = 0.9 * mT + 0.1 * g; vT = 0.999 * vT + 0.001 * g ** 2
        r10 = float(np.mean([h[1] for h in hist[-9:]] + [rew[0]]))
        mult = 1.0
        if anneal:
            if anneal_from is None and len(hist) >= 9 and r10 > anneal:
                anneal_from = it
            if anneal_from is not None:
                mult = max(0.125, 0.5 ** ((it - anneal_from) // 20 + 1))
        th = th + mult * lr * (mT / (1 - 0.9 ** it)) / (np.sqrt(vT / (1 - 0.999 ** it)) + 1e-8)
        if fe_patience:
            if it - fr["start"] >= 10 and r10 > fr["best"] + fe_gain:
                fr["best"], fr["best_it"] = r10, it
            if r10 < fe_below and it - fr["best_it"] >= fe_patience and it - fr["start"] >= fe_patience:
                th = rng.uniform(-np.pi, np.pi, npar); mT[:] = 0; vT[:] = 0; anneal_from = None
                fr.update(start=it, best=-1.0, best_it=it); restarts.append(it)
        for _ in range(3):
            b, tgt = episodes(D[0], rng); rm = dec.step(b, tgt, rng, base=base); base = 0.9 * base + 0.1 * rm
        if restart_every and it % restart_every == 0:
            def score(dd):
                r_ = np.random.default_rng([seed, it, 99]); b, tgt = episodes(D[0], r_, E=6000)
                return act_reward(dd, b, tgt, r_.random(len(b)))
            cur = score(dec); best = None
            for j in range(3):
                dd = Dec(np.random.default_rng([seed, it, j]), nout); b0 = 1 / 15
                for _ in range(400):
                    b, tgt = episodes(D[0], rng, E=1000); rm = dd.step(b, tgt, rng, base=b0); b0 = 0.9 * b0 + 0.1 * rm
                sc = score(dd)
                if best is None or sc > best[0]:
                    best = (sc, dd)
            if best[0] > cur + 0.02:
                dec = best[1]; base = best[0]
        hist.append((it, rew[0]))
        if log_every and it % log_every == 0:
            print(kind, seed, it, round(r10, 3), flush=True)
    p_ideal = probs(kind, th)
    return dict(kind=kind, seed=seed, reward=[h[1] for h in hist], final=float(np.mean([h[1] for h in hist[-10:]])),
                theta=th.tolist(), info=info_bits(p_ideal), restarts=restarts)
