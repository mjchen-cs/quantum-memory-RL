# -*- coding: utf-8 -*-
"""Site-local equivariant front-end: the same two-qubit unitary V(theta) acts on every pair
(qubit i of copy A, qubit i of copy B). This is the structure kept by structured pruning in stage 2.
Outcome probabilities factorize over sites for every Pauli string, so they are computed exactly from 4x4 tables:
  p_P(b) = (1 + eps prod_i gA[P_i,b_i] + eps prod_i gB[P_i,b_i] + eps^2 prod_i gAB[P_i,b_i]) / 4^n.
V(theta) = (R_A x R_B)_post . CRX(phi) . (R_A x R_B)_pre, R = RZ.RY.RZ, 13 parameters."""
import itertools
import numpy as np

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex)
PAULI = [I2.astype(complex), X, Y, Z]
NPAR = 13


def euler(a, b, c):
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    ry = np.array([[np.cos(b / 2), -np.sin(b / 2)], [np.sin(b / 2), np.cos(b / 2)]])
    return rz(c) @ ry @ rz(a)


def V_of(th):
    pre = np.kron(euler(*th[0:3]), euler(*th[3:6]))
    phi = th[6]
    crx = np.eye(4, dtype=complex)
    crx[2:, 2:] = np.cos(phi / 2) * I2 - 1j * np.sin(phi / 2) * X
    post = np.kron(euler(*th[7:10]), euler(*th[10:13]))
    return post @ crx @ pre


def tables(th):
    V = V_of(th)
    g = np.zeros((3, 4, 4))
    for s in range(4):
        for k, O in enumerate((np.kron(PAULI[s], I2), np.kron(I2, PAULI[s]), np.kron(PAULI[s], PAULI[s]))):
            g[k, s] = np.real(np.diag(V @ O @ V.conj().T))
    return g          # g[k, sigma, beta]


class SiteModel:
    def __init__(self, n, eps=0.9, twirl=False):
        self.n, self.eps, self.twirl = n, eps, twirl
        self.P = np.array(list(itertools.product(range(4), repeat=n)))[1:]
        self.B = np.array(list(itertools.product(range(4), repeat=n)))         # outcome digits per site
        self.N, self.D = len(self.P), 4 ** n

    def probs(self, th):
        g = tables(th)
        prod = np.ones((3, self.N, self.D))
        for i in range(self.n):
            prod *= g[:, self.P[:, i]][:, :, self.B[:, i]]       # (3, N, D)
        if self.twirl:
            # the same uniformly random Pauli R is applied to both copies and discarded: R P R = +-P with equal
            # probability, so the single-copy terms average out and only tr(P rho)^2 survives
            p = (1 + self.eps ** 2 * prod[2]) / self.D
        else:
            p = (1 + self.eps * prod[0] + self.eps * prod[1] + self.eps ** 2 * prod[2]) / self.D
        return np.clip(p, 1e-15, None)

    def visibility(self, th):
        g = tables(th)
        prod = np.ones((self.N, self.D))
        for i in range(self.n):
            prod *= g[2][self.P[:, i]][:, self.B[:, i]]
        return float((prod ** 2).mean())


def bell_theta():
    """CNOT(A->B) then H on A, written in this parametrization (used only as a reference check)."""
    th = np.zeros(NPAR); th[6] = np.pi; th[7:10] = (-np.pi / 2, -np.pi / 2, -np.pi)   # from stage-2 grid search
    return th
