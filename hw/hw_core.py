# -*- coding: utf-8 -*-
"""
Hardware layer for the quantum drift-tracking experiment on Origin Quantum WK_C180.

Circuits are kept as plain gate lists [(name, qubits, params)] on logical qubits 0..2n-1
(copy A = 0..n-1, copy B = n..2n-1; site i = (i, n+i)) and translated to pyqpanda3 only at submission.

Preparation of rho_P = (I + eps P)/2^n needs no entangling gate: Pi_+ (the +1 eigenspace of P) is the uniform
mixture of product states that are eigenstates of P_i on the support of P (any computational state elsewhere) with
product of eigenvalues +1. A random draw per copy therefore prepares rho_P exactly in expectation. The shared Pauli
twirl of stage 3 is a shared random sign s for both copies: each copy is drawn from rho_{sP}.
"""
import numpy as np

# ------------------------------------------------------------------ single-qubit eigenstate preparation
# Pauli index 0=I 1=X 2=Y 3=Z; eigenvalue +1 / -1
PREP = {
    (3, +1): [], (3, -1): [("X",)],
    (1, +1): [("H",)], (1, -1): [("X",), ("H",)],
    (2, +1): [("H",), ("S",)], (2, -1): [("X",), ("H",), ("S",)],
}
# basis change before a computational-basis measurement so that outcome 0 <-> eigenvalue +1
MEAS = {3: [], 1: [("H",)], 2: [("SDG",), ("H",)]}


def sample_copy(P, sign, eps, rng):
    """One random product state whose mixture is rho_{sign*P}: returns per-qubit (pauli, eigenvalue) or (0, bit)."""
    n = len(P)
    supp = [i for i in range(n) if P[i] != 0]
    want = sign if rng.random() < (1 + eps) / 2 else -sign          # sign of the product of eigenvalues
    ev = rng.choice([1, -1], size=len(supp))
    if np.prod(ev) != want:
        ev[0] *= -1
    out = []
    k = 0
    for i in range(n):
        if P[i] == 0:
            out.append((0, int(rng.integers(2))))
        else:
            out.append((int(P[i]), int(ev[k]))); k += 1
    return out


def prep_ops(spec, offset):
    ops = []
    for i, (p, v) in enumerate(spec):
        q = offset + i
        if p == 0:
            if v == 1:
                ops.append(("X", (q,), ()))
        else:
            ops += [(g[0], (q,), ()) for g in PREP[(p, v)]]
    return ops


def site_V_ops(theta, a, b):
    """V(theta) of fast_site.py on qubits (a, b): pre Euler (RZ.RY.RZ) on a and b, CRX(phi) a->b, post Euler."""
    t = np.asarray(theta, float)
    ops = []
    for q, (x, y, z) in ((a, t[0:3]), (b, t[3:6])):
        ops += [("RZ", (q,), (x,)), ("RY", (q,), (y,)), ("RZ", (q,), (z,))]
    ops.append(("CRX", (a, b), (t[6],)))
    for q, (x, y, z) in ((a, t[7:10]), (b, t[10:13])):
        ops += [("RZ", (q,), (x,)), ("RY", (q,), (y,)), ("RZ", (q,), (z,))]
    return ops


def _is0(t, tol=1e-9):
    return abs((float(t) + np.pi) % (2 * np.pi) - np.pi) < tol


def native_site_ops(theta, a, b):
    """V(theta) for a PRUNED theta (CRX angle exactly +-pi, see hw_prune.py) in native gates with ONE CZ:
    CRX(+-pi) = RZ_a(-+pi/2) CNOT (global phase), CNOT = H_b CZ H_b; zero-angle rotations are omitted.
    Equal to V(theta) up to a global phase, so fast_site.tables(theta) describes it exactly."""
    t = np.asarray(theta, float)
    phi = (t[6] + np.pi) % (2 * np.pi) - np.pi
    if not (_is0(abs(phi) - np.pi, 1e-6)):
        raise ValueError("native_site_ops needs a pruned theta with CRX angle +-pi")
    sgn = 1.0 if t[6] > 0 else -1.0                       # CRX(pi) = S^dag CNOT, CRX(-pi) = S CNOT
    ops = []

    def euler(q, x, y, z):
        for name, v in (("RZ", x), ("RY", y), ("RZ", z)):
            if not _is0(v):
                ops.append((name, (q,), (float(v),)))
    euler(a, *t[0:3]); euler(b, *t[3:6])
    ops += [("H", (b,), ()), ("CZ", (a, b), ()), ("H", (b,), ())]
    euler(a, t[7] - sgn * np.pi / 2, t[8], t[9]); euler(b, *t[10:13])
    return ops


def quantum_circuit(P, theta, eps, rng, twirl=True, native=False):
    """Two independent copies of rho_{sP} (shared twirl sign s) followed by the site-local policy circuit."""
    n = len(P)
    s = int(rng.choice([1, -1])) if twirl else 1
    ops = prep_ops(sample_copy(P, s, eps, rng), 0) + prep_ops(sample_copy(P, s, eps, rng), n)
    for i in range(n):
        ops += native_site_ops(theta, i, n + i) if native else site_V_ops(theta, i, n + i)
    return ops, dict(sign=s)


def shadow_circuit(P, eps, rng):
    """Two independent copies of rho_P, every qubit measured in a uniformly random Pauli basis (single-copy)."""
    n = len(P)
    ops = prep_ops(sample_copy(P, 1, eps, rng), 0) + prep_ops(sample_copy(P, 1, eps, rng), n)
    bases = rng.integers(1, 4, size=2 * n)
    for q, bs in enumerate(bases):
        ops += [(g[0], (q,), ()) for g in MEAS[int(bs)]]
    return ops, dict(bases=bases.tolist())


def product_site_ops(theta, a, b):
    """Single-copy front-end: Euler rotations theta[0:3] on qubit a and theta[3:6] on qubit b, no two-qubit gate."""
    t = np.asarray(theta, float); ops = []
    for q, (x, y, z) in ((a, t[0:3]), (b, t[3:6])):
        ops += [("RZ", (q,), (x,)), ("RY", (q,), (y,)), ("RZ", (q,), (z,))]
    return ops


def sample_commutant(P, rng):
    """Uniformly random n-qubit Pauli R (per-qubit indices) that commutes with P."""
    P = np.asarray(P)
    while True:
        R = rng.integers(4, size=len(P))
        anti = sum(1 for p, r in zip(P, R) if p and r and p != r)
        if anti % 2 == 0:
            return [int(x) for x in R]


PAULI_GATES = {0: [], 1: [("X",)], 2: [("RZ", np.pi), ("X",)], 3: [("RZ", np.pi)]}   # up to global phases


def pauli_ops(R, qubits):
    ops = []
    for r, q in zip(R, qubits):
        for g in PAULI_GATES[int(r)]:
            ops.append((g[0], (q,), (g[1],) if len(g) > 1 else ()))
    return ops


def channel_circuit(P, theta_in, theta_out, rng, native=False, native_out=None):
    """Channel task: every site (system i, ancilla n+i) starts in |00>, the agent's input circuit W(theta_in) acts on it,
    the hidden-Pauli channel Lambda_P(rho) = (I + tr(P rho) P)/d acts on the system qubits (realized exactly as a
    uniformly random Pauli R from the commutant of P), and the agent's measurement circuit V(theta_out) follows."""
    n = len(P)
    site_in = native_site_ops if native else site_V_ops
    site_out = native_site_ops if (native if native_out is None else native_out) else site_V_ops
    ops = []
    for i in range(n):
        ops += site_in(theta_in, i, n + i)
    R = sample_commutant(P, rng)
    ops += pauli_ops(R, range(n))
    for i in range(n):
        ops += site_out(theta_out, i, n + i)
    return ops, dict(R=R)


# ------------------------------------------------------------------ numpy reference simulator (verification)
_G1 = {
    "X": np.array([[0, 1], [1, 0]], complex),
    "H": np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
    "S": np.diag([1, 1j]),
    "SDG": np.diag([1, -1j]),
}


def _rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
def _ry(t): return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], complex)
def _rx(t): return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]], complex)


def apply_ops(psi, ops, nq):
    psi = psi.reshape([2] * nq)
    for name, qs, ps in ops:
        if name in _G1 or name in ("RZ", "RY", "RX"):
            g = _G1[name] if name in _G1 else {"RZ": _rz, "RY": _ry, "RX": _rx}[name](ps[0])
            psi = np.moveaxis(np.tensordot(g, np.moveaxis(psi, qs[0], 0), axes=1), 0, qs[0])
        elif name == "CRX":
            c, t = qs
            psi = np.moveaxis(psi, (c, t), (0, 1)).copy()
            psi[1] = np.tensordot(_rx(ps[0]), psi[1], axes=1)
            psi = np.moveaxis(psi, (0, 1), (c, t))
        elif name == "CNOT":
            c, t = qs
            psi = np.moveaxis(psi, (c, t), (0, 1)).copy()
            psi[1] = psi[1][::-1].copy()
            psi = np.moveaxis(psi, (0, 1), (c, t))
        elif name == "CZ":
            c, t = qs
            psi = np.moveaxis(psi, (c, t), (0, 1)).copy()
            psi[1, 1] = -psi[1, 1]
            psi = np.moveaxis(psi, (0, 1), (c, t))
        else:
            raise ValueError(name)
    return psi.reshape(-1)


def outcome_probs(ops, nq):
    psi = np.zeros(2 ** nq, complex); psi[0] = 1
    return np.abs(apply_ops(psi, ops, nq)) ** 2           # index: qubit 0 = most significant bit


# ------------------------------------------------------------------ pyqpanda3 translation
def to_qprog(ops, nq, phys=None):
    """Translate to a pyqpanda3 QProg. phys maps logical -> physical qubit (no server mapping);
    classical bit k always holds logical qubit k."""
    from pyqpanda3.core import QProg, H, S, X, RZ, RY, RX, CRX, CNOT, CZ, measure, RZ as _RZ
    phys = list(range(nq)) if phys is None else list(phys)
    prog = QProg()
    for name, qs, ps in ops:
        q = [phys[x] for x in qs]
        if name == "H": prog << H(q[0])
        elif name == "S": prog << S(q[0])
        elif name == "SDG": prog << _RZ(q[0], -np.pi / 2)            # S^dag up to a global phase
        elif name == "X": prog << X(q[0])
        elif name == "RZ": prog << RZ(q[0], float(ps[0]))
        elif name == "RY": prog << RY(q[0], float(ps[0]))
        elif name == "RX": prog << RX(q[0], float(ps[0]))
        elif name == "CRX": prog << CRX(q[0], q[1], float(ps[0]))
        elif name == "CNOT": prog << CNOT(q[0], q[1])
        elif name == "CZ": prog << CZ(q[0], q[1])
        else: raise ValueError(name)
    prog << measure([phys[k] for k in range(nq)], list(range(nq)))
    return prog


def counts_to_logical(counts, nq):
    """pyqpanda3 count keys are bit strings with cbit 0 as the rightmost character; return {index: count} with
    logical qubit 0 as the most significant bit (the convention of fast_site / outcome_probs)."""
    out = {}
    for k, v in counts.items():
        k = str(k).strip()
        if k.lower().startswith("0x"):
            k = format(int(k, 16), "b")
        elif not set(k) <= {"0", "1"}:
            k = format(int(k), "b")                        # decimal outcome index
        k = k[-nq:].zfill(nq)
        bits = [int(ch) for ch in reversed(k)]          # bits[c] = value of cbit c = logical qubit c
        idx = 0
        for c in range(nq):
            idx = (idx << 1) | bits[c]
        out[idx] = out.get(idx, 0) + int(v)
    return out
