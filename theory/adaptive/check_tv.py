"""Numerical check of Lemma (adaptive detection): for random ADAPTIVE single-copy trees with rank-one POVMs,
TV( E_{P,s} p_{P,s} , p_0 ) <= eps^2 M/(d+1), and the per-leaf likelihood ratio is >= 1 - eps^2 M/(d+1).
Exact enumeration over leaves for n=1,2 and depth M<=3."""
import numpy as np, itertools
rng = np.random.default_rng(0)
P1 = [np.eye(2), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.diag([1,-1])]
def paulis(n):
    out=[]
    for idx in itertools.product(range(4), repeat=n):
        if any(idx):
            M=np.array([[1]])
            for i in idx: M=np.kron(M,P1[i])
            out.append(M)
    return out
def rand_povm(d, extra=2):
    # random rank-one POVM with d+extra outcomes: columns of an isometry
    K=d+extra
    A=rng.normal(size=(K,d))+1j*rng.normal(size=(K,d))
    Q,_=np.linalg.qr(A)          # K x d isometry, rows v_j with sum v_j^dag v_j = I
    vs=Q.conj()                   # element j: |v_j><v_j|, v_j = row of Q (conjugated)
    return [np.outer(v, v.conj()) for v in Q]
def run(n, M, eps):
    d=2**n; Ps=paulis(n); N=len(Ps)
    worst_ratio=np.inf; tv=0.0
    # adaptive tree: a fresh random POVM at every node
    def rec(depth, p0, pPs):
        nonlocal worst_ratio, tv
        if depth==M:
            mix=np.mean(pPs); worst_ratio=min(worst_ratio, mix/p0); tv+=max(p0-mix,0); return
        povm=rand_povm(d, extra=int(rng.integers(0,3)))
        for E in povm:
            q0=np.trace(E).real/d
            if q0<1e-12: continue
            new=[]
            for i,P in enumerate(Ps):
                for s in (1,-1):
                    rho=(np.eye(d)+s*eps*P)/d
                    new.append(np.trace(E@rho).real)
            new=np.array(new).reshape(N,2)
            rec(depth+1, p0*q0, (pPs.reshape(N,2)*new).reshape(-1))
    rec(0,1.0,np.ones(2*N))
    return tv, worst_ratio, eps**2*M/(d+1)
for n,M in ((1,2),(1,3),(2,2),(2,3)):
    for eps in (1.0,0.6):
        res=[run(n,M,eps) for _ in range(3)]
        print(f"n={n} M={M} eps={eps}: TV max {max(r[0] for r in res):.4f}  min leaf ratio {min(r[1] for r in res):.4f}  bound eps^2M/(d+1)={res[0][2]:.4f}  -> ratio>=1-bound: {all(r[1]>=1-r[2]-1e-12 for r in res)}")
