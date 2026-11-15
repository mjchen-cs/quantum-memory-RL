"""Natural ADAPTIVE single-copy baseline for eps=1 (ideal hardware): stabilizer-group elimination.
Non-identity Paulis are partitioned into d+1 maximal commuting groups (a complete set of MUBs), each group is
F_2^n minus 0 (d-1 Paulis). Measuring rho_P (eps=1) in the stabilizer basis of group G gives an outcome whose
eigenvalue on every Q in G is a uniformly random character chi(Q) = (-1)^{c.Q}, conditioned on chi(P)=+1 if P in G
(rho_P is supported on the +1 eigenspace of P); if P is not in G every character is equally likely. Candidates Q in
G with chi(Q) = -1 are eliminated (exact, because eps = 1). The posterior is uniform on the surviving candidates,
so the optimal guess succeeds with probability 1/|alive|.
Adaptive rule: always measure the group with the most surviving candidates (ties random).
Non-adaptive rule: cycle through the d+1 groups.
Reports copies for 90 % average success. Sign cocycles are ignored (linear characters), a modelling simplification."""
import numpy as np, sys
rng = np.random.default_rng(1)
popc = np.vectorize(lambda x: bin(x).count("1") & 1)

def run(n, adaptive, trials, mmax):
    d = 2 ** n; G = d + 1; vecs = np.arange(1, d)
    par = popc(np.bitwise_and.outer(np.arange(d), vecs))          # par[c, v] = c.v mod 2
    succ = np.zeros(mmax + 1)
    for _ in range(trials):
        alive = np.ones((G, d - 1), bool)
        g0, v0 = rng.integers(G), rng.integers(d - 1)
        succ[0] += 1 / alive.sum(); ptr = 0
        for m in range(1, mmax + 1):
            cnt = alive.sum(1)
            if adaptive:
                best = np.flatnonzero(cnt == cnt.max()); g = best[rng.integers(len(best))]
            else:
                g = ptr % G; ptr += 1
            while True:
                c = rng.integers(d)
                if g != g0 or par[c, v0] == 0:
                    break
            alive[g] &= par[c] == 0
            succ[m] += 1 / alive.sum()
    return succ / trials

def copies90(acc):
    i = np.flatnonzero(acc >= 0.9)
    return int(i[0]) if len(i) else None

for n in map(int, sys.argv[1:] or ["3", "4", "5", "6"]):
    d = 2 ** n
    tr = 400 if n <= 5 else 150
    mm = 12 * (d + 1) * n // 4
    a = run(n, True, tr, mm); b = run(n, False, tr, mm)
    print(f"n={n}: copies for 90%  adaptive greedy {copies90(a)}   non-adaptive round-robin {copies90(b)}", flush=True)
