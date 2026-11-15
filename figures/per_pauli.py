import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_RUNS = _os.path.join(_ROOT, "hw", "runs")
_OUT = _os.path.join(_os.path.dirname(__file__), "out"); _os.makedirs(_OUT, exist_ok=True)
# Per-Pauli copies for 90% identification (factorized likelihood, works for n<=7), one random shot per circuit.
import sys, json, itertools, numpy as np
_os.makedirs(_os.path.join(_RUNS, 'per_pauli'), exist_ok=True)
sys.path.insert(0, _os.path.join(_ROOT, 'hw'))
import hw_analyze as H
from fast_site import tables
step, pols, trials = sys.argv[1], sys.argv[2].split(','), int(sys.argv[3])
man, raw, _ = H.load(step, 'origin')
n = man['records'][0]['n']
Pall = np.array(list(itertools.product(range(4), repeat=n)))[1:]; N = len(Pall); w = (Pall > 0).sum(1)
index = {tuple(p): i for i, p in enumerate(Pall)}
shift = np.array([2 * (n - 1 - i) for i in range(n)])
q_cal, _ = H.pools(man, raw, 'cal'); q_ev, _ = H.pools(man, raw, 'eval')
from fast_site import bell_theta
th_all = {k: np.array(v) for k, v in man['thetas_extra'].items()}; th_all['bell'] = bell_theta()
rng = np.random.default_rng(1)
out = {}
for pol in pols:
    g = tables(th_all[pol])[2]; s_sq = (g ** 2).mean(1)
    vals = []
    for (p_, P), circs in q_cal.items():
        if p_ != pol: continue
        Pv = np.array(P); dg = (np.concatenate(circs)[:, None] >> shift[None, :]) & 3
        lam = np.prod(g[Pv[None, :].repeat(len(dg), 0), dg], axis=1).mean() / float(np.prod(s_sq[Pv]))
        if lam > 0: vals.append(lam ** (1 / int((Pv > 0).sum())))
    kq = float(np.median(vals)); amp = kq ** w
    def llr(shots):
        dg = (np.asarray(shots)[:, None] >> shift[None, :]) & 3
        V = np.ones((N, len(shots)))
        for i in range(n): V *= g[Pall[:, i]][:, dg[:, i]]
        return np.log(np.clip(1 + amp[:, None] * V, 1e-12, None))
    res = {}
    for (p_, P), circs in q_ev.items():
        if p_ != pol: continue
        K = len(circs); grid = np.unique(np.round(np.logspace(0, np.log10(K), 25)).astype(int)); acc = np.zeros(len(grid))
        t = index[P]
        for _ in range(trials):
            order = rng.permutation(K)
            shots = [circs[i][rng.integers(len(circs[i]))] for i in order]
            L = np.cumsum(llr(shots), axis=1)
            for j, m in enumerate(grid):
                col = L[:, m - 1]; best = np.flatnonzero(col == col.max()); acc[j] += (t in best) / len(best)
        acc /= trials
        c90 = H.interp(acc, 2 * grid)
        res[''.join(map(str, P))] = dict(w=int((np.array(P) > 0).sum()), copies90=c90, acc=acc.tolist(), copies=(2 * grid).tolist(), max_copies=int(2 * K))
        print(step, pol, P, res[''.join(map(str, P))]['w'], c90, flush=True)
    out[pol] = dict(kappa=kq, paulis=res)
json.dump(out, open(_os.path.join(_RUNS, 'per_pauli', f'{step}_perpauli.json'), 'w'))
