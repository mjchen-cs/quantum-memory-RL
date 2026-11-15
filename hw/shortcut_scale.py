import sys, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fast_site import SiteModel, bell_theta
rng = np.random.default_rng(0)
def copies90(th, n, twirl, trials=300, grid=None):
    S = SiteModel(n, eps=1.0, twirl=twirl); p = S.probs(th); p /= p.sum(1, keepdims=True)
    lp = np.log(p); cdf = np.cumsum(p, 1)
    grid = np.unique(np.round(np.logspace(0, 4, 40)).astype(int))   # pairs
    acc = np.zeros(len(grid))
    for _ in range(trials):
        t = rng.integers(S.N); L = np.zeros(S.N); used = 0
        for j, m in enumerate(grid):
            L += lp[:, np.searchsorted(cdf[t], rng.random(m - used)).clip(0, S.D - 1)].sum(1); used = m
            best = np.flatnonzero(L == L.max()); acc[j] += (t in best) / len(best)
    acc /= trials; i = np.flatnonzero(acc >= 0.9)
    return 2 * grid[i[0]] if len(i) else float("nan")      # copies = 2 x pairs
nt = {s: np.array((json.load(open(f"runs/train_nt_s{s}.json")).get("best"))["theta"]) for s in (20, 22)}
tw = {s: np.array((json.load(open(f"runs/train_s{s}.json")).get("best"))["theta"]) for s in (20, 22)}
for n in (2, 3, 4, 5):
    row = {"bell(twirl)": copies90(bell_theta(), n, True)}
    for s in (20, 22):
        row[f"onchip{s}(twirl)"] = copies90(tw[s], n, True)
        row[f"shortcut{s}(no twirl)"] = copies90(nt[s], n, False, trials=200 if n < 5 else 80)
    print(n, row, flush=True)
