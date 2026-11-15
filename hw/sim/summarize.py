# -*- coding: utf-8 -*-
"""Summary of the simulated controls -> results/ablation_summary.json (used by figures/fig7.py and source_data.py):
final rewards, information, site visibility of the learned front-ends, and the site visibility of 20,000 Haar-random
two-qubit unitaries."""
import os, sys, json, glob
import numpy as np
from scipy.stats import unitary_group
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from simlab import g3, V13, su4


def sitevis(kind, t):
    t = np.array(t)
    if kind == "site13": return [float((g3(V13(t))[1:] ** 2).mean())]
    if kind == "su4_shared": return [float((g3(su4(t))[1:] ** 2).mean())]
    if kind == "su4_free": return [float((g3(su4(t[:15]))[1:] ** 2).mean()), float((g3(su4(t[15:]))[1:] ** 2).mean())]
    return None


def rows(pattern, long=False):
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", pattern))):
        r = json.load(open(f)); R = np.convolve(r["reward"], np.ones(10) / 10, "valid")
        d = dict(seed=r["seed"], iters=len(r["reward"]), final=r["final"], info=r["info"], site_vis=sitevis(r["kind"], r["theta"]),
                 reward10=R.tolist(), restarts=r["restarts"])
        if long:
            d["long"] = True
        out.append(d)
    return out


out = {}
for k in ("site13", "su4_shared", "su4_free", "hea4", "product6", "channel26"):
    out[k] = rows(f"abl/{k}_*.json") + rows(f"abl_long/{k}_*.json", True)
for k in ("channel_in13", "channel_echo13"):
    out[k] = rows(f"abl_ch/{k}_*.json")
v = np.array([float((g3(unitary_group.rvs(4, random_state=i))[1:] ** 2).mean()) for i in range(20000)])
out["haar_site_vis"] = dict(n=20000, median=float(np.median(v)), p99=float(np.percentile(v, 99)), max=float(v.max()),
                            frac_above_0p9=float(np.mean(v > 0.9)), hist=np.histogram(v, bins=40, range=(0, 1))[0].tolist())
out["optimum_info"] = dict(site13_bell=0.949, channel=0.962, single_copy=0.191)
json.dump(out, open(os.path.join(HERE, "results", "ablation_summary.json"), "w"))
print("-> results/ablation_summary.json")
