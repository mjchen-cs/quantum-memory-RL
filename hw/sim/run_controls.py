# -*- coding: utf-8 -*-
"""Simulated reward-only training at n = 2 (Methods, 'Simulation controls'; Supplementary Note 7).
Reproduces every run in results/ (seed offsets as used for the paper); runs already present are skipped.
Usage: python run_controls.py [--procs 2]"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import simlab

JOBS = ([("abl", k, 1000 + s, dict(iters=250)) for s in range(12)
         for k in ("site13", "su4_shared", "su4_free", "hea4", "product6", "channel26")]
        + [("abl_long", "channel26", 2000 + s, dict(iters=500, fe_patience=25)) for s in range(12)]
        + [("abl_long", k, 2000 + s, dict(iters=750, fe_patience=40)) for s in range(6) for k in ("su4_free", "hea4")]
        + [("abl_ch", k, 4000 + s, dict(iters=250)) for s in range(8) for k in ("channel_echo13", "channel_in13")])


def run(job):
    group, kind, seed, kw = job
    tag = f"{kind}_{seed - {'abl': 1000, 'abl_long': 2000, 'abl_ch': 4000}[group]}"
    if group == "abl_long":
        tag = f"{kind}_{kw['iters']}_{kw['fe_patience']}_{seed - 2000}"
    out = os.path.join(HERE, "results", group, tag + ".json")
    if os.path.exists(out):
        return
    if kind == "channel_in13":                    # measurement learned on the state task (pruned seed 20)
        pr = json.load(open(os.path.join(HERE, "..", "runs", "prune.json")))["agents"]
        simlab.V_FIXED = np.array(pr["train_s20"]["theta_pruned"], float)
    r = simlab.train(kind, seed, **kw)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(r, open(out, "w"))
    print(group, kind, seed, round(r["final"], 3), round(r["info"], 3), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2); a = ap.parse_args()
    with Pool(a.procs) as p:
        p.map(run, JOBS, chunksize=1)
