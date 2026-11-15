# -*- coding: utf-8 -*-
"""Pruning of channel-task agents (input circuit W(theta_in), measurement V(theta_out), hw_train.py --frontend
channel26): the last Z rotations of V are removed, then every angle is snapped, one at a time, to the nearest multiple
of pi/2 if the ideal information per pair (n = 2) stays within 0.005 bits of the original and the hardware objective
(information times the depolarizing factor of the native circuit) does not drop; the controlled rotations must end at
+-pi (one CZ each). Also writes the analytic reference: Bell-pair input and the matching Bell measurement V = W^dagger.
Usage: python hw_prune_channel.py runs/train_ch_s40.json [more files]  ->  runs/prune_channel.json"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from hw_train import ideal_info, channel_tables

F_CZ, F_1 = 0.987, 0.999
RY = (1, 4, 8, 11)          # RY angles of a 13-angle block (RZ rotations are virtual)


def wrap4(x):                # CRX(phi) has period 4 pi
    return (x + 2 * np.pi) % (4 * np.pi) - 2 * np.pi


def is_pm_pi(x):
    return abs(abs(wrap4(x)) - np.pi) < 1e-9


def cost(t):
    n1 = sum(1 for b in (0, 13) for k in RY if abs((t[b + k] + np.pi) % (2 * np.pi) - np.pi) > 1e-9)
    ncz = sum(1 if is_pm_pi(t[b + 6]) else 2 for b in (0, 13))
    return F_CZ ** ncz * F_1 ** n1


def info(t):
    return ideal_info(t, 2, "channel26")


def prune(theta):
    t = np.array(theta, float); i0 = info(t)
    t[[0, 3]] = 0.0              # first Z rotations of W act on |0>: a phase
    t[[22, 25]] = 0.0            # last Z rotations of V commute with the measurement
    for b in (0, 13):            # controlled rotations to +-pi (one CZ each)
        trial = t.copy(); trial[b + 6] = np.pi if wrap4(t[b + 6]) >= 0 else -np.pi
        if info(trial) >= i0 - 0.005:
            t = trial
    for k in [k for k in range(26) if k not in (6, 19)]:
        trial = t.copy(); trial[k] = np.round(t[k] / (np.pi / 2)) * np.pi / 2
        if info(trial) >= i0 - 0.005 and info(trial) * cost(trial) >= info(t) * cost(t) - 1e-12:
            t = trial
    return t, i0, info(t)


def reference():
    tin = np.zeros(13); tin[1] = np.pi / 2; tin[6] = np.pi            # RY(pi/2) on the system qubit, CRX(pi): Bell pair
    tout = np.zeros(13); tout[6] = -np.pi; tout[8] = -np.pi / 2        # V = W^dagger
    return np.r_[tin, tout]


if __name__ == "__main__":
    out = dict(agents={})
    ref = reference(); out["agents"]["bell_channel"] = dict(theta=ref.tolist(), theta_pruned=ref.tolist(),
                                                            info=ideal_info(ref, 2, "channel26"))
    for f in sys.argv[1:]:
        st = json.load(open(f)); th = np.array((st.get("best") or st)["theta"], float)
        if st.get("theta_out_fixed"):            # channel_in13: only the input circuit was trained
            th = np.r_[th, np.array(st["theta_out_fixed"], float)]
        tp, i0, i1 = prune(th)
        ok = all(is_pm_pi(tp[k]) for k in (6, 19))
        key = os.path.splitext(os.path.basename(f))[0]
        out["agents"][key] = dict(theta=th.tolist(), theta_pruned=tp.tolist(), info=i0, info_pruned=i1, native_ok=ok)
        print(f"{key}: information {i0:.3f} -> {i1:.3f} bits per pair, one CZ per circuit half: {ok}")
    print(f"reference (Bell pair + Bell measurement): {out['agents']['bell_channel']['info']:.3f} bits")
    json.dump(out, open(os.path.join(HERE, "runs", "prune_channel.json"), "w"), indent=1)
