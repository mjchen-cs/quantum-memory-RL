# -*- coding: utf-8 -*-
"""
Hardware-aware pruning of the reward-trained site front-end V(theta) (no QPU time).

V(theta) = post Euler (RZ RY RZ on A and on B) . CRX(phi) A->B . pre Euler. On WK_C180 a general CRX(phi) compiles
into two CZ plus single-qubit pulses, while CRX(+-pi) = S^(-+1) CNOT needs one CZ (hw_core.native_site_ops).
Pruning (greedy, deterministic):
  1. the last RZ on each qubit (theta[9], theta[12]) commutes with the Z measurement: removed (set to 0);
  2. every other angle is a candidate to be snapped to the nearest multiple of pi/2 (RY = 0 / RZ = 0 removes a gate,
     CRX = +-pi removes one CZ);
  3. hardware-aware objective J = V_ideal * (1 - 4/3 (1 - F_site)), with V_ideal = mean over sigma in {X, Y, Z} and
     outcomes of g_AB^2 (two-copy visibility, the quantity the decoder lives on) and F_site = f_CZ^(#CZ) * f_1q^(#pulses)
     (RZ is virtual; calibration of the chip snapshot runs/chip.json, median over the best couplers);
  4. repeatedly apply the snap that increases J most, as long as V_ideal stays within --tol of the original agent
     and J does not decrease; snaps that leave J unchanged (Clifford RZ) are accepted last, for a simpler circuit.
No retraining: the remaining angles are the agent's own.
Usage:  python hw_prune.py                       -> runs/prune.json  (used by hw_run.py steps PR5 / PR6)
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fast_site import tables, bell_theta              # noqa: E402
from hw_noise import load_chip, select_sites          # noqa: E402

RY_IDX = (1, 4, 8, 11); RZ_IDX = (0, 2, 3, 5, 7, 10); FINAL_RZ = (9, 12); CRX = 6
AGENTS = ["train_s1", "train_s11", "train_s12", "train_s20", "train_s22", "train_s30", "train_s31", "train_s32",
          "detect_s50", "detect_s51"]


def wrap(t):
    return (t + np.pi) % (2 * np.pi) - np.pi


def is0(t, tol=1e-9):
    return abs(wrap(t)) < tol


def vis(th):
    return float((tables(th)[2, 1:] ** 2).mean())


def cost(th):
    """(#CZ, #non-virtual single-qubit pulses) per site."""
    crx = wrap(th[CRX])
    if is0(crx):
        cz, p = 0, 0
    elif is0(abs(crx) - np.pi, 1e-9):
        cz, p = 1, 2                                   # H . CZ . H
    else:
        cz, p = 2, 4                                   # CRX(phi) = H_b CNOT RZ_b CNOT RZ_b H_b -> 2 CZ + 4 H
    p += sum(0 if is0(th[i]) else 1 for i in RY_IDX)
    return cz, p


def objective(th, fcz, f1):
    cz, p = cost(th)
    F = fcz ** cz * f1 ** p
    return vis(th) * (1 - 4 / 3 * (1 - F))


def prune(th0, fcz, f1, tol):
    th = np.array(th0, float).copy()
    th[list(FINAL_RZ)] = 0.0
    v0 = vis(np.array(th0, float))
    log = [dict(op="drop final RZ", V=vis(th), J=objective(th, fcz, f1))]
    free = [i for i in range(13) if i not in FINAL_RZ]
    while True:
        J = objective(th, fcz, f1); best = None
        for i in free:
            tgt = np.round(th[i] / (np.pi / 2)) * np.pi / 2
            if abs(tgt - th[i]) < 1e-12:
                continue
            t2 = th.copy(); t2[i] = tgt
            if vis(t2) < v0 - tol:
                continue
            dJ = objective(t2, fcz, f1) - J
            if dJ < -1e-6:
                continue
            key = (dJ > 1e-6, dJ, -abs(tgt - th[i]))
            if best is None or key > best[0]:
                best = (key, i, tgt)
        if best is None:
            break
        _, i, tgt = best
        log.append(dict(op=f"theta[{i}] {th[i]:+.3f} -> {tgt:+.3f}", V=None, J=None))
        th[i] = tgt; free.remove(i)
        log[-1].update(V=vis(th), J=objective(th, fcz, f1))
    th[CRX] = wrap(th[CRX]) if not is0(abs(wrap(th[CRX])) - np.pi, 1e-9) else np.sign(th[CRX]) * np.pi
    return th, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", default=os.path.join(HERE, "runs", "chip.json"))
    ap.add_argument("--tol", type=float, default=0.005, help="largest allowed drop of the ideal two-copy visibility")
    ap.add_argument("--sites", type=int, default=6, help="couplers used for the median calibration values")
    a = ap.parse_args()
    d, s1, f2 = load_chip(a.chip)
    st = select_sites(a.sites, s1, f2)
    fcz = float(np.median([x["f_cz"] for x in st]))
    f1 = float(np.median([s1[q]["g"] for x in st for q in (x["a"], x["b"])]))
    print(f"calibration {d.get('fetched')}: median CZ fidelity {fcz:.4f}, single-qubit gate fidelity {f1:.5f}")
    out = dict(chip=d.get("fetched"), f_cz=fcz, f_1q=f1, tol=a.tol, agents={})
    srcs = [("bell", bell_theta())]
    for f in AGENTS:
        p = os.path.join(HERE, "runs", f + ".json")
        if os.path.exists(p):
            s = json.load(open(p)); srcs.append((f, np.array((s.get("best") or s)["theta"], float)))
    print(f"{'agent':11s} {'V_ideal':>8s} {'->':>3s} {'pruned':>7s}  CZ/site  pulses/site   J (hardware-aware)   snapped")
    for name, th0 in srcs:
        th, log = prune(th0, fcz, f1, a.tol)
        c0, c1 = cost(th0), cost(th)
        out["agents"][name] = dict(theta=th0.tolist(), theta_pruned=th.tolist(), V=vis(th0), V_pruned=vis(th),
                                   cz=[c0[0], c1[0]], pulses=[c0[1], c1[1]],
                                   J=[objective(th0, fcz, f1), objective(th, fcz, f1)], log=log,
                                   clifford=bool(all(is0(th[i] % (np.pi / 2)) or is0(th[i] % (np.pi / 2) - np.pi / 2)
                                                     for i in range(13))))
        r = out["agents"][name]
        print(f"{name:11s} {r['V']:8.4f} -> {r['V_pruned']:7.4f}   {c0[0]} -> {c1[0]}    {c0[1]:2d} -> {c1[1]:2d}      "
              f"{r['J'][0]:.4f} -> {r['J'][1]:.4f}   {len(log) - 1:2d} angles{'  (all Clifford)' if r['clifford'] else ''}")
    json.dump(out, open(os.path.join(HERE, "runs", "prune.json"), "w"), indent=1)
    print("saved runs/prune.json")


if __name__ == "__main__":
    main()
