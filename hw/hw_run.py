# -*- coding: utf-8 -*-
"""
First hardware batch (1 hour) for the quantum drift-tracking experiment on Origin Quantum WK_C180.

Steps (run in order; each is resumable, job ids are written before results are fetched):
  S1  timing and billing test           (~2-3 min of QPU time)
  S2  state-preparation / visibility     (~5 min)
  S3  core comparison, n = 4 (8 qubits)  (~15 min)
  S4  core comparison, n = 3 (6 qubits)  (~15 min)

Usage (on your own computer, Python 3.10+, pip install pyqpanda3 numpy):
  set the API key once per shell:   export ORIGIN_API_KEY=...        (PowerShell: $env:ORIGIN_API_KEY = "...")
  python hw_run.py --fetch_chip                       # fresh calibration snapshot -> runs/chip.json
  python hw_run.py --step S1 --mode dryrun            # offline rehearsal (no QPU time)
  python hw_run.py --step S1 --mode origin            # real run
  python hw_run.py --step S2 --mode origin   ... and so on; then  python hw_analyze.py --step S2

Design
  * qubits are pinned (server mapping off): site i of the policy uses a calibrated coupler (a_i, b_i) selected from
    the fresh snapshot; copy A lives on the a-qubits, copy B on the b-qubits; server-side readout mitigation on;
  * quantum circuits: two copies of rho_{sP} (shared twirl sign s), then the reward-trained site-local policy V(theta);
  * classical circuits: the same two copies, every qubit measured in a product-Pauli basis; all 3^n basis settings
    are used (derandomized shadows), so every Pauli is measured equally often.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hw_core import quantum_circuit, prep_ops, sample_copy, MEAS, to_qprog, outcome_probs, counts_to_logical, site_V_ops  # noqa

BACKEND = "WK_C180"
BATCH = 25
# scaled core steps (billing is about 0.4-0.5 s per 1,000 shots and essentially nothing per circuit)
CORE = {   # episodes use at most one shot per circuit, so circuits (independent preparations) matter more than shots
    "S3": dict(n=4, q_P=128, q_circ=60, q_shots=10, bell_P=32, c_P=8, c_circ=1000, c_shots=10),   # 8 qubits
    "S4": dict(n=3, q_P=32, q_circ=100, q_shots=10, bell_P=16, c_P=8, c_circ=200, c_shots=10),     # 6 qubits, 6,400 circuits
    "S5": dict(n=5, q_P=32, q_circ=320, q_shots=10, bell_P=16, c_P=0, c_circ=0, c_shots=10),       # 10 qubits, 15,360 circuits
    "S6": dict(n=6, q_P=32, q_circ=400, q_shots=10, bell_P=16, c_P=0, c_circ=0, c_shots=10),       # 12 qubits, 19,200 circuits
}
# billing is about 10 ms per circuit (S3: 17,600 circuits cost 199.5 s), so S3P keeps the circuit count moderate
S3P = dict(q_P=64, bell_P=16, q_circ=160, q_shots=10)       # 12,800 circuits, about 130 s
EXTRA_MAN = {}                            # extra manifest fields filled by build()
EXT = dict(DR_T=400, DR_runs=4, DR_q=(0.02, 0.1),      # drift stream: 2 q x 4 runs x 400 rounds x 2 circuits x 3 agents = 19,200
           SC3_circ=800, SC4_circ=1000, SC_bell3=200, SC_bell4=300,   # shortcut check: 2x16x800 + 16x200 = 28,800 (n=3); 2x16x1000 + 16x300 = 36,800 (n=4)
           PR5_circ=320, PR6_circ=400, PR7_circ=1000,                            # PR7: 2 x 16 x 1000 = 32,000                                            # pruning: 6 x 16 x (320 + 400) = 69,120
           DT5_sig=160, DT6_sig=200, DT7_sig=300, DT_null=2000, DTC_sig=3000, DTC_null=3000,   # task 2
           R4_circ=160, R5_circ=320, S5R_circ=320, S6R_circ=400, S7_circ=1000,   # S7: 2 x 16 x 1000 = 32,000   # S5R 20,480 / S6R 25,600 circuits      # 3 x 16 x 160 = 7,680 and 3 x 16 x 320 = 15,360 circuits
           S6X_circ=400,                  # 32 x 400 = 12,800 circuits, about 140 s
           S5C_P=8, S5C_circ=3500)        # 8 x 3,500 = 28,000 circuits, about 300 s (up to 7,000 copies)


# ------------------------------------------------------------------ chip snapshot and site selection
def fetch_chip(backend, out_f):
    from pyqpanda3 import qcloud
    key = os.environ.get("ORIGIN_API_KEY")
    if not key:
        raise SystemExit("set ORIGIN_API_KEY first")
    be = qcloud.QCloudService(api_key=key).backend(backend)
    ci = be.chip_info()

    def call(o, name):
        try:
            return getattr(o, name)()
        except Exception:
            return None
    two = []
    for d in (call(ci, "double_qubits_info") or []):
        qs = call(d, "get_qubits")
        if qs is not None:
            two.append(dict(qubits=[int(qs[0]), int(qs[1])], fidelity=call(d, "get_fidelity")))
    one = []
    for s in (call(ci, "single_qubit_info") or []):
        one.append(dict(qubit=call(s, "get_qubit_id"), gate_fidelity=call(s, "get_single_gate_fidelity"),
                        readout_fidelity=call(s, "get_readout_fidelity"), t1=call(s, "get_t1"), t2=call(s, "get_t2")))
    snap = dict(backend=backend, available_qubits=[int(q) for q in ci.available_qubits()],
                double_qubits=two, single_qubits=one, fetched=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(snap, open(out_f, "w"), indent=1)
    print(f"snapshot: {len(snap['available_qubits'])} qubits, {len(two)} couplers -> {out_f}")


def choose_sites(chip_f, n):
    from hw_noise import load_chip, select_sites
    _, s1, f2 = load_chip(chip_f)
    return select_sites(n, s1, f2)


# ------------------------------------------------------------------ circuit sets for each step
def paulis(n):
    import itertools
    return [list(p) for p in itertools.product(range(4), repeat=n)][1:]


def bell_theta_():
    from fast_site import bell_theta
    return bell_theta()


def classical_ops(P, setting_a, setting_b, eps, rng, sign=1):
    n = len(P)
    ops = prep_ops(sample_copy(P, sign, eps, rng), 0) + prep_ops(sample_copy(P, sign, eps, rng), n)
    for q, bs in enumerate(list(setting_a) + list(setting_b)):
        ops += [(g[0], (q,), ()) for g in MEAS[int(bs)]]
    return ops


def build(step, theta, eps, seed):
    """Return a list of circuit records {key, n, kind, P, ops, shots, meta}."""
    rng = np.random.default_rng(seed)
    recs = []
    if step == "S1":
        P = [1, 3]
        ops, meta = quantum_circuit(P, theta, eps, rng)
        for shots in (100, 1000, 4000):
            recs.append(dict(key=f"S1_single_{shots}", n=2, kind="quantum", P=P, ops=ops, shots=shots, meta=meta, group=f"g{shots}"))
        for j in range(BATCH):
            ops, meta = quantum_circuit(P, theta, eps, rng)
            recs.append(dict(key=f"S1_batch_{j}", n=2, kind="quantum", P=P, ops=ops, shots=100, meta=meta, group="batch"))
    elif step == "S2":
        n = 2
        test = [[1, 0], [0, 3], [2, 2], [1, 3], [3, 2], [2, 1]]
        for P in test:
            for r in range(10):
                ops, meta = quantum_circuit(P, theta, eps, rng)
                recs.append(dict(key=f"S2_q_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=P, ops=ops, shots=200, meta=meta))
                # single-copy check: prepare rho_P on both copies and measure every qubit in the eigenbasis of P_i
                setting = [p if p else 3 for p in P]
                ops2 = classical_ops(P, setting, setting, eps, rng)
                recs.append(dict(key=f"S2_c_{''.join(map(str, P))}_{r}", n=n, kind="classical", P=P, ops=ops2, shots=200,
                                 meta=dict(setting_a=setting, setting_b=setting)))
    elif step == "D1":
        # diagnosis of the n = 4 visibility: site-resolved (weight-one Paulis on every site) and full-weight Paulis,
        # for the block layout used in S3 and an interleaved layout
        n = 4
        Ps = [[p if i == k else 0 for i in range(n)] for k in range(n) for p in (1, 2, 3)]
        Ps += [list(map(int, rng.integers(1, 4, size=n))) for _ in range(4)]
        for layout in ("block", "interleave"):
            for P in Ps:
                for r in range(20):
                    ops, meta = quantum_circuit(P, theta, eps, rng)
                    meta.update(rep=r, policy="learned", layout=layout)
                    recs.append(dict(key=f"D1_{layout[0]}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=P, ops=ops,
                                     shots=10, meta=meta))
    elif step == "S3P":
        # S3 repeated on pinned, calibration-selected qubits: more circuits per Pauli, quantum only.
        # Paulis are taken from the S3 run so the classical-shadow data of S3 (single-qubit gates only) is the
        # matched baseline; every Pauli of the classical set is also run by both quantum policies.
        n = 4
        m = json.load(open(os.path.join(HERE, "runs", "S3_origin_sm", "manifest.json")))
        Pc = sorted({tuple(r["P"]) for r in m["records"] if r["kind"] == "classical"})
        Pold = sorted({tuple(r["P"]) for r in m["records"] if r["kind"] == "quantum"} - set(Pc))
        extra = [Pold[i] for i in sorted(rng.permutation(len(Pold))[:S3P["q_P"] - len(Pc)])]
        Pq = [list(P) for P in Pc] + [list(P) for P in extra]
        Pb = Pq[:S3P["bell_P"]]
        from fast_site import bell_theta
        for pol, Pset, th in (("learned", Pq, theta), ("bell", Pb, bell_theta())):
            for P in Pset:
                for r in range(S3P["q_circ"]):
                    ops, meta = quantum_circuit(P, th, eps, rng)
                    meta.update(rep=r, policy=pol)
                    recs.append(dict(key=f"{step}_{pol[0]}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=P, ops=ops,
                                     shots=S3P["q_shots"], meta=meta))
    elif step in ("R4", "R5"):
        # same-day comparison of three front-ends on the 16 Paulis that the earlier run measured with both policies:
        # learned (trained in simulation at n = 2), onchip (trained on WK_C180 at n = 2 by hw_train.py, reward only),
        # bell (analytic). Also a repetition of S3P / S5 on a different day with freshly selected qubits.
        n = 4 if step == "R4" else 5
        m = json.load(open(os.path.join(HERE, "runs", "S3P_origin" if n == 4 else "S5_origin", "manifest.json")))
        Pb = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "bell"})
        th_on = np.array(json.load(open(os.path.join(HERE, "runs", "train_s1.json")))["theta"], float)
        EXTRA_MAN["theta_onchip"] = th_on.tolist()
        from fast_site import bell_theta
        for pol, th in (("learned", theta), ("onchip", th_on), ("bell", bell_theta())):
            for P in Pb:
                for r in range(EXT[step + "_circ"]):
                    ops, meta = quantum_circuit(list(P), th, eps, rng)
                    meta.update(rep=r, policy=pol)
                    recs.append(dict(key=f"{step}_{pol[0]}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
    elif step in ("T4", "T5"):
        # transfer test of the agents trained ON THE CHIP from scratch with the final method (seeds 20 and 22, n = 2,
        # their best front-end by reward), with the analytic Bell front-end measured the same day as the reference;
        # same 16 Paulis as R4 / R5
        n = 4 if step == "T4" else 5
        m = json.load(open(os.path.join(HERE, "runs", "S3P_origin" if n == 4 else "S5_origin", "manifest.json")))
        Pb = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "bell"})
        extra = {}
        for sd in (20, 22):
            st = json.load(open(os.path.join(HERE, "runs", f"train_s{sd}.json")))
            extra[f"onchip{sd}"] = (st.get("best") or st)["theta"]
        EXTRA_MAN["thetas_extra"] = extra
        from fast_site import bell_theta
        pols = [(k, np.array(v, float), "o" + k[-2:]) for k, v in extra.items()] + [("bell", bell_theta(), "b")]
        for pol, th, code in pols:
            for P in Pb:
                for r in range(EXT["R" + step[1] + "_circ"]):
                    ops, meta = quantum_circuit(list(P), th, eps, rng)
                    meta.update(rep=r, policy=pol)
                    recs.append(dict(key=f"{step}_{code}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
    elif step in ("S5R", "S6R"):
        # repetition of S5 / S6 on another day (freshly selected qubits): the same Paulis, the simulation-trained
        # front-end on all of them and Bell on the 16 / 16 Bell Paulis, plus the front-end trained on the chip with
        # the final method (seed 20, best by reward) on the Bell Paulis
        n = 5 if step == "S5R" else 6
        m = json.load(open(os.path.join(HERE, "runs", f"S{n}_origin", "manifest.json")))
        Pq = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "learned"})
        Pb = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "bell"})
        st = json.load(open(os.path.join(HERE, "runs", "train_s20.json")))
        EXTRA_MAN["thetas_extra"] = {"onchip20": (st.get("best") or st)["theta"]}
        from fast_site import bell_theta
        pols = [("learned", theta, "l", Pq), ("bell", bell_theta(), "b", Pb),
                ("onchip20", np.array(EXTRA_MAN["thetas_extra"]["onchip20"], float), "o20", Pb)]
        for pol, th, code, Pset in pols:
            for P in Pset:
                for r in range(EXT[step + "_circ"]):
                    ops, meta = quantum_circuit(list(P), th, eps, rng)
                    meta.update(rep=r, policy=pol)
                    recs.append(dict(key=f"{step}_{code}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
    elif step == "S7":
        # n = 7 (14 qubits): 16 random Paulis, Bell and the front-end trained on the chip (seed 20, best by reward),
        # S7_circ circuits per Pauli and front-end (half calibration, half evaluation)
        n = 7
        allP = paulis(n)
        Pset = [allP[i] for i in sorted(rng.permutation(len(allP))[:16])]
        st = json.load(open(os.path.join(HERE, "runs", "train_s20.json")))
        EXTRA_MAN["thetas_extra"] = {"onchip20": (st.get("best") or st)["theta"]}
        from fast_site import bell_theta
        for pol, th, code in (("bell", bell_theta(), "b"),
                              ("onchip20", np.array(EXTRA_MAN["thetas_extra"]["onchip20"], float), "o20")):
            for P in Pset:
                for r in range(int(os.environ.get("S7_CIRC", EXT["S7_circ"]))):
                    ops, meta = quantum_circuit(list(P), th, eps, rng)
                    meta.update(rep=r, policy=pol)
                    recs.append(dict(key=f"{step}_{code}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
    elif step in ("PR5", "PR6"):
        # hardware-aware pruning (hw_prune.py -> runs/prune.json): same-day comparison on the 16 Bell Paulis of S5 / S6
        # of the agents trained on the chip (seeds 20, 22) as trained (general CRX, compiled by the cloud) and pruned
        # (native gates, one CZ per site), and of Bell written with CRX(pi) (as in all earlier steps) and in native gates
        n = 5 if step == "PR5" else 6
        m = json.load(open(os.path.join(HERE, "runs", f"S{n}_origin", "manifest.json")))
        Pb = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "bell"})
        pr = json.load(open(os.path.join(HERE, "runs", "prune.json")))["agents"]
        from fast_site import bell_theta
        fronts = [("bell", bell_theta(), "b", False), ("bell_native", pr["bell"]["theta_pruned"], "bn", True)]
        for sd in (20, 22):
            fronts += [(f"onchip{sd}", pr[f"train_s{sd}"]["theta"], f"o{sd}", False),
                       (f"onchip{sd}_pruned", pr[f"train_s{sd}"]["theta_pruned"], f"p{sd}", True)]
        EXTRA_MAN["thetas_extra"] = {k: list(map(float, th)) for k, th, _, _ in fronts if k != "bell"}
        EXTRA_MAN["native"] = [k for k, _, _, nat in fronts if nat]
        for pol, th, code, nat in fronts:
            th = np.array(th, float)
            for P in Pb:
                for r in range(int(os.environ.get("PR_CIRC", EXT[step + "_circ"]))):
                    ops, meta = quantum_circuit(list(P), th, eps, rng, native=nat)
                    meta.update(rep=r, policy=pol, native=nat)
                    recs.append(dict(key=f"{step}_{code}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
    elif step == "PR7":
        # n = 7 with the pruned / native front-ends only, on the 16 Paulis of S7 (same circuit count per Pauli as S7)
        n = 7
        m = json.load(open(os.path.join(HERE, "runs", "S7_origin", "manifest.json")))
        Pset = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "bell"})
        pr = json.load(open(os.path.join(HERE, "runs", "prune.json")))["agents"]
        fronts = [("bell_native", pr["bell"]["theta_pruned"], "bn"),
                  ("onchip20_pruned", pr["train_s20"]["theta_pruned"], "p20")]
        EXTRA_MAN["thetas_extra"] = {k: list(map(float, th)) for k, th, _ in fronts}
        EXTRA_MAN["native"] = [k for k, _, _ in fronts]
        for pol, th, code in fronts:
            th = np.array(th, float)
            for P in Pset:
                for r in range(int(os.environ.get("PR_CIRC", EXT["PR7_circ"]))):
                    ops, meta = quantum_circuit(list(P), th, eps, rng, native=True)
                    meta.update(rep=r, policy=pol, native=True)
                    recs.append(dict(key=f"{step}_{code}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
    elif step == "DR5":
        # drift-tracking task streamed on the chip (n = 5): hidden Pauli P_t uniform on ALL N Paulis, redrawn with
        # probability q per round; each round k = 4 copies = 2 circuits per agent; circuits are listed (and therefore
        # submitted) in time order, round by round, all agents interleaved, every circuit is a fresh preparation.
        # Agents: Bell and the pruned chip-trained front-end (native gates, twirled pairs as in S5) and the classical
        # random-basis agent (as in S5C).
        n = 5
        allP = paulis(n)
        pr = json.load(open(os.path.join(HERE, "runs", "prune.json")))["agents"]
        fq = [("bell_native", pr["bell"]["theta_pruned"], "bn"), ("onchip20_pruned", pr["train_s20"]["theta_pruned"], "p20")]
        EXTRA_MAN["thetas_extra"] = {k: list(map(float, th)) for k, th, _ in fq}
        EXTRA_MAN["native"] = [k for k, _, _ in fq]
        EXTRA_MAN["drift"] = dict(T=EXT["DR_T"], runs=EXT["DR_runs"], q=list(EXT["DR_q"]), k=4, sequences={})
        T = int(os.environ.get("DR_T", EXT["DR_T"]))
        EXTRA_MAN["drift"]["T"] = T
        for qi, q in enumerate(EXT["DR_q"]):
            for run in range(EXT["DR_runs"]):
                r_ = np.random.default_rng([seed, 77, qi, run])
                seq = [int(r_.integers(len(allP)))]
                for t in range(1, T):
                    seq.append(int(r_.integers(len(allP))) if r_.random() < q else seq[-1])
                EXTRA_MAN["drift"]["sequences"][f"{qi}_{run}"] = seq
                for t in range(T):
                    P = allP[seq[t]]
                    for j in range(2):
                        for pol, th, code in fq:
                            ops, meta = quantum_circuit(list(P), np.array(th, float), eps, rng, native=True)
                            meta.update(policy=pol, native=True, q=q, run=run, t=t, j=j, rep=2 * t + j)
                            recs.append(dict(key=f"DR5_{code}_{qi}_{run}_{t}_{j}", n=n, kind="quantum", P=list(P),
                                             ops=ops, shots=10, meta=meta))
                        sa = [int(x) for x in rng.integers(1, 4, size=n)]; sb = [int(x) for x in rng.integers(1, 4, size=n)]
                        recs.append(dict(key=f"DR5_c_{qi}_{run}_{t}_{j}", n=n, kind="classical", P=list(P),
                                         ops=classical_ops(list(P), sa, sb, eps, rng), shots=10,
                                         meta=dict(setting_a=sa, setting_b=sb, q=q, run=run, t=t, j=j, rep=2 * t + j)))
    elif step in ("SC3", "SC4"):
        # "classical shortcut" check on the chip: the front-ends trained on WK_C180 WITHOUT the twirl (train_nt_s20/22,
        # best by reward), deployed at n = 3 / 4 on the task without twirl (copies of rho_P, no random sign), and Bell
        # (native gates) on the same task as the reference; 16 random Paulis, one preparation per circuit
        n = 3 if step == "SC3" else 4
        allP = paulis(n)
        Pset = [allP[i] for i in sorted(rng.permutation(len(allP))[:16])]
        pr = json.load(open(os.path.join(HERE, "runs", "prune.json")))["agents"]
        fronts = []
        for sd in (20, 22):
            st = json.load(open(os.path.join(HERE, "runs", f"train_nt_s{sd}.json")))
            fronts.append((f"shortcut{sd}", (st.get("best") or st)["theta"], f"s{sd}", False, EXT[f"{step}_circ"]))
        fronts.append(("bell_native", pr["bell"]["theta_pruned"], "bn", True, EXT[f"SC_bell{n}"]))
        EXTRA_MAN["thetas_extra"] = {k: list(map(float, th)) for k, th, _, _, _ in fronts}
        EXTRA_MAN["native"] = [k for k, _, _, nat, _ in fronts if nat]
        EXTRA_MAN["twirl"] = False
        for pol, th, code, nat, nc in fronts:
            th = np.array(th, float)
            for P in Pset:
                for r in range(int(os.environ.get("SC_CIRC", nc))):
                    ops, meta = quantum_circuit(list(P), th, eps, rng, twirl=False, native=nat)
                    meta.update(rep=r, policy=pol, native=nat, twirl=False)
                    recs.append(dict(key=f"{step}_{code}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
    elif step in ("DT5", "DT6", "DT7"):
        # task 2 (detection): signal pairs rho_{sP} x rho_{sP} (shared twirl sign, fresh per circuit) on 16 Paulis and
        # null pairs (I/d: every qubit in a uniformly random computational basis state), for the Bell front-end, the
        # task-1 front-end trained on the chip (seed 20) and, if present, front-ends trained on the chip for detection
        n = int(step[2])
        src = {5: "S5_origin", 6: "S6_origin", 7: "S7_origin"}[n]
        m = json.load(open(os.path.join(HERE, "runs", src, "manifest.json")))
        Pb = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "bell"})
        fronts = {"bell": bell_theta_()}
        st = json.load(open(os.path.join(HERE, "runs", "train_s20.json"))); fronts["onchip20"] = (st.get("best") or st)["theta"]
        for sd in (50, 51, 52):
            f = os.path.join(HERE, "runs", f"detect_s{sd}.json")
            if os.path.exists(f):
                sd_ = json.load(open(f)); fronts[f"det{sd}"] = (sd_.get("best") or sd_)["theta"]
        EXTRA_MAN["thetas_extra"] = {k: list(map(float, v)) for k, v in fronts.items() if k != "bell"}
        n_sig, n_null = EXT[f"DT{n}_sig"], EXT["DT_null"]
        if os.environ.get("DT_TEST"):                       # small sizes for an offline test only
            n_sig, n_null = 20, 100
        for pol, th in fronts.items():
            th = np.array(th, float); code = pol[0] + pol[-2:]
            for P in Pb:
                for r in range(n_sig):
                    ops, meta = quantum_circuit(list(P), th, eps, rng)
                    meta.update(rep=r, policy=pol, label="signal")
                    recs.append(dict(key=f"{step}_{code}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=list(P),
                                     ops=ops, shots=10, meta=meta))
            for r in range(n_null):
                bits = rng.integers(2, size=2 * n)
                ops = prep_ops([(0, int(x)) for x in bits[:n]], 0) + prep_ops([(0, int(x)) for x in bits[n:]], n)
                for i in range(n):
                    ops += site_V_ops(th, i, n + i)
                recs.append(dict(key=f"{step}_{code}_null_{r}", n=n, kind="quantum", P=[0] * n, ops=ops, shots=10,
                                 meta=dict(rep=r, policy=pol, label="null")))
    elif step == "DTC5":
        # task 2, classical single-copy baseline at n = 5: both copies measured in independent random Pauli bases;
        # signal pairs carry the shared twirl sign (as in the task), null pairs are I/d
        n = 5
        m = json.load(open(os.path.join(HERE, "runs", "S5C_origin", "manifest.json")))
        Pc = sorted({tuple(r["P"]) for r in m["records"] if r["kind"] == "classical"})
        n_sig, n_null = (60, 200) if os.environ.get("DT_TEST") else (EXT["DTC_sig"], EXT["DTC_null"])
        for P in Pc:
            for j in range(n_sig):
                sa = [int(x) for x in rng.integers(1, 4, size=n)]; sb = [int(x) for x in rng.integers(1, 4, size=n)]
                sg = int(rng.choice([1, -1]))
                ops = classical_ops(list(P), sa, sb, eps, rng, sign=sg)
                recs.append(dict(key=f"{step}_c_{''.join(map(str, P))}_{j}", n=n, kind="classical", P=list(P), ops=ops,
                                 shots=10, meta=dict(setting_a=sa, setting_b=sb, rep=j, label="signal")))
        for j in range(n_null):
            sa = [int(x) for x in rng.integers(1, 4, size=n)]; sb = [int(x) for x in rng.integers(1, 4, size=n)]
            bits = rng.integers(2, size=2 * n)
            ops = prep_ops([(0, int(x)) for x in bits[:n]], 0) + prep_ops([(0, int(x)) for x in bits[n:]], n)
            for q, bs in enumerate(sa + sb):
                ops += [(g[0], (q,), ()) for g in MEAS[int(bs)]]
            recs.append(dict(key=f"{step}_c_null_{j}", n=n, kind="classical", P=[0] * n, ops=ops, shots=10,
                             meta=dict(setting_a=sa, setting_b=sb, rep=j, label="null")))
    elif step == "S6X":
        # S6 extension: 400 more circuits per Pauli for the learned policy on the same 32 Paulis (reps 400..799),
        # so that the accuracy curve reaches 800 copies; analysed together with S6
        m = json.load(open(os.path.join(HERE, "runs", "S6_origin", "manifest.json")))
        Pq = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "learned"})
        r0 = 1 + max(r["meta"]["rep"] for r in m["records"] if r.get("meta", {}).get("policy") == "learned")
        for P in Pq:
            for r in range(r0, r0 + EXT["S6X_circ"]):
                ops, meta = quantum_circuit(list(P), theta, eps, rng)
                meta.update(rep=r, policy="learned")
                recs.append(dict(key=f"S6X_l_{''.join(map(str, P))}_{r}", n=6, kind="quantum", P=list(P), ops=ops,
                                 shots=10, meta=meta))
    elif step == "S5C":
        # classical-shadow baseline at n = 5 on hardware: 8 of the Paulis that S5 ran with both quantum policies,
        # fresh random bases in every circuit; analysed together with S5
        m = json.load(open(os.path.join(HERE, "runs", "S5_origin", "manifest.json")))
        Pb = sorted({tuple(r["P"]) for r in m["records"] if r.get("meta", {}).get("policy") == "bell"})
        Pc = [Pb[i] for i in sorted(rng.permutation(len(Pb))[:EXT["S5C_P"]])]
        n = 5
        for P in Pc:
            for j in range(EXT["S5C_circ"]):
                sa = [int(x) for x in rng.integers(1, 4, size=n)]; sb = [int(x) for x in rng.integers(1, 4, size=n)]
                ops = classical_ops(list(P), sa, sb, eps, rng)
                recs.append(dict(key=f"S5C_c_{''.join(map(str, P))}_{j}", n=n, kind="classical", P=list(P), ops=ops,
                                 shots=10, meta=dict(setting_a=sa, setting_b=sb, rep=j)))
    elif step in CORE:
        cfg = CORE[step]; n = cfg["n"]
        allP = paulis(n)
        by_w = {}
        for P in allP:
            by_w.setdefault(sum(1 for x in P if x), []).append(list(P))

        def stratified(k):
            pool = {w: list(v) for w, v in by_w.items()}; out = []
            while len(out) < k:
                for w in sorted(pool):
                    if pool[w] and len(out) < k:
                        out.append(pool[w].pop(int(rng.integers(len(pool[w])))))
            return out
        Pc = allP if cfg["c_P"] == "all" else (stratified(cfg["c_P"]) if cfg["c_P"] else [])
        if cfg["q_P"] == "all":
            Pq = allP
        else:
            rest = [P for P in allP if P not in Pc]
            Pq = Pc + [rest[i] for i in sorted(rng.permutation(len(rest))[:cfg["q_P"] - len(Pc)])]
        Pb = [] if not cfg["bell_P"] else (allP if cfg["bell_P"] == "all" else
                                          Pc + [P for P in Pq if P not in Pc][:cfg["bell_P"] - len(Pc)])
        from fast_site import bell_theta
        for pol, Pset, th in (("learned", Pq, theta), ("bell", Pb, bell_theta())):
            for P in Pset:
                for r in range(cfg["q_circ"]):
                    ops, meta = quantum_circuit(P, th, eps, rng)
                    meta.update(rep=r, policy=pol)
                    recs.append(dict(key=f"{step}_{pol[0]}_{''.join(map(str, P))}_{r}", n=n, kind="quantum", P=P, ops=ops,
                                     shots=cfg["q_shots"], meta=meta))
        for P in Pc:                                        # classical shadows: fresh random bases in every circuit
            for j in range(cfg["c_circ"]):
                sa = [int(x) for x in rng.integers(1, 4, size=n)]; sb = [int(x) for x in rng.integers(1, 4, size=n)]
                ops = classical_ops(P, sa, sb, eps, rng)
                recs.append(dict(key=f"{step}_c_{''.join(map(str, P))}_{j}", n=n, kind="classical", P=P, ops=ops,
                                 shots=cfg["c_shots"], meta=dict(setting_a=sa, setting_b=sb, rep=j)))
    return recs


def program_qubits(r):
    """Logical -> program qubit for cloud-mapped runs. 'block': A_0..A_{n-1}, B_0..B_{n-1};
    'interleave': A_0, B_0, A_1, B_1, ... so that the two qubits of every site are neighbours in the program."""
    n = r["n"]
    if r.get("meta", {}).get("layout") == "interleave":
        return [2 * i for i in range(n)] + [2 * i + 1 for i in range(n)]
    return list(range(2 * n))


# ------------------------------------------------------------------ runners
class JobFailed(RuntimeError):
    """The cloud reported FAILED (nothing is billed); the circuits are re-submitted on the next run."""


class DryRunner:
    """Offline rehearsal: exact outcome probabilities plus symmetric readout flips from the snapshot
    (flips: {n: per-logical-qubit flip probabilities})."""
    def __init__(self, flips, seed=0):
        self.flips, self.rng, self.jobs = flips, np.random.default_rng(seed), {}

    def submit(self, recs, shots):
        jid = f"dry{len(self.jobs)}"
        out = []
        for r in recs:
            nq = 2 * r["n"]
            p = outcome_probs(r["ops"], nq)
            idx = self.rng.choice(len(p), size=shots, p=p / p.sum())
            bits = (idx[:, None] >> np.arange(nq - 1, -1, -1)) & 1
            bits ^= (self.rng.random(bits.shape) < np.array(self.flips[r["n"]])[None, :])
            cnt = {}
            for row in bits:
                k = "".join(str(b) for b in row[::-1])              # cbit 0 rightmost, as pyqpanda3
                cnt[k] = cnt.get(k, 0) + 1
            out.append(cnt)
        self.jobs[jid] = (out, time.time())
        return jid

    def fetch(self, jid):
        out, t = self.jobs[jid]
        return out, dict(submitted=t, finished=time.time())


class OriginRunner:
    def __init__(self, backend, phys_list, server_mapping=False):
        from pyqpanda3 import qcloud
        key = os.environ.get("ORIGIN_API_KEY")
        if not key:
            raise SystemExit("set ORIGIN_API_KEY first (the key is never written to disk by this script)")
        self.qc = qcloud
        self.backend = qcloud.QCloudService(api_key=key).backend(backend)
        o = qcloud.QCloudOptions()
        o.set_amend(True)          # server-side readout-error mitigation
        o.set_mapping(server_mapping)   # default False: qubits pinned to the selected couplers
        o.set_optimization(True)
        self.options, self.phys, self.server_mapping = o, phys_list, server_mapping
        self.n_expected, self.shots_of = {}, {}

    def submit(self, recs, shots):
        progs = [to_qprog(r["ops"], 2 * r["n"], program_qubits(r) if self.server_mapping else self.phys[r["n"]])
                 for r in recs]
        for attempt in range(3):
            try:
                jid = self.backend.run(progs, shots, self.options).job_id()
                self.n_expected[jid], self.shots_of[jid] = len(recs), shots
                return jid
            except Exception as e:
                print(f"  submit retry {attempt + 1}: {e}", flush=True); time.sleep(15 * (attempt + 1))
        raise RuntimeError("submission failed repeatedly")

    def status(self, jid):
        try:
            st = self.qc.QCloudJob(jid).status()
            return getattr(st, "name", str(st))
        except Exception as e:
            return f"status query failed ({e})"

    def error(self, jid):
        try:
            return self.qc.QCloudJob(jid).query().error_message()
        except Exception:
            return "?"

    def fetch(self, jid):
        job = self.qc.QCloudJob(jid)
        t0 = time.time(); last = None; polls = 0
        while True:
            name = self.status(jid); polls += 1
            if name == "FINISHED":
                break
            if polls % 6 == 0 and name not in ("FAILED",):
                # the status endpoint sometimes keeps reporting COMPUTING for a job the console shows as finished:
                # every 2 minutes try to read the result directly and accept it if it is complete
                try:
                    r_ = job.result(); c_ = []
                    for g_ in ("get_counts_list", "get_probs_list"):
                        try:
                            c_ = getattr(r_, g_)() or []
                        except Exception:
                            c_ = []
                        if c_:
                            break
                    if len(c_) == self.n_expected.get(jid, 1):
                        print(f"    {jid}: status says {name}, but the complete result is available -> collecting",
                              flush=True)
                        break
                except Exception:
                    pass
            if name == "FAILED":
                raise JobFailed(f"job {jid} failed: {self.error(jid)}")
            if name != last or time.time() - t0 > 60 and int(time.time() - t0) % 300 < 20:
                print(f"    {jid}: {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                last = name
            time.sleep(20)
        res = job.result()
        n_exp = self.n_expected.get(jid, 1)
        counts = []
        try:
            counts = res.get_counts_list() or []
        except Exception:
            pass
        if len(counts) != n_exp:                 # with readout mitigation the cloud may return probabilities only
            try:
                probs = res.get_probs_list() or []
                if len(probs) == n_exp:
                    counts = [{k: int(round(float(v) * self.shots_of[jid])) for k, v in p.items()} for p in probs]
            except Exception:
                pass
        if len(counts) != n_exp and n_exp == 1:
            for getter in ("get_counts", "get_probs"):
                try:
                    c = getattr(res, getter)()
                    if c:
                        counts = [c if getter == "get_counts" else
                                  {k: int(round(float(v) * self.shots_of[jid])) for k, v in c.items()}]
                        break
                except Exception:
                    pass
        if len(counts) != n_exp:
            raise RuntimeError(f"job {jid}: expected {n_exp} results, got {len(counts)}")
        info = dict(polled_s=time.time() - t0)
        info["result_methods"] = [x for x in dir(res) if not x.startswith("_")]
        for attr in ("mapping_qprog", "get_mapping_qprog", "mapping_qubits", "get_mapping", "origin_ir", "compiled_prog"):
            try:
                v = getattr(res, attr)
                info[attr] = str(v() if callable(v) else v)[:200000]       # compiled program / physical qubits
            except Exception:
                pass
        try:
            q = job.query()
            info["query_methods"] = [x for x in dir(q) if not x.startswith("_")]
            for attr in ("mapping_qubits", "get_mapping", "compiled_prog", "qubits", "get_qubits"):
                try:
                    v = getattr(q, attr); info["query_" + attr] = str(v() if callable(v) else v)[:200000]
                except Exception:
                    pass
        except Exception:
            pass
        for attr in ("execution_time", "running_time", "time", "get_running_time"):  # noqa   # record whatever is offered
            try:
                v = getattr(res, attr); info[attr] = v() if callable(v) else v
            except Exception:
                pass
        return counts, info


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["S1", "S2", "S3", "S4", "S5", "S6", "D1", "S3P", "S6X", "S5C", "R4", "R5", "T4", "T5", "S5R", "S6R", "S7", "DT5", "DT6", "DT7", "DTC5", "PR5", "PR6", "PR7", "SC3", "SC4", "DR5"])
    ap.add_argument("--batch", type=int, default=BATCH, help="circuits per cloud job (split automatically on errors)")
    ap.add_argument("--mode", choices=["dryrun", "origin"], default="dryrun")
    ap.add_argument("--fetch_chip", action="store_true")
    ap.add_argument("--qpu", default="WK_C180", help="Origin backend name (python list_backends.py prints them)")
    ap.add_argument("--chip", default="", help="default runs/chip.json (WK_C180) or runs/chip_<qpu>.json")
    ap.add_argument("--agent", default=os.path.join(HERE, "agent_n2.json"))
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--abandon", default="", help="comma-separated job ids stuck on the cloud: their circuits are "
                    "submitted again as new jobs (the stuck job is ignored even if it finishes later)")
    ap.add_argument("--status", action="store_true", help="only print the status of the submitted jobs of this step")
    ap.add_argument("--server_mapping", action="store_true",
                    help="fallback if pinned qubits are rejected: let the cloud choose the physical qubits")
    a = ap.parse_args()
    if not a.chip:
        a.chip = os.path.join(HERE, "runs", "chip.json" if a.qpu == BACKEND else f"chip_{a.qpu}.json")
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    if a.fetch_chip:
        fetch_chip(a.qpu, a.chip); return
    if not os.path.exists(a.chip):
        raise SystemExit(f"{a.chip} missing: run  python hw_run.py --fetch_chip  (or copy a snapshot there)")
    theta = np.array(json.load(open(a.agent))["theta"])
    recs = build(a.step, theta, a.eps, a.seed)
    nmax = max(r["n"] for r in recs)
    tag = a.mode + ("_sm" if a.server_mapping else "")      # server-mapped runs are kept apart from pinned runs
    if a.qpu != BACKEND:
        tag += "_" + a.qpu                                    # runs on another chip get their own folder
    out_dir = os.path.join(HERE, "runs", f"{a.step}_{tag}")
    man_f = os.path.join(out_dir, "manifest.json")
    base = {"S6X": "S6", "S5C": "S5"}.get(a.step)
    base_f = os.path.join(HERE, "runs", f"{base}_{tag}", "manifest.json") if base else None
    if base_f and os.path.exists(base_f):
        old = json.load(open(base_f))                         # extension steps run on the qubits of the step they extend
        sites, chip_fetched = old["sites"], old.get("chip_fetched")
        print(f"using the qubits of {base} (calibration {chip_fetched})")
    elif os.path.exists(man_f) and json.load(open(man_f)).get("sites"):
        old = json.load(open(man_f))                          # resuming: keep the qubits this step started on
        sites, chip_fetched = old["sites"], old.get("chip_fetched")
        print(f"resuming on the qubits chosen at the start of this step (calibration {chip_fetched})")
    else:
        sites, chip_fetched = choose_sites(a.chip, max(nmax, 4)), json.load(open(a.chip)).get("fetched")
    phys = {n: [s["a"] for s in sites[:n]] + [s["b"] for s in sites[:n]] for n in range(2, max(nmax, 4) + 1)}
    os.makedirs(out_dir, exist_ok=True)
    raw_f, jobs_f = os.path.join(out_dir, "raw.json"), os.path.join(out_dir, "jobs.json")
    raw = json.load(open(raw_f)) if os.path.exists(raw_f) else {}
    jobs = json.load(open(jobs_f)) if os.path.exists(jobs_f) else {}
    if a.abandon:
        ff = os.path.join(out_dir, "failed_jobs.json")
        fl = json.load(open(ff)) if os.path.exists(ff) else {}
        for jid in [x.strip() for x in a.abandon.split(",") if x.strip()]:
            if jid in jobs:
                fl[jid] = dict(jobs.pop(jid), error="abandoned by the user: stuck in COMPUTING on the cloud")
                print(f"abandoned {jid}: its {len(fl[jid]['keys'])} circuits will be submitted again", flush=True)
            else:
                print(f"--abandon: {jid} is not an open job of this step", flush=True)
        json.dump(fl, open(ff, "w"), indent=1); json.dump(jobs, open(jobs_f, "w"), indent=1, default=str)
    json.dump(dict(EXTRA_MAN, step=a.step, mode=a.mode, eps=a.eps, seed=a.seed, theta=theta.tolist(), sites=sites, phys=phys,
                   chip_fetched=chip_fetched, n_circuits=len(recs),
                   records=[{k: v for k, v in r.items() if k != "ops"} for r in recs]),
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    if a.mode == "origin" and not os.environ.get("ORIGIN_API_KEY"):
        raise SystemExit("set ORIGIN_API_KEY first")
    print(f"{a.step} [{a.mode}]: {len(recs)} circuits, {sum(r['shots'] for r in recs)} shots; "
          f"qubits {phys[nmax]}", flush=True)
    if a.mode == "origin":
        runner = OriginRunner(a.qpu, phys, a.server_mapping)
    else:
        from hw_noise import load_chip
        _, s1, _ = load_chip(a.chip)
        flips = {n: [1 - s1[q]["r"] for q in phys[n]] for n in phys}
        runner = DryRunner(flips)
    if a.status:
        for jid, j in jobs.items():
            done = all(k in raw for k in j["keys"])
            st = "COLLECTED" if done else (runner.status(jid) if a.mode == "origin" else "dryrun")
            extra = f"  error: {runner.error(jid)}" if st == "FAILED" else ""
            print(f"  {jid}: {len(j['keys'])} x {j['shots']} shots, submitted {j['submitted']}: {st}{extra}")
        return
    pending = {k for j in jobs.values() for k in j["keys"]}
    groups = {}
    for r in recs:
        if r["key"] not in raw and r["key"] not in pending:
            groups.setdefault((r.get("group", ""), r["shots"]), []).append(r)
    for (_, shots), rs in groups.items():
        queue = [rs[i:i + a.batch] for i in range(0, len(rs), a.batch)]
        while queue:
            part = queue.pop(0)
            try:
                jid = runner.submit(part, shots)
            except Exception as e:
                if len(part) == 1:
                    raise
                print(f"  submission of {len(part)} circuits failed ({e}); splitting", flush=True)
                h = len(part) // 2; queue[:0] = [part[:h], part[h:]]
                continue
            jobs[jid] = dict(keys=[r["key"] for r in part], shots=shots, submitted=time.strftime("%Y-%m-%d %H:%M:%S"),
                             submitted_unix=time.time())
            json.dump(jobs, open(jobs_f, "w"), indent=1)
            print(f"  submitted {jid}: {len(part)} circuits x {shots} shots", flush=True)
    bykey = {r["key"]: r for r in recs}
    failed_f = os.path.join(out_dir, "failed_jobs.json")
    failed = json.load(open(failed_f)) if os.path.exists(failed_f) else {}
    n_failed = 0
    for jid, j in list(jobs.items()):
        if all(k in raw for k in j["keys"]):
            continue
        if hasattr(runner, "n_expected"):
            runner.n_expected[jid], runner.shots_of[jid] = len(j["keys"]), j["shots"]
        try:
            counts, info = runner.fetch(jid)
        except JobFailed as e:
            print(f"  {e} -> its {len(j['keys'])} circuits will be re-submitted when you run this command again", flush=True)
            failed[jid] = dict(j, error=str(e))
            n_failed += 1
            continue
        for k, c in zip(j["keys"], counts):
            raw[k] = dict(counts=counts_to_logical(c, 2 * bykey[k]["n"]), job=jid)
        j.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), finished_unix=time.time(), info=info)
        json.dump(raw, open(raw_f, "w")); json.dump(jobs, open(jobs_f, "w"), indent=1, default=str)
        print(f"  collected {jid} ({len(j['keys'])} circuits)", flush=True)
    for jid in failed:
        jobs.pop(jid, None)
    json.dump(failed, open(failed_f, "w"), indent=1); json.dump(jobs, open(jobs_f, "w"), indent=1, default=str)
    if n_failed:
        print(f"{n_failed} job(s) FAILED on the cloud (not billed); first error: {next(iter(failed.values()))['error']}")
        print("run the same command again to re-submit their circuits")
    print(f"done: {len(raw)}/{len(recs)} circuits stored in {out_dir}")


if __name__ == "__main__":
    main()
