# -*- coding: utf-8 -*-
"""Test whether jobs with PINNED physical qubits (cloud mapping off) run on WK_C180, and whether the gate set matters.
Takes the best coupler from today's calibration (runs/chip.json) and submits three 1-circuit jobs, 100 shots each:
  A  H + CNOT, mapping off           (what hung before, but on one coupler)
  B  H + CZ + H, mapping off          (native two-qubit gate only)
  C  same as B, mapping on            (control, should finish like probe.py)
Then polls every 20 s for up to 15 minutes. Cost: about 0.1 s of QPU time in total.
Usage:  python hw_run.py --fetch_chip      (refresh calibration first)
        python probe_pinned.py            (ORIGIN_API_KEY must be set)"""
import os
import time
import json
from pyqpanda3 import qcloud
from pyqpanda3.core import QProg, H, CNOT, CZ, measure
from hw_noise import load_chip, select_sites

HERE = os.path.dirname(os.path.abspath(__file__))
key = os.environ.get("ORIGIN_API_KEY")
if not key:
    raise SystemExit("set ORIGIN_API_KEY first")
chip = os.path.join(HERE, "runs", "chip.json")
d, s1, f2 = load_chip(chip)
site = select_sites(1, s1, f2)[0]
a, b = site["a"], site["b"]
print(f"calibration {d.get('fetched')}: best coupler ({a},{b})  CZ fidelity {site['f_cz']}")

be = qcloud.QCloudService(api_key=key).backend("WK_C180")


def opts(mapping):
    o = qcloud.QCloudOptions(); o.set_amend(True); o.set_mapping(mapping); o.set_optimization(True)
    return o


pA = QProg(); pA << H(a) << CNOT(a, b) << measure([a, b], [0, 1])
pB = QProg(); pB << H(a) << H(b) << CZ(a, b) << H(b) << measure([a, b], [0, 1])
pC = QProg(); pC << H(0) << H(1) << CZ(0, 1) << H(1) << measure([0, 1], [0, 1])
jobs = {}
for tag, prog, mapping in (("A_pinned_CNOT", pA, False), ("B_pinned_CZ", pB, False), ("C_mapped_CZ", pC, True)):
    try:
        jobs[tag] = be.run([prog], 100, opts(mapping)).job_id()
        print(tag, "submitted", jobs[tag])
    except Exception as e:
        print(tag, "REJECTED at submission:", e)

t0, state = time.time(), {}
while jobs and time.time() - t0 < 900:
    for tag, jid in jobs.items():
        if state.get(tag) in ("FINISHED", "FAILED"):
            continue
        st = qcloud.QCloudJob(jid).status(); state[tag] = getattr(st, "name", str(st))
    print(time.strftime("%H:%M:%S"), "  ".join(f"{t}:{s}" for t, s in state.items()), flush=True)
    if all(s in ("FINISHED", "FAILED") for s in state.values()):
        break
    time.sleep(20)

out = dict(coupler=[a, b], jobs=jobs, final=state)
for tag, jid in jobs.items():
    if state.get(tag) == "FINISHED":
        r = qcloud.QCloudJob(jid).result()
        try:
            out[tag] = dict(probs=r.get_probs_list(), timing=r.timing_info())
        except Exception as e:
            out[tag] = str(e)
    elif state.get(tag) == "FAILED":
        out[tag] = qcloud.QCloudJob(jid).query().error_message()
    print(tag, state.get(tag), out.get(tag, "still running"))
json.dump(out, open(os.path.join(HERE, "runs", "probe_pinned.json"), "w"), indent=1, default=str)
print("saved runs/probe_pinned.json   (ideal: A gives 00/11 at 50/50; B and C give 00/11 at 50/50)")
