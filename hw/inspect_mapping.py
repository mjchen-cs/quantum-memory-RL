# -*- coding: utf-8 -*-
"""Read back how the cloud compiled finished jobs: which physical qubits each logical qubit went to,
how many two-qubit gates the compiled circuit has, and the platform's timing info.
Only queries finished jobs - it submits nothing and uses no QPU time.

Usage:  python inspect_mapping.py D1_origin_sm            (ORIGIN_API_KEY must be set)
        python inspect_mapping.py S3_origin_sm --max 3
Writes runs/<dir>/mapping_info.json"""
import os
import re
import sys
import json
import argparse
from pyqpanda3 import qcloud

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--max", type=int, default=0, help="inspect at most this many jobs (0 = all)")
a = ap.parse_args()

key = os.environ.get("ORIGIN_API_KEY")
if not key:
    raise SystemExit("set ORIGIN_API_KEY first")
qcloud.QCloudService(api_key=key)                     # authenticates the session for QCloudJob

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", a.run_dir)
jobs = json.load(open(os.path.join(d, "jobs.json")))
ids = list(jobs)[: a.max] if a.max else list(jobs)

TWOQ = re.compile(r"\b(CZ|CNOT|CX|ISWAP|SWAP|CRX|CR|ECR|CP)\b", re.I)


def safe(obj, name):
    try:
        v = getattr(obj, name)
        return v() if callable(v) else v
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def plain(v, limit=4000):
    try:
        json.dumps(v)
        return v
    except Exception:
        s = str(v)
        return s if len(s) <= limit else s[:limit] + " ...<truncated>"


out = {}
for jid in ids:
    res = qcloud.QCloudJob(jid).result()
    rec = {"first_key": jobs[jid]["keys"][0], "n_circuits": len(jobs[jid]["keys"])}
    for name in ("mapping_qubit", "src_qubits", "measure_qubits", "target_cbits", "timing_info"):
        rec[name] = plain(safe(res, name))
    mq = safe(res, "mapping_qprog")
    progs = mq if isinstance(mq, (list, tuple)) else [mq]
    rec["n_mapping_qprog"] = len(progs)
    texts = [str(p) for p in progs[:2]]
    rec["twoq_gates_first_circuit"] = {}
    if texts:
        for g in TWOQ.findall(texts[0]):
            g = g.upper(); rec["twoq_gates_first_circuit"][g] = rec["twoq_gates_first_circuit"].get(g, 0) + 1
        rec["mapping_qprog_first"] = texts[0][:6000]
    out[jid] = rec
    print(f"\n=== {jid}  ({rec['first_key']}, {rec['n_circuits']} circuits)")
    for name in ("mapping_qubit", "src_qubits", "measure_qubits", "timing_info", "twoq_gates_first_circuit"):
        s = str(rec[name])
        print(f"  {name}: {s[:600]}")

path = os.path.join(d, "mapping_info.json")
json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
print("\nsaved", path)
