# -*- coding: utf-8 -*-
"""Probe whether WK_C180 is executing jobs right now.
Submits ONE 2-qubit circuit with 100 shots exactly as in the hardware run of October (cloud chooses the qubits),
then prints its status every 20 s for up to 20 minutes. Cost: about one second of QPU time.
Usage:  python probe.py            (ORIGIN_API_KEY must be set)"""
import os
import time
from pyqpanda3 import qcloud
from pyqpanda3.core import QProg, H, CNOT, measure

key = os.environ.get("ORIGIN_API_KEY")
if not key:
    raise SystemExit("set ORIGIN_API_KEY first")
be = qcloud.QCloudService(api_key=key).backend("WK_C180")
for name in ("status", "queue_size", "get_queue_size", "is_available"):        # print whatever the SDK offers
    try:
        v = getattr(be, name); print(f"backend.{name}:", v() if callable(v) else v)
    except Exception:
        pass
prog = QProg(); prog << H(0) << CNOT(0, 1) << measure([0, 1], [0, 1])
o = qcloud.QCloudOptions(); o.set_amend(True); o.set_mapping(True); o.set_optimization(True)
job = be.run([prog], 100, o)
jid = job.job_id(); t0 = time.time()
print("probe job", jid, "submitted", time.strftime("%H:%M:%S"))
while time.time() - t0 < 1200:
    st = qcloud.QCloudJob(jid).status(); name = getattr(st, "name", str(st))
    print(f"  {time.strftime('%H:%M:%S')}  {name}", flush=True)
    if name in ("FINISHED", "FAILED"):
        break
    time.sleep(20)
if name == "FINISHED":
    res = qcloud.QCloudJob(jid).result()
    for g in ("get_counts_list", "get_probs_list", "get_counts", "get_probs"):
        try:
            print(g, ":", getattr(res, g)())
        except Exception as e:
            print(g, ": not available", e)
elif name == "FAILED":
    print("error:", qcloud.QCloudJob(jid).query().error_message())
else:
    print("still not finished after 20 minutes")
