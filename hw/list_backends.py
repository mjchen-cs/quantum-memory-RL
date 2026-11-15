# -*- coding: utf-8 -*-
"""Print the backend names available to this Origin Quantum account (no QPU time).
Usage:  python list_backends.py        (ORIGIN_API_KEY must be set)"""
import os
from pyqpanda3 import qcloud

key = os.environ.get("ORIGIN_API_KEY")
if not key:
    raise SystemExit("set ORIGIN_API_KEY first")
svc = qcloud.QCloudService(api_key=key)
for name in ("backends", "get_backends", "list_backends"):
    f = getattr(svc, name, None)
    if f is None:
        continue
    try:
        res = f() if callable(f) else f
        print(f"{name}():")
        items = res.items() if isinstance(res, dict) else enumerate(res)
        for k, v in items:
            print("  ", k, v)
        break
    except Exception as e:
        print(name, "failed:", e)
else:
    print("no backend listing method found; attributes:", [x for x in dir(svc) if not x.startswith("_")])
