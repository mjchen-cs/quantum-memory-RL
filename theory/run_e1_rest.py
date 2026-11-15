import json, sys, time
from run_stage2 import fit
part=sys.argv[1]; out=sys.argv[2]; res=[]; t0=time.time()
if part=="A":
    jobs=[(4,"equiv",sd,150,"vis") for sd in range(1,6)]
else:
    jobs=[(n,"equiv",sd,300,"info") for n in (2,3) for sd in range(10)]+[(4,"full",sd,150,"vis") for sd in range(3)]
for n,var,sd,steps,obj in jobs:
    m,vis=fit(n,var,1,sd,steps,objective=obj)
    res.append(dict(n=n,variant=var,objective=obj,seed=sd,steps=steps,vis=vis))
    print(n,var,obj,sd,round(vis,4),f"{time.time()-t0:.0f}s",flush=True)
    json.dump(res,open(out,"w"),indent=1)
