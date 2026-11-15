# Reward-only curriculum: agent trained at n=2 (reward only, with twirl) is deployed at larger n with the same
# n-independent parameters (zero-shot), then fine-tuned with rewards only.
import json, sys, time, numpy as np, torch
from e2e_site import run, Decoder, sample, act, cdf_of, bayes_accuracy
from fast_site import SiteModel, bell_theta
src = json.load(open("es_n2_tw1_saved.json"))
best = max(src["runs"], key=lambda r: r["final"]["acc"])
print("source seed", best["seed"], "acc", best["final"]["acc"], "vis", best["final"]["vis"], flush=True)
M = {2: 8, 3: 12, 4: 14, 5: 18}
FT = {3: 150, 4: 150}
out = []
for n in (3, 4, 5):
    S = SiteModel(n, twirl=True)
    dec = Decoder(n)
    with torch.no_grad():
        dec.h.copy_(torch.tensor(best["h"])); dec.c.copy_(torch.tensor(best["c"]))
    rng = np.random.default_rng(11)
    th = np.array(best["theta"])
    Pe = rng.integers(S.N, size=1000)
    with torch.no_grad():
        acc0 = float((act(dec.scores(torch.tensor(sample(cdf_of(S, th), S.B, Pe, rng.random((1000, M[n]))))), greedy=True) == Pe).mean())
    ref = bayes_accuracy(SiteModel(n), bell_theta(), M[n], 1000, np.random.default_rng(1))
    rec = dict(n=n, m=M[n], zero_shot_acc=acc0, vis=S.visibility(th), bell_ref=ref, source_seed=best["seed"])
    print("n", n, "zero-shot acc", round(acc0, 3), "vis", round(rec["vis"], 4), "Bell ref", round(ref, 3), flush=True)
    if n in FT:
        th2, dec2, hist = run(n, M[n], FT[n], 0, twirl=True, init=best, log=25, tag=f"[ft n={n}]")
        rec.update(finetune_hist=hist, finetune_final=hist[-1])
    out.append(rec)
    json.dump(dict(source=dict(seed=best["seed"], acc=best["final"]["acc"], vis=best["final"]["vis"]), runs=out), open("es_transfer.json", "w"), indent=1)
