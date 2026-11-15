import json, numpy as np, os
E1=json.load(open("E1.json"))
for f in ("E1a.json","E1b.json"):
    if os.path.exists(f): E1+=json.load(open(f))
json.dump(E1,open("E1_all.json","w"),indent=1)
E2=json.load(open("E2.json")); E3=json.load(open("E3.json"))
rows=[]
for n in (2,3,4):
    for var,lab in (("full","Unconstrained"),("equiv","Equivariant")):
        for obj,ol in (("vis","visibility"),("info","information")):
            r=[x for x in E1 if x["n"]==n and x["variant"]==var and x["objective"]==obj]
            if not r: continue
            s=sum(x["vis"]>0.99 for x in r)
            rows.append(f"{n} & {2*n} & {lab} & {ol} & {s}/{len(r)} & {max(x['vis'] for x in r):.3f} \\\\")
t1=r"""\begin{table}[ht]\centering\small
\caption{\textbf{E1, trainability.} Runs reaching the optimum ($V>0.99$) out of independent random initializations (one entangling layer; 300 steps for $n\le3$, 150 for $n=4$). $V$, two-copy visibility (1 for a Bell-type measurement).}
\begin{tabular}{rrllrr}\toprule $n$ & Qubits & Ansatz & Objective & Success & Best $V$\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}"""
rows=[]
for x in E2:
    names={0:"cross, same position",1:"cross, other positions",2:"within A",3:"within B"}
    orb="; ".join(f"L{l+1}: {names[k]}" for l,k in x["kept_orbits"]) or "none"
    rows.append(f"{x['seed']} & {x['vis_final']:.3f} & {x['n_two_qubit']} & {orb} \\\\")
t2=r"""\begin{table}[ht]\centering\small
\footnotesize\caption{\textbf{E2, structured pruning} ($n=3$, two layers, 30 candidate entanglers, $L_1$ weight 0.01, threshold 0.1, then fine-tuning).}
\begin{tabular}{rrrl}\toprule Seed & Final $V$ & Two-qubit gates & Kept entangler orbits\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}"""
rows=[f"{x['n']} & {2*x['n']} & {x['vis']:.4f} & {x['info']:.4f} & {x['info_bell']:.4f} & {x['copies90']:.0f} & {x['regret']:.1f} $\pm$ {x['se']:.1f} \\\\" for x in E3]
t3=r"""\begin{table}[ht]\centering\small
\caption{\textbf{E3, zero-shot size transfer.} Equivariant circuit trained once at $n=2$ (best of 10 runs), evaluated without retraining. Information per pair in nats; copies for 90\% identification (static task, 300 trials); regret over 2,000 rounds of the drifting task ($k=4$, $q=0.01$, 40 trials).}
\begin{tabular}{rrrrrrr}\toprule $n$ & Qubits & $V$ & Info & Info (Bell) & Copies (90\%) & Regret\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}"""
open("stage2_tables.tex","w").write(t1+"\n"+t2+"\n"+t3+"\n")
print(open("stage2_tables.tex").read())
