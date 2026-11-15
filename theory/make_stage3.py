import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Liberation Sans","font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
TW,NT,REF,ZS="#16956a","#d95a24","#2a78d6","#8fd3b6"
runs={}; ref={}
for n in (2,3):
    for tw in (0,1):
        d=json.load(open(f"es_n{n}_tw{tw}.json")); runs[(n,tw)]=d["runs"]; ref[n]=d["ref"]["bell"]
T=json.load(open("es_transfer.json"))
fig,ax=plt.subplots(1,4,figsize=(7.2,2.45),gridspec_kw=dict(width_ratios=[1,1,1.05,0.8]))
plt.subplots_adjust(wspace=0.45)
for j,n in enumerate((2,3)):
    a=ax[j]
    for tw,col,lab in ((0,NT,"no twirl"),(1,TW,"shared Pauli twirl")):
        for r in runs[(n,tw)]:
            h=r["hist"]; a.plot([x["it"] for x in h],[x["acc"] for x in h],color=col,lw=0.8,alpha=0.8,label=lab); lab=None
    a.axhline(ref[n],color=REF,ls="--",lw=1,label="Bell agent (Bayes)")
    a.set_ylim(0,1); a.set_xlabel("Iteration"); a.set_title(f"n = {n} ({2*n} qubits)",fontsize=8.5)
    if j==0: a.set_ylabel("Accuracy = expected reward"); a.legend(fontsize=6,frameon=False,loc="upper left")
    a.text(-0.3,1.05,"ab"[j],transform=a.transAxes,fontsize=11,fontweight="bold")
a=ax[2]; ns=[r["n"] for r in T["runs"]]; x=np.arange(len(ns)); w=0.27
z=[r["zero_shot_acc"] for r in T["runs"]]
f=[np.mean([h["acc"] for h in r["finetune_hist"][-3:]]) if "finetune_hist" in r else np.nan for r in T["runs"]]
b=[r["bell_ref"] for r in T["runs"]]
a.bar(x-w,z,w*0.9,color=ZS,label="zero-shot"); a.bar(x,f,w*0.9,color=TW,label="+150 reward-only iterations"); a.bar(x+w,b,w*0.9,color=REF,label="Bell agent (Bayes)")
a.set_xticks(x); a.set_xticklabels([f"n = {n}" for n in ns]); a.set_ylim(0,1.3); a.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); a.set_ylabel("Accuracy")
a.set_title("Trained at n = 2",fontsize=8.5); a.legend(fontsize=5.8,frameon=False,loc="upper left")
a.text(-0.3,1.05,"c",transform=a.transAxes,fontsize=11,fontweight="bold")
a=ax[3]
for k,n in enumerate((2,3)):
    for tw,col,dx in ((0,NT,-0.15),(1,TW,0.15)):
        v=[r["final"]["vis"] for r in runs[(n,tw)]]
        a.scatter(np.full(len(v),k+dx)+np.random.default_rng(k+5*tw).uniform(-0.06,0.06,len(v)),v,color=col,s=12)
a.set_xticks([0,1]); a.set_xticklabels(["n = 2","n = 3"]); a.set_ylim(-0.03,1.05); a.set_xlim(-0.6,1.6)
a.set_ylabel("Two-copy visibility V"); a.set_title("Learned circuit",fontsize=8.5)
a.text(-0.45,1.05,"d",transform=a.transAxes,fontsize=11,fontweight="bold")
fig.savefig("fig_stage3.pdf",bbox_inches="tight"); fig.savefig("fig_stage3.png",dpi=200,bbox_inches="tight")
rows=[]
for n in (2,3):
    for tw,lab in ((0,"no"),(1,"yes")):
        R=runs[(n,tw)]; succ=sum(r["final"]["vis"]>0.95 for r in R); acc=[r["final"]["acc"] for r in R]
        good=[r["final"]["acc"] for r in R if r["final"]["vis"]>0.95]
        rows.append(f"{n} & {2*n} & {R[0]['m']} & {lab} & {succ}/{len(R)} & {max(acc):.3f} & {np.median(acc):.3f} & {ref[n]:.3f} \\\\")
t1=r"""\begin{table}[ht]\centering\small
\caption{\textbf{Reward-only training from random initialization} (600 iterations). Success: learned circuit with two-copy visibility $V>0.95$ (Bell type). Accuracy: greedy accuracy of the learned agent (circuit and decoder) at the end of training, 1,000 test episodes. Reference: Bayes-optimal Bell agent with the same number of pairs.}
\begin{tabular}{rrrlrrrr}\toprule $n$ & Qubits & Pairs & Twirl & Success & Best acc. & Median acc. & Reference\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}"""
rows=[]
for r,fv in zip(T["runs"],f):
    rows.append(f"{r['n']} & {2*r['n']} & {r['m']} & {r['vis']:.3f} & {r['zero_shot_acc']:.3f} & {'--' if np.isnan(fv) else f'{fv:.3f}'} & {r['bell_ref']:.3f} \\\\")
t2=r"""\begin{table}[ht]\centering\small
\caption{\textbf{Reward-only curriculum.} The agent trained with rewards at $n=2$ (best of four runs; all its parameters are independent of $n$) is deployed without retraining at larger $n$ (zero-shot) and then fine-tuned for 150 iterations with rewards only (mean of the last three evaluations, 1,000 episodes each).}
\begin{tabular}{rrrrrrr}\toprule $n$ & Qubits & Pairs & $V$ & Zero-shot & Fine-tuned & Reference\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}"""
open("stage3_table.tex","w").write(t1+"\n"+t2+"\n")
print(open("stage3_table.tex").read())
