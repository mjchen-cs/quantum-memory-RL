import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from qdrift_sim import lower_bound_copies, upper_bound_copies
plt.rcParams.update({"font.family":"Liberation Sans","font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
S=json.load(open("static_eps0.9.json")); D=json.load(open("drift_eps0.9.json"))
C,Q,G="#d95a24","#2a78d6","#5f5f59"
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); plt.subplots_adjust(wspace=0.32)
a=ax[0]
for kind,col,lab in (("classical",C,"Random-Pauli shadows, Bayes-optimal"),("quantum",Q,"Bell pairs, Bayes-optimal")):
    r=[x for x in S if x["kind"]==kind]; a.plot([x["n"] for x in r],[x["m90"] for x in r],"o-",color=col,label=lab,ms=4)
nn=np.arange(2,8)
a.plot(nn,[lower_bound_copies(n,0.9) for n in nn],"--",color=C,lw=1,label="Lower bound, any non-adaptive single-copy")
a.plot(nn,[upper_bound_copies(n,0.9) for n in nn],":",color=Q,lw=1.2,label="Upper bound, Bell (Thm 2)")
a.set_yscale("log"); a.set_ylim(7,2.5e4); a.set_xlabel("Qubits n"); a.set_ylabel("Copies for 90% identification"); a.legend(fontsize=6.5,frameon=False)
a.set_title("Static task (q = 0)",fontsize=8.5)
a.text(-0.2,1.03,"a",transform=a.transAxes,fontsize=11,fontweight="bold")
b=ax[1]; ns=sorted({x["n"] for x in D})
for kind,col,lab in (("classical",C,"Random-Pauli shadows, Bayes filter"),("quantum",Q,"Bell pairs, Bayes filter")):
    r=[x for x in D if x["kind"]==kind]; b.errorbar([x["n"] for x in r],[x["regret"] for x in r],yerr=[x["se"] for x in r],fmt="o-",color=col,label=lab,ms=4,capsize=2)
r=[x for x in D if x["kind"]=="classical"]; b.plot([x["n"] for x in r],[x["lb"] for x in r],"--",color=C,lw=1,label="Lower bound, non-adaptive single-copy (Thm 1)")
b.set_yscale("log"); b.set_xlabel("Qubits n"); b.set_ylabel("Regret over T = 2,000 rounds"); b.set_xticks(ns); b.legend(fontsize=6.5,frameon=False)
b.set_title("Drifting task (k = 4, q = 0.01)",fontsize=8.5)
b.text(-0.2,1.03,"b",transform=b.transAxes,fontsize=11,fontweight="bold")
fig.savefig("fig_theory_checks.pdf",bbox_inches="tight"); fig.savefig("fig_theory_checks.png",dpi=200,bbox_inches="tight")
# table
rows=[]
for n in range(1,8):
    c=[x for x in S if x["kind"]=="classical" and x["n"]==n]; q=[x for x in S if x["kind"]=="quantum" and x["n"]==n]
    rows.append(f"{n} & {c[0]['m90']:.0f} & {q[0]['m90']:.0f} & {lower_bound_copies(n,0.9):.0f} & {upper_bound_copies(n,0.9):.0f} \\\\" if c else
                f"{n} & -- & {q[0]['m90']:.0f} & {lower_bound_copies(n,0.9):.0f} & {upper_bound_copies(n,0.9):.0f} \\\\")
drows=[]
for n in ns:
    c=[x for x in D if x["kind"]=="classical" and x["n"]==n][0]; q=[x for x in D if x["kind"]=="quantum" and x["n"]==n][0]
    drows.append(f"{n} & {c['regret']:.1f} $\\pm$ {c['se']:.1f} & {q['regret']:.1f} $\\pm$ {q['se']:.1f} & {c['lb']:.1f} & {q['ub']:.0f} \\\\")
open("sim_table.tex","w").write(r"""\begin{table}[ht]\centering\small
\caption{\textbf{Simulation results ($\varepsilon=0.9$).} Left: copies for 90\% identification (static task). Right: regret over $T=2{,}000$ rounds ($k=4$, $q=0.01$). LB: lower bound for every non-adaptive single-copy agent; UB: upper bound for the Bell agent. For $n=1$ the static lower bound is nearly vacuous, and Theorem~1 assumes $n\ge2$.}
\begin{tabular}{rrrrr}\toprule
$n$ & Classical & Bell & LB & UB\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\hspace{1.5em}
\begin{tabular}{rrrrr}\toprule
$n$ & Classical & Bell & LB & UB\\\midrule
"""+"\n".join(drows)+r"""
\bottomrule\end{tabular}
\end{table}
""")
