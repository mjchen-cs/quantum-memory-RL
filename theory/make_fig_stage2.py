import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qdrift_sim import lower_bound_copies

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
C, Q, LQ, G = "#d95a24", "#2a78d6", "#16956a", "#5f5f59"
E1 = json.load(open("E1_all.json")); E3 = json.load(open("E3.json"))
S = json.load(open("static_eps0.9.json")); D = json.load(open("drift_eps0.9.json"))

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
plt.subplots_adjust(wspace=0.42)

# a: trainability
a = ax[0]
ns = [2, 3, 4]; w = 0.36
for j, (var, col, lab) in enumerate((("full", G, "Unconstrained ansatz"), ("equiv", LQ, "Equivariant ansatz"))):
    rate, cnt = [], []
    for n in ns:
        r = [x for x in E1 if x["n"] == n and x["variant"] == var and x["objective"] == "vis"]
        rate.append(np.mean([x["vis"] > 0.99 for x in r]) if r else np.nan); cnt.append(len(r))
    xs = np.arange(len(ns)) + (j - 0.5) * w
    a.bar(xs, rate, width=w * 0.9, color=col, label=lab)
    for x_, r_, c_ in zip(xs, rate, cnt):
        if c_:
            a.text(x_, (0 if np.isnan(r_) else r_) + 0.02, f"{int(round(r_ * c_))}/{c_}", ha="center", fontsize=6.3)
a.set_xticks(range(len(ns))); a.set_xticklabels([f"n = {n}\n({2*n} qubits)" for n in ns])
a.set_ylim(0, 1.1); a.set_ylabel("Training runs reaching the optimum")
a.legend(fontsize=6.3, frameon=False, loc="upper right")
a.set_title("Trainability", fontsize=8.5)
a.text(-0.3, 1.04, "a", transform=a.transAxes, fontsize=11, fontweight="bold")

# b: identification copies
b = ax[1]
r = [x for x in S if x["kind"] == "classical"]; b.plot([x["n"] for x in r], [x["m90"] for x in r], "o-", color=C, ms=3.5, label="Classical shadows")
r = [x for x in S if x["kind"] == "quantum" and x["n"] <= 5]; b.plot([x["n"] for x in r], [x["m90"] for x in r], "s--", color=Q, ms=3.5, label="Bell agent (analytic)")
b.plot([x["n"] for x in E3], [x["copies90"] for x in E3], "D-", color=LQ, ms=4, label="Learned (trained at n = 2)")
nn = np.arange(2, 6); b.plot(nn, [lower_bound_copies(n, 0.9) for n in nn], ":", color=C, lw=1, label="Single-copy lower bound")
b.set_yscale("log"); b.set_xticks([1, 2, 3, 4, 5]); b.set_xlabel("Qubits per copy n"); b.set_ylabel("Copies for 90% identification")
b.legend(fontsize=6, frameon=False); b.set_title("Zero-shot size transfer", fontsize=8.5)
b.text(-0.3, 1.04, "b", transform=b.transAxes, fontsize=11, fontweight="bold")

# c: regret
c = ax[2]
for kind, col, lab, mk in (("classical", C, "Classical shadows", "o-"), ("quantum", Q, "Bell agent (analytic)", "s--")):
    r = [x for x in D if x["kind"] == kind]
    c.errorbar([x["n"] for x in r], [x["regret"] for x in r], yerr=[x["se"] for x in r], fmt=mk, color=col, ms=3.5, capsize=2, label=lab)
c.errorbar([x["n"] for x in E3], [x["regret"] for x in E3], yerr=[x["se"] for x in E3], fmt="D-", color=LQ, ms=4, capsize=2, label="Learned (trained at n = 2)")
r = [x for x in D if x["kind"] == "classical"]; c.plot([x["n"] for x in r], [x["lb"] for x in r], ":", color=C, lw=1, label="Single-copy lower bound")
c.set_yscale("log"); c.set_xticks([2, 3, 4, 5]); c.set_xlabel("Qubits per copy n"); c.set_ylabel("Regret over 2,000 rounds")
c.legend(fontsize=6, frameon=False); c.set_title("Drifting task (k = 4, q = 0.01)", fontsize=8.5)
c.text(-0.3, 1.04, "c", transform=c.transAxes, fontsize=11, fontweight="bold")
fig.savefig("fig_stage2.pdf", bbox_inches="tight"); fig.savefig("fig_stage2.png", dpi=200, bbox_inches="tight")
print("ok")
