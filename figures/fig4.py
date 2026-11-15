import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_RUNS = _os.path.join(_ROOT, "hw", "runs")
_OUT = _os.path.join(_os.path.dirname(__file__), "out"); _os.makedirs(_OUT, exist_ok=True)
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from style import *
sys.path.insert(0, _os.path.join(_ROOT, "hw")); from fast_site import tables, bell_theta
from matplotlib.patches import Patch
R = _RUNS; A = _os.path.join(_RUNS, "per_pauli"); Hd = _RUNS
pr = json.load(open(os.path.join(R, "prune.json")))["agents"]
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(3, 4, hspace=0.7, wspace=0.62, left=0.07, right=0.985, top=0.955, bottom=0.07)
# a angles
ax = fig.add_subplot(gs[0, :2])
names = ["train_s1", "train_s11", "train_s12", "train_s20", "train_s22", "train_s30", "train_s31", "train_s32", "detect_s50", "detect_s51"]
groups = [("CRX φ", [6], BLUE), ("R_Y", [1, 4, 8, 11], ORA), ("R_Z", [0, 2, 3, 5, 7, 10], MUT)]
for j, (l, idx, col) in enumerate(groups):
    v = np.array([(np.array(pr[nm]["theta"])[i] + np.pi) % (2 * np.pi) - np.pi for nm in names for i in idx])
    y = j + np.random.default_rng(j).uniform(-0.2, 0.2, len(v)); ax.scatter(v, y, s=8, color=col, lw=0, alpha=0.85)
for k in range(-2, 3): ax.axvline(k * np.pi / 2, color=GRID, lw=0.6, zorder=0)
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]); ax.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
ax.set_yticks(range(3)); ax.set_yticklabels([g[0] for g in groups]); ax.set_ylim(-0.6, 2.6); ax.invert_yaxis()
ax.set_xlabel("learned angle (10 front-ends trained on the chip; final R_Z angles omitted)"); ax.set_title("learned angles cluster at Clifford values", loc="left"); lab(ax, "a", x=-0.16)
# b response tables
ths = [("Bell", bell_theta()), ("seed 20, as trained", np.array(pr["train_s20"]["theta"])), ("seed 20, pruned", np.array(pr["train_s20"]["theta_pruned"])),
       ("shortcut (no sign)", np.array((json.load(open(os.path.join(Hd, "train_nt_s20.json")))["best"])["theta"]))]
sub = gs[0, 2:].subgridspec(1, 4, wspace=0.15)
for i, (t, th) in enumerate(ths):
    ax = fig.add_subplot(sub[i]); g = tables(th)[2][1:]
    im = ax.imshow(g, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(4)); ax.set_xticklabels(["00", "01", "10", "11"], fontsize=5, rotation=90)
    ax.set_yticks(range(3)); ax.set_yticklabels(["XX", "YY", "ZZ"] if i == 0 else [], fontsize=5.5)
    ax.set_title(t, fontsize=5.6, pad=2)
    for (r, c), v in np.ndenumerate(g): ax.text(c, r, f"{v:+.1f}".replace("+0.0", "0").replace("-0.0", "0"), ha="center", va="center", fontsize=4.3, color="white" if abs(v) > 0.6 else INK)
    for s_ in ax.spines.values(): s_.set_visible(False)
    if i == 0: lab(ax, "b", x=-0.9, y=1.25)
fig.text(0.535, 0.735, "two-copy response g₃[σ, β] = ⟨β|V (σ⊗σ) V†|β⟩ (±1 everywhere = Bell)", fontsize=5.8, color=MUT)
# c visibility and J before/after per agent
ax = fig.add_subplot(gs[1, :2])
order = ["train_s1", "train_s11", "train_s12", "train_s20", "train_s22", "train_s30", "train_s31", "train_s32", "detect_s50", "detect_s51"]
labs = ["s1", "s11", "s12", "s20", "s22", "s30", "s31", "s32", "det50", "det51"]
x = np.arange(len(order))
for i, k in enumerate(order):
    v = pr[k]; ax.plot([i, i], [v["V"], v["V_pruned"]], color=GRID, lw=1.5, zorder=1)
    ax.plot([i + 0.25, i + 0.25], [v["J"][0], v["J"][1]], color=GRID, lw=1.5, zorder=1)
ax.scatter(x, [pr[k]["V"] for k in order], s=12, color=LBLUE, label="visibility V, as trained", zorder=2)
ax.scatter(x, [pr[k]["V_pruned"] for k in order], s=12, color=BLUE, label="V, pruned", zorder=2)
ax.scatter(x + 0.25, [pr[k]["J"][0] for k in order], s=12, color="#f2a27f", marker="D", label="hardware objective J, as trained", zorder=2)
ax.scatter(x + 0.25, [pr[k]["J"][1] for k in order], s=12, color=ORA, marker="D", label="J, pruned", zorder=2)
ax.set_xticks(x + 0.12); ax.set_xticklabels(labs); ax.set_ylim(0.84, 1.01); ax.set_ylabel("value"); ax.set_title("pruning raises visibility and hardware objective (no retraining)", loc="left")
ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=5.3); lab(ax, "c", x=-0.16)
# d gates per site
ax = fig.add_subplot(gs[1, 2])
cz0 = [pr[k]["cz"][0] for k in order]; cz1 = [pr[k]["cz"][1] for k in order]; p0 = [pr[k]["pulses"][0] for k in order]; p1 = [pr[k]["pulses"][1] for k in order]
ax.bar([0, 1], [np.mean(cz0), np.mean(cz1)], color=[LBLUE, BLUE], width=0.6)
ax.bar([2.5, 3.5], [np.mean(p0), np.mean(p1)], color=[LBLUE, BLUE], width=0.6)
ax.errorbar([3.5], [np.mean(p1)], yerr=[[np.mean(p1) - min(p1)], [max(p1) - np.mean(p1)]], color=INK, capsize=2, lw=0.8)
ax.set_xticks([0.5, 3]); ax.set_xticklabels(["CZ gates", "1-qubit pulses"]); ax.set_ylabel("per site (mean of 10)")
ax.legend([Patch(color=LBLUE), Patch(color=BLUE)], ["as trained", "pruned"], frameon=False, loc="upper left"); ax.set_title("native gate count", loc="left"); lab(ax, "d")
# e kappa by step
ax = fig.add_subplot(gs[1, 3])
S = lambda s: json.load(open(os.path.join(R, f"{s}_summary_origin.json")))
for i, st in enumerate(("PR5", "PR6", "PR7")):
    kq = S(st)["kappa_q"]
    for k, c, o in (("bell", LBLUE, -0.27), ("bell_native", BLUE, -0.09), ("onchip20", "#f2a27f", 0.09), ("onchip20_pruned", ORA, 0.27)):
        if k in kq: ax.bar(i + o, kq[k], 0.17, color=c)
ax.set_xticks(range(3)); ax.set_xticklabels(["n = 5", "n = 6", "n = 7"]); ax.set_ylim(0.8, 0.92); ax.set_ylabel("fitted per-site visibility κ")
ax.legend([Patch(color=c) for c in (LBLUE, BLUE, "#f2a27f", ORA)], ["Bell CRX", "Bell native", "s20 trained", "s20 pruned"], frameon=False, fontsize=5, loc="upper right", ncol=2)
ax.set_ylim(0.8, 0.95); ax.set_title("κ measured on the chip", loc="left"); lab(ax, "e")
# f paired bars
SUF = "_bell_bell_native_onchip20_onchip20_pruned_onchip22_onchip22_pruned.json"
def get(step, k):
    if step == "PR7":
        d = json.load(open(os.path.join(R, f"PR7_bootstrap_large_origin_{k}_bayes.json"))); a = np.array(d["results"][f"{k}_bayes"], float); b = a[1:]
        return a[0], np.percentile(b, 2.5), np.percentile(b, 97.5), d["bound"], a
    d = json.load(open(os.path.join(R, f"{step}_bootstrap_pauli_origin{SUF}"))); r = d["results"][f"{k}_bayes"]
    return r["point"], r["ci95"][0], r["ci95"][1], d["single_copy_lower_bound_ideal"], np.array(r["reps"], float)
sub = gs[2, :2].subgridspec(1, 3, wspace=0.5, width_ratios=[1, 1, 0.6])
for p, (step, n) in enumerate((("PR5", 5), ("PR6", 6), ("PR7", 7))):
    ax = fig.add_subplot(sub[p])
    G = [("Bell", "bell", "bell_native"), ("s20", "onchip20", "onchip20_pruned"), ("s22", "onchip22", "onchip22_pruned")] if n < 7 else [("Bell", None, "bell_native"), ("s20", None, "onchip20_pruned")]
    for j, (l, full, prn) in enumerate(G):
        for dx, k, col in ((-0.2, full, LBLUE), (0.2, prn, BLUE)):
            if k is None: continue
            dx = 0 if n == 7 else dx
            y, lo, hi, b, _ = get(step, k)
            ax.bar(j + dx, y, 0.38 if n < 7 else 0.55, color=col); ax.errorbar(j + dx, y, yerr=[[y - lo], [hi - y]], color=INK, lw=0.7, capsize=1.5)
    ax.axhline(b, color=INK, ls="--", lw=0.8); ax.set_xticks(range(len(G))); ax.set_xticklabels([g[0] for g in G]); ax.set_ylim(0, b * 1.12)
    ax.set_title(f"n = {n}", loc="left")
    if p == 0: ax.set_ylabel("copies for 90%"); lab(ax, "f", x=-0.55)
fig.legend([Patch(color=LBLUE), Patch(color=BLUE)], ["as trained / Bell via CRX", "pruned / Bell native"], loc="lower left", ncol=2, frameon=False, fontsize=5.5, bbox_to_anchor=(0.07, 0.0))
# g forest plot of paired ratios
ax = fig.add_subplot(gs[2, 2])
rows = []
for step, n in (("PR5", 5), ("PR6", 6)):
    for l, a_, b_ in (("Bell", "bell", "bell_native"), ("s20", "onchip20", "onchip20_pruned"), ("s22", "onchip22", "onchip22_pruned")):
        X = get(step, a_)[4]; Y = get(step, b_)[4]; r = (Y / X)[1:]; r = r[np.isfinite(r)]
        rows.append((f"{l}, n = {n}", Y[0] / X[0], np.percentile(r, 2.5), np.percentile(r, 97.5)))
for i, (l, m, lo, hi) in enumerate(rows):
    c = BLUE if hi < 1 else MUT
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=3, capsize=1.5, lw=0.9)
ax.axvline(1, color=INK, lw=0.7, ls="--"); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.2); ax.invert_yaxis()
ax.set_xlabel("copies, pruned ÷ as trained"); ax.set_title("paired effect (95% CI)", loc="left"); lab(ax, "g", x=-0.5)
# h per-Pauli paired scatter
ax = fig.add_subplot(gs[2, 3])
for st, mk in (("PR5", "o"), ("PR6", "s")):
    q = json.load(open(os.path.join(A, f"{st}_perpauli.json")))
    for a_, b_, c in (("onchip20", "onchip20_pruned", ORA), ("onchip22", "onchip22_pruned", "#f2a27f"), ("bell", "bell_native", BLUE)):
        ks = [k for k in q[a_]["paulis"] if k in q[b_]["paulis"]]
        x_ = np.array([q[a_]["paulis"][k]["copies90"] for k in ks], float); y_ = np.array([q[b_]["paulis"][k]["copies90"] for k in ks], float)
        ax.scatter(x_, y_, s=7, marker=mk, color=c, lw=0, alpha=0.85)
lim = [40, 700]; ax.plot(lim, lim, color=GRID, lw=0.8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
for A_ in (ax.xaxis, ax.yaxis):
    A_.set_major_locator(FixedLocator([50, 100, 200, 400])); A_.set_major_formatter(FixedFormatter(["50", "100", "200", "400"])); A_.set_minor_locator(NullLocator())
ax.set_xlabel("copies, as trained / CRX"); ax.set_ylabel("copies, pruned / native")
ax.scatter([], [], s=8, color=ORA, label="s20"); ax.scatter([], [], s=8, color="#f2a27f", label="s22"); ax.scatter([], [], s=8, color=BLUE, label="Bell"); ax.legend(frameon=False, loc="upper left", fontsize=5.2)
ax.set_title("per Pauli (n = 5 ●, 6 ■)", loc="left"); lab(ax, "h")
fig.savefig(_os.path.join(_OUT, "fig4_pruning.pdf")); fig.savefig(_os.path.join(_OUT, "fig4.png"), dpi=170)
