import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_RUNS = _os.path.join(_ROOT, "hw", "runs")
_OUT = _os.path.join(_os.path.dirname(__file__), "out"); _os.makedirs(_OUT, exist_ok=True)
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from style import *
from matplotlib.patches import Rectangle, FancyArrowPatch
R = _RUNS; A = _os.path.join(_RUNS, "per_pauli")
S = lambda s: json.load(open(os.path.join(R, f"{s}_summary_origin.json")))
bound = {n: (2 ** n + 1) * (0.9 * np.log(4 ** n - 1) - np.log(2)) for n in range(2, 8)}
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(3, 4, hspace=0.68, wspace=0.45, left=0.07, right=0.985, top=0.955, bottom=0.065)
# a: schematic train small, deploy large
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); lab(ax, "a", x=-0.05)
ax.text(0, 9.6, "train at n = 2 (4 qubits)", fontsize=6.3, fontweight="bold")
for i in range(2):
    ax.add_patch(Rectangle((0.3 + i * 1.6, 7.0), 1.3, 1.6, fc="white", ec=BLUE, lw=0.8)); ax.text(0.95 + i * 1.6, 7.8, "V", ha="center", va="center", fontsize=6.5)
ax.text(3.7, 7.8, "θ (13 angles)", fontsize=6, va="center", color=MUT)
ax.add_patch(FancyArrowPatch((2.0, 6.8), (2.0, 5.6), arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.8))
ax.text(2.5, 6.0, "same θ, no retraining", fontsize=5.8, color=MUT)
ax.text(0, 5.35, "deploy at n = 3–7\n(up to 14 qubits)", fontsize=6.3, fontweight="bold", va="top")
for i in range(7):
    ax.add_patch(Rectangle((0.15 + i * 1.38, 1.75), 1.15, 1.4, fc="white", ec=BLUE, lw=0.8)); ax.text(0.72 + i * 1.38, 2.45, "V", ha="center", va="center", fontsize=6)
ax.text(0, 0.55, "decoder built from V's response\ntables, 1 calibrated visibility κ", fontsize=5.6, color=MUT, va="center")
# b-f: accuracy curves
COL = {"bell": LBLUE, "bell_native": BLUE, "learned": AQUA, "onchip20": "#f2a27f", "onchip20_pruned": ORA, "onchip22": "#f2a27f", "classical": RED}
LAB = {"bell": "Bell (CRX)", "bell_native": "Bell (native)", "learned": "sim.-trained", "onchip20": "chip-trained", "onchip20_pruned": "chip-trained, pruned", "classical": "random bases"}
panels = [("S4", ["bell", "learned", "classical"], 3), ("S3P", ["bell", "learned", "classical"], 4),
          ("PR5", ["bell_native", "onchip20_pruned", "onchip20"], 5), ("PR6", ["bell_native", "onchip20_pruned", "onchip20"], 6), ("PR7", ["bell_native", "onchip20_pruned"], 7)]
pos = [gs[0, 1], gs[0, 2], gs[0, 3], gs[1, 0], gs[1, 1]]
for (st, keys, n), p, L in zip(panels, pos, "bcdef"):
    ax = fig.add_subplot(p); d = S(st)
    for k in keys:
        r = d["results"].get(f"{k}_bayes")
        if r is None: continue
        g = np.array(d["grid_c"] if k == "classical" else d["grid_q"], float)
        acc = np.array(r["acc"], float); ok = ~np.isnan(acc)
        ax.plot(g[ok], acc[ok], color=COL[k], label=LAB[k], lw=1.3 if "native" in k or "pruned" in k else 1.0)
    if st == "PR5":
        c = S("S5C")["results"]["classical_bayes"]; g = np.array(S("S5C")["grid_c"], float); acc = np.array(c["acc"], float); ok = ~np.isnan(acc)
        ax.plot(g[ok], acc[ok], color=RED, label="random bases")
    ax.axvspan(1, bound[n], color=SURF, lw=0, zorder=0); ax.axvline(bound[n], color=INK, ls="--", lw=0.8)
    ax.axhline(0.9, color=MUT, ls=":", lw=0.6)
    ax.set_xscale("log"); ax.set_xlim(2, 1e4); ax.set_ylim(0, 1.02); ax.set_xlabel("copies")
    ax.set_title(f"n = {n} ({2 * n} qubits)", loc="left")
    if L in "be": ax.set_ylabel("identification accuracy")
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, fontsize=5.3)
    lab(ax, L)
# g: copies vs n
ax = fig.add_subplot(gs[1, 2:])
def bs(f, k):
    d = json.load(open(os.path.join(R, f)))["results"][k]
    if isinstance(d, dict): return d["point"], d["ci95"][0], d["ci95"][1]
    a = np.array(d, float); b = a[1:][np.isfinite(a[1:])]; return a[0], np.percentile(b, 2.5), np.percentile(b, 97.5)
PR = "_bell_bell_native_onchip20_onchip20_pruned_onchip22_onchip22_pruned.json"
series = [("Bell", BLUE, "o", {3: bs("S4_bootstrap_pauli_origin.json", "bell_bayes"), 4: bs("S3P_bootstrap_pauli_origin.json", "bell_bayes"), 5: bs("PR5_bootstrap_pauli_origin" + PR, "bell_native_bayes"), 6: bs("PR6_bootstrap_pauli_origin" + PR, "bell_native_bayes"), 7: bs("PR7_bootstrap_large_origin_bell_native_bayes.json", "bell_native_bayes")}),
          ("chip-trained (n = 2)", ORA, "D", {4: bs("T4_bootstrap_pauli_origin.json", "onchip20_bayes"), 5: bs("PR5_bootstrap_pauli_origin" + PR, "onchip20_pruned_bayes"), 6: bs("PR6_bootstrap_pauli_origin" + PR, "onchip20_pruned_bayes"), 7: bs("PR7_bootstrap_large_origin_onchip20_pruned_bayes.json", "onchip20_pruned_bayes")}),
          ("simulation-trained (n = 2)", AQUA, "^", {3: bs("S4_bootstrap_pauli_origin.json", "learned_bayes"), 4: bs("S3P_bootstrap_pauli_origin.json", "learned_bayes"), 5: bs("S5_bootstrap_pauli_origin.json", "learned_bayes"), 6: bs("S6X_bootstrap_pauli_origin_learned.json", "learned_bayes")}),
          ("random bases", RED, "s", {3: bs("S4_bootstrap_pauli_origin.json", "classical_bayes"), 4: bs("S3P_bootstrap_pauli_origin.json", "classical_bayes"), 5: bs("S5C_bootstrap_pauli_origin_classical.json", "classical_bayes")})]
ns = np.arange(3, 8)
ax.fill_between(ns, [bound[n] for n in ns], 1e5, color=SURF, lw=0, zorder=0)
ax.plot(ns, [bound[n] for n in ns], color=INK, ls="--", lw=1, label="single-copy limit (non-adaptive)")
ax.plot([3, 4, 5, 6], [50, 108, 235, 517], color=MUT, ls=":", lw=1, label="adaptive elimination (ideal)")
off = [-0.08, 0.08, 0, 0]
for (l, c, m, pts), o in zip(series, off):
    x = np.array(sorted(pts)); y = np.array([pts[n][0] for n in x]); lo = np.array([pts[n][1] for n in x]); hi = np.array([pts[n][2] for n in x])
    ax.errorbar(x + o, y, yerr=[y - lo, hi - y], color=c, marker=m, ms=3.5, lw=1.1, capsize=1.5, label=l)
    for xi, yi in zip(x, y):
        if l in ("Bell", "chip-trained (n = 2)") and xi >= 5: ax.text(xi + o + (0.12 if o > 0 else -0.12), yi, f"{yi / bound[xi]:.2f}×", fontsize=5.3, color=c, ha="left" if o > 0 else "right", va="center")
ax.set_yscale("log"); ax.set_ylim(20, 2e4); ax.set_xlim(2.7, 7.4); ax.set_xticks(ns); ax.set_xlabel("qubits per copy, n"); ax.set_ylabel("copies for 90%")
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.3); ax.set_title("copies vs size (×: fraction of the limit)", loc="left"); lab(ax, "g", x=-0.12)
# h: per-Pauli copies at n=5
pp = json.load(open(os.path.join(A, "PR5_perpauli.json")))
ax = fig.add_subplot(gs[2, :2])
keys = sorted(pp["bell_native"]["paulis"], key=lambda k: (pp["bell_native"]["paulis"][k]["w"], k))
x = np.arange(len(keys))
for k_, c, o, l in (("bell", LBLUE, -0.27, "Bell (CRX)"), ("bell_native", BLUE, -0.09, "Bell (native)"), ("onchip20", "#f2a27f", 0.09, "chip-trained"), ("onchip20_pruned", ORA, 0.27, "chip-trained, pruned")):
    y = [pp[k_]["paulis"][k]["copies90"] for k in keys]
    ax.scatter(x + o, y, s=9, color=c, label=l, zorder=3, lw=0)
ax.axhline(bound[5], color=INK, ls="--", lw=0.8); ax.text(-0.4, bound[5] * 1.03, "single-copy limit 183", ha="left", va="bottom", fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(["".join("IXYZ"[int(c)] for c in k) for k in keys], rotation=90, fontsize=5, family="monospace")
ax.set_ylabel("copies for 90%"); ax.set_ylim(0, 270); ax.set_title("per Pauli, n = 5 (same day, same circuits per Pauli)", loc="left")
ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=5.1, bbox_to_anchor=(1, 1.04)); lab(ax, "h", x=-0.08)
# i: copies vs weight (n=5 and 6)
ax = fig.add_subplot(gs[2, 2])
for st, mk in (("PR5", "o"), ("PR6", "s"), ("PR7", "^")):
    f = os.path.join(A, f"{st}_perpauli.json")
    if not os.path.exists(f): continue
    q = json.load(open(f))
    for k_, c in (("bell_native", BLUE), ("onchip20_pruned", ORA)):
        ws = np.array([v["w"] for v in q[k_]["paulis"].values()]); cs = np.array([v["copies90"] for v in q[k_]["paulis"].values()], float)
        for wv in sorted(set(ws)):
            m = cs[(ws == wv) & np.isfinite(cs)]
            if len(m): ax.scatter(wv + (0.12 if k_ == "onchip20_pruned" else -0.12), np.mean(m), color=c, marker=mk, s=12, lw=0)
ax.scatter([], [], color=BLUE, s=10, label="Bell (native)"); ax.scatter([], [], color=ORA, s=10, label="chip-trained, pruned")
ax.scatter([], [], color=MUT, marker="o", s=10, label="n = 5"); ax.scatter([], [], color=MUT, marker="s", s=10, label="n = 6"); ax.scatter([], [], color=MUT, marker="^", s=10, label="n = 7")
ax.set_yscale("log"); ax.set_xlabel("Pauli weight w"); ax.set_ylabel("mean copies for 90%")
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
ax.yaxis.set_major_locator(FixedLocator([50, 100, 200, 400])); ax.yaxis.set_major_formatter(FixedFormatter(["50", "100", "200", "400"])); ax.yaxis.set_minor_locator(NullLocator()); ax.set_ylim(45, 700); ax.set_title("cost grows with weight", loc="left"); ax.legend(frameon=False, fontsize=5.2, loc="upper left"); lab(ax, "i")
# j: reproducibility across days (n=5)
ax = fig.add_subplot(gs[2, 3])
days = [("S5", "bell_bayes", BLUE), ("R5", "bell_bayes", BLUE), ("T5", "bell_bayes", BLUE), ("S5R", "bell_bayes", BLUE), ("PR5", "bell_bayes", BLUE),
        ("T5", "onchip20_bayes", ORA), ("T5", "onchip22_bayes", ORA), ("S5R", "onchip20_bayes", ORA), ("PR5", "onchip20_bayes", ORA), ("PR5", "onchip22_bayes", ORA)]
for i, (st, k, c) in enumerate(days):
    f = f"{st}_bootstrap_pauli_origin.json" if st != "PR5" else "PR5_bootstrap_pauli_origin" + PR
    y, lo, hi = bs(f, k)
    ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], color=c, marker="o", ms=3, capsize=1.5, lw=0.9)
ax.axhline(bound[5], color=INK, ls="--", lw=0.8)
ax.set_xticks(range(len(days))); ax.set_xticklabels([d[0] for d in days], rotation=90, fontsize=5.3)
ax.set_ylabel("copies for 90% (n = 5)"); ax.set_ylim(0, 230); ax.set_title("repeat runs, n = 5", loc="left")
ax.text(1.5, 15, "Bell (CRX)", color=BLUE, fontsize=5.5, ha="center"); ax.text(7, 15, "chip-trained", color=ORA, fontsize=5.5, ha="center"); lab(ax, "j")
fig.savefig(_os.path.join(_OUT, "fig3_scaling.pdf")); fig.savefig(_os.path.join(_OUT, "fig3.png"), dpi=170)
