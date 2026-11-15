import json, os, sys, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
R = os.path.join(_HERE, "..", "hw", "runs"); AB = os.path.join(_HERE, "..", "hw", "sim", "results", "ablation_summary.json")
OUTD = os.path.join(_HERE, "out"); os.makedirs(OUTD, exist_ok=True)
J = lambda f: json.load(open(os.path.join(R, f)))
def bound(n): return (2 ** n + 1) * (0.9 * np.log(4 ** n - 1) - np.log(2))
fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(3, 4, hspace=0.72, wspace=0.55, left=0.075, right=0.985, top=0.955, bottom=0.07)

# ---------------- a: copies vs n, 5..10
ax = fig.add_subplot(gs[0, 0:2])
PR = "_bell_bell_native_onchip20_onchip20_pruned_onchip22_onchip22_pruned.json"
def bs(f, k):
    d = J(f)["results"][k]
    if isinstance(d, dict):
        a = np.array(d["reps"], float)
    else:
        a = np.array(d, float)
    b = a[1:][np.isfinite(a[1:])]; return a[0], np.percentile(b, 2.5), np.percentile(b, 97.5)
def bg(step, pol):
    p = J(f"{step}_bign_origin.json")["policies"][pol]; return p["point"], p["ci"][0], p["ci"][1]
ser = {"bell_native": {5: bs("PR5_bootstrap_pauli_origin" + PR, "bell_native_bayes"), 6: bs("PR6_bootstrap_pauli_origin" + PR, "bell_native_bayes"),
                       7: bs("PR7_bootstrap_large_origin_bell_native_bayes.json", "bell_native_bayes")},
       "onchip20_pruned": {5: bs("PR5_bootstrap_pauli_origin" + PR, "onchip20_pruned_bayes"), 6: bs("PR6_bootstrap_pauli_origin" + PR, "onchip20_pruned_bayes"),
                           7: bs("PR7_bootstrap_large_origin_onchip20_pruned_bayes.json", "onchip20_pruned_bayes")}}
for st, n in (("X8", 8), ("X9", 9), ("X10R", 10)):
    for pol in ser:
        try:
            ser[pol][n] = bg(st, pol)
        except Exception:
            pass
ns = np.arange(5, 11)
ax.fill_between(ns, [bound(n) for n in ns], 1e5, color=SURF, lw=0, zorder=0)
ax.plot(ns, [bound(n) for n in ns], color=INK, ls="--", lw=1, label="single-copy limit (non-adaptive, ideal)")
for (pol, d), col, mk, lb, off in ((("bell_native", ser["bell_native"]), BLUE, "o", "Bell (native)", -0.08),
                                   (("onchip20_pruned", ser["onchip20_pruned"]), ORA, "D", "chip-trained at n = 2, pruned", 0.08)):
    xs = sorted(d); y = np.array([d[x][0] for x in xs]); lo = np.array([d[x][1] for x in xs]); hi = np.array([d[x][2] for x in xs])
    ax.errorbar(np.array(xs) + off, y, yerr=[y - lo, hi - y], color=col, marker=mk, ms=3.5, lw=1.1, capsize=1.5, label=lb)
for n in ns:
    if n in ser["onchip20_pruned"]:
        y0 = ser["onchip20_pruned"][n]
        ax.text(n + 0.08, y0[2] * 1.35, f"{y0[0] / bound(n):.2f}", ha="center", fontsize=5.2, color=ORA)
ax.set_yscale("log"); ax.set_ylim(40, 3e4); ax.set_xlim(4.6, 10.4); ax.set_xticks(ns)
ax.set_xticklabels([f"{n}\n({2 * n})" for n in ns]); ax.set_xlabel("qubits per copy n (total qubits)")
ax.set_ylabel("copies for 90 % identification"); ax.legend(frameon=False, loc="upper left", fontsize=5.3)
ax.text(0.99, 0.03, "numbers: chip-trained ÷ limit", transform=ax.transAxes, ha="right", fontsize=5.2, color=ORA)
lab(ax, "a", x=-0.12)

# ---------------- b: accuracy vs copies at n = 8-10 with Fano ceilings
ax = fig.add_subplot(gs[0, 2])
cols = {8: AQUA, 9: VIO, 10: BLUE}
for st, n in (("X8", 8), ("X9", 9), ("X10R", 10)):
    try:
        d = J(f"{st}_bign_origin.json"); p = d["policies"]["bell_native"]
    except Exception:
        continue
    g = np.array(d["grid"], float); c = np.array(p["curve"], float); ok = np.isfinite(c)
    ax.plot(g[ok], c[ok], color=cols[n], lw=1.2, label=f"n = {n}")
    dd = 2 ** n; M = np.logspace(1, 4.3, 60); ceil = np.minimum(1, (M / (dd + 1) + np.log(2)) / np.log(dd * dd - 1))
    ax.plot(M, ceil, color=cols[n], ls="--", lw=0.8)
try:
    d = J("X10_bign_origin.json"); p = d["policies"]["bell_native"]; g = np.array(d["grid"], float); c = np.array(p["curve"], float); ok = np.isfinite(c)
    ax.plot(g[ok], c[ok], color=MUT, lw=0.9, ls="-.", label="n = 10, first run")
except Exception:
    pass
ax.set_xscale("log"); ax.set_xlim(50, 2e4); ax.set_ylim(0, 1.02); ax.axhline(0.9, color=MUT, ls=":", lw=0.6)
ax.set_xlabel("copies"); ax.set_ylabel("accuracy (Bell, native)"); ax.legend(frameon=False, fontsize=5, loc="upper left", handlelength=1.2)
ax.plot([], [], color=MUT, ls="--", lw=0.8, label="best single-copy\n(same colour)")
ax.legend(frameon=False, fontsize=5, loc="upper left", handlelength=1.2)
lab(ax, "b")

# ---------------- c: simulation controls, fraction of seeds that learn
ab = json.load(open(AB))
ax = fig.add_subplot(gs[0, 3])
fams = [("site13", "V (13)"), ("su4_shared", "SU(4) (15)"), ("su4_free", "SU(4) per site (30)"), ("hea4", "pairing-agnostic (40)"), ("product6", "single-copy (6)")]
for i, (k, lb) in enumerate(fams):
    rows = [r for r in ab[k] if not r.get("long")]
    fin = np.array([r["final"] for r in rows])
    ax.bar(i, np.mean(fin > 0.5), color=[BLUE, AQUA, VIO, MUT, RED][i], width=0.7)
    ax.scatter(i + (np.random.default_rng(i).random(len(fin)) - 0.5) * 0.45, fin, s=4, color=INK, zorder=3, lw=0)
    ax.text(i, 1.03, f"{int(np.sum(fin > 0.5))}/{len(fin)}", ha="center", fontsize=5.2)
ax.set_xticks(range(len(fams))); ax.set_xticklabels([f[1] for f in fams], fontsize=5, rotation=40, ha="right", rotation_mode="anchor")
ax.set_ylim(0, 1.12); ax.set_ylabel("fraction learned (bars)\nfinal reward (dots)")
ax.set_title("simulation, 250 iterations", loc="left", fontsize=6)
lab(ax, "c", x=-0.38)

# ---------------- d: site visibility, learned vs random two-qubit circuits
ax = fig.add_subplot(gs[1, 0])
h = np.array(ab["haar_site_vis"]["hist"], float); edges = np.linspace(0, 1, 41)
ax.bar(edges[:-1], h / h.sum(), width=0.025, align="edge", color=SURF, edgecolor=MUT, lw=0.3, label="random SU(4)")
for k, col, lb in (("site13", BLUE, "paper's V"), ("su4_shared", AQUA, "general SU(4)")):
    v = [r["site_vis"][0] for r in ab[k] if not r.get("long") and r["final"] > 0.5]
    for x in v:
        ax.axvline(x, color=col, lw=0.7, ymax=0.85, alpha=0.8)
    ax.plot([], [], color=col, lw=1, label=f"learned, {lb}")
ax.set_xlim(0, 1.02); ax.set_xlabel("site two-copy visibility"); ax.set_ylabel("fraction")
ax.legend(frameon=False, fontsize=4.9, loc="center", bbox_to_anchor=(0.66, 0.5))
lab(ax, "d", x=-0.32)

# ---------------- e: single-copy agent on the chip
ax = fig.add_subplot(gs[1, 1])
def hist(f):
    return np.array([x["reward"] for x in J(f)["hist"]])
for f, col in (("train_s30.json", BLUE), ("train_s31.json", BLUE), ("train_s32.json", BLUE)):
    r = smooth(hist(f)); ax.plot(np.arange(1, len(r) + 1), r, color=col, lw=0.9, alpha=0.8)
for f in ("train_sc_s60.json", "train_sc_s61.json"):
    r = smooth(hist(f)); ax.plot(np.arange(1, len(r) + 1), r, color=RED, lw=1.1)
ax.plot([], [], color=BLUE, label="two-copy agent (seeds 30–32)"); ax.plot([], [], color=RED, label="single-copy agent (seeds 60, 61)")
ax.axhline(1 / 15, color=RED, ls=":", lw=0.6)
ax.set_xlabel("iteration"); ax.set_ylabel("reward (WK_C180)"); ax.set_ylim(0, 1.18); ax.legend(frameon=False, fontsize=5, loc="upper left")
lab(ax, "e", x=-0.32)

# ---------------- f: channel task schematic
ax = fig.add_subplot(gs[1, 2:]); ax.set_xlim(0, 40); ax.set_ylim(0, 14); ax.axis("off")
ax.text(0, 13.2, "Second task: name a hidden Pauli channel", fontsize=6.4, fontweight="bold")
ys, ya = 9.0, 4.2
for y, l in ((ys, "system $S_i$"), (ya, "ancilla $A_i$")):
    ax.plot([5.0, 33.5], [y, y], color=INK, lw=0.6); ax.text(4.6, y, l, ha="right", va="center", fontsize=5.5)
    ax.text(5.6, y + 0.9, "|0⟩", fontsize=5.5)
ax.add_patch(FancyBboxPatch((7.5, 3.0), 6.0, 7.2, boxstyle="round,pad=0,rounding_size=0.4", fc="#fff4e8", ec=ORA, lw=0.9))
ax.text(10.5, 6.6, "W(θ)\ninput\n(learned)", ha="center", va="center", fontsize=5.5)
ax.add_patch(FancyBboxPatch((15.3, 7.6), 5.6, 2.8, boxstyle="round,pad=0,rounding_size=0.4", fc=SURF, ec=INK, lw=0.8))
ax.text(18.1, 9.0, "$\\Lambda_P$", ha="center", va="center", fontsize=7)
ax.add_patch(FancyBboxPatch((22.7, 3.0), 6.0, 7.2, boxstyle="round,pad=0,rounding_size=0.4", fc="#eef4fc", ec=BLUE, lw=0.9))
ax.text(25.7, 6.6, "V\nmeasurement\n(reused from\ntask 1)", ha="center", va="center", fontsize=5.2)
ax.text(34.0, 6.6, "→ $\\hat P$\nreward\n$\\mathbf{1}\\{\\hat P=P\\}$", va="center", fontsize=5.6)
ax.text(18.1, 1.2, r"$\Lambda_P(\rho)=(I+\mathrm{tr}(P\rho)\,P)/d$: without the ancilla every strategy obeys the same limit (Theorem 1)", ha="center", fontsize=5.2, color=MUT)
lab(ax, "f", x=-0.02)

# ---------------- g: channel training on the chip
ax = fig.add_subplot(gs[2, 0])
for f, col, lb in (("train_ch_s70.json", AQUA, "seed 70"), ("train_ch_s71.json", MUT, "seed 71")):
    r = smooth(hist(f)); ax.plot(np.arange(1, len(r) + 1), r, color=col, lw=1.1, label=lb)
ax.axhline(1 / 15, color=RED, ls=":", lw=0.6); ax.set_ylim(0, 1); ax.set_xlabel("iteration"); ax.set_ylabel("reward (WK_C180)")
ax.legend(frameon=False, fontsize=5.2, loc="upper left")
lab(ax, "g", x=-0.32)

# ---------------- h, i: channel identification at n = 5, 6
for j, (n, L) in enumerate(((5, "h"), (6, "i"))):
    ax = fig.add_subplot(gs[2, 1 + j])
    for st, pol, col, lb in ((f"CH{n}", "bell_channel", BLUE, "Bell pair + Bell meas."), (f"CH{n}A", "train_ch_s70", AQUA, "learned input (seed 70)")):
        try:
            d = J(f"{st}_channel_origin.json"); p = d["policies"][pol]
        except Exception:
            continue
        g = np.array(d["grid"], float); c = np.array(p["curve"], float); ok = np.isfinite(c)
        ax.plot(g[ok], c[ok], color=col, lw=1.2, label=f"{lb}: {p['point']:.0f}")
    ax.axvspan(1, bound(n), color=SURF, lw=0, zorder=0); ax.axvline(bound(n), color=INK, ls="--", lw=0.8)
    ax.axhline(0.9, color=MUT, ls=":", lw=0.6); ax.set_xscale("log"); ax.set_xlim(3, 2000); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("channel uses"); ax.set_title(f"n = {n} system qubits", loc="left", fontsize=6)
    if j == 0: ax.set_ylabel("identification accuracy")
    ax.text(bound(n) * 1.15, 0.45, "no-ancilla\nlimit", fontsize=5, color=INK, ha="left", va="center")
    ax.legend(frameon=False, fontsize=4.9, loc="upper left", bbox_to_anchor=(-0.02, 1.0))
    lab(ax, L)
# ---------------- j: info per use
ax = fig.add_subplot(gs[2, 3])
vals = [("no ancilla\n(optimum)", 0.191, RED), ("learned\n(seed 70)", None, AQUA), ("Bell pair,\nBell meas.", 0.962, BLUE)]
try:
    st = J("train_ch_s70.json"); vals[1] = (vals[1][0], st["hist"][-1]["info_ideal"], AQUA)
except Exception:
    pass
for i, (lb, v, col) in enumerate(vals):
    if v is not None:
        ax.bar(i, v, color=col, width=0.65); ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.2)
ax.set_xticks(range(3)); ax.set_xticklabels([v[0] for v in vals], fontsize=5); ax.set_ylim(0, 1.1)
ax.set_ylabel("information per use (bits, n = 2)")
lab(ax, "j", x=-0.35)
fig.savefig(os.path.join(OUTD, "fig7_generality.pdf")); fig.savefig(os.path.join(OUTD, "fig7.png"), dpi=170)
