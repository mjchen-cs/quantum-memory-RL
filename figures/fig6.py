import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_RUNS = _os.path.join(_ROOT, "hw", "runs")
_OUT = _os.path.join(_os.path.dirname(__file__), "out"); _os.makedirs(_OUT, exist_ok=True)
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from style import *
sys.path.insert(0, _os.path.join(_ROOT, "hw")); from fast_site import tables, SiteModel
R = _RUNS; Hd = _RUNS
def hist(f): return json.load(open(os.path.join(Hd, f + ".json")))["hist"]
def vis2(th): return SiteModel(2).visibility(np.asarray(th, float)) if hasattr(SiteModel(2), "visibility") else None
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(3, 4, hspace=0.72, wspace=0.75, left=0.07, right=0.985, top=0.955, bottom=0.065)
# a adaptation reward
ax = fig.add_subplot(gs[0, :2])
for f, c, l in (("adapt_s20", BLUE, "agent s20"), ("adapt_s22", LBLUE, "agent s22")):
    h = hist(f); ax.plot([x["it"] for x in h], [x["reward"] for x in h], color=c, marker="o", ms=2.2, label=l)
ax.axvspan(10.5, 45.5, color=SURF, lw=0, zorder=0); ax.axvline(10.5, color=INK, lw=0.7, ls="--")
ax.text(11.2, 0.95, "coherent error injected: R_Z(π/2) on every qubit of copy B", fontsize=5.8, va="top")
ax.axhline(1 / 15, color=RED, lw=0.6, ls=":"); ax.set_xlim(0.5, 45.5); ax.set_ylim(0, 1)
ax.set_xlabel("training iteration on the chip (n = 2)"); ax.set_ylabel("reward"); ax.set_title("online adaptation", loc="left")
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.1)); lab(ax, "a", x=-0.1)
# b step size and overlap with the pre-drift optimum
ax = fig.add_subplot(gs[0, 2])
for f, c in (("adapt_s20", BLUE), ("adapt_s22", LBLUE)):
    h = hist(f); ax.plot([x["it"] for x in h], [x["vis_ideal"] for x in h], color=c)
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
h = hist("adapt_s20"); ax2.step([x["it"] for x in h], [x["lr_mult"] for x in h], color=ORA, lw=0.9, where="mid")
ax2.set_ylabel("step-size multiplier", color=ORA, fontsize=6); ax2.tick_params(axis="y", colors=ORA, labelsize=5.5); ax2.set_ylim(0, 1.1)
ax.set_ylim(0, 1.05); ax.set_xlabel("iteration"); ax.set_ylabel("two-copy visibility\nwithout the error"); ax.set_title("the front-end re-orients", loc="left"); lab(ax, "b")
# c effective visibility with the error before/after adaptation
ax = fig.add_subplot(gs[0, 3])
S2 = SiteModel(2)
def veff(th, dB=np.pi / 2):
    t = np.array(th, float).copy(); t[3] = t[3] + dB      # R_Z(dB) before the pre-rotation of copy B
    g = tables(t)[2][1:]; return float((g ** 2).mean())
vals = []
for f0, f1 in (("train_s20", "adapt_s20"), ("train_s22", "adapt_s22")):
    t0 = json.load(open(os.path.join(Hd, f0 + ".json")))["best"]["theta"]; d1 = json.load(open(os.path.join(Hd, f1 + ".json"))); t1 = d1["theta"]
    vals.append((veff(t0, 0), veff(t0), veff(t1)))
x = np.arange(2)
for j, (l, c) in enumerate((("no error", MUT), ("error, not adapted", RED), ("error, adapted", BLUE))):
    ax.bar(x + (j - 1) * 0.27, [v[j] for v in vals], 0.25, color=c, label=l)
ax.set_xticks(x); ax.set_xticklabels(["s20", "s22"]); ax.set_ylim(0, 1.45); ax.set_ylabel("two-copy visibility")
ax.legend(frameon=False, fontsize=5, loc="upper left"); ax.set_title("recovered", loc="left"); lab(ax, "c")
# d,e drift regret
d = json.load(open(os.path.join(R, "DR5_drift_origin.json"))); T, l0 = d["T"], d["l0"]
COL = dict(bell_native=BLUE, onchip20_pruned=ORA, classical=RED); LAB = dict(bell_native="Bell (native)", onchip20_pruned="chip-trained, pruned", classical="random bases")
for p, (qi, q) in enumerate(enumerate(d["q"])):
    ax = fig.add_subplot(gs[1, p])
    for ag in ("classical", "onchip20_pruned", "bell_native"):
        C = np.array([d["results"][ag][f"{qi}_{r}"]["curve"] for r in range(4)]); t = np.arange(C.shape[1]) * 5 + 1
        m = C.mean(0); s = C.std(0, ddof=1) / 2
        ax.fill_between(t, m - s, m + s, color=COL[ag], alpha=0.18, lw=0); ax.plot(t, m, color=COL[ag], label=LAB[ag])
    tt = np.arange(1, T + 1); ax.plot(tt, np.where(tt > l0, 0.5 * (tt - l0) * (1 - (1 - q) ** l0), 0), color=INK, ls="--", lw=0.8, label="single-copy limit")
    ax.plot(tt, tt, color=GRID, ls=":", lw=0.7)
    ax.set_xlim(0, T); ax.set_ylim(0, T); ax.set_xlabel("round"); ax.set_title(f"tracking, n = 5, q = {q}", loc="left")
    if p == 0: ax.set_ylabel("cumulative regret"); ax.legend(frameon=False, fontsize=5, loc="upper left")
    lab(ax, "de"[p])
# f per-run paired final regret
ax = fig.add_subplot(gs[1, 2])
for qi, q in enumerate(d["q"]):
    for r in range(4):
        y = [d["results"][ag][f"{qi}_{r}"]["final"] for ag in ("bell_native", "onchip20_pruned", "classical")]
        x0 = qi * 4
        ax.plot([x0, x0 + 1, x0 + 2], y, color=GRID, lw=0.7, zorder=1)
        ax.scatter([x0, x0 + 1, x0 + 2], y, color=[BLUE, ORA, RED], s=10, zorder=2, lw=0)
    ax.hlines(d[f"lower_bound_q{q}"], x0 - 0.4, x0 + 2.4, color=INK, ls="--", lw=0.8)
ax.set_xticks([0, 1, 2, 4, 5, 6]); ax.set_xticklabels(["Bell", "chip", "rand.", "Bell", "chip", "rand."], fontsize=5.3, rotation=90)
ax.text(1, 405, "q = 0.02", ha="center", fontsize=5.8); ax.text(5, 405, "q = 0.1", ha="center", fontsize=5.8)
ax.set_ylim(0, 430); ax.set_ylabel("regret after 400 rounds"); ax.set_title("8/8 paired runs", loc="left"); lab(ax, "f")
# g simulation vs visibility
ax = fig.add_subplot(gs[1, 3])
ks = ["1.0", "0.95", "0.866\n(chip)"]; sim = {0.02: [40, 69, 142], 0.1: [133, 220, 318]}
for j, (q, c) in enumerate(((0.02, BLUE), (0.1, LBLUE))):
    ax.bar(np.arange(3) + (j - 0.5) * 0.36, sim[q], 0.34, color=c, label=f"q = {q}")
    ax.hlines(d[f"lower_bound_q{q}"], -0.5, 2.5, color=c, ls="--", lw=0.8)
ax.scatter([2 - 0.18, 2 + 0.18], [170, 332], color=INK, marker="x", s=14, zorder=3, label="measured")
ax.set_xticks(range(3)); ax.set_xticklabels(ks, fontsize=5.5); ax.set_xlabel("κ (simulated Bell)"); ax.set_ylabel("regret after 400 rounds")
ax.legend(frameon=False, fontsize=5, loc="upper left"); ax.set_title("noise sets the gap", loc="left"); lab(ax, "g")
# h-j detection
D = {n: json.load(open(os.path.join(R, f"DT{n}_detect_origin.json"))) for n in (5, 6, 7)}
DC = json.load(open(os.path.join(R, "DTC5_detect_origin.json")))
DCOL = dict(bell=LBLUE, onchip20="#f2a27f", det50=AQUA, det51=ORA)
DLAB = dict(bell="Bell (CRX)", onchip20="chip-trained (identification)", det50="detection agent s50", det51="detection agent s51")
for p, n in enumerate((5, 6, 7)):
    ax = fig.add_subplot(gs[2, p]); dd = D[n]; g = np.array(dd["grid"], float); dim = 2 ** n
    Ms = np.logspace(0, 4.2, 200); ax.fill_between(Ms, 0.4, np.minimum(1, 0.5 + Ms / (4 * (dim + 1))), color=SURF, lw=0)
    ax.plot(Ms, np.minimum(1, 0.5 + Ms / (4 * (dim + 1))), color=INK, ls="--", lw=0.8)
    for k in ("bell", "onchip20", "det50", "det51"):
        a = np.array(dd["agents"][k]["acc_cv"], float); ok = ~np.isnan(a); ax.plot(g[ok], a[ok], color=DCOL[k], label=DLAB[k], lw=0.9)
    if n == 5:
        a = np.array(DC["agents"]["classical"]["acc_cv"], float); gc = np.array(DC["grid"], float); ok = ~np.isnan(a); ax.plot(gc[ok], a[ok], color=RED, label="random bases", lw=0.9)
    ax.set_xscale("log"); ax.set_xlim(2, 1e4); ax.set_ylim(0.4, 1.02); ax.set_xlabel("copies"); ax.set_title(f"detection, n = {n}", loc="left")
    if p == 0: ax.set_ylabel("balanced accuracy"); ax.legend(frameon=False, fontsize=4.8, loc="upper left")
    lab(ax, "hij"[p])
ax = fig.add_subplot(gs[2, 3])
for j, k in enumerate(("bell", "onchip20", "det50", "det51")):
    y = [2 * D[n]["agents"][k]["delay_pairs_mean"] for n in (5, 6, 7)]; e = [2 * D[n]["agents"][k]["delay_pairs_sem"] for n in (5, 6, 7)]
    ax.bar(np.arange(3) + (j - 1.5) * 0.2, y, 0.19, yerr=e, color=DCOL[k], capsize=1, error_kw=dict(lw=0.6))
ax.bar(-0.75, 2 * DC["agents"]["classical"]["delay_pairs_mean"], 0.19, color=RED)
ax.axhline(1000, color=INK, ls=":", lw=0.7); ax.text(2.5, 1030, "false-alarm horizon (500 pairs)", fontsize=5, ha="right")
ax.set_xticks([-0.75, 0, 1, 2]); ax.set_xticklabels(["rand.\n(n=5)", "5", "6", "7"], fontsize=5.5); ax.set_xlabel("n"); ax.set_ylabel("mean detection delay (copies)")
ax.set_title("CUSUM delay", loc="left"); lab(ax, "k")
fig.savefig(_os.path.join(_OUT, "fig6_adapt_drift.pdf")); fig.savefig(_os.path.join(_OUT, "fig6.png"), dpi=170)
print(vals)
