import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_RUNS = _os.path.join(_ROOT, "hw", "runs")
_OUT = _os.path.join(_os.path.dirname(__file__), "out"); _os.makedirs(_OUT, exist_ok=True)
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from style import *
sys.path.insert(0, _os.path.join(_ROOT, "hw")); from fast_site import tables, bell_theta
R = _RUNS; Hd = _RUNS
def hist(f): return json.load(open(os.path.join(Hd, f + ".json")))["hist"]
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.48, left=0.07, right=0.985, top=0.94, bottom=0.085)
# a reward
ax = fig.add_subplot(gs[0, 0])
for f, c, ls, l in (("train_s20", BLUE, "-", "s20, shared sign"), ("train_s22", LBLUE, "-", "s22, shared sign"), ("train_nt_s20", ORA, "--", "s20, no sign"), ("train_nt_s22", "#f2a27f", "--", "s22, no sign")):
    h = hist(f)[:120]; ax.plot([x["it"] for x in h], smooth([x["reward"] for x in h]), color=c, ls=ls, label=l)
ax.axhline(1 / 15, color=RED, lw=0.6, ls=":"); ax.set_ylim(0, 1); ax.set_xlabel("iteration (n = 2, on the chip)"); ax.set_ylabel("reward"); ax.set_title("reward", loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=5.2); lab(ax, "a")
# b two-copy visibility
ax = fig.add_subplot(gs[0, 1])
for f, c, ls in (("train_s20", BLUE, "-"), ("train_s22", LBLUE, "-"), ("train_nt_s20", ORA, "--"), ("train_nt_s22", "#f2a27f", "--")):
    h = hist(f)[:120]; ax.plot([x["it"] for x in h], [x["vis_ideal"] for x in h], color=c, ls=ls)
ax.set_ylim(0, 1.02); ax.set_xlabel("iteration"); ax.set_ylabel("two-copy visibility"); ax.set_title("two-copy visibility", loc="left"); lab(ax, "b")
# c final front-ends: single-copy vs two-copy visibility (n = 2 model quantities)
ax = fig.add_subplot(gs[0, 2])
def single_vis(th, n=2):
    g = tables(np.asarray(th, float)); import itertools
    P = np.array(list(itertools.product(range(4), repeat=n)))[1:]; B = np.array(list(itertools.product(range(4), repeat=n)))
    out = []
    for k in (0, 1):
        pr_ = np.ones((len(P), len(B)))
        for i in range(n): pr_ *= g[k][P[:, i]][:, B[:, i]]
        out.append(float((pr_ ** 2).mean()))
    return max(out)
def two_vis(th, n=2):
    g = tables(np.asarray(th, float)); import itertools
    P = np.array(list(itertools.product(range(4), repeat=n)))[1:]; B = np.array(list(itertools.product(range(4), repeat=n)))
    pr_ = np.ones((len(P), len(B)))
    for i in range(n): pr_ *= g[2][P[:, i]][:, B[:, i]]
    return float((pr_ ** 2).mean())
prj = json.load(open(os.path.join(R, "prune.json")))["agents"]
for k in ("train_s1", "train_s11", "train_s12", "train_s20", "train_s22", "train_s30", "train_s31", "train_s32"):
    th = prj[k]["theta"]; ax.scatter(single_vis(th), two_vis(th), color=BLUE, s=14, lw=0)
for f, c in (("train_nt_s20", ORA), ("train_nt_s22", "#f2a27f")):
    th = json.load(open(os.path.join(Hd, f + ".json")))["best"]["theta"]; ax.scatter(single_vis(th), two_vis(th), color=c, s=18, marker="D", lw=0)
th = bell_theta(); ax.scatter(single_vis(th), two_vis(th), color=INK, s=30, marker="*", lw=0)
ax.text(0.02, 0.93, "Bell", fontsize=5.5); ax.set_xlim(-0.02, 0.4); ax.set_ylim(-0.03, 1.05)
ax.scatter([], [], color=BLUE, s=12, label="trained with sign (8)"); ax.scatter([], [], color=ORA, marker="D", s=12, label="trained without sign (2)")
ax.legend(frameon=False, fontsize=5.1, loc="center right"); ax.set_xlabel("single-copy visibility"); ax.set_ylabel("two-copy visibility"); ax.set_title("what was learned (n = 2)", loc="left"); lab(ax, "c")
# d response tables of the shortcut front-end
th = np.array(json.load(open(os.path.join(Hd, "train_nt_s20.json")))["best"]["theta"]); g = tables(th)
sub = gs[0, 3].subgridspec(3, 1, hspace=0.55)
for i, (t, k) in enumerate((("copy A", 0), ("copy B", 1), ("both", 2))):
    a = fig.add_subplot(sub[i]); m = g[k][1:]
    a.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    for (r, c), v in np.ndenumerate(m): a.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=4.5, color="white" if abs(v) > 0.6 else INK)
    a.set_xticks([]); a.set_yticks(range(3)); a.set_yticklabels(["X", "Y", "Z"], fontsize=5); a.set_ylabel(t, fontsize=5.5, rotation=0, ha="right", va="center", labelpad=8)
    for s_ in a.spines.values(): s_.set_visible(False)
    if i == 0: lab(a, "d", x=-0.9, y=1.25); a.set_title("shortcut s20: g_k[σ, β]", fontsize=5.6, loc="left", pad=3)
# e,f accuracy curves on the chip
for p, (st, n) in enumerate((("SC3", 3), ("SC4", 4))):
    ax = fig.add_subplot(gs[1, p]); d = json.load(open(os.path.join(R, f"{st}_shortcut_origin.json")))
    for k, c, l in (("shortcut20", ORA, "shortcut s20"), ("shortcut22", "#f2a27f", "shortcut s22"), ("bell_native", BLUE, "Bell")):
        r = d["agents"][k]; x = np.array(r["copies"], float); y = np.array(r["acc"], float); ok = ~np.isnan(y)
        ax.fill_between(x[ok], np.array(r["acc_lo"])[ok], np.array(r["acc_hi"])[ok], color=c, alpha=0.18, lw=0)
        ax.plot(x[ok], y[ok], color=c, label=l); ax.plot(x[ok], np.array(r["model_acc"])[ok], color=c, ls=":", lw=0.8)
    ax.axhline(0.9, color=MUT, ls=":", lw=0.6); ax.set_xscale("log"); ax.set_xlim(2, 1200); ax.set_ylim(0, 1.02); ax.set_xlabel("copies")
    ax.set_title(f"on the chip, n = {n} (no sign)", loc="left")
    if p == 0: ax.set_ylabel("identification accuracy"); ax.legend(frameon=False, loc="upper left", fontsize=5.2)
    lab(ax, "ef"[p])
# g scaling: ideal extrapolation + measured
ax = fig.add_subplot(gs[1, 2])
sc = json.load(open(os.path.join(R, "shortcut_scaling_ideal.json")))["copies90"]
ns = [2, 3, 4, 5]
ax.plot(ns, [sc[str(n)]["shortcut20"] for n in ns], color=ORA, ls="--", marker="o", ms=3, mfc="white", label="shortcut s20, ideal")
ax.plot(ns, [sc[str(n)]["shortcut22"] for n in ns], color="#f2a27f", ls="--", marker="o", ms=3, mfc="white", label="shortcut s22, ideal")
ax.plot(ns, [sc[str(n)]["bell"] for n in ns], color=BLUE, ls="--", marker="o", ms=3, mfc="white", label="Bell, ideal")
ax.scatter([3, 3], [415, 384], color=[ORA, "#f2a27f"], s=16, zorder=4, label="measured on chip")
ax.annotate("", xy=(4, 1000), xytext=(4, 600), arrowprops=dict(arrowstyle="->", color=ORA, lw=0.9)); ax.scatter([4], [1000], marker="_", s=40, color=ORA)
ax.text(4.08, 700, ">1,000\n(not reached)", fontsize=5.2, color=ORA)
ax.scatter([3, 4], [41, 58], color=BLUE, s=16, zorder=4)
ax.set_yscale("log"); ax.set_xticks(ns); ax.set_xlabel("qubits per copy, n"); ax.set_ylabel("copies for 90%"); ax.set_title("the shortcut does not scale", loc="left")
ax.legend(frameon=False, fontsize=5, loc="upper left"); lab(ax, "g")
# h fitted visibilities
ax = fig.add_subplot(gs[1, 3])
for p, st in enumerate(("SC3", "SC4")):
    d = json.load(open(os.path.join(R, f"{st}_shortcut_origin.json")))["agents"]
    for j, k in enumerate(("shortcut20", "shortcut22")):
        x0 = p * 2.6 + j * 1.1
        ax.bar(x0 - 0.22, d[k]["a"], 0.4, color=ORA); ax.bar(x0 + 0.22, d[k]["b"], 0.4, color=LBLUE)
ax.set_xticks([0, 1.1, 2.6, 3.7]); ax.set_xticklabels(["s20", "s22", "s20", "s22"]); ax.set_ylim(0.7, 1.0)
ax.text(0.55, 0.985, "n = 3", ha="center", fontsize=5.8); ax.text(3.15, 0.985, "n = 4", ha="center", fontsize=5.8)
from matplotlib.patches import Patch
ax.legend([Patch(color=ORA), Patch(color=LBLUE)], ["single-copy a", "two-copy b"], frameon=False, fontsize=5.2, loc="lower right")
ax.set_ylabel("fitted per-site visibility"); ax.set_title("fitted noise levels", loc="left"); lab(ax, "h")
fig.savefig(_os.path.join(_OUT, "fig5_shortcut.pdf")); fig.savefig(_os.path.join(_OUT, "fig5.png"), dpi=170)
