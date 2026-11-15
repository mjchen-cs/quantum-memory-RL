import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_RUNS = _os.path.join(_ROOT, "hw", "runs")
_OUT = _os.path.join(_os.path.dirname(__file__), "out"); _os.makedirs(_OUT, exist_ok=True)
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from style import *
H = _RUNS
def hist(f): return json.load(open(os.path.join(H, f + ".json")))["hist"]
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.42, left=0.07, right=0.985, top=0.94, bottom=0.08)
def curves(ax, runs, title, marks=None):
    for f, col, l in runs:
        h = hist(f); it = [x["it"] for x in h]; r = smooth([x["reward"] for x in h])
        ax.plot(it, r, color=col, label=l)
    ax.axhline(1 / 15, color=RED, lw=0.6, ls=":")
    ax.set_ylim(0, 1); ax.set_xlabel("iteration"); ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
# a: protocol v1 -> v2 (decoder restarts from it ~151)
ax = fig.add_subplot(gs[0, 0])
curves(ax, [("train_s1", MUT, "seed 1 (v1)"), ("train_s10", RED, "seed 10"), ("train_s11", BLUE, "seed 11"), ("train_s12", AQUA, "seed 12")], "v1 → v2: decoder restarts")
ax.axvline(151, color=INK, lw=0.6, ls="--"); ax.text(153, 0.05, "restarts\nstart", fontsize=5.5, color=INK)
ax.set_ylabel("reward (5-iteration mean)"); lab(ax, "a")
ax = fig.add_subplot(gs[0, 1])
curves(ax, [("train_s20", BLUE, "seed 20"), ("train_s21", RED, "seed 21"), ("train_s22", LBLUE, "seed 22")], "v3: + annealing, unscreened"); lab(ax, "b")
ax = fig.add_subplot(gs[0, 2])
curves(ax, [("train_s30", BLUE, "seed 30"), ("train_s31", ORA, "seed 31"), ("train_s32", AQUA, "seed 32")], "v4: + front-end restart"); lab(ax, "c"); ax.legend(frameon=False, loc="lower right", handlelength=1.2)
for f, col in (("train_s30", BLUE), ("train_s31", ORA)):
    for x in hist(f):
        if x.get("fe_restart"): ax.axvline(x["it"], color=col, lw=0.7, ls=":")
# d: success per protocol
ax = fig.add_subplot(gs[0, 3])
vers = ["v1", "v2", "v3", "v4"]; learned = [1, 2, 2, 3]; tot = [4, 3, 3, 3]
ax.bar(vers, tot, color=SURF, edgecolor=GRID, width=0.6, label="runs")
ax.bar(vers, learned, color=BLUE, width=0.6, label="learned (reward > 0.6)")
for i, (l, t) in enumerate(zip(learned, tot)): ax.text(i, t + 0.08, f"{l}/{t}", ha="center", fontsize=6)
ax.set_ylim(0, 5); ax.set_ylabel("training runs"); ax.set_title("success by protocol", loc="left")
ax.legend(frameon=False, loc="upper right"); lab(ax, "d")
# e: two-copy visibility of the front-end during training
ax = fig.add_subplot(gs[1, 0])
for f, col in (("train_s20", BLUE), ("train_s21", RED), ("train_s22", LBLUE), ("train_s30", BLUE), ("train_s31", ORA), ("train_s32", AQUA)):
    h = hist(f); ax.plot([x["it"] for x in h], [x["vis_ideal"] for x in h], color=col, lw=0.8, alpha=0.9)
ax.set_ylim(0, 1.02); ax.set_xlabel("iteration"); ax.set_ylabel("two-copy visibility of V(θ)"); ax.set_title("two-copy visibility", loc="left"); lab(ax, "e")
# f: reward vs visibility (all iterations, all twirl runs)
ax = fig.add_subplot(gs[1, 1])
for f, col in (("train_s20", BLUE), ("train_s21", RED), ("train_s22", LBLUE), ("train_s30", BLUE), ("train_s31", ORA), ("train_s32", AQUA)):
    h = hist(f); ax.scatter([x["vis_ideal"] for x in h], [x["reward"] for x in h], s=2, color=col, alpha=0.5, lw=0)
ax.set_xlabel("two-copy visibility"); ax.set_ylabel("reward"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1); ax.set_title("reward vs visibility", loc="left"); lab(ax, "f")
# g: decoder restarts: current vs best fresh candidate
ax = fig.add_subplot(gs[1, 2])
for f in ("train_s10", "train_s11", "train_s12", "train_s20", "train_s21", "train_s22", "train_s30", "train_s31", "train_s32"):
    for x in hist(f):
        d = x.get("dec_restart")
        if d and "current" in d:
            ax.scatter(d["current"], max(d["cands"]), s=7, color=ORA if d["accepted"] else MUT, lw=0, alpha=0.85)
ax.plot([0, 1], [0, 1], color=GRID, lw=0.7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.scatter([], [], s=7, color=ORA, label="accepted"); ax.scatter([], [], s=7, color=MUT, label="kept current"); ax.legend(frameon=False, loc="upper left")
ax.set_xlabel("reward, current decoder"); ax.set_ylabel("reward, best fresh decoder"); ax.set_title("decoder restarts", loc="left"); lab(ax, "g")
# h: processor time per iteration
ax = fig.add_subplot(gs[1, 3])
q = [x["info"]["qpu_ms"] / 1000 for f in ("train_s20", "train_s21", "train_s22", "train_s30", "train_s31", "train_s32") for x in hist(f) if x.get("info", {}).get("qpu_ms")]
ax.hist(q, bins=30, color=BLUE, alpha=0.85)
ax.axvline(np.median(q), color=INK, lw=0.7, ls="--"); ax.text(np.median(q) * 1.02, ax.get_ylim()[1] * 0.85, f"median {np.median(q):.1f} s", fontsize=5.8)
ax.set_xlabel("processor time per iteration (s)"); ax.set_ylabel("iterations"); ax.set_title("processor time", loc="left"); lab(ax, "h")
fig.savefig(_os.path.join(_OUT, "fig2_training.pdf")); fig.savefig(_os.path.join(_OUT, "fig2.png"), dpi=170)
print(np.median(q), len(q), np.sum(q) / 3600)
