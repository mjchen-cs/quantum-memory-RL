import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_RUNS = _os.path.join(_ROOT, "hw", "runs")
_OUT = _os.path.join(_os.path.dirname(__file__), "out"); _os.makedirs(_OUT, exist_ok=True)
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
R = _RUNS
W, H = 7.2, 6.2
fig = plt.figure(figsize=(W, H))
def canvas(x, y, w, h):   # figure fractions; data units = 0.1 inch
    ax = fig.add_axes([x, y, w, h]); ax.set_xlim(0, w * W * 10); ax.set_ylim(0, h * H * 10); ax.axis("off"); return ax
def box(ax, x, y, w, h, fc="white", ec=GRID, lw=0.8, r=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=1))
def arrow(ax, p0, p1, col=INK, lw=0.9, cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=col, lw=lw, connectionstyle=cs, zorder=4))
def gate(ax, x, y, t, w=2.2, h=1.6, fs=5):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec=INK, lw=0.6, zorder=5)); ax.text(x, y, t, ha="center", va="center", fontsize=fs, zorder=6)
# ================= a (left top): 0..0.47 x 0.53..1
ax = canvas(0.0, 0.54, 0.46, 0.45)    # 33 x 28 units
ax.text(0.5, 26.5, "a", fontweight="bold", fontsize=9); ax.text(2.5, 26.5, "Task: name a hidden Pauli from pairs of copies", fontsize=6.6, fontweight="bold")
box(ax, 0.5, 3.0, 15.5, 21.5, fc=SURF)
ax.text(8.25, 22.6, "quantum data source", ha="center", fontsize=5.8, color=MUT)
ax.text(8.25, 20.6, r"hidden $P\in\{I,X,Y,Z\}^{\otimes n}\setminus I$", ha="center", fontsize=5.8)
for j, y in enumerate((17.4, 14.1, 10.8)):
    for x0 in (1.5, 8.9):
        ax.add_patch(Rectangle((x0, y - 1.1), 5.2, 2.2, fc="white", ec=BLUE, lw=0.8, zorder=3)); ax.text(x0 + 2.6, y, r"$\rho_{s_jP}$", ha="center", va="center", fontsize=5.8, zorder=4)
    ax.text(7.8, y, "⊗", ha="center", va="center", fontsize=6); ax.text(14.4, y - 2.0, ["pair 1", "pair 2", "pair m"][j], fontsize=4.8, color=MUT, ha="center")
ax.text(8.25, 5.6, r"$\rho_{sP}=(I+sP)/2^n$, sign $s_j=\pm1$ shared" + "\nwithin a pair: one copy alone carries\nno information about P", ha="center", va="center", fontsize=5, color=MUT)
arrow(ax, (16.3, 14.1), (19.0, 14.1), lw=1.1)
box(ax, 19.2, 10.6, 7.0, 7.0, ec=INK, lw=1)
ax.text(22.7, 15.6, "agent", ha="center", fontsize=6.5, fontweight="bold"); ax.text(22.7, 12.7, "64 parameters\nat every n", ha="center", fontsize=5.2, color=MUT, va="center")
arrow(ax, (26.4, 14.1), (28.3, 14.1), lw=1.1); ax.text(28.5, 14.1, r"$\hat P$", va="center", fontsize=6.5)
ax.text(27.0, 8.4, "reward\n" + r"$r=\mathbb{1}\{\hat P=P\}$", ha="center", va="center", fontsize=5.6, color=ORA)
arrow(ax, (27.0, 6.8), (22.7, 10.4), col=ORA, cs="arc3,rad=-0.3")
ax.text(24.8, 3.6, "the agent never sees P;\nthe reward is its only feedback", fontsize=5, color=ORA, ha="center", va="center")
# ================= b (right top)
ax = canvas(0.47, 0.54, 0.53, 0.45)   # 38 x 28
ax.text(0.3, 26.5, "b", fontweight="bold", fontsize=9); ax.text(2.3, 26.5, "Agent: site-local two-copy front-end + product decoder", fontsize=6.6, fontweight="bold")
ya, yb = 20.0, 13.0
for y, l in ((ya, r"$A_i$"), (yb, r"$B_i$")):
    ax.plot([1.6, 24.6], [y, y], color=INK, lw=0.6, zorder=2); ax.text(0.8, y, l, va="center", ha="center", fontsize=6)
for y in (ya, yb):
    for x, g in zip((3.8, 6.4, 9.0), ("RZ", "RY", "RZ")): gate(ax, x, y, g)
    for x, g in zip((15.4, 18.0, 20.6), ("RZ", "RY", "RZ")): gate(ax, x, y, g)
    ax.add_patch(Rectangle((22.6, y - 0.8), 1.8, 1.6, fc="white", ec=INK, lw=0.6, zorder=5)); ax.text(23.5, y, "◠", ha="center", va="center", fontsize=5.5, zorder=6)
ax.plot([12.2, 12.2], [yb, ya], color=INK, lw=0.7, zorder=3); ax.add_patch(Circle((12.2, ya), 0.38, color=INK, zorder=6)); gate(ax, 12.2, yb, "RX(φ)", w=3.0)
ax.add_patch(Rectangle((2.3, 10.8), 19.9, 11.6, fill=False, ec=BLUE, lw=0.8, ls="--"))
ax.text(12.2, 9.3, r"$V(\theta)$: 13 angles, the same on every pair $(A_i,B_i)$, $i=1,\dots,n$", ha="center", fontsize=5.1, color=BLUE)
arrow(ax, (24.8, 16.5), (27.0, 16.5)); ax.text(25.9, 17.4, r"$\beta_i$", ha="center", fontsize=5.8)
box(ax, 27.2, 10.6, 10.4, 12.0, ec=INK)
ax.text(32.4, 21.0, "decoder", ha="center", fontsize=6, fontweight="bold"); ax.text(32.4, 19.5, "3 tables h_k (4×4) + 3 weights", ha="center", fontsize=4.6, color=MUT)
ax.text(32.4, 15.6, r"$\mathrm{score}(a)=$" + "\n" + r"$\sum_j\sum_k c_k\prod_i h_k[a_i,\beta_{ji}]$", ha="center", va="center", fontsize=5.2)
ax.text(32.4, 12.2, r"$\hat P\sim\mathrm{softmax}(\mathrm{score})$", ha="center", va="center", fontsize=5.2)
for y, t in ((6.2, "Deploy: the same 13 angles at any n; decoder from V's response tables + one fitted visibility κ"),
             (3.9, "Prune: snap angles to Clifford values when the hardware objective does not drop → 1 CZ per site"),
             (1.6, "Interpret: response tables g[σ, β] = ⟨β|V(σ⊗σ)V†|β⟩ = ±1 identify a Bell measurement")):
    ax.text(1.0, y, "• " + t, fontsize=5.1, color=INK, va="center")
# ================= c (bottom left): closed loop
ax = canvas(0.0, 0.02, 0.5, 0.49)  # 36 x 30
ax.text(0.5, 28.5, "c", fontweight="bold", fontsize=9); ax.text(2.5, 28.5, "Closed-loop training on the WK_C180 cloud processor (n = 2)", fontsize=6.6, fontweight="bold")
nodes = [(6.5, 21.5, "5 front-ends\nθ, θ ± σd₁, θ ± σd₂"), (18.0, 21.5, "600 circuits\n15 Paulis × 8 random\npreparations × 5"), (29.5, 21.5, "WK_C180\npinned couplers\n20 shots/circuit"),
         (29.5, 7.0, "counts → 2,000\nepisodes per front-end\n(8 distinct circuits)"), (18.0, 7.0, "rewards r(θ ± σd)\n(only feedback)"), (6.5, 7.0, "update θ: ES + Adam\ndecoder: REINFORCE\n+ restarts, annealing")]
for x, y, t in nodes:
    wk = t.startswith("WK")
    box(ax, x - 5.3, y - 3.8, 10.6, 7.6, fc="#eef4fc" if wk else "white", ec=BLUE if wk else GRID, lw=1 if wk else 0.8)
    ax.text(x, y, t, ha="center", va="center", fontsize=5.3)
for p0, p1 in (((11.9, 21.5), (12.6, 21.5)), ((23.4, 21.5), (24.1, 21.5)), ((29.5, 17.6), (29.5, 10.9)), ((24.1, 7.0), (23.4, 7.0)), ((12.6, 7.0), (11.9, 7.0)), ((6.5, 10.9), (6.5, 17.6))):
    arrow(ax, p0, p1, lw=1.0)
ax.text(18.0, 14.2, "per iteration ≈ 7.9 s of processor time\n120–250 iterations per run\nsame preparations for all front-ends\n(common random numbers)", ha="center", va="center", fontsize=5.2, color=MUT)
# ================= d chip calibration
ax = fig.add_axes([0.575, 0.085, 0.17, 0.36])
chip = json.load(open(os.path.join(R, "chip.json")))
fz = np.array([x["fidelity"] for x in chip["double_qubits"] if (x.get("fidelity") or 0) > 0.8])
man = json.load(open(_os.path.join(_RUNS, "PR7_origin", "manifest.json"))); sel = [s["f_cz"] for s in man["sites"][:7]]
ax.hist(fz, bins=np.linspace(0.94, 1.0, 25), color=SURF, edgecolor=MUT, lw=0.4)
for f in sel: ax.axvline(f, color=BLUE, lw=0.8, ymax=0.8)
ax.set_xlim(0.94, 1.0); ax.set_xticks([0.95, 0.97, 0.99]); ax.set_xlabel("CZ fidelity"); ax.set_ylabel("couplers")
ax.text(0.03, 0.97, f"all {len(fz)} couplers", transform=ax.transAxes, fontsize=5, va="top", color=MUT)
ax.text(0.03, 0.89, "7 used at n = 7", transform=ax.transAxes, fontsize=5, va="top", color=BLUE)
ax.set_title("WK_C180 calibration", loc="left"); ax.text(-0.38, 1.05, "d", transform=ax.transAxes, fontweight="bold", fontsize=9)
# ================= e results at a glance
ax = fig.add_axes([0.875, 0.085, 0.115, 0.36])
rows = [("identif., n = 5", 84 / 183), ("identif., n = 6", 167 / 442), ("identif., n = 7", 291 / 1037),
        ("tracking, q = 0.02", 175 / 68), ("tracking, q = 0.1", 325 / 170), ("detection, n = 5", 152 / 53)]
for i, (l, v) in enumerate(rows):
    c = BLUE if v < 1 else MUT
    ax.barh(i, v, color=c, height=0.62); ax.text(max(v, 1) + 0.08, i, f"{v:.2f}", va="center", fontsize=5)
ax.axvline(1, color=INK, ls="--", lw=0.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.2); ax.invert_yaxis()
ax.set_xlim(0, 3.6); ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel("copies or regret\n÷ single-copy limit")
ax.set_title("at a glance", loc="left"); ax.text(-1.15, 1.05, "e", transform=ax.transAxes, fontweight="bold", fontsize=9)
fig.savefig(_os.path.join(_OUT, "fig1_concept.pdf")); fig.savefig(_os.path.join(_OUT, "fig1.png"), dpi=170)
