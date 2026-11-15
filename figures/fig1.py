import json, os, sys, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
R = os.path.join(_HERE, "..", "hw", "runs"); OUTD = os.path.join(_HERE, "out"); os.makedirs(OUTD, exist_ok=True)
PHOTO = os.path.join(_HERE, "chip_photo.png")  # photograph of the processor package (not distributed; see README)
NAVY, NAVY2, GOLD, TEAL = "#0d1b2e", "#1d3557", "#d9a520", "#1baf7a"
W, H = 7.2, 7.5
fig = plt.figure(figsize=(W, H))
def canvas(x, y, w, h):
    ax = fig.add_axes([x, y, w, h]); ax.set_xlim(0, w * W * 10); ax.set_ylim(0, h * H * 10); ax.axis("off"); return ax
def box(ax, x, y, w, h, fc="white", ec=GRID, lw=0.8, r=0.8, z=1, alpha=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))
def shadowbox(ax, x, y, w, h, fc="white", ec=GRID, lw=0.7, r=0.8):
    box(ax, x + 0.25, y - 0.3, w, h, fc="#000000", ec="none", r=r, z=0.5, alpha=0.08); box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=r)
def arrow(ax, p0, p1, col=INK, lw=0.9, cs="arc3,rad=0", ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw, connectionstyle=cs, zorder=4))
def gate(ax, x, y, t, w=2.2, h=1.6, fs=4.9, fc="white", ec=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=fc, ec=ec, lw=0.6, zorder=5)); ax.text(x, y, t, ha="center", va="center", fontsize=fs, zorder=6)
def head(ax, x, y, l, t, col=INK):
    ax.text(x, y, l, fontweight="bold", fontsize=9, va="baseline"); ax.text(x + 2.0, y, t, fontsize=6.6, fontweight="bold", color=col, va="baseline")
def gradient(ax, x0, x1, y0, y1, c0, c1, z=0):
    g = np.linspace(0, 1, 256)[:, None]; cm = LinearSegmentedColormap.from_list("g", [c0, c1])
    ax.imshow(g, extent=(x0, x1, y0, y1), cmap=cm, aspect="auto", zorder=z, origin="lower")

chip = json.load(open(os.path.join(R, "chip.json")))
man = json.load(open(os.path.join(R, "PR7_origin", "manifest.json"))); sites = man["sites"][:7]
used = {q for s in sites for q in (s["a"], s["b"])}

# ======================= a: processor photo on dark panel
ax = canvas(0.0, 0.655, 0.255, 0.335)       # 18.4 x 25.1
head(ax, 0.3, 23.6, "a", "The processor")
x0, x1, y0, y1 = 0.3, 18.1, 2.2, 22.2
gradient(ax, x0, x1, y0, y1, NAVY, NAVY2, z=0)
ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.0", fc="none", ec=NAVY, lw=0.8, zorder=1))
clip = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.0", transform=ax.transData, fc="none", ec="none")
ax.add_patch(clip); ax.images[0].set_clip_path(clip)
s = 14.2; cx, cy = 9.2, 13.4
if os.path.exists(PHOTO):
    img = mpimg.imread(PHOTO)
    ax.add_patch(FancyBboxPatch((cx - s / 2 + 0.35, cy - s / 2 - 0.45), s, s, boxstyle="round,pad=0,rounding_size=1.0", fc="black", ec="none", alpha=0.35, zorder=2))
    ax.imshow(img, extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2), zorder=3, interpolation="lanczos")
else:
    ax.text(cx, cy, "photograph of the\nprocessor package\n(chip_photo.png)", ha="center", va="center", color="#b9c7da", fontsize=5.5, zorder=3)
ax.set_xlim(0, 18.4); ax.set_ylim(0, 25.1)
ax.text(9.2, 4.6, "Origin Quantum WK_C180", ha="center", fontsize=5.8, color="white", fontweight="bold", zorder=4)
ax.text(9.2, 3.0, "superconducting · 180 qubits · cloud access", ha="center", fontsize=4.6, color="#b9c7da", zorder=4)

# ======================= b: coupling graph with CZ fidelities and the sites used
ax = fig.add_axes([0.275, 0.735, 0.30, 0.215])
def pos(q):
    r, c = divmod(q, 9); return r, c - 0.5 * (r % 2)
E = {}
for d in chip["double_qubits"]:
    a, b = sorted(d["qubits"]); E[(a, b)] = d.get("fidelity") or 0
alive = {int(x["qubit"]) for x in chip["single_qubits"] if x["gate_fidelity"] > 0}
cmap = LinearSegmentedColormap.from_list("cz", ["#f0e2c0", "#86b6ef", "#1d4e9a"])
segs, cols = [], []
for (a, b), f in E.items():
    segs.append([pos(a), pos(b)]); cols.append(cmap(np.clip((f - 0.95) / 0.05, 0, 1)) if f > 0.9 else (0.85, 0.85, 0.83, 1))
ax.add_collection(LineCollection(segs, colors=cols, lw=1.5, zorder=1, capstyle="round"))
P = np.array([pos(q) for q in range(180)])
al = np.array([q in alive for q in range(180)])
ax.scatter(P[al, 0], P[al, 1], s=5, c="white", edgecolors="#55606e", linewidths=0.4, zorder=2)
ax.scatter(P[~al, 0], P[~al, 1], s=4, marker="x", c="#b0b0aa", linewidths=0.5, zorder=2)
for i, st in enumerate(sites):
    pa, pb = pos(st["a"]), pos(st["b"])
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=ORA, lw=3.0, zorder=3, solid_capstyle="round")
    for p, col in ((pa, ORA), (pb, ORA)): ax.scatter(*p, s=16, c=col, edgecolors="white", linewidths=0.5, zorder=4)
ax.set_xlim(-1, 20); ax.set_ylim(-1.4, 9.0); ax.set_aspect("equal"); ax.axis("off")
ax.text(-1, 9.9, "b", fontweight="bold", fontsize=9); ax.text(0.5, 9.9, "Coupling graph and the 7 sites used", fontsize=6.6, fontweight="bold")
cax = fig.add_axes([0.30, 0.752, 0.12, 0.010])
cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.95, 1.0)), cax=cax, orientation="horizontal")
cb.set_ticks([0.95, 0.975, 1.0]); cb.ax.tick_params(labelsize=4.8, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label("CZ fidelity", fontsize=5, labelpad=1)
lx = fig.add_axes([0.44, 0.735, 0.14, 0.035]); lx.axis("off"); lx.set_xlim(0, 10); lx.set_ylim(0, 3)
lx.plot([0.2, 1.4], [2.2, 2.2], color=ORA, lw=3, solid_capstyle="round"); lx.text(1.8, 2.2, "site (Aᵢ, Bᵢ), n = 7", va="center", fontsize=4.8)
lx.scatter([0.8], [0.7], s=4, marker="x", c="#b0b0aa", linewidths=0.5); lx.text(1.8, 0.7, "unavailable qubit", va="center", fontsize=4.8, color=MUT)

# ======================= c: task
ax = canvas(0.595, 0.655, 0.405, 0.335)     # 29.2 x 25.1
head(ax, 0.3, 23.6, "c", "Task: name a hidden Pauli from pairs of copies")
box(ax, 0.3, 3.0, 13.6, 18.9, fc="#f4f6f9", ec=GRID)
ax.text(7.1, 20.2, "quantum data source", ha="center", fontsize=5.4, color=MUT)
ax.text(7.1, 18.3, r"hidden $P\in\{I,X,Y,Z\}^{\otimes n}\setminus I$", ha="center", fontsize=5.5)
for j, y in enumerate((15.3, 12.3, 9.3)):
    for x0_ in (1.1, 7.6):
        ax.add_patch(FancyBboxPatch((x0_, y - 1.0), 5.2, 2.0, boxstyle="round,pad=0,rounding_size=0.4", fc="white", ec=BLUE, lw=0.8, zorder=3))
        ax.text(x0_ + 2.6, y, r"$\rho_{s_jP}$", ha="center", va="center", fontsize=5.6, zorder=4)
    ax.text(6.95, y, "⊗", ha="center", va="center", fontsize=6)
ax.text(7.1, 7.6, "⋮", ha="center", fontsize=6, color=MUT)
ax.text(7.1, 4.8, r"$\rho_{sP}=(I+sP)/2^n$; sign $s_j=\pm1$" + "\nshared within a pair: one copy\nalone carries no information on P", ha="center", va="center", fontsize=4.7, color=MUT)
arrow(ax, (14.1, 12.3), (16.4, 12.3), lw=1.1)
gradient(ax, 16.6, 23.4, 9.0, 15.6, NAVY2, NAVY, z=2)
ax.add_patch(FancyBboxPatch((16.6, 9.0), 6.8, 6.6, boxstyle="round,pad=0,rounding_size=0.8", fc="none", ec=NAVY, lw=0.8, zorder=3))
cl = FancyBboxPatch((16.6, 9.0), 6.8, 6.6, boxstyle="round,pad=0,rounding_size=0.8", transform=ax.transData, fc="none", ec="none"); ax.add_patch(cl); ax.images[-1].set_clip_path(cl)
ax.set_xlim(0, 29.2); ax.set_ylim(0, 25.1)
ax.text(20.0, 13.6, "agent", ha="center", fontsize=6.5, fontweight="bold", color="white", zorder=4)
ax.text(20.0, 11.0, "64 parameters\nat every n", ha="center", fontsize=5.0, color="#b9c7da", va="center", zorder=4)
arrow(ax, (23.6, 12.3), (25.4, 12.3), lw=1.1); ax.text(25.7, 12.3, r"$\hat P$", va="center", fontsize=6.5)
ax.text(24.7, 6.6, "reward\n" + r"$r=\mathbf{1}\{\hat P=P\}$", ha="center", va="center", fontsize=5.4, color=ORA)
arrow(ax, (24.7, 5.0), (20.0, 8.8), col=ORA, cs="arc3,rad=-0.3")
ax.text(21.8, 2.3, "the agent never sees P;\nthe reward is its only feedback", fontsize=4.8, color=ORA, ha="center", va="center")

# ======================= d: agent
ax = canvas(0.0, 0.335, 0.575, 0.30)      # 41.4 x 22.5
head(ax, 0.3, 21.0, "d", "Agent: site-local two-copy front-end + product decoder")
ya, yb = 15.4, 9.4
for y, l in ((ya, r"$A_i$"), (yb, r"$B_i$")):
    ax.plot([1.8, 25.4], [y, y], color=INK, lw=0.6, zorder=2); ax.text(1.0, y, l, va="center", ha="center", fontsize=6)
box(ax, 2.6, 7.6, 20.4, 9.6, fc="#eef4fc", ec=BLUE, lw=0.7, r=0.6)
for y in (ya, yb):
    for x, g in zip((4.4, 7.0, 9.6), ("RZ", "RY", "RZ")): gate(ax, x, y, g, fc="white", ec=BLUE)
    for x, g in zip((16.2, 18.8, 21.4), ("RZ", "RY", "RZ")): gate(ax, x, y, g, fc="white", ec=BLUE)
    gate(ax, 24.3, y, "◠", w=1.8, fc="#fff4e8", ec=ORA)
ax.plot([12.9, 12.9], [yb, ya], color=INK, lw=0.7, zorder=3); ax.add_patch(Circle((12.9, ya), 0.38, color=INK, zorder=6)); gate(ax, 12.9, yb, "RX(φ)", w=3.0, fc="white", ec=BLUE)
ax.text(12.8, 6.3, r"$V(\theta)$: 13 angles, identical on every pair $(A_i,B_i)$, $i=1,\dots,n$", ha="center", fontsize=5.0, color=BLUE)
arrow(ax, (25.6, 12.4), (28.0, 12.4)); ax.text(26.8, 13.3, r"$\beta_i$", ha="center", fontsize=5.8)
shadowbox(ax, 28.2, 6.0, 12.7, 12.4, ec=INK, lw=0.7)
ax.text(34.55, 16.8, "decoder", ha="center", fontsize=6, fontweight="bold")
ax.text(34.55, 15.3, "3 tables $h_k$ (4×4) + 3 weights", ha="center", fontsize=4.7, color=MUT)
ax.text(34.55, 11.6, r"$\mathrm{score}(a)=\sum_j\sum_k c_k\prod_i h_k[a_i,\beta_{ji}]$", ha="center", va="center", fontsize=5.0)
ax.text(34.55, 8.4, r"$\hat P\sim\mathrm{softmax}(\mathrm{score})$", ha="center", va="center", fontsize=5.2)
for i, (t, b, c) in enumerate((("Deploy", "same 13 angles at any n; decoder from response tables + fitted κ", BLUE),
                                ("Prune", "snap angles to Clifford values → one CZ per site", TEAL),
                                ("Interpret", "response tables g[σ,β] = ±1 identify a Bell measurement", ORA))):
    y = 4.1 - i * 1.55
    ax.add_patch(FancyBboxPatch((1.0, y - 0.6), 5.0, 1.2, boxstyle="round,pad=0,rounding_size=0.5", fc=c, ec="none", zorder=3))
    ax.text(3.5, y, t, ha="center", va="center", fontsize=4.9, color="white", fontweight="bold", zorder=4)
    ax.text(6.6, y, b, va="center", fontsize=5.0, color=INK)

# ======================= e: closed-loop training cycle
ax = canvas(0.595, 0.335, 0.405, 0.30)    # 29.2 x 22.5
head(ax, 0.3, 21.0, "e", "Closed-loop training on the chip (n = 2)")
cx, cy, rx, ry = 14.6, 10.6, 9.6, 6.6
ang = np.deg2rad([90, 30, -30, -90, -150, 150])
labels = ["5 front-ends\nθ, θ ± σd₁, θ ± σd₂", "600 circuits\n15 Paulis × 40", "WK_C180\n20 shots/circuit", "counts → 2,000\nepisodes / front-end", "rewards\nr(θ ± σd)", "ES + Adam (θ)\nREINFORCE (dec.)"]
E_ = lambda a: (cx + rx * np.cos(a), cy + ry * np.sin(a))
th = np.linspace(0, 2 * np.pi, 200); ax.plot(cx + rx * np.cos(th), cy + ry * np.sin(th), color="#e4e8ee", lw=2.2, zorder=1)
for k in range(6):
    a0, a1 = ang[k] - 0.42, ang[k] - (np.pi / 3) + 0.42
    arrow(ax, E_(a0), E_(a1), col=BLUE, lw=1.1, cs="arc3,rad=-0.12", ms=7)
bw, bh = 7.6, 3.0
for k, (a, t) in enumerate(zip(ang, labels)):
    x, y = E_(a)
    isq = t.startswith("WK")
    if isq:
        gradient(ax, x - bw / 2, x + bw / 2, y - bh / 2, y + bh / 2, NAVY2, NAVY, z=2)
        cl = FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.6", transform=ax.transData, fc="none", ec="none"); ax.add_patch(cl); ax.images[-1].set_clip_path(cl)
        ax.set_xlim(0, 29.2); ax.set_ylim(0, 22.5)
    else:
        shadowbox(ax, x - bw / 2, y - bh / 2, bw, bh, ec=GRID, r=0.6)
    ax.text(x, y, t, ha="center", va="center", fontsize=4.6, color="white" if isq else INK, zorder=6)
ax.text(cx, cy + 0.9, "≈ 7.9 s", ha="center", va="center", fontsize=8, fontweight="bold", color=NAVY2)
ax.text(cx, cy - 0.9, "processor time\nper iteration", ha="center", va="center", fontsize=4.8, color=MUT)
ax.text(cx, 0.6, "2,469 iterations, 1.5 million circuits on the chip in total", ha="center", fontsize=4.6, color=MUT)

# ======================= f: CZ fidelity
ax = fig.add_axes([0.065, 0.07, 0.22, 0.20])
fz = np.array([v for v in E.values() if v > 0.9])
ax.hist(fz, bins=np.linspace(0.94, 1.0, 25), color="#c9d7ea", edgecolor="white", lw=0.4)
for st in sites: ax.axvline(st["f_cz"], color=ORA, lw=0.8, ymax=0.82)
ax.set_xlim(0.94, 1.0); ax.set_xticks([0.95, 0.97, 0.99]); ax.set_xlabel("CZ fidelity"); ax.set_ylabel("couplers"); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); ax.set_ylim(0, 25)
ax.text(0.03, 0.97, f"{len(fz)} calibrated couplers", transform=ax.transAxes, fontsize=5, va="top", color=MUT)
ax.text(0.03, 0.88, "7 sites used at n = 7", transform=ax.transAxes, fontsize=5, va="top", color=ORA)
lab(ax, "f", x=-0.3, y=1.04)
# ======================= g: single-qubit calibration
ax = fig.add_axes([0.37, 0.07, 0.22, 0.20])
sq = [x for x in chip["single_qubits"] if x["gate_fidelity"] > 0]
t1 = np.array([x["t1"] for x in sq]); ro = np.array([x["readout_fidelity"] for x in sq]); u = np.array([int(x["qubit"]) in used for x in sq])
ax.scatter(t1[~u], ro[~u], s=5, c="#c9d7ea", edgecolors="#7f97b8", linewidths=0.3, label="all qubits")
ax.scatter(t1[u], ro[u], s=12, c=ORA, edgecolors="white", linewidths=0.4, label="used (n = 7)", zorder=3)
ax.set_xlabel(r"$T_1$ (µs)"); ax.set_ylabel("readout fidelity"); ax.set_ylim(max(0.0, ro.min() - 0.03), 1.0)
ax.legend(loc="center right", bbox_to_anchor=(1.02, 0.36), frameon=False, handletextpad=0.2, borderaxespad=0.2)
lab(ax, "g", x=-0.3, y=1.04)
# ======================= h: results at a glance
ax = fig.add_axes([0.765, 0.07, 0.22, 0.20])
rows = [("identification, n = 5", 84 / 183), ("identification, n = 6", 167 / 442), ("identification, n = 7", 291 / 1037),
        ("drift tracking, q = 0.02", 175 / 68), ("drift tracking, q = 0.1", 325 / 170), ("detection, n = 5", 152 / 53)]
ax.axvspan(0.1, 1, color="#eef4fc", zorder=0); ax.axvline(1, color=INK, ls="--", lw=0.7)
for i, (l, v) in enumerate(rows):
    c = BLUE if v < 1 else "#8a8a82"
    ax.plot([1, v], [i, i], color=c, lw=1.4, solid_capstyle="round"); ax.scatter(v, i, s=22, c=c, edgecolors="white", linewidths=0.5, zorder=3)
    ax.text(v * (0.78 if v < 1 else 1.18), i, f"{v:.2f}", va="center", ha="right" if v < 1 else "left", fontsize=5)
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.set_xlim(0.11, 6); ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=5.2); ax.invert_yaxis(); ax.set_ylim(5.6, -0.9)
ax.text(0.17, 5.45, "beats the\nsingle-copy\nlimit", fontsize=4.6, color=BLUE, va="bottom")
ax.set_xlabel("copies or regret ÷ single-copy limit")
lab(ax, "h", x=-0.95, y=1.04)
fig.savefig(os.path.join(OUTD, "fig1_concept.pdf")); fig.savefig(os.path.join(OUTD, "fig1.png"), dpi=200)
