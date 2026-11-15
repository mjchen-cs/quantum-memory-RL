import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.8, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "lines.linewidth": 1.1, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
BLUE, LBLUE, ORA, AQUA, YEL, MAG, GRN, VIO, RED = "#2a78d6", "#86b6ef", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, MUT, GRID, SURF = "#262624", "#77776f", "#d9d9d4", "#ebebe6"
def lab(ax, s, x=-0.28, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
def smooth(y, k=5):
    import numpy as np
    y = np.asarray(y, float); return np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])
