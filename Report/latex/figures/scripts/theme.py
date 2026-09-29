"""SwarmFlow viewer palette (src/swarmflow/web/style.css and app.js) for report figures."""
import matplotlib.pyplot as plt

BG = "#0c1115"
SURFACE = "#11191f"
PANEL = "#0e161c"
LINE = "#26323a"
GRID = "#20323e"
TEXT = "#e5edee"
MUTED = "#889ba4"
ACCENT = "#79d9bf"
AMBER = "#e7bd79"
BLUE = "#9fb7ff"
RED = "#ee9389"
DRONE = "#83ddc7"
OBJECT = "#dc918b"
STAGES = [ACCENT, AMBER, BLUE, RED]
DRONES = ["#79d9bf", "#e7bd79", "#9fb7ff", "#ee9389", "#b8e986", "#f5a3d0", "#7fd0f0", "#d7c7ff"]
MONO = ["Menlo", "DejaVu Sans Mono", "monospace"]


def apply():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "grid.color": GRID, "grid.alpha": 1.0, "axes.grid": True,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.facecolor": SURFACE, "legend.edgecolor": LINE, "legend.labelcolor": TEXT,
        "savefig.dpi": 220, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    })


def apply_print():
    """Print-friendly style for graphs and diagrams (white background)."""
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.titlesize": 10, "savefig.dpi": 220, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})


PRINT_DRONES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
PRINT_STAGES = ["#1b9e77", "#d95f02", "#7570b3"]


def mono(ax, text, x, y, color=MUTED, size=7, **kw):
    ax.text(x, y, text, color=color, fontsize=size, family=MONO, transform=ax.transAxes, **kw)
