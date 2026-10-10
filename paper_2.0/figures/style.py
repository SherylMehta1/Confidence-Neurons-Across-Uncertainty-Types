"""Shared style for all paper figures: Okabe-Ito palette, NeurIPS text width, 9 pt type, vector PDF."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = next(p for p in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[2] / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
HERE = Path(__file__).resolve().parent
TEXTWIDTH = 5.5  # inches, NeurIPS single column

# One colour per concept, the same in every figure.
FAM = "#E69F00"      # orange: familiarity / popularity
KNOW = "#009E73"     # bluish green: knowledge
KNOWN = "#0072B2"    # blue: HighlyKnown items
UNKNOWN = "#D55E00"  # vermillion: Unknown items
NONE = "#000000"     # black: no intervention
RAND = "#999999"     # grey: random directions
NOHEDGE = "#CC79A7"  # reddish purple: familiarity direction fitted on never-hedged items
MODEL = {"llama3.1-8b": ("#0072B2", "-", "o", "Llama-3.1-8B"),
         "gemma-2-9b": ("#009E73", "--", "s", "Gemma-2-9B"),
         "gemma-2-2b": ("#56B4E9", "-.", "^", "Gemma-2-2B"),
         "olmo-3-1025-7b": ("#CC79A7", ":", "D", "OLMo-3-7B")}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.02, "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.wspace": 0.04,
    "axes.titlesize": 9, "font.family": "DejaVu Sans",
})


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="right")


def save(fig, name, inputs, plotted):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=200)
    prov = {"inputs": {str(p): os.path.getmtime(p) for p in inputs}, "plotted": plotted}
    (HERE / f"{name}.provenance.json").write_text(json.dumps(prov, indent=1, default=float), encoding="utf-8")
