"""X2 figure: abstention against within-relation popularity decile, inside knowledge bands. Reads results/x2/gradient.csv only.
Writes results/x2/fig_x2_gradient.{pdf,png}. Same colours per concept as the paper's figures."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "paper_2.0/figures"))
from style import KNOWN, UNKNOWN, panel_label, plt  # noqa: E402

G = pd.read_csv(REPO / "results/x2/gradient.csv")
BANDS = [("0 of 10 correct", UNKNOWN, "s", "--"), ("1 to 8 of 10", "#666666", "^", ":"), ("9 or 10 of 10", KNOWN, "o", "-")]
fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True)
for a, (y, t) in zip(ax, (("both", "Two judges, 96-token replies"), ("regex24", "Regex, 20 short replies"))):
    for i, (b, c, m, ls) in enumerate(BANDS):
        g = G[(G.outcome == y) & (G.band == b)].sort_values("decile")
        x = g.decile + (i - 1) * 0.12
        a.errorbar(x, g["mean"], yerr=[g["mean"] - g.lo, g.hi - g["mean"]], color=c, marker=m, ls=ls, lw=1.3, ms=3.8, capsize=0, elinewidth=0.8, label=b)
    a.set_title(t); a.set_xticks(range(1, 11)); a.set_xlabel("popularity decile within relation\n(1 = least viewed)"); a.set_ylim(-0.02, 1.0)
    a.grid(axis="y", lw=0.4, color="#dddddd"); a.set_axisbelow(True)
ax[0].set_ylabel("share of “I don't know”")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, title="forced answers correct", title_fontsize=8, ncol=3, loc="outside lower center", frameon=False)
panel_label(ax[0], "a"); panel_label(ax[1], "b")
for e in ("pdf", "png"):
    fig.savefig(REPO / f"results/x2/fig_x2_gradient.{e}", dpi=200)
