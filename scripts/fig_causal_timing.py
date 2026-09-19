"""Causal-timing figure: recovery vs layer (resid patched at layers 0..l at the suffix positions), both
directions, hedge log-odds beside entropy, with per-pair bootstrap bands, the random-set null floor
from faithfulness.csv, the 0.7 circuit threshold, and the rank-1 direction-patching window shaded
as the reference the curve should corroborate. Writes paper/figures/<name>.pdf; refuses to
overwrite without --overwrite.

  python scripts/fig_causal_timing.py --circuit-dir results/circuit_familiarity_v2 --tag up_to_suffix --window 13-19
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, AMBER, GREY = "#3B4FA8", "#D97706", "#6B7280"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit-dir", default="results/circuit_familiarity_v2")
    ap.add_argument("--tag", default="up_to_suffix", help="causal_timing_<tag>_summary.csv to plot")
    ap.add_argument("--window", default="13-19", help="direction-patching layer window to shade")
    ap.add_argument("--name", default="fig_causal_timing")
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    pdf = os.path.join(args.out_dir, f"{args.name}.pdf")
    if os.path.exists(pdf) and not args.overwrite:
        raise SystemExit(f"{pdf} exists; pass --overwrite to replace it")

    s = pd.read_csv(os.path.join(args.circuit_dir, f"causal_timing_{args.tag}_summary.csv"))
    fa = pd.read_csv(os.path.join(args.circuit_dir, "faithfulness.csv"))
    null = fa[fa["set"].str.startswith("random") & (fa.direction == "control_to_uncertain")].groupby("set").logodds_rec.mean()
    w0, w1 = (int(x) for x in args.window.split("-"))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    panels = (("logodds_rec_mean", "ci_lo", "ci_hi", "hedge log-odds recovery", null.mean()),
              ("entropy_rec_mean", "entropy_ci_lo", "entropy_ci_hi", "entropy recovery", None))
    for ax, (m, lo, hi, ylabel, floor) in zip(axes, panels):
        ax.axvspan(w0 - 0.5, w1 + 0.5, color=GREY, alpha=0.12, lw=0, label=f"rank-1 direction window (L{w0}-{w1})")
        ax.axhline(0.7, color=GREY, lw=0.8, ls=":", label="0.7 circuit threshold")
        if floor is not None:
            ax.axhline(floor, color=GREY, lw=0.8, ls="--", label=f"random-set null ({floor:+.2f})")
        for direc, col, lab in (("uncertain_to_control", BLUE, "inject unknown (u\u2192c)"),
                                ("control_to_uncertain", AMBER, "remove unknown (c\u2192u)")):
            g = s[s.direction == direc].sort_values("layer")
            ax.plot(g.layer, g[m], color=col, lw=1.6, marker="o", ms=2.5, label=lab)
            ax.fill_between(g.layer, g[lo], g[hi], color=col, alpha=0.18, lw=0)
            first = g[g[m] > 0.7].layer.min()
            if pd.notna(first):
                ax.axvline(first, color=col, lw=0.7, ls="-.", alpha=0.7)
        ax.set_xlabel("residual stream patched at layers 0..l (suffix positions)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(s.layer.min() - 0.5, s.layer.max() + 0.5)
        ax.set_ylim(-0.12, 1.18)
    axes[0].legend(fontsize=6.6, loc="upper left")
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
    print(f"{args.name} written from {args.circuit_dir} ({args.tag})")


if __name__ == "__main__":
    main()
