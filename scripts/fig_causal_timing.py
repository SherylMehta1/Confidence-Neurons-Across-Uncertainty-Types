"""Causal-timing figure: log-odds recovery vs layer (resid patched at layers 0..l), both directions,
with per-pair bootstrap bands, the random-set null floor from faithfulness.csv, the 0.7 circuit
threshold, and the rank-1 direction-patching window shaded as the reference the curve should
corroborate. Writes paper/figures/<name>.pdf; refuses to overwrite without --overwrite.

  python scripts/fig_causal_timing.py --circuit-dir results/circuit_familiarity_v2 --window 13-19
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
    ap.add_argument("--mode", default="up_to")
    ap.add_argument("--window", default="13-19", help="direction-patching layer window to shade")
    ap.add_argument("--name", default="fig_causal_timing")
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    pdf = os.path.join(args.out_dir, f"{args.name}.pdf")
    if os.path.exists(pdf) and not args.overwrite:
        raise SystemExit(f"{pdf} exists; pass --overwrite to replace it")

    s = pd.read_csv(os.path.join(args.circuit_dir, f"causal_timing_{args.mode}_summary.csv"))
    fa = pd.read_csv(os.path.join(args.circuit_dir, "faithfulness.csv"))
    null = fa[fa["set"].str.startswith("random") & (fa.direction == "control_to_uncertain")].groupby("set").logodds_rec.mean()
    w0, w1 = (int(x) for x in args.window.split("-"))

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.axvspan(w0 - 0.5, w1 + 0.5, color=GREY, alpha=0.12, lw=0, label=f"rank-1 direction window (L{w0}-{w1})")
    ax.axhline(0.7, color=GREY, lw=0.8, ls=":", label="0.7 circuit threshold")
    ax.axhline(null.mean(), color=GREY, lw=0.8, ls="--", label=f"random-set null ({null.mean():+.2f})")
    for direc, col, lab in (("uncertain_to_control", BLUE, "inject unknown (u→c)"),
                            ("control_to_uncertain", AMBER, "remove unknown (c→u)")):
        g = s[s.direction == direc].sort_values("layer")
        ax.plot(g.layer, g.logodds_rec_mean, color=col, lw=1.6, marker="o", ms=2.5, label=lab)
        ax.fill_between(g.layer, g.ci_lo, g.ci_hi, color=col, alpha=0.18, lw=0)
    ax.set_xlabel("residual stream patched at layers 0..l" if args.mode == "up_to" else "residual stream patched at layers l..L-1")
    ax.set_ylabel("hedge log-odds recovery")
    ax.set_xlim(s.layer.min() - 0.5, s.layer.max() + 0.5)
    ax.set_ylim(min(-0.1, s.ci_lo.min() - 0.05), max(1.1, s.ci_hi.max() + 0.05))
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
    print(f"{args.name} written from {args.circuit_dir} (mode={args.mode})")


if __name__ == "__main__":
    main()
