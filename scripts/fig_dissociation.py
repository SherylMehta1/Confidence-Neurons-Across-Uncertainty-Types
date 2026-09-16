"""The dissociation figure: representation x readout x direction, with per-pair bootstrap CIs.

Rows of make_paper_figures.py's fig6 (rank-1 direction vs random vs projection-matched vs the
full component set), parameterized by circuit directory and extended with an entropy-recovery
panel beside the hedge-log-odds one. If the component set and the rank-1 direction, read out
as hedge log-odds and as entropy, all tell one story, that supports "one signal, two readouts";
cells that differ are evidence against it.

Real-set bars (rank-1 direction, full set) carry percentile-bootstrap 95% CIs over held-out
pairs (10k resamples, seed 0 -- the same estimator as stats_upgrades.boot_ci, replicated here
because that module runs its whole analysis at import). Null bars (random unit directions,
projection-matched random directions) keep the spread across null sets, which is what a null
distribution's whisker should show.

Writes paper/figures/<name>.pdf and paper/figures/<name>_cells.csv (every plotted mean and CI,
so captions can cite exact numbers). Refuses to overwrite an existing figure without --overwrite.

  python scripts/fig_dissociation.py --circuit-dir results/circuit_familiarity_v2 --name fig6_familiarity
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, AMBER, TEAL, GREY, PURPLE = "#3B4FA8", "#D97706", "#0E7C86", "#6B7280", "#9A6FB0"
DIRECTIONS = (("uncertain_to_control", "switch hedge ON\n(inject unknown)"),
              ("control_to_uncertain", "switch hedge OFF\n(remove unknown)"))
READOUTS = (("logodds_rec", "hedge log-odds recovery"), ("entropy_rec", "entropy recovery"))
B, SEED = 10_000, 0


def boot_ci(v, rng):
    v = np.asarray(v, dtype=float)
    idx = rng.integers(0, len(v), (B, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])


def real_bar(df, setname, direc, metric, rng):
    v = df[(df["set"] == setname) & (df.direction == direc)][metric].to_numpy()
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    lo, hi = boot_ci(v, rng)
    return float(v.mean()), float(lo), float(hi), len(v)


def null_bar(df, prefix, direc, metric):
    m = df[df["set"].str.startswith(prefix) & (df.direction == direc)].groupby("set")[metric].mean()
    if len(m) == 0:
        return np.nan, np.nan, np.nan, 0
    half = 1.96 * m.std(ddof=1)
    return float(m.mean()), float(m.mean() - half), float(m.mean() + half), len(m)


def build(circuit_dir, primary=BLUE):
    dp = pd.read_csv(os.path.join(circuit_dir, "direction_patch_u2c_null.csv"))
    hm = pd.read_csv(os.path.join(circuit_dir, "direction_patch_u2c_hedgematched.csv"))
    fa = pd.read_csv(os.path.join(circuit_dir, "faithfulness.csv"))
    rng = np.random.default_rng(SEED)
    series = [("rank-1 direction", AMBER, lambda d, m: real_bar(dp, "direction", d, m, rng)),
              ("random unit directions", GREY, lambda d, m: null_bar(dp, "random", d, m)),
              ("projection-matched\nrandom directions", PURPLE, lambda d, m: null_bar(hm, "hedgematched", d, m)),
              ("full set (20h+100n)", primary, lambda d, m: real_bar(fa, "circuit", d, m, rng))]
    cells = []
    for metric, _ in READOUTS:
        for direc, _ in DIRECTIONS:
            for label, _, fn in series:
                mean, lo, hi, n = fn(direc, metric)
                cells.append(dict(readout=metric, direction=direc, series=label.replace("\n", " "),
                                  mean=mean, ci_lo=lo, ci_hi=hi, n=n,
                                  ci_type="bootstrap_pairs" if n and "random" not in label else "normal_across_sets"))
    return series, pd.DataFrame(cells)


def draw(series, cells, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=False)
    w = 0.2
    for ax, (metric, ylabel) in zip(axes, READOUTS):
        for j, (label, col, _) in enumerate(series):
            xs, ms, lo, hi = [], [], [], []
            for gi, (direc, _) in enumerate(DIRECTIONS):
                c = cells[(cells.readout == metric) & (cells.direction == direc)
                          & (cells.series == label.replace("\n", " "))].iloc[0]
                if np.isnan(c["mean"]):
                    continue
                xs.append(gi + (j - 1.5) * w); ms.append(c["mean"]); lo.append(c["ci_lo"]); hi.append(c["ci_hi"])
            if not xs:
                continue
            ms, lo, hi = map(np.array, (ms, lo, hi))
            ax.bar(xs, ms, width=w, color=col, alpha=0.6 if col == GREY else 1.0,
                   yerr=np.vstack([ms - lo, hi - ms]), capsize=2.5, error_kw=dict(lw=0.9), label=label)
            for xi, mi, li, hi_ in zip(xs, ms, lo, hi):
                ax.text(xi, hi_ + 0.03 if mi >= 0 else li - 0.10, f"{mi:.2f}", ha="center", fontsize=6.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels([lab for _, lab in DIRECTIONS], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        allhi = cells[cells.readout == metric].ci_hi.max(); alllo = cells[cells.readout == metric].ci_lo.min()
        ax.set_ylim(min(-0.35, alllo - 0.15), max(1.25, allhi + 0.2))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, fontsize=8, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit-dir", default="results/circuit_familiarity_v2")
    ap.add_argument("--name", default="fig6_familiarity")
    ap.add_argument("--primary-color", default=BLUE)
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    pdf = os.path.join(args.out_dir, f"{args.name}.pdf")
    if os.path.exists(pdf) and not args.overwrite:
        raise SystemExit(f"{pdf} exists; pass --overwrite to replace it")
    os.makedirs(args.out_dir, exist_ok=True)
    series, cells = build(args.circuit_dir, args.primary_color)
    fig = draw(series, cells)
    fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
    cells.to_csv(os.path.join(args.out_dir, f"{args.name}_cells.csv"), index=False)
    print(f"{args.name} written from {args.circuit_dir}")
    real = cells[cells.ci_type == "bootstrap_pairs"]
    print(real.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
