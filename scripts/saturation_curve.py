"""
L2e saturation: how the recovery estimate behaves as the held-out evaluation set grows.

No GPU. circuit_faithfulness.py / circuit_direction_patch.py take pairs[:limit] and a pair's
recovery does not depend on how many other pairs were evaluated, so a --limit N run is exactly the
first N rows of a --limit M run for N <= M. Verified on the committed v3 files: the --limit 60 run
reproduces the first 60 pairs of the --limit 141 run to 0.00e+00, random sets included. The whole
sweep is therefore recoverable from the largest committed run, and re-running the model for each N
would only reproduce numbers already on disk.

Two curves per series, because they answer different questions:
  prefix      mean over pairs[:n] -- exactly what `--limit n` prints, deterministic, one path
  subsample   mean over --n-draws random n-subsets, with a percentile band -- the sampling
              distribution of the estimate at that n, which is what "has it saturated" actually asks
A flat prefix curve alone can look stable by luck; the band is what shows the estimate tightening.

Writes <circuit-dir>/saturation_curve.csv (one row per n x cell x direction x metric, with a
`recovery` column equal to the prefix mean so the file reads as a plain recovery-vs-n curve) and
saturation_summary.txt. Refuses to overwrite without --overwrite.

  python scripts/saturation_curve.py --circuit-dir results/circuit_familiarity_v3
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse

import numpy as np
import pandas as pd

from _common import REPO_ROOT, guard_output

SERIES = (("circuit", "faithfulness_limit141.csv", "circuit"),
          ("direction", "direction_patch_limit141.csv", "direction"))
METRICS = ("logodds_rec", "entropy_rec")
DIRECTIONS = ("control_to_uncertain", "uncertain_to_control")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit-dir", default="results/circuit_familiarity_v3")
    ap.add_argument("--grid", default="10,20,30,40,50,60,80,100,120,141")
    ap.add_argument("--n-draws", type=int, default=2000, help="random n-subsets per grid point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    d = REPO_ROOT / args.circuit_dir
    out_csv = d / "saturation_curve.csv"
    guard_output(out_csv, args.overwrite)

    rng = np.random.default_rng(args.seed)
    rows = []
    for cell, fname, setname in SERIES:
        p = d / fname
        if not p.exists():
            print(f"skip {cell}: {fname} not present")
            continue
        df = pd.read_csv(p)
        sub = df[df["set"] == setname]
        for direction in DIRECTIONS:
            for metric in METRICS:
                s = sub[sub.direction == direction].sort_values("pair")
                v = s[metric].to_numpy(float)
                if not len(v):
                    continue
                grid = [n for n in (int(x) for x in args.grid.split(",")) if n <= len(v)]
                if grid and grid[-1] != len(v):
                    grid.append(len(v))
                for n in grid:
                    draws = np.array([rng.choice(v, n, replace=False).mean() for _ in range(args.n_draws)]) \
                        if n < len(v) else np.array([v.mean()])
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    rows.append(dict(n=n, cell=cell, direction=direction, metric=metric,
                                     recovery=float(v[:n].mean()),          # == what `--limit n` prints
                                     subsample_mean=float(draws.mean()), subsample_lo=float(lo),
                                     subsample_hi=float(hi), subsample_sd=float(draws.std(ddof=1)) if len(draws) > 1 else 0.0,
                                     ci_width=float(hi - lo), n_total=len(v), n_draws=len(draws)))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    lines = [f"Saturation -- {args.circuit_dir}; prefix curve == `--limit n`; band over {args.n_draws} random n-subsets"]
    for cell, _, _ in SERIES:
        for direction in DIRECTIONS:
            g = out[(out.cell == cell) & (out.direction == direction) & (out.metric == "logodds_rec")].sort_values("n")
            if not len(g):
                continue
            lines.append(f"  [{cell}] {direction} hedge log-odds recovery:")
            lines.append("    " + "  ".join(f"n={int(r.n)}:{r.recovery:+.2f}" for r in g.itertuples()))
            lines.append("    band width " + "  ".join(f"n={int(r.n)}:{r.ci_width:.2f}" for r in g.itertuples()))
            full = g.iloc[-1]
            drift = max(abs(r.recovery - full.recovery) for r in g.itertuples() if r.n >= 40) if (g.n >= 40).any() else float("nan")
            lines.append(f"    max |prefix mean - full-set mean| for n>=40: {drift:.3f}; "
                         f"band width {g.iloc[0].ci_width:.2f} (n={int(g.iloc[0].n)}) -> {full.ci_width:.2f} (n={int(full.n)})")
    summary = "\n".join(lines)
    print(summary)
    (d / "saturation_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"wrote {out_csv.relative_to(REPO_ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
