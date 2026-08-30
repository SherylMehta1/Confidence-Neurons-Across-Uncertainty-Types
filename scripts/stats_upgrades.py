"""Post-review statistical upgrades (30 Aug 2026), from the committed faithfulness CSVs.

1. Percentile-bootstrap 95% CIs (10k resamples, seed 0) for circuit logodds recovery,
   both directions, all three cells -- compared against the normal 1.96*sd/sqrt(n) CIs.
2. Clip fractions: share of per-pair recoveries at the [-1, 2] clip bounds.
3. Paired sign-flip permutation test (20k flips) of the claim-7 entropy asymmetry:
   entropy_rec(uncertain_to_control) - entropy_rec(control_to_uncertain) per pair,
   within each cell, with a bootstrap CI on the mean difference.

Run from the repo root: python scripts/stats_upgrades.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
CELLS = {
    "Llama familiarity": "results/circuit_familiarity/faithfulness.csv",
    "Llama contested": "results/circuit_conflict/faithfulness.csv",
    "Qwen contested": "results/second_model/qwen25_7b_instruct/results/circuit_conflict/faithfulness.csv",
}
B = 10_000
P = 20_000


def boot_ci(v):
    idx = rng.integers(0, len(v), (B, len(v)))
    means = v[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


print("=== 1+2. circuit logodds recovery: normal vs bootstrap CI, clip fractions ===")
for name, path in CELLS.items():
    df = pd.read_csv(path)
    for d in ["control_to_uncertain", "uncertain_to_control"]:
        v = df[(df["set"] == "circuit") & (df["direction"] == d)]["logodds_rec"].to_numpy()
        m, sd, n = v.mean(), v.std(ddof=1), len(v)
        lo_n, hi_n = m - 1.96 * sd / np.sqrt(n), m + 1.96 * sd / np.sqrt(n)
        lo_b, hi_b = boot_ci(v)
        clip = np.mean((v <= -1 + 1e-9) | (v >= 2 - 1e-9))
        print(f"{name:18s} {d:22s} n={n} mean={m:.3f} "
              f"normal=[{lo_n:.3f},{hi_n:.3f}] boot=[{lo_b:.3f},{hi_b:.3f}] clip={clip:.1%}")

print("\n=== 3. claim-7 entropy asymmetry: paired sign-flip permutation ===")
for name, path in CELLS.items():
    df = pd.read_csv(path)
    c = df[df["set"] == "circuit"].pivot(index="pair", columns="direction", values="entropy_rec").dropna()
    diff = (c["uncertain_to_control"] - c["control_to_uncertain"]).to_numpy()
    obs = diff.mean()
    signs = rng.choice([-1, 1], (P, len(diff)))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (P + 1)
    lo, hi = boot_ci(diff)
    print(f"{name:18s} n={len(diff)} mean c2u={c['control_to_uncertain'].mean():+.3f} "
          f"u2c={c['uncertain_to_control'].mean():+.3f} diff={obs:+.3f} "
          f"boot95=[{lo:+.3f},{hi:+.3f}] perm p={p:.4g}")
