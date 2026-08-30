"""Decider (a) for the Qwen aleatoric half-failure, from local per-pair lens trajectories.

Question: does the late-emerging hedge log-odds gap RIDE the entropy gap pair-by-pair
(leakage: channels entangled) or not (a genuine partial verdict)?

Per pair: gap_x(pair) = x[uncertain] - x[control] at the late layers, for
x in {lens_logodds, lens_entropy}. Correlate across pairs (Spearman + Pearson),
with a sign-flip permutation p on the Spearman rho. Reference arms: Qwen contested
(a real verdict arm, same model) and Llama aleatoric (where the prediction held).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

rng = np.random.default_rng(0)

ARMS = {
    "Qwen aleatoric  (late L25-26)": ("results/circuit_aleatoric_qwen/jlens_trajectory.csv", [25, 26]),
    "Qwen contested  (late L25-27)": ("results/circuit_conflict_qwen/jlens_trajectory.csv.gz", [25, 26, 27]),
    "Llama aleatoric (late L29-31)": ("results/circuit_aleatoric/jlens_trajectory.csv", [29, 30, 31]),
}


def perm_p(x, y, n=20000):
    obs = spearmanr(x, y).statistic
    cnt = 0
    for _ in range(n):
        if abs(spearmanr(rng.permutation(x), y).statistic) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


for name, (path, layers) in ARMS.items():
    df = pd.read_csv(path)
    lmax = df["layer"].max()
    late = df[df["layer"].isin([l for l in layers if l <= lmax])]
    g = late.groupby(["pair", "arm"])[["lens_logodds", "lens_entropy"]].mean().unstack("arm")
    lo = g[("lens_logodds", "uncertain")] - g[("lens_logodds", "control")]
    en = g[("lens_entropy", "uncertain")] - g[("lens_entropy", "control")]
    ok = lo.notna() & en.notna()
    lo, en = lo[ok].to_numpy(), en[ok].to_numpy()
    pr = pearsonr(lo, en)
    rho, p = perm_p(en, lo, 5000)
    print(f"{name}: n={len(lo)} layers<= {lmax} | "
          f"pearson r={pr.statistic:+.3f} (p={pr.pvalue:.3g}) | spearman rho={rho:+.3f} perm p={p:.4g} | "
          f"mean logodds gap {lo.mean():+.2f}, mean entropy gap {en.mean():+.2f}")
