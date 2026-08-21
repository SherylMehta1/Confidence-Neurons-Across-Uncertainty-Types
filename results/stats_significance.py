"""
results/stats_significance.py -- per-cell significance, equivalence (TOST),
retrospective power/MDE, and cross-category correlation significance.

Runs from ANY working directory (all paths are resolved relative to this
file) and writes every output into results/.

INPUT (live data): person_*/results/results_v3.csv -- the per-prompt
ablation results for the current candidate set (candidate_neurons.json).
The number of neurons / categories / prompts is READ FROM THE DATA, never
hard-coded. Schema v3 columns are required; schema v4 columns
(is_control, orig_activation, mean_val, mean_source, precision) are
optional -- when is_control is absent it is derived as split == "control".

OUTPUTS (all in results/):
  significance_results.csv          pooled working + held_out prompts
  significance_results_heldout.csv  held_out split only (stricter generalization test)
  correlation_significance.csv      cross-category correlation of per-neuron mean shifts (pooled)
  correlation_significance_heldout.csv   same, held_out only
  power_mde.csv                     per-cell SE and minimum detectable effect (both subsets)

FDR FAMILIES (Benjamini-Hochberg at FDR_ALPHA). Each family is corrected
SEPARATELY because each tests a different null hypothesis; p-values are
never pooled across families:
  F1  per-cell shifts: one sign-flip permutation p per (neuron, category),
      H0: mean entropy_shift = 0.                      -> significant_fdr
      (the pooled and the held-out runs are two separate F1 families)
  F2  TOST equivalence: one TOST p per (neuron, category) and per SESOI,
      H0: |true dz| >= SESOI.                           -> equivalent_at_sesoi*
  F3  cross-category correlations: the 3 category-pair permutation p-values.
                                                        -> significant_fdr
  F4  mixed-model per-neuron coefficients  -- results/mixed_model_stats.py
  F5  candidate-vs-control                 -- results/mixed_model_stats.py

PER-CELL SEEDING: every (neuron, category) cell gets its own RNG seeded by
RNG_SEED + stable sha256 hash of (neuron_id, category), so re-running or
changing a single cell never changes any other cell's bootstrap or
permutation result.

TOST: two one-sided paired t-tests on the per-prompt shifts against the
bounds +/- SESOI * sd(shift); the smallest effect size of interest is
declared in paired Cohen's dz units (SESOI_DZ_PRIMARY = 0.2, also reported
at SESOI_DZ_SECONDARY = 0.1). tost_p = max(p_lower, p_upper); equivalence
is claimed when tost_p survives BH within the TOST family.

POWER / MDE (retrospective): minimum detectable mean shift at 80% power for
a two-sided paired t-test, MDE = (t_{1-alpha/2, n-1} + t_{0.80, n-1}) * SE,
for alpha = 0.05 and alpha = 0.01 / n_cells (a Bonferroni-level stand-in
for the FDR threshold).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

RESULT_FILES = {
    "ambiguity": REPO_ROOT / "person_A_ambiguity/results/results_v3.csv",
    "lack_of_knowledge": REPO_ROOT / "person_B_lack_of_knowledge/results/results_v3.csv",
    "contradictory_context": REPO_ROOT / "person_C_contradictory_context/results/results_v3.csv",
}

N_BOOT = 10_000
N_PERM = 20_000
FDR_ALPHA = 0.01
RNG_SEED = 42
CI_LEVEL = 95.0
SESOI_DZ_PRIMARY = 0.2     # "small" paired effect; primary equivalence bound
SESOI_DZ_SECONDARY = 0.1   # stricter bound, reported alongside
POWER_TARGET = 0.80


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def cell_rng(*keys, seed: int = RNG_SEED) -> np.random.Generator:
    """Per-cell RNG: seed + stable hash (independent of PYTHONHASHSEED)."""
    key = "|".join(str(k) for k in keys).encode()
    offset = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed + offset)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    """Boolean mask of p-values rejected by BH at `alpha` within ONE family."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    max_i = np.max(np.where(passed))
    significant = np.zeros(n, dtype=bool)
    significant[order[: max_i + 1]] = True
    return significant


def bootstrap_ci_mean(values: np.ndarray, rng: np.random.Generator,
                      n_boot: int = N_BOOT, ci: float = CI_LEVEL):
    """Vectorised percentile bootstrap CI on the mean."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


def sign_flip_permutation_test(values: np.ndarray, rng: np.random.Generator,
                               n_perm: int = N_PERM) -> float:
    """Vectorised two-sided sign-flip permutation test, H0: mean = 0."""
    observed = abs(values.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(values)))
    perm_means = np.abs((signs * values).mean(axis=1))
    return float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))


def tost_paired(values: np.ndarray, sesoi_dz: float):
    """
    Two one-sided paired t-tests against +/- sesoi_dz * sd(values).
    Returns (p_lower, p_upper, tost_p = max). H0: |true mean| >= bound.
    """
    n = len(values)
    sd = values.std(ddof=1)
    if n < 2 or sd == 0 or np.isnan(sd):
        return np.nan, np.nan, np.nan
    se = sd / np.sqrt(n)
    bound = sesoi_dz * sd
    m = values.mean()
    df = n - 1
    t_lower = (m + bound) / se          # H0: mean <= -bound
    t_upper = (m - bound) / se          # H0: mean >= +bound
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    return float(p_lower), float(p_upper), float(max(p_lower, p_upper))


def mde_paired(sd: float, n: int, alpha: float, power: float = POWER_TARGET) -> float:
    """Minimum detectable mean shift at `power` for a two-sided paired t-test."""
    df = n - 1
    se = sd / np.sqrt(n)
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se)


# ---------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------

def load_results(held_out_only: bool = False) -> pd.DataFrame:
    frames = []
    for category, path in RESULT_FILES.items():
        if not path.exists():
            print(f"WARNING: {path} not found -- skipping {category}.")
            continue
        df = pd.read_csv(path)
        if "is_control" not in df.columns:           # schema v3 -> derive
            df["is_control"] = df["split"].eq("control")
        df = df[~df["is_control"].astype(bool)]
        if held_out_only:
            df = df[df["split"] == "held_out"]
            assert len(df) > 0, (f"held_out filter returned 0 rows for {category} ({path}); "
                                 f"check that split == 'held_out' exists.")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No results_v3.csv files found.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Part 1: per-cell significance, TOST, power
# ---------------------------------------------------------------------

def compute_significance_table(df: pd.DataFrame):
    rows, power_rows = [], []
    n_cells = df.groupby(["neuron_id", "category"]).ngroups
    alphas = {"alpha05": 0.05, f"alpha01_over_{n_cells}": 0.01 / n_cells}
    sec = f"dz{SESOI_DZ_SECONDARY}"

    for (neuron_id, category), group in df.groupby(["neuron_id", "category"]):
        shifts = group["entropy_shift"].to_numpy(dtype=float)
        n = len(shifts)
        if n < 5:
            print(f"WARNING: {neuron_id}/{category} has only {n} prompts -- unstable.")
        rng = cell_rng(neuron_id, category)
        mean_shift, ci_lo, ci_hi = bootstrap_ci_mean(shifts, rng)
        p_value = sign_flip_permutation_test(shifts, rng)
        sd = shifts.std(ddof=1)
        se = sd / np.sqrt(n)
        dz = mean_shift / sd if sd > 0 else np.nan
        _, _, tost_p1 = tost_paired(shifts, SESOI_DZ_PRIMARY)
        _, _, tost_p2 = tost_paired(shifts, SESOI_DZ_SECONDARY)
        rows.append({
            "neuron_id": neuron_id, "category": category, "n_prompts": n,
            "mean_shift": mean_shift, "sd_shift": sd, "se_shift": se, "dz": dz,
            "ci_lower": ci_lo, "ci_upper": ci_hi,
            "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
            "p_value_raw": p_value,
            "tost_p": tost_p1, f"tost_p_{sec}": tost_p2,
        })
        prow = {"neuron_id": neuron_id, "category": category, "n": n,
                "observed_mean_shift": mean_shift, "observed_dz": dz,
                "sd_shift": sd, "se_shift": se}
        for name, a in alphas.items():
            mde = mde_paired(sd, n, a)
            prow[f"mde_{name}"] = mde
            prow[f"mde_dz_{name}"] = mde / sd if sd > 0 else np.nan
        power_rows.append(prow)

    res = pd.DataFrame(rows)
    # F1: per-cell shifts
    res["significant_fdr"] = benjamini_hochberg(res["p_value_raw"].to_numpy())
    # F2: TOST (one family per SESOI)
    res["equivalent_at_sesoi"] = benjamini_hochberg(res["tost_p"].fillna(1.0).to_numpy())
    res[f"equivalent_at_sesoi_{sec}"] = benjamini_hochberg(res[f"tost_p_{sec}"].fillna(1.0).to_numpy())
    res["sesoi_dz_primary"] = SESOI_DZ_PRIMARY
    return res.sort_values("p_value_raw").reset_index(drop=True), pd.DataFrame(power_rows)


# ---------------------------------------------------------------------
# Part 2: cross-category correlation significance
# ---------------------------------------------------------------------

def permutation_test_correlation(x, y, rng, n_perm: int = N_PERM) -> float:
    observed = abs(np.corrcoef(x, y)[0, 1])
    n = len(x)
    perm_idx = rng.permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)
    yp = y[perm_idx]                                   # (n_perm, n)
    xc = x - x.mean()
    ypc = yp - yp.mean(axis=1, keepdims=True)
    r = (ypc @ xc) / (np.sqrt((xc ** 2).sum()) * np.sqrt((ypc ** 2).sum(axis=1)))
    return float((np.sum(np.abs(r) >= observed) + 1) / (n_perm + 1))


def bootstrap_ci_correlation(x, y, rng, n_boot: int = N_BOOT, ci: float = CI_LEVEL):
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    xs, ys = x[idx], y[idx]
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    denom = np.sqrt((xc ** 2).sum(axis=1)) * np.sqrt((yc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc * yc).sum(axis=1) / denom
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def compute_correlation_significance(wide: pd.DataFrame) -> pd.DataFrame:
    categories = wide.columns.tolist()
    rows = []
    for i, cat_a in enumerate(categories):
        for cat_b in categories[i + 1:]:
            x = wide[cat_a].to_numpy(dtype=float)
            y = wide[cat_b].to_numpy(dtype=float)
            rng = cell_rng("corr", cat_a, cat_b)
            p = permutation_test_correlation(x, y, rng)
            lo, hi = bootstrap_ci_correlation(x, y, rng)
            rows.append({
                "category_a": cat_a, "category_b": cat_b, "n_neurons": len(x),
                "observed_r": float(np.corrcoef(x, y)[0, 1]),
                "ci_lower": lo, "ci_upper": hi, "p_value": p,
            })
    out = pd.DataFrame(rows)
    out["significant_fdr"] = benjamini_hochberg(out["p_value"].to_numpy())   # F3
    return out


# ---------------------------------------------------------------------

def run_subset(label: str, held_out_only: bool):
    df = load_results(held_out_only=held_out_only)
    n_neurons, n_cats = df["neuron_id"].nunique(), df["category"].nunique()
    print(f"\n===== {label}: {len(df)} rows, {n_neurons} neurons x {n_cats} categories "
          f"= {n_neurons * n_cats} cells =====")

    sig, power = compute_significance_table(df)
    power.insert(0, "subset", label)
    suffix = "_heldout" if held_out_only else ""
    sig.to_csv(RESULTS_DIR / f"significance_results{suffix}.csv", index=False)
    sec = f"dz{SESOI_DZ_SECONDARY}"
    print(f"\n--- F1 per-cell shifts + F2 TOST ({label}) ---")
    print(sig.to_string(index=False))
    print(f"\nF1: {sig['significant_fdr'].sum()} / {len(sig)} cells significant at FDR {FDR_ALPHA}.")
    print(f"F2: {sig['equivalent_at_sesoi'].sum()} / {len(sig)} cells equivalent to 0 "
          f"at SESOI dz={SESOI_DZ_PRIMARY} (FDR {FDR_ALPHA}); "
          f"{sig[f'equivalent_at_sesoi_{sec}'].sum()} / {len(sig)} at dz={SESOI_DZ_SECONDARY}.")
    print(f"    paired dz range: {sig['dz'].min():.4f} .. {sig['dz'].max():.4f}")

    wide = df.pivot_table(index="neuron_id", columns="category", values="entropy_shift", aggfunc="mean")
    corr = compute_correlation_significance(wide)
    corr.to_csv(RESULTS_DIR / f"correlation_significance{suffix}.csv", index=False)
    print(f"\n--- F3 cross-category correlations ({label}; n_neurons={len(wide)}) ---")
    print(corr.to_string(index=False))
    print(f"F3: {corr['significant_fdr'].sum()} / {len(corr)} pairs significant at FDR {FDR_ALPHA}.")
    return sig, power, corr


def main():
    print(f"FDR families (each BH-corrected separately, FDR_ALPHA = {FDR_ALPHA}):")
    print("  F1 per-cell shifts | F2 TOST equivalence | F3 cross-category correlations")
    print("  F4 mixed-model per-neuron | F5 candidate-vs-control  (F4/F5: mixed_model_stats.py)")
    print(f"N_BOOT={N_BOOT}, N_PERM={N_PERM}, per-cell seeding from RNG_SEED={RNG_SEED}")

    _, power_pooled, _ = run_subset("pooled(working+held_out)", held_out_only=False)
    _, power_heldout, _ = run_subset("held_out_only", held_out_only=True)

    power = pd.concat([power_pooled, power_heldout], ignore_index=True)
    power.to_csv(RESULTS_DIR / "power_mde.csv", index=False)
    print(f"\n--- Retrospective power / MDE ({POWER_TARGET:.0%} power, two-sided paired t) ---")
    mde_cols = [c for c in power.columns if c.startswith("mde_")]
    summ = power.groupby("subset")[["se_shift"] + mde_cols].agg(["min", "median", "max"])
    print(summ.T.to_string())
    print(f"\nWrote outputs into {RESULTS_DIR}")


if __name__ == "__main__":
    main()
