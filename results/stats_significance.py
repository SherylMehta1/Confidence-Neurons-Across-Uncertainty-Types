"""
stats_significance.py

Adds proper statistical backing on top of merge_and_analyze.py's output,
addressing two open questions:

1. Are the per-neuron, per-category mean entropy shifts actually
   significant, or could eyeballed "shift != 0" calls be sampling noise
   from a small held-out split? -> sign-flip permutation test + bootstrap
   CI per (neuron, category), with FDR correction across the 45 tests
   (15 neurons x 3 categories) instead of relying on a raw threshold.

2. Is the 0.55-0.69 cross-category correlation real, or could it be
   driven by 2-3 large-magnitude neurons out of only 15 data points?
   -> permutation test (neuron-label shuffle) + bootstrap CI on the
   correlation itself.

Reads the raw per-prompt results.csv files (NOT merged_summary.csv --
that's already averaged, and averages are exactly what needs a CI/p-value
computed on their underlying variability, so we go back to the per-prompt
source).

Expected schema per results.csv (per project convention):
neuron_id, layer, neuron_idx, category, prompt_id, orig_entropy,
ablated_entropy, entropy_shift, orig_top1_prob, ablated_top1_prob,
direct_effect_score, split
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULT_FILES = {
    "ambiguity": "person_A_ambiguity/results/results.csv",
    "lack_of_knowledge": "person_B_lack_of_knowledge/results/results.csv",
    "contradictory_context": "person_C_contradictory_context/results/results.csv",
}

# Set to True to only use the held-out split (the stricter generalization
# test per your eval criteria). False uses everything available.
HELD_OUT_ONLY = False

N_BOOT = 10_000
N_PERM = 10_000
FDR_ALPHA = 0.01
RNG_SEED = 42

rng = np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------
# Part 1: per-neuron, per-category significance of the mean entropy shift
# ---------------------------------------------------------------------

def bootstrap_ci_mean(values: np.ndarray, n_boot: int = N_BOOT, ci: float = 95.0):
    """Percentile bootstrap CI on the mean of `values`."""
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return values.mean(), lower, upper


def sign_flip_permutation_test(values: np.ndarray, n_perm: int = N_PERM) -> float:
    """
    Two-sided sign-flip permutation test for H0: true mean shift = 0.
    Under H0, the sign of each prompt's shift is exchangeable (equally
    likely positive or negative), since ablation would have no
    systematic effect. Randomly flip signs, recompute the mean, and see
    how often the permuted mean is as extreme as the observed one.
    """
    observed = np.abs(values.mean())
    n = len(values)
    count_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = np.abs((values * signs).mean())
        if perm_mean >= observed:
            count_extreme += 1
    return (count_extreme + 1) / (n_perm + 1)  # +1 avoids p=0


def benjamini_hochberg(pvals: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    """
    Returns a boolean array: which p-values survive FDR correction at
    `alpha`, across all tests jointly (not per-category separately) --
    correct since you're running 45 tests (15 neurons x 3 categories) and
    want to control the overall false-discovery rate, not just each
    category's raw threshold in isolation.
    """
    n = len(pvals)
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


def load_results() -> pd.DataFrame:
    frames = []
    for category, path in RESULT_FILES.items():
        df = pd.read_csv(path)
        if HELD_OUT_ONLY:
            df = df[df["split"] == "held-out"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_significance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (neuron_id, category), group in df.groupby(["neuron_id", "category"]):
        shifts = group["entropy_shift"].to_numpy()
        n_prompts = len(shifts)
        if n_prompts < 5:
            print(f"WARNING: {neuron_id}/{category} has only {n_prompts} "
                  f"held-out prompts -- CI/p-value will be unstable.")

        mean_shift, ci_lo, ci_hi = bootstrap_ci_mean(shifts)
        p_value = sign_flip_permutation_test(shifts)

        rows.append({
            "neuron_id": neuron_id,
            "category": category,
            "n_prompts": n_prompts,
            "mean_shift": mean_shift,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
            "p_value_raw": p_value,
        })

    result = pd.DataFrame(rows)
    result["significant_fdr"] = benjamini_hochberg(result["p_value_raw"].to_numpy())
    return result.sort_values("p_value_raw")


# ---------------------------------------------------------------------
# Part 2: significance of the cross-category correlation matrix
# ---------------------------------------------------------------------

def permutation_test_correlation(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM) -> float:
    """
    Two-sided permutation test for H0: no association between the two
    categories' per-neuron effect vectors. Shuffle one vector relative to
    the other (breaking the neuron-to-neuron pairing) and see how often
    the permuted |correlation| is as extreme as observed.
    """
    observed = np.abs(np.corrcoef(x, y)[0, 1])
    n = len(x)
    count_extreme = 0
    y_perm = y.copy()
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        perm_corr = np.abs(np.corrcoef(x, y_perm)[0, 1])
        if perm_corr >= observed:
            count_extreme += 1
    return (count_extreme + 1) / (n_perm + 1)


def bootstrap_ci_correlation(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, ci: float = 95.0):
    """
    Bootstrap CI on the correlation itself, resampling NEURONS (pairs)
    with replacement. With only 15 neurons this CI will be wide -- that's
    expected and worth reporting honestly rather than hiding.
    """
    n = len(x)
    idx = np.arange(n)
    boot_corrs = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        xs, ys = x[sample_idx], y[sample_idx]
        if xs.std() == 0 or ys.std() == 0:
            boot_corrs[i] = np.nan  # degenerate resample, skip
        else:
            boot_corrs[i] = np.corrcoef(xs, ys)[0, 1]
    boot_corrs = boot_corrs[~np.isnan(boot_corrs)]
    lower = np.percentile(boot_corrs, (100 - ci) / 2)
    upper = np.percentile(boot_corrs, 100 - (100 - ci) / 2)
    return lower, upper


def compute_correlation_significance(merged_summary: pd.DataFrame) -> pd.DataFrame:
    """
    merged_summary: wide-format DataFrame, index=neuron_id, columns=
    category, values=mean entropy_shift -- i.e. exactly what
    merge_and_analyze.py already prints/saves.
    """
    categories = merged_summary.columns.tolist()
    rows = []
    for i, cat_a in enumerate(categories):
        for cat_b in categories[i + 1:]:
            x = merged_summary[cat_a].to_numpy()
            y = merged_summary[cat_b].to_numpy()
            observed_r = np.corrcoef(x, y)[0, 1]
            p_value = permutation_test_correlation(x, y)
            ci_lo, ci_hi = bootstrap_ci_correlation(x, y)
            rows.append({
                "category_a": cat_a,
                "category_b": cat_b,
                "n_neurons": len(x),
                "observed_r": observed_r,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": p_value,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------

def main():
    print(f"Loading raw per-prompt results "
          f"({'held-out split only' if HELD_OUT_ONLY else 'all splits'})...")
    df = load_results()
    print(f"Loaded {len(df)} rows across "
          f"{df['neuron_id'].nunique()} neurons x {df['category'].nunique()} categories.")

    print("\n--- Part 1: per-neuron, per-category significance ---")
    sig_table = compute_significance_table(df)
    sig_table.to_csv("significance_results.csv", index=False)
    print(sig_table.to_string(index=False))
    n_sig = sig_table["significant_fdr"].sum()
    print(f"\n{n_sig} / {len(sig_table)} (neuron, category) shifts survive "
          f"FDR correction at alpha={FDR_ALPHA}.")
    print("Compare this to your original overlap sets -- if far fewer "
          "survive here, the raw-threshold overlap sets were likely "
          "inflated by multiple-comparisons / small-sample noise.")

    print("\n--- Part 2: cross-category correlation significance ---")
    # Rebuild the wide-format table merge_and_analyze.py already computes,
    # here from the same (possibly held-out-filtered) data for consistency.
    wide = df.pivot_table(index="neuron_id", columns="category",
                           values="entropy_shift", aggfunc="mean")
    corr_table = compute_correlation_significance(wide)
    corr_table.to_csv("correlation_significance.csv", index=False)
    print(corr_table.to_string(index=False))

    print("\nInterpretation guide:")
    print("- p_value here tests whether the correlation is distinguishable")
    print("  from chance pairing of only 15 neurons -- NOT the same as the")
    print("  correlation being 'large'. A significant p-value with a wide")
    print("  CI still means shared direction, uncertain magnitude.")
    print("- If a category pair's CI comfortably excludes 0, that's your")
    print("  strongest evidence for H1-leaning shared effect on that pair.")
    print("- Cross-reference against Part 1: a pair can show a significant")
    print("  correlation while having ZERO individually-significant shared")
    print("  neurons -- that's the H3 (partial overlap) signature: a weak,")
    print("  diffuse shared trend plus category-specific standouts, rather")
    print("  than a few neurons carrying the whole effect.")


if __name__ == "__main__":
    main()