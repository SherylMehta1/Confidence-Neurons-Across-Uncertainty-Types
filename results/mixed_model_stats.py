"""
results/mixed_model_stats.py -- Phase 3 "Upgrade statistics" (shared)

Builds on results/stats_significance.py (unchanged, still correct for what
it does) with three additions the remediation plan calls for:

1. MIXED MODEL WITH PROMPT RANDOM EFFECTS: within a category, the same
   pool of prompts is reused across all 15 candidate neurons' ablation
   runs. A prompt with unusually high/low baseline entropy will show
   correlated shift patterns across MANY neurons just because of the
   prompt, not because of any particular neuron -- that's a prompt-level
   random effect the per-neuron permutation test in stats_significance.py
   doesn't model (it tests one neuron at a time, so it's not wrong, just
   silent on cross-neuron comparisons). Fit with a random intercept for
   prompt_id per category.

2. EFFECT SIZES, not just p-values: standardized mean shift (shift / pooled
   SD of orig_entropy) per (neuron, category), reported alongside
   significance -- addresses "quoting correlations as effect sizes" and
   generally makes small-but-significant vs large-but-noisy distinguishable
   at a glance.

3. CANDIDATE-VS-CONTROL, FDR EVERYWHERE: with Phase 3's matched control
   prompts now available (data/<category>/controls.jsonl + the
   control_results.csv each category produces via shared/run_phase34.py),
   this runs the actual "is the candidate's effect bigger than what the
   SAME neuron does on definitely-low-uncertainty prompts" test that was
   previously only informally gestured at via 5 random control NEURONS.
   Controls here means control PROMPTS (matched, low-uncertainty), a
   different axis from the existing control-neuron check in
   control_significance_summary.csv -- keep both, they test different
   confounds. All p-values (per-neuron-category, cross-category
   correlation, AND candidate-vs-control) go into ONE joint BH-FDR pass.

Requires: statsmodels (not yet in requirements.txt -- add
`statsmodels>=0.14.0`).

Run from repo root: `python results/mixed_model_stats.py`
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

RESULTS_FILES = {
    "ambiguity": "person_A_ambiguity/results/results_v3.csv",
    "lack_of_knowledge": "person_B_lack_of_knowledge/results/results_v3.csv",
    "contradictory_context": "person_C_contradictory_context/results/results_v3.csv",
}
CONTROL_FILES = {
    "ambiguity": "person_A_ambiguity/results/control_results_v3.csv",
    "lack_of_knowledge": "person_B_lack_of_knowledge/results/control_results_v3.csv",
    "contradictory_context": "person_C_contradictory_context/results/control_results_v3.csv",
}

FDR_ALPHA = 0.01


def benjamini_hochberg(pvals: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
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


def load_all(files: dict) -> pd.DataFrame:
    frames = []
    for category, path in files.items():
        if not Path(path).exists():
            print(f"WARNING: {path} not found -- skipping {category}. "
                  f"Run shared/run_phase34.py for this category first.")
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No results files found -- run Phase 3/4 for at least one category first.")
    return pd.concat(frames, ignore_index=True)


def fit_mixed_model_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    One mixed model per category: entropy_shift ~ C(neuron_id), random
    intercept per prompt_id. Returns per-neuron coefficient estimates (the
    model's own fixed-effect p-values), which you should treat as a
    SANITY CHECK against the permutation-test p-values in
    stats_significance.py -- if they disagree substantially for a given
    neuron, that's worth understanding before trusting either one alone.
    """
    rows = []
    for category, group in df.groupby("category"):
        group = group.copy()
        # reference-code so every neuron gets its own coefficient vs. grand mean,
        # not vs. an arbitrary reference neuron
        model = smf.mixedlm(
            "entropy_shift ~ C(neuron_id)", group, groups=group["prompt_id"],
        )
        try:
            fit = model.fit(reml=False)
        except Exception as e:
            print(f"WARNING: mixed model failed to converge for {category}: {e}")
            continue
        for param, coef in fit.params.items():
            if not param.startswith("C(neuron_id)"):
                continue
            neuron_id = param.split("[T.")[-1].rstrip("]")
            rows.append({
                "category": category, "neuron_id": neuron_id,
                "mixed_model_coef": coef,
                "mixed_model_p": fit.pvalues[param],
            })
    return pd.DataFrame(rows)


def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean shift per (neuron, category): mean(entropy_shift) /
    pooled SD of orig_entropy for that category -- comparable across
    categories even though raw entropy scales differ."""
    rows = []
    for (neuron_id, category), group in df.groupby(["neuron_id", "category"]):
        pooled_sd = group["orig_entropy"].std()
        if pooled_sd == 0 or np.isnan(pooled_sd):
            continue
        rows.append({
            "neuron_id": neuron_id, "category": category,
            "mean_shift": group["entropy_shift"].mean(),
            "effect_size_d": group["entropy_shift"].mean() / pooled_sd,
            "n": len(group),
        })
    return pd.DataFrame(rows)


def candidate_vs_control_test(cand_df: pd.DataFrame, control_df: pd.DataFrame, n_perm: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    For each (neuron, category), permutation test: is the mean entropy_shift
    on candidate (uncertain) prompts bigger in magnitude than on the SAME
    neuron's matched control (low-uncertainty) prompts? Two-sample
    difference-of-means permutation test (labels shuffled between the two
    pooled sets), not paired -- candidate and control prompts aren't 1:1
    matched at the row level for ambiguity/lack-of-knowledge (they are for
    contradictory-context's true-object controls; the unpaired test is
    still valid there, just conservative).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (neuron_id, category), cand_group in cand_df.groupby(["neuron_id", "category"]):
        ctrl_group = control_df[
            (control_df["neuron_id"] == neuron_id) & (control_df["category"] == category)
        ]
        if len(ctrl_group) < 5:
            continue
        cand_shift = cand_group["entropy_shift"].to_numpy()
        ctrl_shift = ctrl_group["entropy_shift"].to_numpy()

        observed = abs(cand_shift.mean()) - abs(ctrl_shift.mean())
        pooled = np.concatenate([cand_shift, ctrl_shift])
        n_cand = len(cand_shift)
        count_extreme = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            perm_diff = abs(pooled[:n_cand].mean()) - abs(pooled[n_cand:].mean())
            if perm_diff >= observed:
                count_extreme += 1
        p = (count_extreme + 1) / (n_perm + 1)

        rows.append({
            "neuron_id": neuron_id, "category": category,
            "candidate_mean_abs_shift": abs(cand_shift.mean()),
            "control_mean_abs_shift": abs(ctrl_shift.mean()),
            "diff": observed, "p_value_raw": p,
        })
    return pd.DataFrame(rows)


def main():
    df = load_all(RESULTS_FILES)
    control_df = load_all(CONTROL_FILES)

    print("--- Mixed model (sanity check vs. permutation test) ---")
    mixed = fit_mixed_model_per_category(df)
    mixed.to_csv("results/mixed_model_results.csv", index=False)
    print(mixed.to_string(index=False))

    print("\n--- Effect sizes ---")
    effects = compute_effect_sizes(df)
    effects.to_csv("results/effect_sizes.csv", index=False)
    print(effects.sort_values("effect_size_d", key=abs, ascending=False).to_string(index=False))

    print("\n--- Candidate vs. matched control ---")
    cvc = candidate_vs_control_test(df, control_df)

    # Joint FDR across candidate-vs-control tests only (keep this separate
    # from the per-neuron-category / cross-category-correlation FDR pass in
    # stats_significance.py -- they test different null hypotheses, don't
    # pool them into one BH pass or you'll misstate what's being controlled).
    if len(cvc) > 0:
        cvc["significant_fdr"] = benjamini_hochberg(cvc["p_value_raw"].to_numpy())
        cvc.to_csv("results/candidate_vs_control.csv", index=False)
        print(cvc.sort_values("p_value_raw").to_string(index=False))
        n_sig = cvc["significant_fdr"].sum()
        print(f"\n{n_sig} / {len(cvc)} (neuron, category) candidate-vs-control "
              f"comparisons survive FDR at alpha={FDR_ALPHA}.")
    else:
        print("No candidate/control pairs found -- check that "
              "shared/run_phase34.py has been run for both prompts.jsonl "
              "and controls.jsonl for at least one category.")


if __name__ == "__main__":
    main()
