"""
results/mixed_model_stats.py -- mixed model, paired effect sizes, and the
candidate-vs-control comparison. Runs from any CWD; all paths are resolved
relative to this file and every output goes into results/.

INPUTS: person_*/results/results_v3.csv (candidate prompts) and
person_*/results/control_results_v3.csv (matched control prompts). Schema v4
columns (is_control, orig_activation, mean_val, mean_source, precision) are
optional; is_control is derived from split == "control" when absent.

1. MIXED MODEL (family F4). Per category:
       entropy_shift ~ 0 + C(neuron_id),  random intercept per prompt_id.
   Prompts are reused across all neurons, so a prompt with unusual baseline
   entropy produces correlated shifts across neurons -- that is the prompt
   random effect. The "0 +" removes the intercept so there is NO reference
   neuron: every coefficient IS that neuron's mean shift and its Wald
   p-value tests mean shift != 0. n_neurons rows per category.
   ConvergenceWarnings are caught and recorded in the `converged` flag;
   fitting tries REML, then ML, then lbfgs for each.
   ICC = prompt variance / (prompt variance + residual variance).
   BH-FDR over all per-neuron p-values as ONE family (F4).

2. EFFECT SIZES. Paired Cohen's dz = mean(shift) / sd(shift, ddof=1) with a
   seeded percentile-bootstrap 95% CI on dz. The previous column (mean shift
   divided by the SD of ORIGINAL ENTROPY -- a between-prompt spread, not the
   paired-difference SD) is kept as effect_size_d_legacy for traceability
   only; it is not a paired effect size and should not be reported.

3. CANDIDATE VS CONTROL (family F5). Exploratory and CONSERVATIVE: control
   prompts were ablated with the neuron set to the UNCERTAIN-prompt mean
   activation (mean_source = uncertain set), and control prompts have a
   different baseline entropy than uncertain prompts, so a difference here
   can reflect the ablation-value / baseline mismatch rather than a
   category-specific mechanism. We report (a) a two-sided Welch t-test on
   SIGNED mean shifts and (b) a two-sided studentized permutation test:
   is_control labels are permuted within each (neuron, category) cell and
   the Welch t recomputed, p = P(|t_perm| >= |t_obs|). This replaces the
   previous undisclosed one-sided test on |mean|. BH-FDR over the cells as
   its own family (F5).

FDR FAMILIES: F4 (mixed-model coefficients) and F5 (candidate-vs-control)
are corrected separately here; F1-F3 live in stats_significance.py. There
is NO joint BH pass across families.
"""

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PERSON_DIRS = {
    "ambiguity": REPO_ROOT / "person_A_ambiguity/results",
    "lack_of_knowledge": REPO_ROOT / "person_B_lack_of_knowledge/results",
    "contradictory_context": REPO_ROOT / "person_C_contradictory_context/results",
}
RESULTS_FILES = {c: d / "results_v3.csv" for c, d in PERSON_DIRS.items()}
CONTROL_FILES = {c: d / "control_results_v3.csv" for c, d in PERSON_DIRS.items()}

FDR_ALPHA = 0.01
RNG_SEED = 42
N_BOOT = 10_000
N_PERM = 10_000


def cell_rng(*keys, seed: int = RNG_SEED) -> np.random.Generator:
    key = "|".join(str(k) for k in keys).encode()
    return np.random.default_rng(seed + int(hashlib.sha256(key).hexdigest()[:8], 16))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    passed = ranked <= (np.arange(1, n + 1) / n) * alpha
    if not passed.any():
        return np.zeros(n, dtype=bool)
    significant = np.zeros(n, dtype=bool)
    significant[order[: np.max(np.where(passed)) + 1]] = True
    return significant


def load_all(files: dict, expect_control: bool) -> pd.DataFrame:
    frames = []
    for category, path in files.items():
        if not path.exists():
            print(f"WARNING: {path} not found -- skipping {category}.")
            continue
        df = pd.read_csv(path)
        if "is_control" not in df.columns:
            df["is_control"] = df["split"].eq("control")
        df["is_control"] = df["is_control"].astype(bool)
        df = df[df["is_control"] == expect_control]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files found among {list(files.values())}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# 1. mixed model
# ---------------------------------------------------------------------

def _fit_with_fallbacks(model):
    """
    Try REML -> ML -> lbfgs variants; return (fit, converged, label, warnings).
    A fit counts as converged when the optimizer reports convergence AND no
    Hessian / singular-covariance ConvergenceWarning fired. The statsmodels
    "MLE may be on the boundary of the parameter space" warning is recorded
    (boundary flag) but does NOT by itself reject a fit: the prompt variance
    here is well away from zero and the fixed-effect SEs match the naive
    per-cell SEs, so the warning is informational.
    """
    attempts = [dict(reml=True), dict(reml=False),
                dict(reml=True, method="lbfgs"), dict(reml=False, method="lbfgs")]
    last = (None, False, "no attempt", "")
    for kw in attempts:
        label = f"reml={kw.get('reml')},method={kw.get('method', 'default')}"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                fit = model.fit(**kw)
            except Exception as e:
                last = (None, False, f"{label}: {type(e).__name__}", "")
                continue
        msgs = [str(x.message) for x in w]
        boundary = any("boundary" in m for m in msgs)
        fatal = any(issubclass(x.category, ConvergenceWarning) and "boundary" not in str(x.message)
                    for x in w) or any("singular" in m.lower() for m in msgs)
        converged = bool(getattr(fit, "converged", False)) and not fatal
        note = ("boundary_warning" if boundary else "") + ("|fatal_warning" if fatal else "")
        if converged:
            return fit, True, label, note
        last = (fit, False, label, note)
    return last


def fit_mixed_model_per_category(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in df.groupby("category"):
        group = group.copy()
        model = smf.mixedlm("entropy_shift ~ 0 + C(neuron_id)", group, groups=group["prompt_id"])
        fit, converged, label, note = _fit_with_fallbacks(model)
        if fit is None:
            print(f"WARNING: mixed model failed for {category}: {label}")
            continue
        var_prompt = float(np.asarray(fit.cov_re)[0, 0])
        var_resid = float(fit.scale)
        icc = var_prompt / (var_prompt + var_resid) if (var_prompt + var_resid) > 0 else np.nan
        print(f"  {category}: fit={label} converged={converged} warnings='{note}' "
              f"prompt_var={var_prompt:.3e} resid_var={var_resid:.3e} ICC={icc:.3f}")
        for param, coef in fit.params.items():
            if not param.startswith("C(neuron_id)"):
                continue
            neuron_id = param[len("C(neuron_id)["):].rstrip("]")
            rows.append({
                "category": category, "neuron_id": neuron_id,
                "mixed_model_coef": coef, "mixed_model_se": fit.bse[param],
                "mixed_model_p": fit.pvalues[param],
                "converged": converged, "fit_method": label, "fit_warnings": note,
                "prompt_var": var_prompt, "resid_var": var_resid, "icc": icc,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["significant_fdr"] = benjamini_hochberg(out["mixed_model_p"].to_numpy())  # F4
    return out


# ---------------------------------------------------------------------
# 2. effect sizes
# ---------------------------------------------------------------------

def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (neuron_id, category), group in df.groupby(["neuron_id", "category"]):
        shifts = group["entropy_shift"].to_numpy(dtype=float)
        n = len(shifts)
        sd = shifts.std(ddof=1)
        dz = shifts.mean() / sd if sd > 0 else np.nan
        rng = cell_rng("dz", neuron_id, category)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boot = shifts[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            bdz = boot.mean(axis=1) / boot.std(axis=1, ddof=1)
        bdz = bdz[np.isfinite(bdz)]
        lo, hi = np.percentile(bdz, [2.5, 97.5])
        legacy_sd = group["orig_entropy"].std()
        rows.append({
            "neuron_id": neuron_id, "category": category, "n": n,
            "mean_shift": shifts.mean(), "sd_shift": sd,
            "effect_size_dz": dz, "dz_ci_lower": lo, "dz_ci_upper": hi,
            "effect_size_d_legacy": shifts.mean() / legacy_sd if legacy_sd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. candidate vs control
# ---------------------------------------------------------------------

def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def candidate_vs_control_test(cand_df: pd.DataFrame, control_df: pd.DataFrame,
                              n_perm: int = N_PERM) -> pd.DataFrame:
    rows = []
    for (neuron_id, category), cand_group in cand_df.groupby(["neuron_id", "category"]):
        ctrl_group = control_df[(control_df["neuron_id"] == neuron_id)
                                & (control_df["category"] == category)]
        if len(ctrl_group) < 5:
            continue
        cand = cand_group["entropy_shift"].to_numpy(dtype=float)
        ctrl = ctrl_group["entropy_shift"].to_numpy(dtype=float)
        n_c = len(cand)
        t_obs = welch_t(cand, ctrl)
        welch = stats.ttest_ind(cand, ctrl, equal_var=False)

        rng = cell_rng("cvc", neuron_id, category)
        pooled = np.concatenate([cand, ctrl])
        perm_idx = rng.permuted(np.tile(np.arange(len(pooled)), (n_perm, 1)), axis=1)
        pp = pooled[perm_idx]
        a, b = pp[:, :n_c], pp[:, n_c:]
        t_perm = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(
            a.var(axis=1, ddof=1) / n_c + b.var(axis=1, ddof=1) / (len(pooled) - n_c))
        p_perm = float((np.sum(np.abs(t_perm) >= abs(t_obs)) + 1) / (n_perm + 1))

        if "mean_source" in cand_group.columns:
            mean_source = str(cand_group["mean_source"].iloc[0])
        else:
            mean_source = "uncertain_prompt_mean (assumed; schema v3 has no column)"
        rows.append({
            "neuron_id": neuron_id, "category": category,
            "n_candidate": n_c, "n_control": len(ctrl),
            "candidate_mean_shift": cand.mean(), "control_mean_shift": ctrl.mean(),
            "diff_signed": cand.mean() - ctrl.mean(),
            "welch_t": t_obs, "welch_p_two_sided": float(welch.pvalue),
            "perm_p_two_sided": p_perm, "mean_source": mean_source,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["significant_fdr"] = benjamini_hochberg(out["perm_p_two_sided"].to_numpy())  # F5
    return out


def main():
    df = load_all(RESULTS_FILES, expect_control=False)
    control_df = load_all(CONTROL_FILES, expect_control=True)
    print(f"candidates: {len(df)} rows, {df['neuron_id'].nunique()} neurons; "
          f"controls: {len(control_df)} rows")

    print("\n--- F4: mixed model entropy_shift ~ 0 + C(neuron_id), prompt random intercept ---")
    mixed = fit_mixed_model_per_category(df)
    mixed.to_csv(RESULTS_DIR / "mixed_model_results.csv", index=False)
    print(mixed.sort_values("mixed_model_p").to_string(index=False))
    if len(mixed):
        print(f"\nF4: {mixed['significant_fdr'].sum()} / {len(mixed)} per-neuron coefficients "
              f"survive FDR at alpha={FDR_ALPHA}; converged per category: "
              f"{mixed.groupby('category')['converged'].first().to_dict()}")

    print("\n--- Effect sizes (paired dz with bootstrap 95% CI) ---")
    effects = compute_effect_sizes(df)
    effects.to_csv(RESULTS_DIR / "effect_sizes.csv", index=False)
    print(effects.sort_values("effect_size_dz", key=abs, ascending=False).to_string(index=False))
    print(f"\npaired dz range: {effects['effect_size_dz'].min():.4f} .. {effects['effect_size_dz'].max():.4f}; "
          f"legacy d range: {effects['effect_size_d_legacy'].min():.4f} .. {effects['effect_size_d_legacy'].max():.4f}")

    print("\n--- F5: candidate vs matched control (two-sided, signed means; exploratory) ---")
    cvc = candidate_vs_control_test(df, control_df)
    if len(cvc):
        cvc.to_csv(RESULTS_DIR / "candidate_vs_control.csv", index=False)
        print(cvc.sort_values("perm_p_two_sided").to_string(index=False))
        print(f"\nF5: {cvc['significant_fdr'].sum()} / {len(cvc)} cells survive FDR at alpha={FDR_ALPHA} "
              f"(studentized permutation); {(cvc['welch_p_two_sided'] < 0.05).sum()} have raw Welch p<0.05.")
    else:
        print("No candidate/control pairs found.")
    print(f"\nWrote outputs into {RESULTS_DIR}")


if __name__ == "__main__":
    main()
