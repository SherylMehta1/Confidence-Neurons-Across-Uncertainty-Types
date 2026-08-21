"""
results/test_h1_interaction.py -- does the per-neuron ablation effect
pattern differ across uncertainty categories? (neuron x category interaction)

Runs from any CWD; reads person_*/results/results_v3.csv; writes
results/h1_interaction.csv and results/h1_interaction.json.

WHY NOT A MIXED MODEL OR A PLAIN TWO-WAY ANOVA
  * MixedLM entropy_shift ~ C(neuron_id) * C(category) + (1 | prompt_id) is
    not identifiable for the category main effect: prompts are nested in
    category, so category is a between-prompt factor carried entirely by
    the prompt random intercepts. (The interaction is estimable in
    principle, but the fit is fragile and the Wald p assumes normality of
    shifts that are heavy-tailed here.)
  * The old plain OLS ANOVA (no prompt term) treated the 17 rows of each
    prompt as independent observations, inflating the interaction df.

WHAT THIS SCRIPT DOES (prompt-level permutation test of the interaction)
  The independent unit is the PROMPT: each prompt contributes a 17-vector of
  shifts (one per neuron). Within-prompt centering removes the prompt
  random intercept exactly (balanced design: every prompt has every
  neuron). The interaction statistic is the F-ratio for neuron x category
  computed from those centered values:
      SS_int  = sum_cells n_cell * (m_nc - m_n. - m_.c + m_..)^2
      SS_res  = sum over rows of (y_centered - cell mean)^2
      F       = (SS_int / df_int) / (SS_res / df_res),
      df_int = (n_neurons-1)(n_cats-1),
      df_res = N - n_prompts - (n_neurons-1) - df_int.
  NULL: "the expected neuron profile does not depend on category". Under
  this null, prompt blocks are exchangeable across categories, so the
  permutation shuffles CATEGORY LABELS ACROSS PROMPTS (whole 17-row blocks
  move together, preserving within-prompt dependence), keeping the number
  of prompts per category fixed. p = P(F_perm >= F_obs).
  Caveat: exchangeability also assumes similar within-prompt variance
  across categories; the statistic is reported with the parametric F
  p-value for reference only (it ignores heteroscedasticity). We also
  report partial eta^2 = SS_int / (SS_int + SS_res) as the effect size.

  The alternative suggested in the remediation notes -- shuffling neuron_id
  within each prompt's 17 rows -- tests a DIFFERENT null (that neurons are
  exchangeable, i.e. no neuron main effect AND no interaction), so it can
  reject merely because some neuron has a nonzero mean shift in all
  categories. It is therefore not a test of H1 vs H2/H3 and is not used.

  Run for pooled (working + held_out) and held_out-only prompts.
"""

import json
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
N_PERM = 20_000
RNG_SEED = 42
ALPHA = 0.01


def load(held_out_only: bool) -> pd.DataFrame:
    frames = []
    for category, path in RESULT_FILES.items():
        df = pd.read_csv(path)
        if "is_control" not in df.columns:
            df["is_control"] = df["split"].eq("control")
        df = df[~df["is_control"].astype(bool)]
        if held_out_only:
            df = df[df["split"] == "held_out"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_block_matrix(df: pd.DataFrame):
    """Returns Y (n_prompts x n_neurons) of within-prompt-centered shifts, and
    the category label per prompt. Requires a complete balanced design."""
    wide = df.pivot_table(index="prompt_id", columns="neuron_id", values="entropy_shift")
    assert not wide.isna().any().any(), "unbalanced design: some prompt lacks a neuron"
    cat = df.groupby("prompt_id")["category"].first().loc[wide.index].to_numpy()
    Y = wide.to_numpy(dtype=float)
    Y = Y - Y.mean(axis=1, keepdims=True)       # remove prompt intercept
    return Y, cat, wide.columns.tolist()


def interaction_stats(Y: np.ndarray, cat: np.ndarray, cats: np.ndarray):
    """F, SS_int, SS_res, df_int, df_res for neuron x category on centered Y."""
    n_prompts, n_neurons = Y.shape
    n_cats = len(cats)
    grand = Y.mean(axis=0)                                   # per-neuron mean (m_n.)
    ss_int, ss_res = 0.0, 0.0
    for c in cats:
        Yc = Y[cat == c]
        mc = Yc.mean(axis=0)                                 # cell means m_nc (for this c)
        # m_.c is 0 after within-prompt centering (each row sums to 0), m_.. = 0
        ss_int += len(Yc) * np.sum((mc - grand) ** 2)
        ss_res += np.sum((Yc - mc) ** 2)
    df_int = (n_neurons - 1) * (n_cats - 1)
    df_res = n_prompts * n_neurons - n_prompts - (n_neurons - 1) - df_int
    F = (ss_int / df_int) / (ss_res / df_res)
    return F, ss_int, ss_res, df_int, df_res


def run(label: str, held_out_only: bool) -> dict:
    df = load(held_out_only)
    Y, cat, neurons = to_block_matrix(df)
    cats = np.array(sorted(set(cat)))
    F_obs, ss_int, ss_res, df_int, df_res = interaction_stats(Y, cat, cats)
    rng = np.random.default_rng(RNG_SEED)
    F_perm = np.empty(N_PERM)
    for i in range(N_PERM):
        F_perm[i] = interaction_stats(Y, rng.permutation(cat), cats)[0]
    p_perm = float((np.sum(F_perm >= F_obs) + 1) / (N_PERM + 1))
    p_param = float(stats.f.sf(F_obs, df_int, df_res))
    out = {
        "subset": label, "n_prompts": int(Y.shape[0]), "n_neurons": int(Y.shape[1]),
        "n_categories": int(len(cats)), "F_interaction": float(F_obs),
        "df_int": int(df_int), "df_res": int(df_res),
        "partial_eta2": float(ss_int / (ss_int + ss_res)),
        "p_permutation": p_perm, "p_parametric_reference": p_param,
        "n_perm": N_PERM, "significant_at_alpha": bool(p_perm < ALPHA), "alpha": ALPHA,
    }
    print(f"\n=== {label}: {out['n_prompts']} prompts x {out['n_neurons']} neurons, "
          f"{out['n_categories']} categories ===")
    print(f"interaction F({df_int},{df_res}) = {F_obs:.3f}, partial eta^2 = {out['partial_eta2']:.4f}")
    print(f"prompt-level permutation p = {p_perm:.5f} (n_perm={N_PERM}); parametric F p (reference) = {p_param:.3g}")
    print("Result: " + ("neuron x category interaction detected -- per-neuron effect pattern differs by category"
                        if out["significant_at_alpha"] else
                        f"no neuron x category interaction at alpha = {ALPHA}"))
    return out


def main():
    rows = [run("pooled(working+held_out)", False), run("held_out_only", True)]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "h1_interaction.csv", index=False)
    with open(RESULTS_DIR / "h1_interaction.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote h1_interaction.csv / .json into {RESULTS_DIR}")


if __name__ == "__main__":
    main()
