"""
results/merge_and_analyze.py -- merge the three categories' per-prompt
results and compute the per-neuron x category mean-shift table and its
cross-category correlation (the descriptive H1/H2/H3 summary; significance
for these correlations is in stats_significance.py, family F3).

Runs from any CWD; paths resolve relative to this file; outputs go to results/:
  merged_summary.csv             candidate prompts: mean entropy_shift per neuron x category
  merged_summary_controls.csv    control prompts: same table
  cross_category_correlation.csv Pearson r between category columns (candidate prompts)

Schema v4 column is_control is optional; derived from split == "control".
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PERSON_DIRS = {
    "ambiguity": REPO_ROOT / "person_A_ambiguity/results",
    "lack_of_knowledge": REPO_ROOT / "person_B_lack_of_knowledge/results",
    "contradictory_context": REPO_ROOT / "person_C_contradictory_context/results",
}


def load_all_results() -> pd.DataFrame:
    dfs = []
    for category, d in PERSON_DIRS.items():
        for fname in ("results_v3.csv", "control_results_v3.csv"):
            path = d / fname
            if not path.exists():
                print(f"WARNING: {path} missing -- skipped")
                continue
            df = pd.read_csv(path)
            assert (df["category"] == category).all(), f"Category mismatch in {path}"
            if "is_control" not in df.columns:
                df["is_control"] = df["split"].eq("control")
            df["is_control"] = df["is_control"].astype(bool)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarize(all_results: pd.DataFrame):
    cand = all_results[~all_results["is_control"]]
    ctrl = all_results[all_results["is_control"]]
    summary = cand.groupby(["neuron_id", "category"])["entropy_shift"].mean().unstack()
    ctrl_summary = ctrl.groupby(["neuron_id", "category"])["entropy_shift"].mean().unstack()
    print(f"Per-neuron, per-category mean entropy shift (candidate prompts; "
          f"{cand['neuron_id'].nunique()} neurons):")
    print(summary)
    corr = summary.corr()
    print("\nCross-category Pearson correlation of per-neuron mean shifts (descriptive; "
          "see stats_significance.py F3 for permutation p / bootstrap CI):")
    print(corr)
    return summary, ctrl_summary, corr


if __name__ == "__main__":
    all_results = load_all_results()
    summary, ctrl_summary, corr = summarize(all_results)
    summary.to_csv(RESULTS_DIR / "merged_summary.csv")
    ctrl_summary.to_csv(RESULTS_DIR / "merged_summary_controls.csv")
    corr.to_csv(RESULTS_DIR / "cross_category_correlation.csv")
    print(f"\nSaved merged_summary.csv, merged_summary_controls.csv, cross_category_correlation.csv to {RESULTS_DIR}")
