"""
Week 9-10 synthesis: merge all three people's results and answer H1/H2/H3.

Run this once everyone's results.csv files are finalized. Depends on the exact
schema in RESULTS_SCHEMA.md — do not run this if anyone's columns don't match.
"""

import pandas as pd

PATHS = {
    "ambiguity": "../person_A_ambiguity/results/results_v3.csv",
    "lack_of_knowledge": "../person_B_lack_of_knowledge/results/results_v3.csv",
    "contradictory_context": "../person_C_contradictory_context/results/results_v3.csv",
}


def load_all_results():
    dfs = []
    for category, path in PATHS.items():
        df = pd.read_csv(path)
        assert (df["category"] == category).all(), f"Category mismatch in {path}"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarize(all_results):
    # Average entropy shift per neuron, per category
    summary = all_results.groupby(["neuron_id", "category"])["entropy_shift"].mean().unstack()
    print("Per-neuron, per-category average entropy shift:")
    print(summary)
    print()

    # Pairwise correlation across categories -- this is the H1/H2/H3 answer
    corr = summary.corr()
    print("Cross-category correlation of per-neuron effect sizes:")
    print(corr)
    print()
    print("High correlation (>0.6ish) between two categories -> shared mechanism (H1-leaning)")
    print("Low/near-zero correlation -> specialized mechanism (H2-leaning)")
    print("Mixed pattern across neurons -> partial overlap (H3-leaning)")

    return summary, corr


if __name__ == "__main__":
    all_results = load_all_results()
    summary, corr = summarize(all_results)
    summary.to_csv("merged_summary.csv")
    corr.to_csv("cross_category_correlation.csv")
    print("\nSaved merged_summary.csv and cross_category_correlation.csv")
