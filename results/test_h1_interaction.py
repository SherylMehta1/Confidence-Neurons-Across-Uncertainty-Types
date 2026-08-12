import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load all prompt-level results
files = {
    "ambiguity": "person_A_ambiguity/results/results.csv",
    "lack_of_knowledge": "person_B_lack_of_knowledge/results/results.csv",
    "contradictory_context": "person_C_contradictory_context/results/results.csv",
}

frames = []

for category, path in files.items():
    df = pd.read_csv(path)
    df["category"] = category
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

print(f"Loaded {len(df)} rows")
print(f"Neurons: {df['neuron_id'].nunique()}")
print(f"Categories: {df['category'].nunique()}")
print(f"Prompts per neuron/category:")
print(
    df.groupby(["neuron_id", "category"])
      .size()
      .value_counts()
)

# --------------------------------------------------
# H1 TEST: neuron × category interaction
# --------------------------------------------------

model = ols(
    "entropy_shift ~ C(neuron_id) * C(category)",
    data=df
).fit()

anova_table = sm.stats.anova_lm(model, typ=2)

print("\n=== TWO-WAY ANOVA ===")
print(anova_table)

interaction_row = "C(neuron_id):C(category)"
p_value = anova_table.loc[interaction_row, "PR(>F)"]
f_value = anova_table.loc[interaction_row, "F"]

print("\n=== H1 TEST ===")
print(f"Interaction F = {f_value:.4f}")
print(f"Interaction p = {p_value:.6g}")

if p_value < 0.01:
    print(
        "\nResult: Significant neuron × category interaction."
    )
    print(
        "The data provide evidence that neuron effects "
        "depend on uncertainty category."
    )
else:
    print(
        "\nResult: No significant neuron × category interaction "
        "at alpha = 0.01."
    )