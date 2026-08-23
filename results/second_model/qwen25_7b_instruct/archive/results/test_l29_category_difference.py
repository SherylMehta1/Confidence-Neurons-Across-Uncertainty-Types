from scipy import stats
import pandas as pd

df = pd.concat([
    pd.read_csv("person_A_ambiguity/results/results.csv"),
    pd.read_csv("person_B_lack_of_knowledge/results/results.csv"),
    pd.read_csv("person_C_contradictory_context/results/results.csv")
], ignore_index=True)

# Targeted test for L29_N8568:
# Contradictory context vs. ambiguity

c_shifts = df[
    (df["neuron_id"] == "L29_N8568") &
    (df["category"] == "contradictory_context")
]["entropy_shift"]

a_shifts = df[
    (df["neuron_id"] == "L29_N8568") &
    (df["category"] == "ambiguity")
]["entropy_shift"]

print(f"Contradiction prompts: {len(c_shifts)}")
print(f"Ambiguity prompts: {len(a_shifts)}")

print(f"Contradiction mean shift: {c_shifts.mean():.6f}")
print(f"Ambiguity mean shift: {a_shifts.mean():.6f}")

t_stat, p_val = stats.ttest_ind(
    c_shifts,
    a_shifts,
    equal_var=False
)

print(f"\nL29_N8568 Contradiction vs Ambiguity")
print(f"t-statistic: {t_stat:.6f}")
print(f"p-value: {p_val:.6f}")

# Test L26_N2788:
# Lack of Knowledge vs Contradictory Context

lok_shifts = df[
    (df["neuron_id"] == "L26_N2788") &
    (df["category"] == "lack_of_knowledge")
]["entropy_shift"]

cc_shifts = df[
    (df["neuron_id"] == "L26_N2788") &
    (df["category"] == "contradictory_context")
]["entropy_shift"]

print(f"Lack of Knowledge prompts: {len(lok_shifts)}")
print(f"Contradiction prompts: {len(cc_shifts)}")

print(f"Lack of Knowledge mean shift: {lok_shifts.mean():.6f}")
print(f"Contradiction mean shift: {cc_shifts.mean():.6f}")

t_stat, p_val = stats.ttest_ind(
    lok_shifts,
    cc_shifts,
    equal_var=False
)

print(f"\nL26_N2788 Lack of Knowledge vs Contradiction")
print(f"t-statistic: {t_stat:.6f}")
print(f"p-value: {p_val:.6f}")