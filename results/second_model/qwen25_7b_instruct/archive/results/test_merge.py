from scipy.stats import pearsonr
import pandas as pd

summary = pd.read_csv("merged_summary.csv", index_col="neuron_id")
print(summary.columns.tolist())

pairs = [
    ("ambiguity", "contradictory_context"),
    ("ambiguity", "lack_of_knowledge"),
    ("contradictory_context", "lack_of_knowledge"),
]

for a, b in pairs:
    r, p = pearsonr(summary[a], summary[b])
    print(f"{a} vs {b}: r={r:.3f}, p={p:.4f} {'**significant**' if p < 0.05 else ''}")


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = summary[["ambiguity", "contradictory_context", "lack_of_knowledge"]].values
X_scaled = StandardScaler().fit_transform(X)  # zero mean, unit variance per column

pca = PCA()
pca.fit(X_scaled)

print("Explained variance ratio per component:", pca.explained_variance_ratio_)
print("Loadings (standardized):")
print(dict(zip(["ambiguity", "contradictory_context", "lack_of_knowledge"], pca.components_[0])))

# Rank each neuron's effect size within each category
ranks = summary[
    ["ambiguity", "contradictory_context", "lack_of_knowledge"]
].rank(ascending=False)

summary["avg_rank"] = ranks.mean(axis=1)

print("\nNeuron ranking by average effect size:")
print(
    summary
    .sort_values("avg_rank")
    [["ambiguity", "contradictory_context", "lack_of_knowledge", "avg_rank"]]
)