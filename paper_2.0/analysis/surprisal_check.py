"""Chair note N6 (post hoc): why the regex surprisal coefficient falls when the popularity arm is added although
surprisal is uncorrelated with log page views over the pool. Writes analysis/surprisal_check.txt."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pointbiserialr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parent.parent, HERE.parent.parent / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
S = {}
for l in open(REPO / "results/round3/e2_surprisal.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l); S[r["subj"]] = r["name_surprisal"]
rows = []
for l in open(REPO / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("subj") in S and r.get("s_pop"):
        rows.append((S[r["subj"]], np.log(float(r["s_pop"])), r["arm"] == "uncertain", r["slick"]))
out = []
su, lp, ob, sl = map(np.array, zip(*rows))
out.append(f"all pool items n={len(su)}: Spearman(surprisal, log page views) {spearmanr(su, lp).correlation:+.3f}")
for nm, ar in (("obscure arm", ob), ("popular arm", ~ob)):
    out.append(f"  within {nm} n={ar.sum()}: Spearman {spearmanr(su[ar], lp[ar]).correlation:+.3f}; mean surprisal {su[ar].mean():.2f}")
lab = np.isin(sl, ["HighlyKnown", "Unknown"])
out.append(f"labelled (regex model) n={lab.sum()}: point-biserial(surprisal, obscure arm) {pointbiserialr(ob[lab], su[lab]).correlation:+.3f}; "
           f"Spearman(surprisal, log page views) {spearmanr(su[lab], lp[lab]).correlation:+.3f}")
(HERE / "surprisal_check.txt").write_text("\n".join(out) + "\n", encoding="utf-8"); print("\n".join(out))

# partial correlation of surprisal and the obscure arm given the S2 covariates (relation fixed effects, unknown, name tokens)
import pandas as pd

rows = []
for l in open(REPO / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("subj") in S and r.get("s_pop") and r["slick"] in ("HighlyKnown", "Unknown"):
        rows.append(dict(prop=r["prop"], surp=S[r["subj"]], obscure=float(r["arm"] == "uncertain"), unknown=float(r["slick"] == "Unknown"),
                         ntok=len(r["subj"].split()), regex=r["hedge_rate"]))
D = pd.DataFrame(rows)
X = pd.get_dummies(D[["unknown", "prop"]], columns=["prop"], dtype=float).to_numpy(float)
res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
rs, ro = res(D.surp.to_numpy(float)), res(D.obscure.to_numpy(float))
line = f"partial correlation(surprisal, obscure | relation, unknown) n={len(D)}: {np.corrcoef(rs, ro)[0, 1]:+.3f}; " \
       f"correlation(surprisal, unknown) {np.corrcoef(D.surp, D.unknown)[0, 1]:+.3f}"
print(line)
with open(HERE / "surprisal_check.txt", "a", encoding="utf-8") as f:
    f.write(line + "\n")
