"""X2 step 1: four-level popularity gradient from replies that already exist (the paper's 387 questions and E14's 400 adjacent ones).
Descriptive only; known/unknown labels as stored. CPU. Output results/x2/existing_gradient.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/hedge_measure_v2"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rows = []
for tag, band in (("llama", {"uncertain": "1: 0-10th percentile", "control": "4: 90-100th percentile"}), ("llama_adj", {"uncertain": "2: 10-30th percentile", "control": "3: 70-90th percentile"})):
    J = {j: {(r["key"], r["s"]): r["label"] == "DONT_KNOW" for r in read(D / f"judge_{j}_{tag}.jsonl")} for j in ("qwen32b", "gemma27b")}
    for g in read(D / f"gens_{tag}.jsonl"):
        rows.append(dict(band=band[g["arm"]], knowledge="known" if g["slick"] == "HighlyKnown" else "unknown", both=np.mean([J["qwen32b"][g["key"], s] and J["gemma27b"][g["key"], s] for s in range(3)])))
T = pd.DataFrame(rows)
rng = np.random.default_rng(0)
out = []
for (k, b), g in T.groupby(["knowledge", "band"]):
    v = g.both.to_numpy(); bt = v[rng.integers(0, len(v), (10_000, len(v)))].mean(1)
    out.append(dict(knowledge=k, popularity_band=b, n=len(v), two_judge_abstention=v.mean(), lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5), draws=10_000, seed=0))
O = pd.DataFrame(out); O.to_csv(REPO / "results/x2/existing_gradient.csv", index=False, float_format="%.6g"); print(O.round(3).to_string(index=False))
