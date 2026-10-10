"""Score the two hedge judges against human labels (results/hedge_measure_v2/HUMAN_VALIDATION_README.md).
Usage: python scripts/hedge_measure_v2_human_score.py <labelled sheet.csv> [<second labeller's sheet.csv>]"""
import sys
import numpy as np
import pandas as pd

D = "results/hedge_measure_v2/"
COL = "label_DONT_KNOW_or_CLARIFY_or_ANSWER"


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


key = pd.read_csv(D + "human_validation_KEY_do_not_open_before_labelling.csv")
sheets = [pd.read_csv(p, encoding="utf-8-sig") for p in sys.argv[1:]]
for i, s in enumerate(sheets):
    m = key.merge(s[["id", COL]], on="id")
    m[COL] = m[COL].astype(str).str.strip().str.upper()
    m = m[m[COL].isin(["DONT_KNOW", "CLARIFY", "ANSWER"])]
    h = m[COL] == "DONT_KNOW"
    both = (m.qwen32b == "DONT_KNOW") & (m.gemma27b == "DONT_KNOW")
    print(f"labeller {i + 1}: {len(m)} labelled; human DONT_KNOW rate {h.mean():.2f}")
    print(f"  kappa human vs qwen32b {kappa(h, m.qwen32b == 'DONT_KNOW'):.2f} | vs gemma27b {kappa(h, m.gemma27b == 'DONT_KNOW'):.2f} | vs both-judges rule {kappa(h, both):.2f}")
    print(f"  both-judges rule: precision {h[both].mean():.2f}, recall {both[h].mean():.2f}")
    print("  per cell, human vs both-judges DONT_KNOW rate:", {c: (round(h[m.cell == c].mean(), 2), round(both[m.cell == c].mean(), 2)) for c in sorted(m.cell.unique())})
if len(sheets) == 2:
    a = sheets[0].set_index("id")[COL].astype(str).str.upper().str.strip() == "DONT_KNOW"
    b = sheets[1].set_index("id")[COL].astype(str).str.upper().str.strip() == "DONT_KNOW"
    a, b = a.align(b, join="inner")
    print(f"inter-human kappa: {kappa(a, b):.2f}")
