"""Exploratory (future work, NOT in the paper): OLMo-2-7B-Instruct through the E5 pipeline (own SliCK pool, up to 100
questions per cell, 3 free replies, three judges). Same read-off as round3_analysis.e5. Writes analysis/explore_olmo.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parent.parent, HERE.parent.parent / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
sys.path.insert(0, str(REPO / "scripts"))
from round3_analysis import cohen, effects, fleiss, fmt, read_jsonl  # noqa: E402

D = REPO / "results/hedge_measure_v2"
TAG = "olmo2_7b"
out = []
meta = {r["key"]: r for r in read_jsonl(D / f"items_{TAG}.jsonl")}
J = {}
for jt in ("qwen32b", "gemma27b", "llama70b"):
    for x in read_jsonl(D / f"judge_{jt}_{TAG}.jsonl"):
        J.setdefault((x["key"], x["s"]), {})[jt] = x["label"] == "DONT_KNOW"
full = {k: v for k, v in J.items() if len(v) == 3}
fk = fleiss([[v["qwen32b"], v["gemma27b"], v["llama70b"]] for v in full.values()])
k2 = cohen([v["qwen32b"] for v in full.values()], [v["gemma27b"] for v in full.values()])
out.append(f"OLMo-2-7B-Instruct (exploratory): {len(meta)} questions, {len(full)} replies; two-judge kappa {k2:.3f}; Fleiss kappa {fk:.3f}")
out.append("cell counts: " + str(pd.Series([r["arm"] + "/" + r["slick"] for r in meta.values()]).value_counts().to_dict()))
for rule, f in (("majority of 3", lambda v: float(sum(v.values()) >= 2)), ("both of two", lambda v: float(v["qwen32b"] and v["gemma27b"]))):
    it = {}
    for (k, s), v in full.items():
        it.setdefault(k, []).append(f(v))
    df = pd.DataFrame([dict(key=k, arm=meta[k]["arm"], know=meta[k]["slick"], y=np.mean(v)) for k, v in it.items() if k in meta])
    e, n = effects(df, "y")
    if e is None:
        out.append(f"{rule}: a cell is empty, n {n}"); continue
    verdict = ("FOLLOWS POPULARITY" if e["diff"][1] > 0 else ("FOLLOWS KNOWLEDGE" if e["diff"][2] < 0 else "NEITHER"))
    out.append(f"{rule}: {fmt(e)}  n {n}\n  cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})} -> {verdict}"
               + ("" if fk >= 0.60 else "  (judges below the 0.60 bar: UNMEASURED by the E5 rule)"))
(HERE / "explore_olmo.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
