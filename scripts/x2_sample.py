"""X2 sample: up to 300 PopQA questions per relation, spread evenly over within-relation popularity deciles, and the 200-question pilot.
PopQA is deduplicated exactly as in scripts/build_familiarity_twins.py (first row per relation and subject; null subjects dropped).
Output results/x2/{sample.jsonl, sample_summary.csv}; seed 0."""
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/x2"
SEED, PER_DECILE, PILOT = 0, 30, 200
ds = load_dataset("akariasai/PopQA", split="test")
D = pd.DataFrame([r for r in ds if r["subj"] is not None]).drop_duplicates(["prop", "subj"])
rng = random.Random(SEED)
out = []
for prop, g in sorted(D.groupby("prop")):
    n = len(g)
    rank = (g.s_pop.rank(method="average") - 0.5) / n  # 0 = least viewed, 1 = most viewed; ties share a rank
    dec = (g.sort_values(["s_pop", "id"]).reset_index()["index"].to_frame().assign(d=lambda x: (x.index * 10 // n) + 1).set_index("index").d)
    for d in range(1, 11):
        ids = sorted(dec.index[dec == d])
        for i in sorted(rng.sample(ids, min(PER_DECILE, len(ids)))):
            r = g.loc[i]
            out.append(dict(key=f"{prop}::{r.subj}", prop=prop, subj=r.subj, question=r.question.strip(), possible_answers=json.loads(r.possible_answers), s_pop=int(r.s_pop),
                            pop_rank=round(float(rank[i]), 6), decile=d, n_relation=n))
by = {d: [x["key"] for x in out if x["decile"] == d] for d in range(1, 11)}
prng = random.Random(SEED + 1)
pilot = {k for d in by for k in prng.sample(by[d], PILOT // 10)}
for x in out:
    x["pilot"] = x["key"] in pilot
assert len({x["key"] for x in out}) == len(out) and len(pilot) == PILOT
OUT.mkdir(exist_ok=True)
p = OUT / "sample.jsonl"
p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
S = pd.DataFrame(out)
T = S.groupby("prop").agg(sampled=("key", "size"), in_popqa=("n_relation", "first"), pilot=("pilot", "sum"), min_per_decile=("decile", lambda s: s.value_counts().min())).reset_index()
T.to_csv(OUT / "sample_summary.csv", index=False)
print(T.to_string(index=False)); print(len(S), "questions;", S.decile.value_counts().sort_index().to_dict(), "\nsha256", hashlib.sha256(p.read_bytes()).hexdigest())
