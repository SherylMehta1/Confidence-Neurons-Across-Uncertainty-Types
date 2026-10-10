"""Exploratory checks requested by review round 2 (methods referee F1, F2; framing referee Q1). Post hoc.
(1) Free-reply correctness: in each popularity x knowledge cell, how often Llama's free (96-token) replies contain a gold
    alias, overall and among replies the two judges did not label DONT_KNOW.
(2) Does the popularity effect on abstention survive controlling for answer entropy (and relation)? Linear probability
    models of the item abstention rate on obscure, unknown and standardised entropy, with relation fixed effects;
    95% CIs from 2,000 item bootstraps (seed 0). Two-judge measure (387 items) and regex measure (full pool).
Writes analysis/round2_checks.txt."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = next(p for p in (HERE.parents[1], HERE.parents[1] / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
pool = {}
for l in open(R / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        if r.get("subj") and r.get("slick") in ("HighlyKnown", "Unknown"):
            pool.setdefault(f"{r['prop']}::{r['subj']}", r)
def aliases(r):
    a = r.get("aliases") or json.loads(r.get("possible_answers") or "[]")
    return [norm(x) for x in a if x and len(norm(x)) >= 2]
out = []
say = lambda s="": (print(s), out.append(s))

# (1) free-reply correctness
gens = [json.loads(l) for l in open(R / "results/hedge_measure_v2/gens_llama.jsonl", encoding="utf-8") if l.strip()]
lab = {}
for j in ("qwen32b", "gemma27b"):
    for l in open(R / f"results/hedge_measure_v2/judge_{j}_llama.jsonl", encoding="utf-8"):
        if l.strip():
            x = json.loads(l); lab.setdefault((x["key"], x["s"]), []).append(x["label"])
rows = []
for g in gens:
    al = aliases(pool[g["key"]])
    for s, txt in enumerate(g["samples"]):
        t = " " + norm(txt) + " "
        correct = any((" " + a + " ") in t for a in al)
        dk = lab.get((g["key"], s), []) == ["DONT_KNOW", "DONT_KNOW"]
        rows.append(dict(key=g["key"], arm=g["arm"], slick=g["slick"], correct=correct, abstain=dk))
d = pd.DataFrame(rows)
say("(1) Llama free replies (3 per question, 96 tokens): share containing a gold alias")
for (a, k), g in d.groupby(["arm", "slick"]):
    na = g[~g.abstain]
    items = g.groupby("key").agg(any_correct=("correct", "max"), any_abstain=("abstain", "max"))
    say(f"  {('obscure' if a == 'uncertain' else 'popular'):8s} {k:11s} replies {len(g):4d}  correct {g.correct.mean():.2f}  "
        f"| non-abstaining replies {len(na):4d} correct {na.correct.mean() if len(na) else float('nan'):.2f}  "
        f"| questions with both a correct reply and an abstention: {int((items.any_correct & items.any_abstain).sum())} of {len(items)}")

# (2) popularity effect controlling for entropy
def lpm(df, y, cols, B=2000):
    X = pd.get_dummies(df[cols + ["prop"]], columns=["prop"], drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    Xv, yv = X.to_numpy(float), df[y].to_numpy(float)
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(yv), len(yv))
        bs.append(np.linalg.lstsq(Xv[i], yv[i], rcond=None)[0])
    bs = np.array(bs)
    return {c: (beta[X.columns.get_loc(c)], *np.percentile(bs[:, X.columns.get_loc(c)], [2.5, 97.5])) for c in cols}
P = pd.DataFrame([dict(key=k, prop=r["prop"], obscure=float(r["arm"] == "uncertain"), unknown=float(r["slick"] == "Unknown"),
                       entropy=r["entropy"], regex=r["hedge_rate"]) for k, r in pool.items()])
P["entropy_z"] = (P.entropy - P.entropy.mean()) / P.entropy.std()
J = pd.read_csv(R / "results/hedge_measure_v2/item_rates.csv").query("model == 'llama'")[["key", "prim"]].merge(P, on="key")
say("\n(2) Linear probability models of the abstention rate, relation fixed effects, 95% bootstrap CI")
for name, df, y in (("two judges (n=%d)" % len(J), J, "prim"), ("regex, full pool (n=%d)" % len(P), P, "regex")):
    for cols in (["obscure", "unknown"], ["obscure", "unknown", "entropy_z"]):
        res = lpm(df, y, cols)
        say(f"  {name:28s} " + "  ".join(f"{c} {b:+.3f} [{lo:+.3f},{hi:+.3f}]" for c, (b, lo, hi) in res.items()))
(HERE / "round2_checks.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
