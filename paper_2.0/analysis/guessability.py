"""Exploratory (post hoc) guessability control, prompted by review round 1 (attack P2/P3, chair I6). Writes analysis/guessability.csv. Two measures:
two-judge primary (results/hedge_measure_v2/item_rates.csv, llama) and regex (data/familiarity_v3/all_measured.jsonl).
guessable = a gold alias of >= 3 chars appears in the subject name, OR the answer is its relation's modal answer
in the pool and that answer covers >= 20% of the relation. Effects with 10,000 within-cell item bootstraps (seed 0)."""
import json, collections, re, csv
import numpy as np
from pathlib import Path
R = next(p for p in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[2] / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
rows = [json.loads(l) for l in open(R / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8") if l.strip()]
pool = {}
for r in rows:
    if r.get("subj") and r.get("slick") in ("HighlyKnown", "Unknown"):
        pool.setdefault((r["prop"], r["subj"]), r)
norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()
def aliases(r):
    a = r.get("aliases") or json.loads(r.get("possible_answers") or "[]")
    return [x for x in a if x]
mode = {}
for prop in {k[0] for k in pool}:
    c = collections.Counter(r["obj"] for k, r in pool.items() if k[0] == prop)
    top, n = c.most_common(1)[0]; mode[prop] = (top, n / sum(c.values()))
def guess(r):
    s = " " + norm(r["subj"]) + " "
    nm = any(len(norm(a)) >= 3 and (" " + norm(a) + " ") in s for a in aliases(r))
    top, share = mode[r["prop"]]
    return nm or (r["obj"] == top and share >= 0.2)
OUT = []
rng = np.random.default_rng(0)
def effects(items, B=10000, tag=None):
    cells = {(a, k): np.array([y for (aa, kk, y) in items if aa == a and kk == k], float)
             for a in ("uncertain", "control") for k in ("Unknown", "HighlyKnown")}
    def stat(c):
        m = {k: v.mean() for k, v in c.items()}
        pop = ((m["uncertain", "Unknown"] - m["control", "Unknown"]) + (m["uncertain", "HighlyKnown"] - m["control", "HighlyKnown"])) / 2
        know = ((m["uncertain", "Unknown"] - m["uncertain", "HighlyKnown"]) + (m["control", "Unknown"] - m["control", "HighlyKnown"])) / 2
        popU = m["uncertain", "Unknown"] - m["control", "Unknown"]
        return pop, know, pop - know, popU
    pt = stat(cells)
    bs = np.array([stat({k: rng.choice(v, len(v)) for k, v in cells.items()}) for _ in range(B)])
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    n = {f"{a[:3]}/{k[:4]}": len(v) for (a, k), v in cells.items()}
    if tag:
        OUT.append(dict(tag=tag, **{f"{nm}{s}": v for nm, p, l, h in zip(("pop", "know", "diff", "popU"), pt, lo, hi) for s, v in (("", p), ("_lo", l), ("_hi", h))}, n_total=sum(len(v) for v in cells.values()), n_obs_known=len(cells["uncertain", "HighlyKnown"]), mean_obs_known=float(cells["uncertain", "HighlyKnown"].mean())))
    means = {f"{a[:3]}/{k[:4]}": round(float(v.mean()), 3) for (a, k), v in cells.items()}
    return " ".join(f"{nm} {p:+.3f} [{l:+.3f},{h:+.3f}]" for nm, p, l, h in zip(("pop", "know", "pop-know", "popUnknownOnly"), pt, lo, hi)) + f"\n      n {n} means {means}"
# two-judge measure
judge = []
for r in csv.DictReader(open(R / "results/hedge_measure_v2/item_rates.csv", encoding="utf-8")):
    if r["model"] != "llama":
        continue
    prop, subj = r["key"].split("::", 1)
    it = pool.get((prop, subj))
    judge.append((r["arm"], r["slick"], float(r["prim"]), guess(it) if it else None))
print("judge items matched to pool:", sum(1 for j in judge if j[3] is not None), "of", len(judge))
print("TWO JUDGES, all     :", effects([(a, k, y) for a, k, y, g in judge], tag="judge_all"))
print("TWO JUDGES, no guess:", effects([(a, k, y) for a, k, y, g in judge if g is False], tag="judge_noguess"))
reg = [(r["arm"], r["slick"], r["hedge_rate"], guess(r)) for r in pool.values()]
print("REGEX, all          :", effects([(a, k, y) for a, k, y, g in reg], tag="regex_all"))
print("REGEX, no guess     :", effects([(a, k, y) for a, k, y, g in reg if not g], tag="regex_noguess"))
ent = [(r["arm"], r["slick"], r["entropy"], guess(r)) for r in pool.values()]
print("ENTROPY, no guess   :", effects([(a, k, y) for a, k, y, g in ent if not g], tag="entropy_noguess"))

import pandas as pd
g_ok = sum(1 for r in pool.values() if r["arm"] == "uncertain" and r["slick"] == "HighlyKnown" and guess(r))
n_ok = sum(1 for r in pool.values() if r["arm"] == "uncertain" and r["slick"] == "HighlyKnown")
df = pd.DataFrame(OUT); df["obs_known_guessable"] = g_ok; df["obs_known_total"] = n_ok
df.to_csv(Path(__file__).resolve().parent / "guessability.csv", index=False)
print(df.round(3).to_string())
