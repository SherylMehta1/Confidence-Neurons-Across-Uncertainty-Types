"""MERe audit (post hoc, existing data only). Popularity preference inside and outside the three relations that hold
most obscure-known facts (capital, country, sport), with within-cell bootstraps (10,000, seed 0).
Writes analysis/mere_checks.txt."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE.parent.parent / "Confidence-Neurons-Across-Uncertainty-Types" / "results" / "hedge_measure_v2"
TOP = {"capital", "country", "sport"}

items = {}
with open(D / "items_llama.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        items[r["key"]] = r
rates = pd.read_csv(D / "item_rates.csv").query("model == 'llama'")[["key", "prim"]]
rates["arm"] = [items[k]["arm"] for k in rates.key]
rates["know"] = [items[k]["slick"] for k in rates.key]
rates["prop"] = [k.split("::")[0] for k in rates.key]


def pref(df, B=10_000, seed=0):
    a = df[(df.arm == "uncertain") & (df.know == "HighlyKnown")].prim.to_numpy()
    b = df[(df.arm == "control") & (df.know == "Unknown")].prim.to_numpy()
    rng = np.random.default_rng(seed)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a.mean() - b.mean(), lo, hi, len(a), len(b), a.mean(), b.mean()


top = rates[rates.prop.isin(TOP)]
out = rates[~rates.prop.isin(TOP)]
lines = []
for name, df in (("all", rates), ("inside", top), ("outside", out)):
    d, lo, hi, na, nb, ra, rb = pref(df)
    lines.append(f"{name}: preference {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; obscure known n={na} rate {ra:.3f}; popular unknown n={nb} rate {rb:.3f}")
k = top[top.know == "HighlyKnown"]
o, p = k[k.arm == "uncertain"].prim.to_numpy(), k[k.arm == "control"].prim.to_numpy()
rng = np.random.default_rng(0)
bs = [rng.choice(o, len(o)).mean() - rng.choice(p, len(p)).mean() for _ in range(10_000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
lines.append(f"inside known gap: {o.mean() - p.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]; obscure known n={len(o)} rate {o.mean():.3f}; popular known n={len(p)} rate {p.mean():.3f}")
text = "\n".join(lines) + "\n"
(HERE / "mere_checks.txt").write_text(text, encoding="utf-8")
print(text)
