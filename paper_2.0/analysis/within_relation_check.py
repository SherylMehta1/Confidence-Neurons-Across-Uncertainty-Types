"""Main-conference planning (post hoc, existing data only). Is the popularity preference present within relations?
Uses Llama's full labeled pool and the 24-token regex rate (20 replies per question), the only abstention measure
available on all 2,303 labeled questions. Writes analysis/within_relation_check.txt."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "Confidence-Neurons-Across-Uncertainty-Types"
rows = {}
with open(REPO / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("subj") and r.get("slick") in ("HighlyKnown", "Unknown") and r.get("arm") in ("uncertain", "control"):
            rows.setdefault(f"{r['prop']}::{r['subj']}", (r["prop"], r["arm"], r["slick"], float(r["hedge_rate"])))
df = pd.DataFrame(rows.values(), columns=["prop", "arm", "know", "rate"])
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def effects(d):
    """Equal-weight 2x2 in one group of questions; nan when a cell is empty."""
    m = {(a, k): d[(d.arm == a) & (d.know == k)].rate.mean() for a in ("uncertain", "control") for k in ("HighlyKnown", "Unknown")}
    ok, ou, pk, pu = m["uncertain", "HighlyKnown"], m["uncertain", "Unknown"], m["control", "HighlyKnown"], m["control", "Unknown"]
    return dict(pop_known=ok - pk, pop_unknown=ou - pu, know_obscure=ou - ok, know_popular=pu - pk, pref=ok - pu)


say(f"Llama labeled questions {len(df)}; regex rate, 20 replies, 24 tokens")
tab = df.groupby(["prop", "arm", "know"]).size().unstack(["arm", "know"]).fillna(0).astype(int)
say(tab.to_string())
say()
say("pooled (all relations): " + "  ".join(f"{k} {v:+.3f}" for k, v in effects(df).items()))
for k in (1, 3, 5, 10):
    keep = [p for p in tab.index if (tab.loc[p] >= k).all()]
    say(f"relations with every cell >= {k}: {len(keep)} {keep}")
keep = [p for p in tab.index if (tab.loc[p] >= 3).all()]
res = pd.DataFrame({p: effects(df[df.prop == p]) for p in keep}).T
say()
say("within-relation effects (relations with every cell >= 3):")
say(res.round(3).to_string())
sub = df[df.prop.isin(keep)]
rng = np.random.default_rng(0)
boots = []
for _ in range(5000):
    e = []
    for p in keep:
        g = sub[sub.prop == p]
        b = pd.concat([c.sample(len(c), replace=True, random_state=int(rng.integers(1 << 31))) for _, c in g.groupby(["arm", "know"])])
        e.append(effects(b))
    boots.append(pd.DataFrame(e).mean())
boots = pd.DataFrame(boots)
est = res.mean()
say()
say(f"mean over those {len(keep)} relations, equal weight, question bootstrap within relation and cell (5,000):")
for c in est.index:
    lo, hi = np.percentile(boots[c], [2.5, 97.5])
    say(f"  {c}: {est[c]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
(HERE / "within_relation_check.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
