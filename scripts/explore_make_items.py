"""Exploratory (future work, not in the paper): sample the abstention-measurement items for a new model exactly as
hedge_measure_v2_criterion.md did for the four families: up to 100 items per popularity x knowledge cell from the
model's own pool (HighlyKnown/Unknown only), seed 0.

  python scripts/explore_make_items.py --pool data/familiarity_v3_<tag> --tag <tag>"""
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
rows, seen = [], set()
for l in open(REPO / a.pool / "all_measured.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r.get("subj") or r.get("slick") not in ("HighlyKnown", "Unknown"):
        continue
    k = f"{r['prop']}::{r['subj']}"
    if k not in seen:
        seen.add(k); rows.append(dict(model=a.tag, key=k, q=r["q"], arm=r["arm"], slick=r["slick"]))
rng = random.Random(0)
out = []
for arm in ("uncertain", "control"):
    for kn in ("Unknown", "HighlyKnown"):
        cell = [r for r in rows if r["arm"] == arm and r["slick"] == kn]
        out += rng.sample(cell, min(100, len(cell)))
p = REPO / "results/hedge_measure_v2" / f"items_{a.tag}.jsonl"
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
print(f"{a.tag}: {len(out)} items; cells " + ", ".join(f"{arm[:3]}/{kn[:4]} {sum(1 for r in out if r['arm'] == arm and r['slick'] == kn)}"
                                               for arm in ("uncertain", "control") for kn in ("Unknown", "HighlyKnown")))
