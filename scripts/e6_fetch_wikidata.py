"""E6 (paper/PREREG/e6_entity_knowledge.md, sha 937b46e3): fetch English Wikidata descriptions for every pool subject.

  python scripts/e6_fetch_wikidata.py  ->  results/round3/e6_wikidata.jsonl"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
items = {}
for l in open(REPO / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("subj") and r.get("s_uri"):
        items.setdefault(f"{r['prop']}::{r['subj']}", r["s_uri"].rsplit("/", 1)[-1])
qids = sorted(set(items.values()))
desc = {}
for i in range(0, len(qids), 50):
    url = ("https://www.wikidata.org/w/api.php?action=wbgetentities&props=descriptions|labels&languages=en&format=json&ids="
           + urllib.parse.quote("|".join(qids[i:i + 50])))
    j = {}
    for t in range(8):
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-paper-pilot/1.0"}), timeout=60).read())
            break
        except Exception as e:
            print("retry", i, e); time.sleep(10 * (t + 1))
    for q, e in j.get("entities", {}).items():
        d = e.get("descriptions", {}).get("en", {}).get("value")
        if d:
            desc[q] = d
    time.sleep(2.0)
out = REPO / "results/round3/e6_wikidata.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for k, q in items.items():
        if q in desc:
            f.write(json.dumps(dict(key=k, qid=q, description=desc[q]), ensure_ascii=False) + "\n")
print(f"items {len(items)}, qids {len(qids)}, with English description {sum(q in desc for q in items.values())}")
