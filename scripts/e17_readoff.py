"""E17 read-off (paper/PREREG/e17_nla_qwen.md, sha 2d6712e1; exploratory). Per position: per-question rates (mean over
3 explanations) of labels A (unfamiliar entity), B (uncertainty / not knowing), C (names an answer) and a gold alias in
the explanation (subject-name echoes excluded); popularity and knowledge effects with 10,000 within-cell item bootstraps;
and the AR critic cosine by cell. Writes results/round3/e17_summary.txt.

  python scripts/e17_readoff.py
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import json
import re

import numpy as np
import pandas as pd

from _common import REPO_ROOT
from round3_analysis import CELLS, effects, fmt

O = REPO_ROOT / "results/round3/e17"
LINES = []


def say(s=""):
    print(s, flush=True); LINES.append(s)


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def nrm(x):
    return " " + re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip() + " "


meta = read(O / "acts_meta.jsonl")
ex = {(x["i"], x["s"]): x for x in read(O / "explanations.jsonl")}
lab = {(x["i"], x["s"]): x for x in read(O / "labels.jsonl")}
cri = {(x["i"], x["s"]): x for x in read(O / "critic.jsonl")}
say("E17 (exploratory): NLA readout of Qwen2.5-7B-Instruct layer 20 (PREREG e17_nla_qwen.md, sha 2d6712e1)")
say(f"activations {len(meta)}; explanations {len(ex)} (with <explanation> tags: {sum(1 for x in ex.values() if x['explanation'])}); "
    f"labelled {len(lab)}; critic-scored {len(cri)}")
rows = []
for (i, s), x in ex.items():
    m = meta[i]
    e = x["explanation"]
    if not e:
        continue
    subj = nrm(m["subj"])
    al = [nrm(a).strip() for a in m["aliases"] if len(str(a).strip()) >= 3]
    L = lab.get((i, s), {})
    rows.append(dict(i=i, key=m["key"], pos=m["pos"], arm=m["arm"], know=m["slick"],
                     A=L.get("A"), B=L.get("B"), C=L.get("C"),
                     gold=any(" " + a + " " in nrm(e) for a in al if (" " + a + " ") not in subj),
                     cos=cri.get((i, s), {}).get("cos")))
D = pd.DataFrame(rows)
for pos, name in (("P1", "last prompt token"), ("P2", "subject's last token")):
    G = D[D.pos == pos]
    say(f"\n[{pos}: {name}] explanations {len(G)}; critic cos mean {G.cos.mean():.3f} "
        + "(" + "  ".join(f"{a[:3]}/{k[:4]} {G[(G.arm == a) & (G.know == k)].cos.mean():.3f}" for a, k in CELLS) + ")")
    for col, desc in (("A", "A unfamiliar entity"), ("B", "B uncertainty / not knowing"), ("C", "C names an answer"), ("gold", "gold alias in explanation")):
        H = G.dropna(subset=[col])
        it = H.groupby(["key", "arm", "know"])[col].mean().reset_index()
        e, n = effects(it, col)
        if e is None:
            say(f"  {desc}: a cell is empty"); continue
        say(f"  {desc}: {fmt(e)}\n    cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}")
(REPO_ROOT / "results/round3/e17_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
