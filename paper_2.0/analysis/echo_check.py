"""Round-4 review (post hoc): reconcile App. A's free-reply alias share (round2_checks, which counts aliases that are
part of the subject name) with Sec. 3.1's alias share inside abstentions (round3_label_checks, which excludes them).
Recomputes, per cell, the share of free replies containing a gold alias with and without subject-name echoes.
Writes analysis/echo_check.txt."""
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parent.parent, HERE.parent.parent / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
D = REPO / "results/hedge_measure_v2"


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(x):
    return " " + re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip() + " "


pool = {}
for r in read(REPO / "data/familiarity_v3/all_measured.jsonl"):
    if r.get("subj"):
        pool.setdefault(f"{r['prop']}::{r['subj']}", r)


def aliases(r):
    a = r.get("aliases") or json.loads(r.get("possible_answers") or "[]")
    a = a if isinstance(a, list) else json.loads(a)
    return [norm(x).strip() for x in a if len(str(x).strip()) >= 3]


lab = {}
for jt in ("qwen32b", "gemma27b"):
    for x in read(D / f"judge_{jt}_llama.jsonl"):
        lab.setdefault((x["key"], x["s"]), []).append(x["label"])
rows = []
for g in read(D / "gens_llama.jsonl"):
    if g["key"] not in pool:
        continue
    al, subj = aliases(pool[g["key"]]), norm(pool[g["key"]]["subj"])
    for s, txt in enumerate(g["samples"]):
        t = norm(txt)
        rows.append(dict(arm=g["arm"], slick=g["slick"], key=g["key"],
                         any_alias=any(" " + a + " " in t for a in al),
                         alias_no_echo=any(" " + a + " " in t for a in al if (" " + a + " ") not in subj),
                         abstain=lab.get((g["key"], s), []) == ["DONT_KNOW", "DONT_KNOW"]))
R = pd.DataFrame(rows)
out = []
for (arm, k), g in R.groupby(["arm", "slick"]):
    na = g[~g.abstain]; ab = g[g.abstain]
    out.append(f"{'obscure' if arm == 'uncertain' else 'popular'} {k:11s} replies {len(g):4d} | alias incl. name echoes {g.any_alias.mean():.2f} "
               f"(non-abstaining {na.any_alias.mean():.2f}, abstaining {ab.any_alias.mean() if len(ab) else float('nan'):.2f}) | "
               f"alias excl. echoes {g.alias_no_echo.mean():.2f} (non-abstaining {na.alias_no_echo.mean():.2f}, "
               f"abstaining {ab.alias_no_echo.mean() if len(ab) else float('nan'):.2f})")
ok = R[(R.arm == "uncertain") & (R.slick == "HighlyKnown")]
it = ok.groupby("key").agg(c=("alias_no_echo", "max"), a=("abstain", "max"))
out.append(f"obscure known questions with both an answering reply (alias, echoes excluded) and an abstention: {int((it.c & it.a).sum())} of {len(it)}")
(HERE / "echo_check.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
