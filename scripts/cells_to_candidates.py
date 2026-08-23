"""
Adapter: turn external evaluation "cell" files (e.g. the R1 / R2 PopQA popularity-quintile cells of the
uncertainty-estimation grid) into twin candidates for scripts/gate_twins.py, so both projects share one
PopQA sample.

Input: one or two files (jsonl or csv) with, per item, a question, a gold alias list, and optionally a
relation / template field and a popularity field. Column names are configurable; sensible defaults cover
PopQA-derived files (question, possible_answers | answers | aliases, prop | relation, s_pop | popularity).

Pairing: every item of the LOW file (uncertain candidates) is paired with an unused item of the HIGH
file (control candidates) sharing the same relation, nearest in question length (cross-relation pairing only
with --allow-cross-relation).
With a single file and --split-by-popularity, items are split at the median popularity within relation.

Writes data/<category>/candidates.jsonl in the gate_twins schema (uncertain.gold is kept: the gate then
requires Unknown on that side, as for the familiarity arm). Split labels from the cell files (train /
calibration / test) are carried in `meta.cell_split` so downstream work can respect them.
Usage: python scripts/cells_to_candidates.py --low cells/R2.jsonl --high cells/R1.jsonl --category familiarity_grid
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import ast
import csv
import json
import random
from collections import defaultdict

from _common import REPO_ROOT


def read_rows(path):
    path = _Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    with open(path, encoding="utf-8") as f:
        txt = f.read().strip()
    return json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]


def first_present(row, names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x) for x in ast.literal_eval(s)]
        except Exception:
            try:
                return [str(x) for x in json.loads(s)]
            except Exception:
                pass
    return [a.strip() for a in s.split("|") if a.strip()] if "|" in s else [s]


def normalize_rows(rows, a):
    out = []
    for i, r in enumerate(rows):
        q = first_present(r, a.question_cols)
        gold = as_list(first_present(r, a.gold_cols))
        if not q or not gold:
            continue
        q = str(q).strip()
        q = q if q.endswith("?") else q + "?"
        pop = first_present(r, a.pop_cols)
        out.append(dict(question=q, gold=gold, relation=str(first_present(r, a.relation_cols) or "any"),
                        pop=float(pop) if pop not in (None, "") else None, src_id=str(first_present(r, a.id_cols) or i),
                        cell_split=first_present(r, a.split_cols)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", required=True, help="uncertain candidates file (e.g. R2, bottom popularity)")
    ap.add_argument("--high", default=None, help="control candidates file (e.g. R1, top popularity); omit with --split-by-popularity")
    ap.add_argument("--split-by-popularity", action="store_true")
    ap.add_argument("--category", default="familiarity_grid")
    ap.add_argument("--source", default="external_cells")
    ap.add_argument("--question-cols", dest="question_cols", default="question,prompt,q")
    ap.add_argument("--gold-cols", dest="gold_cols", default="possible_answers,answers,aliases,gold,answer")
    ap.add_argument("--relation-cols", dest="relation_cols", default="prop,relation,template,prop_id")
    ap.add_argument("--pop-cols", dest="pop_cols", default="s_pop,popularity,pop")
    ap.add_argument("--id-cols", dest="id_cols", default="id,item_id,qid")
    ap.add_argument("--split-cols", dest="split_cols", default="split,cell_split")
    ap.add_argument("--allow-cross-relation", action="store_true", help="pair across relations when a relation pool is exhausted (breaks the twin rule; off by default)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for k in ("question_cols", "gold_cols", "relation_cols", "pop_cols", "id_cols", "split_cols"):
        setattr(args, k, [c.strip() for c in getattr(args, k).split(",") if c.strip()])

    low = normalize_rows(read_rows(args.low), args)
    if args.high:
        high = normalize_rows(read_rows(args.high), args)
    elif args.split_by_popularity:
        by_rel = defaultdict(list)
        for r in low:
            if r["pop"] is not None:
                by_rel[r["relation"]].append(r)
        low, high = [], []
        for rel, items in by_rel.items():
            items.sort(key=lambda r: r["pop"])
            k = len(items) // 2
            low += items[:k]; high += items[k:]
    else:
        raise SystemExit("give --high or --split-by-popularity")

    rng = random.Random(args.seed)
    pool = defaultdict(list)
    for r in high:
        pool[r["relation"]].append(r)
    for v in pool.values():
        rng.shuffle(v)
    out, unmatched = [], 0
    for r in low:
        cands = pool.get(r["relation"]) or (pool.get("any") if "any" in pool else None)
        if not cands and args.allow_cross_relation:
            cands = next((v for v in pool.values() if v), None)
        if not cands:
            unmatched += 1
            continue
        j = min(range(len(cands)), key=lambda i: abs(len(cands[i]["question"].split()) - len(r["question"].split())))
        c = cands.pop(j)
        out.append(dict(twin_id=f"grid_{len(out):05d}", category=args.category, source=args.source,
                        uncertain=dict(question=r["question"], gold=r["gold"], meta=dict(src_id=r["src_id"], relation=r["relation"], popularity=r["pop"], cell_split=r["cell_split"], cell="low")),
                        control=dict(question=c["question"], gold=c["gold"], meta=dict(src_id=c["src_id"], relation=c["relation"], popularity=c["pop"], cell_split=c["cell_split"], cell="high"))))
    d = REPO_ROOT / "data" / args.category
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "candidates.jsonl", "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} candidate pairs to {d / 'candidates.jsonl'} ({unmatched} low items unmatched; {len(low)} low / {len(high)} high items read)")


if __name__ == "__main__":
    main()
