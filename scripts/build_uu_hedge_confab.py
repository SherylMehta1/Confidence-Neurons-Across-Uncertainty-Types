"""
U/U arm: hedgers vs confabulators, both drawn from genuinely-unknown items.

Source population is the retained screen output (all_measured.jsonl, the file L0's
Fix 1 added): every item the familiarity screen scored, including the ones that
never became a twin. Rows with arm == "uncertain" and slick == "Unknown" are items
the model was tested on and genuinely does not know -- the population the U/U
design calls for. Everything needed (hedge_rate, entropy, n_distinct, s_pop,
n_q_tokens) is already computed there, so this is pure data wrangling: no model.

Classification (thresholds are flags; defaults follow the U/U spec):
  hedger       : hedge_rate >= --hedger-min
  confabulator : hedge_rate <= --confab-max AND n_distinct <= --confab-max-distinct

Matching, within relation: question token length within --max-len-diff, first-token
entropy within --entropy-tol nats, then closest popularity. Confabulators are the
scarce side, so the loop iterates over them and consumes the closest eligible
hedger, which maximizes pair yield.

Writes data/<out-category>/{prompts,controls}.jsonl in the repo schema
(hedgers are prompts/is_control=False, confabulators are controls/is_control=True,
which is what load_category() and direction_bridge.py's two-group split expect),
plus gate_report.json and provenance.

Usage:
  python scripts/build_uu_hedge_confab.py --overwrite
  python scripts/build_uu_hedge_confab.py --confab-max-distinct 10 --overwrite
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np

from _common import REPO_ROOT, guard_output
from shared.prompt_format import seeded_shuffle
from shared.provenance import write_provenance

SOURCE = "data/familiarity_v2/all_measured.jsonl"


def load_pool(path):
    """Genuinely-unknown items from the retained screen output."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("subj") is not None]  # PopQA rows with no subject have no usable question
    return rows, [r for r in rows if r.get("arm") == "uncertain" and r.get("slick") == "Unknown"]


def classify(pool, hedger_min, confab_max, confab_max_distinct):
    hedgers = [r for r in pool if float(r["hedge_rate"]) >= hedger_min]
    confabs = [r for r in pool if float(r["hedge_rate"]) <= confab_max
              and int(r["n_distinct"]) <= confab_max_distinct]
    return hedgers, confabs


def match(hedgers, confabs, max_len_diff, entropy_tol):
    """Pair each confabulator (the scarce side) with the closest-popularity hedger
    of the same relation that is within the length and entropy tolerances."""
    by_rel = {}
    for h in hedgers:
        by_rel.setdefault(h["prop"], []).append(h)
    pairs, fails = [], {"no_hedger_in_relation": 0, "len": 0, "entropy": 0}
    for c in sorted(confabs, key=lambda r: (r["prop"], r["subj"])):
        pool = by_rel.get(c["prop"])
        if not pool:
            fails["no_hedger_in_relation"] += 1
            continue
        eligible = [h for h in pool if abs(h["n_q_tokens"] - c["n_q_tokens"]) <= max_len_diff]
        if not eligible:
            fails["len"] += 1
            continue
        eligible = [h for h in eligible if abs(float(h["entropy"]) - float(c["entropy"])) <= entropy_tol]
        if not eligible:
            fails["entropy"] += 1
            continue
        best = min(eligible, key=lambda h: abs(float(h["s_pop"]) - float(c["s_pop"])))
        pool.remove(best)
        pairs.append((best, c))  # (hedger, confabulator)
    return pairs, fails


def smd(a, b, key):
    """Standardized mean difference between the two matched groups on `key`."""
    x = np.array([float(r[key]) for r in a], float)
    y = np.array([float(r[key]) for r in b], float)
    sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) if len(x) > 1 and len(y) > 1 else 0.0
    return float((x.mean() - y.mean()) / sd) if sd > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE, help="retained screen output to draw the U/U pool from")
    ap.add_argument("--out-category", default="uu_hedge_confab")
    ap.add_argument("--hedger-min", type=float, default=0.8)
    ap.add_argument("--confab-max", type=float, default=0.1)
    ap.add_argument("--confab-max-distinct", type=int, default=2,
                    help="U/U spec says 2; the data may not support it -- see gate_report")
    ap.add_argument("--max-len-diff", type=int, default=3)
    ap.add_argument("--entropy-tol", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report counts and pair yield, write nothing")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "data" / args.out_category
    if not args.dry_run:
        guard_output(out_dir / "prompts.jsonl", args.overwrite)

    all_rows, pool = load_pool(REPO_ROOT / args.source)
    hedgers, confabs = classify(pool, args.hedger_min, args.confab_max, args.confab_max_distinct)
    pairs, fails = match(hedgers, confabs, args.max_len_diff, args.entropy_tol)
    print(f"source {args.source}: {len(all_rows)} rows -> {len(pool)} genuinely-unknown uncertain items")
    print(f"  hedgers (hedge_rate >= {args.hedger_min}): {len(hedgers)}")
    print(f"  confabulators (hedge_rate <= {args.confab_max}, n_distinct <= {args.confab_max_distinct}): {len(confabs)}")
    print(f"  matched pairs: {len(pairs)} | unmatched confabulators: {fails}")
    if args.dry_run:
        return

    idx = seeded_shuffle(list(range(len(pairs))), args.seed)
    n_work = int(0.7 * len(pairs))
    split = {i: ("working" if k < n_work else "held_out") for k, i in enumerate(idx)}
    out_dir.mkdir(parents=True, exist_ok=True)

    def rec(x, pid, twin, is_ctrl, spl):
        return dict(prompt_id=pid, category=args.out_category, raw_prompt=x["q"],
                    chat_formatted_prompt=x["chat"], source_dataset=f"{args.source} (U/U hedger vs confabulator)",
                    split=spl, is_control=is_ctrl, twin_id=twin, relation=x["prop"], subject=x["subj"],
                    s_pop=x["s_pop"], gold=x["aliases"], slick_class=x["slick"], frac_correct=x["frac_correct"],
                    greedy=x["greedy"], samples=x["samples"], entropy=x["entropy"], top1=x["top1"],
                    hedge_rate=x["hedge_rate"], n_distinct=x["n_distinct"],
                    uu_class=("hedger" if not is_ctrl else "confabulator"))

    with open(out_dir / "prompts.jsonl", "w", encoding="utf-8") as fh, \
         open(out_dir / "controls.jsonl", "w", encoding="utf-8") as fc:
        for i, (h, c) in enumerate(pairs):
            t = f"uu_{i:04d}"
            fh.write(json.dumps(rec(h, t + "_h", t, False, split[i]), ensure_ascii=False) + "\n")
            fc.write(json.dumps(rec(c, t + "_c", t, True, split[i]), ensure_ascii=False) + "\n")

    hs, cs = [h for h, _ in pairs], [c for _, c in pairs]
    report = dict(source=args.source, n_source_rows=len(all_rows), n_pool=len(pool),
                  thresholds=dict(hedger_min=args.hedger_min, confab_max=args.confab_max,
                                  confab_max_distinct=args.confab_max_distinct,
                                  max_len_diff=args.max_len_diff, entropy_tol=args.entropy_tol),
                  n_hedgers=len(hedgers), n_confabulators=len(confabs), n_pairs=len(pairs),
                  n_working=n_work, n_held_out=len(pairs) - n_work, unmatched=fails,
                  pairs_per_relation={p: sum(1 for h, _ in pairs if h["prop"] == p)
                                      for p in sorted({h["prop"] for h, _ in pairs})},
                  balance_smd={k: smd(hs, cs, k) for k in ("s_pop", "n_q_tokens", "entropy")},
                  means=dict(hedger={k: float(np.mean([float(r[k]) for r in hs])) for k in ("hedge_rate", "entropy", "n_distinct", "s_pop")},
                             confabulator={k: float(np.mean([float(r[k]) for r in cs])) for k in ("hedge_rate", "entropy", "n_distinct", "s_pop")}) if pairs else {})
    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    write_provenance(out_dir / "prompts.jsonl", dict(script="scripts/build_uu_hedge_confab.py", **{k: v for k, v in vars(args).items()}))
    print(json.dumps({k: v for k, v in report.items() if k != "pairs_per_relation"}, indent=1))


if __name__ == "__main__":
    main()
