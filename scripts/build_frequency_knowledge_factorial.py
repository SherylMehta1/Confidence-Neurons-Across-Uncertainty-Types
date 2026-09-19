"""
Frequency x knowledge factorial twins from the retained screen output (L4a).

The standard familiarity pair contrasts an obscure-and-Unknown item with a popular-and-HighlyKnown
one, so popularity and knownness move together. all_measured.jsonl (L0 Fix 1) keeps every scored
item, including the off-diagonal ones the pairing loop discards: popular-but-Unknown and
obscure-but-HighlyKnown. Crossing the popularity tail an item was drawn from (arm: uncertain =
bottom decile, control = top decile) with its scored SliCK class gives four cells, and pairing
across ONE axis while holding the other fixed gives four contrasts:

  pop_given_known    obscure&Known    vs popular&Known    popularity moves, knowledge held Known
  pop_given_unknown  obscure&Unknown  vs popular&Unknown  popularity moves, knowledge held Unknown
  know_given_hi      popular&Unknown  vs popular&Known    knowledge moves, popularity held high
  know_given_lo      obscure&Unknown  vs obscure&Known    knowledge moves, popularity held low

If the circuit tracks knownness, the know_* contrasts should transfer and the pop_* ones should
not; if it tracks popularity, the reverse. The side expected to hedge more (Unknown, or obscure)
is written as prompts / is_control=False, the other as controls / is_control=True.

Matching is within relation on question token length (--max-len-diff); ties are broken by the
closest log popularity for the know_* contrasts (to hold popularity as equal as possible) and by
the closest length for the pop_* contrasts. There is deliberately NO entropy or hedging gate: the
entropy and hedge-rate gaps are the outcomes the factorial measures, so they are recorded per
contrast in gate_report.json rather than used to select pairs. Maybe-class items are excluded.

Writes data/<prefix>_<contrast>/{prompts,controls}.jsonl + gate_report.json (+ provenance) in the
repo schema, so the circuit pipeline runs on each with --category <prefix>_<contrast>, and
data/<prefix>_summary.csv. Refuses to overwrite without --overwrite.

  python scripts/build_frequency_knowledge_factorial.py --source data/familiarity_v2/all_measured.jsonl --prefix factorial_v2
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json
import math

import numpy as np
import pandas as pd

from _common import REPO_ROOT, guard_output
from shared.prompt_format import seeded_shuffle
from shared.provenance import write_provenance

CELLS = {"obscure_known": ("uncertain", "HighlyKnown"), "obscure_unknown": ("uncertain", "Unknown"),
         "popular_known": ("control", "HighlyKnown"), "popular_unknown": ("control", "Unknown")}
CONTRASTS = {  # name: (prompts-side cell, controls-side cell, tiebreak)
    "pop_given_known": ("obscure_known", "popular_known", "length"),
    "pop_given_unknown": ("obscure_unknown", "popular_unknown", "length"),
    "know_given_hi": ("popular_unknown", "popular_known", "logpop"),
    "know_given_lo": ("obscure_unknown", "obscure_known", "logpop"),
}


def bucket(rows):
    cells = {k: [] for k in CELLS}
    for r in rows:
        for name, (arm, slick) in CELLS.items():
            if r.get("arm") == arm and r.get("slick") == slick:
                cells[name].append(r)
    return cells


def match(a_rows, b_rows, max_len_diff, tiebreak):
    """Pair each item of the scarcer side with the closest eligible item of the other, within relation."""
    a_scarce = len(a_rows) <= len(b_rows)
    scarce, pool_rows = (a_rows, b_rows) if a_scarce else (b_rows, a_rows)
    pool = {}
    for r in pool_rows:
        pool.setdefault(r["prop"], []).append(r)
    pairs, unmatched = [], {"no_partner_in_relation": 0, "len": 0}
    for s in sorted(scarce, key=lambda r: (r["prop"], r["subj"])):
        cands = pool.get(s["prop"], [])
        if not cands:
            unmatched["no_partner_in_relation"] += 1; continue
        elig = [c for c in cands if abs(c["n_q_tokens"] - s["n_q_tokens"]) <= max_len_diff]
        if not elig:
            unmatched["len"] += 1; continue
        if tiebreak == "logpop":
            key = lambda c: abs(math.log1p(float(c["s_pop"])) - math.log1p(float(s["s_pop"])))
        else:
            key = lambda c: abs(c["n_q_tokens"] - s["n_q_tokens"])
        best = min(elig, key=key)
        cands.remove(best)
        pairs.append((s, best) if a_scarce else (best, s))  # always (a_side, b_side)
    return pairs, unmatched


def smd(a, b, key, log=False):
    f = (lambda r: math.log1p(float(r[key]))) if log else (lambda r: float(r[key]))
    x, y = np.array([f(r) for r in a]), np.array([f(r) for r in b])
    sd = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) if len(x) > 1 and len(y) > 1 else 0.0
    return float((x.mean() - y.mean()) / sd) if sd > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/familiarity_v2/all_measured.jsonl")
    ap.add_argument("--prefix", default="factorial_v2")
    ap.add_argument("--max-len-diff", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for name in CONTRASTS:
        guard_output(REPO_ROOT / "data" / f"{args.prefix}_{name}" / "prompts.jsonl", args.overwrite)

    rows = [json.loads(l) for l in open(REPO_ROOT / args.source, encoding="utf-8") if l.strip()]
    cells = bucket(rows)
    print(f"{args.source}: {len(rows)} rows -> cells " + ", ".join(f"{k}={len(v)}" for k, v in cells.items()))

    summary = []
    for name, (a_cell, b_cell, tiebreak) in CONTRASTS.items():
        pairs, unmatched = match(cells[a_cell], cells[b_cell], args.max_len_diff, tiebreak)
        out_dir = REPO_ROOT / "data" / f"{args.prefix}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = seeded_shuffle(list(range(len(pairs))), args.seed)
        n_work = int(0.7 * len(pairs))
        split = {i: ("working" if k < n_work else "held_out") for k, i in enumerate(idx)}

        def rec(x, pid, twin, is_ctrl, spl):
            return dict(prompt_id=pid, category=f"{args.prefix}_{name}", raw_prompt=x["q"], chat_formatted_prompt=x["chat"],
                        source_dataset=f"{args.source} (frequency x knowledge factorial: {name})", split=spl, is_control=is_ctrl,
                        twin_id=twin, relation=x["prop"], subject=x["subj"], s_pop=x["s_pop"], gold=x["aliases"],
                        slick_class=x["slick"], frac_correct=x["frac_correct"], greedy=x["greedy"], samples=x["samples"],
                        entropy=x["entropy"], top1=x["top1"], hedge_rate=x["hedge_rate"], n_distinct=x["n_distinct"],
                        factorial_cell=(b_cell if is_ctrl else a_cell))

        with open(out_dir / "prompts.jsonl", "w", encoding="utf-8") as fp, open(out_dir / "controls.jsonl", "w", encoding="utf-8") as fc:
            for i, (a, b) in enumerate(pairs):
                t = f"{name}_{i:04d}"
                fp.write(json.dumps(rec(a, t + "_u", t, False, split[i]), ensure_ascii=False) + "\n")
                fc.write(json.dumps(rec(b, t + "_c", t, True, split[i]), ensure_ascii=False) + "\n")
        A, Bs = [a for a, _ in pairs], [b for _, b in pairs]
        gaps = dict(entropy=float(np.mean([float(a["entropy"]) - float(b["entropy"]) for a, b in pairs])) if pairs else None,
                    hedge_rate=float(np.mean([float(a["hedge_rate"]) - float(b["hedge_rate"]) for a, b in pairs])) if pairs else None,
                    log_pop=float(np.mean([math.log1p(float(a["s_pop"])) - math.log1p(float(b["s_pop"])) for a, b in pairs])) if pairs else None)
        report = dict(contrast=name, prompts_cell=a_cell, controls_cell=b_cell, source=args.source,
                      n_available={a_cell: len(cells[a_cell]), b_cell: len(cells[b_cell])}, n_pairs=len(pairs),
                      n_working=n_work, n_held_out=len(pairs) - n_work, unmatched=unmatched, tiebreak=tiebreak,
                      pairs_per_relation={p: sum(1 for a, _ in pairs if a["prop"] == p) for p in sorted({a["prop"] for a, _ in pairs})},
                      mean_gap_prompts_minus_controls=gaps,
                      balance_smd={"log_pop": smd(A, Bs, "s_pop", log=True), "n_q_tokens": smd(A, Bs, "n_q_tokens")} if pairs else {},
                      note="no entropy/hedge gate: those gaps are the factorial's outcomes, recorded here, not selection criteria")
        (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
        write_provenance(out_dir / "prompts.jsonl", dict(script="scripts/build_frequency_knowledge_factorial.py", contrast=name, **vars(args)))
        summary.append(dict(contrast=name, prompts_cell=a_cell, controls_cell=b_cell, n_pairs=len(pairs), n_working=n_work,
                            n_held_out=len(pairs) - n_work, n_relations=len(report["pairs_per_relation"]),
                            gap_entropy=gaps["entropy"], gap_hedge_rate=gaps["hedge_rate"], gap_log_pop=gaps["log_pop"],
                            smd_log_pop=report["balance_smd"].get("log_pop"), smd_len=report["balance_smd"].get("n_q_tokens")))
        print(f"  {name:<18} {len(pairs):>4} pairs ({n_work}/{len(pairs) - n_work}) over {len(report['pairs_per_relation'])} relations | "
              f"gap entropy {gaps['entropy']:+.2f} hedge {gaps['hedge_rate']:+.2f} logpop {gaps['log_pop']:+.2f} | unmatched {unmatched}"
              if pairs else f"  {name:<18}    0 pairs | unmatched {unmatched}")
    pd.DataFrame(summary).to_csv(REPO_ROOT / "data" / f"{args.prefix}_summary.csv", index=False)
    print(f"wrote data/{args.prefix}_summary.csv")


if __name__ == "__main__":
    main()
