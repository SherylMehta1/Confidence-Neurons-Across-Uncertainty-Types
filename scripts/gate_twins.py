"""
Generic model-relative gate for twin candidates (UNCERTAINTY_DEFINITION.md section 3).

Input: a candidates JSONL where each line is ONE twin pair:
  {"twin_id": ..., "category": ..., "uncertain": {"question": ..., "gold": [aliases] | null, "meta": {...}},
   "control":   {"question": ..., "gold": [aliases],        "meta": {...}},
   "prefill": "The answer is"   (optional, default), "context": {"uncertain": str|null, "control": str|null} (optional)}
`gold` may be null for the uncertain side when no correct answer exists (fabricated entity, aleatoric);
then the correctness rule for that side is skipped and only the control is required HighlyKnown.

For each side: greedy + N sampled answers under the final prompt (chat template [+ context] + prefill),
exact-match grading against aliases, SliCK class, first-token entropy, free-generation hedging rate.
Gate (per pair): control HighlyKnown; uncertain Unknown (if gold given); entropy gap >= --entropy-gap;
hedging gap >= --hedge-gap; question token-length difference <= --max-len-diff (contexts are excluded: they differ by design). Consistently-wrong uncertain
items (one wrong answer in >= 80% of samples) are written to a separate negative-control file.

Writes data/<category>/{prompts,controls,consistently_wrong}.jsonl in the repo schema and gate_report.json.
Usage: python scripts/gate_twins.py --candidates data/situated/candidates.jsonl --category situated --overwrite
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from behavioral_test import HEDGE_RE
from build_familiarity_twins import greedy_answer, is_correct, normalize, sample_answers, slick_class
from shared.model_utils import compute_entropy, get_next_token_probs
from shared.prompt_format import build_completion_prompt, format_chat_prompt, seeded_shuffle
from shared.provenance import build_provenance, write_provenance


def measure_side(model, tokenizer, question, gold, context, prefill, n_samples, n_free):
    user = (context.rstrip() + "\n\n" + question) if context else question
    chat = build_completion_prompt(tokenizer, user, cloze_suffix=prefill, ensure_question_mark=not bool(context))["chat_formatted_prompt"]
    g = greedy_answer(model, tokenizer, chat)
    samples = sample_answers(model, tokenizer, chat, n_samples)
    if gold:
        oks = [is_correct(s, gold) for s in samples]
        frac = float(np.mean(oks)); cls = slick_class(is_correct(g, gold), frac)
    else:
        frac, cls = None, "NoGold"
    probs = get_next_token_probs(model, tokenizer, chat)
    free = sample_answers(model, tokenizer, format_chat_prompt(tokenizer, user), n_free, temperature=0.7, max_new_tokens=24)
    return dict(question=question, chat=chat, greedy=g, samples=samples, frac_correct=frac, slick=cls,
                entropy=compute_entropy(probs), top1=float(probs.max()),
                hedge_rate=float(np.mean([bool(HEDGE_RE.search(f)) for f in free])), free=free,
                n_distinct=len({normalize(s) for s in samples if normalize(s)}),
                n_tokens=len(tokenizer(question, add_special_tokens=False)["input_ids"]),  # twin length rule applies to the question
                n_tokens_total=len(tokenizer(user, add_special_tokens=False)["input_ids"]), gold=gold)  # contexts differ by design


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--n-free", type=int, default=3)
    ap.add_argument("--entropy-gap", type=float, default=0.5)
    ap.add_argument("--hedge-gap", type=float, default=0.3)
    ap.add_argument("--max-len-diff", type=int, default=3)
    ap.add_argument("--require-hedge", action="store_true", default=True)
    ap.add_argument("--no-require-hedge", dest="require_hedge", action="store_false", help="for dissociation controls (aleatoric)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out_dir = REPO_ROOT / "data" / args.category
    guard_output(out_dir / "prompts.jsonl", args.overwrite)
    cands = [json.loads(l) for l in open(REPO_ROOT / args.candidates, encoding="utf-8") if l.strip()]
    if args.limit:
        cands = cands[: args.limit]
    model, tokenizer = load_model_from_args(args)

    pairs, cwrong, fails = [], [], {"control_not_known": 0, "uncertain_not_unknown": 0, "len": 0, "entropy": 0, "hedge": 0}
    measured = []
    for i, c in enumerate(cands):
        prefill = c.get("prefill", "The answer is"); ctx = c.get("context") or {}
        u = measure_side(model, tokenizer, c["uncertain"]["question"], c["uncertain"].get("gold"), ctx.get("uncertain"), prefill, args.n_samples, args.n_free)
        k = measure_side(model, tokenizer, c["control"]["question"], c["control"].get("gold"), ctx.get("control"), prefill, args.n_samples, args.n_free)
        measured.append((c, u, k))
        if k["slick"] != "HighlyKnown":
            fails["control_not_known"] += 1; continue
        if u["gold"] and u["slick"] != "Unknown":
            fails["uncertain_not_unknown"] += 1; continue
        if u["gold"] and u["n_distinct"] <= 1 and u["frac_correct"] == 0.0:
            cwrong.append((c, u, k)); continue
        if abs(u["n_tokens"] - k["n_tokens"]) > args.max_len_diff:
            fails["len"] += 1; continue
        if u["entropy"] - k["entropy"] < args.entropy_gap:
            fails["entropy"] += 1; continue
        if args.require_hedge and u["hedge_rate"] - k["hedge_rate"] < args.hedge_gap:
            fails["hedge"] += 1; continue
        pairs.append((c, u, k))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cands)} measured; {len(pairs)} passing so far")
    print(f"[{args.category}] pairs passing: {len(pairs)} / {len(cands)} | consistently wrong: {len(cwrong)} | fails: {fails}")

    idx = seeded_shuffle(list(range(len(pairs))), args.seed); n_work = int(0.7 * len(pairs))
    split = {i: ("working" if r < n_work else "held_out") for r, i in enumerate(idx)}
    out_dir.mkdir(parents=True, exist_ok=True)

    def rec(c, side, pid, is_ctrl, spl):
        return dict(prompt_id=pid, category=args.category, raw_prompt=side["question"], chat_formatted_prompt=side["chat"],
                    source_dataset=c.get("source", args.candidates), split=spl, is_control=is_ctrl, twin_id=c["twin_id"],
                    gold=side["gold"], slick_class=side["slick"], frac_correct=side["frac_correct"], greedy=side["greedy"],
                    samples=side["samples"], entropy=side["entropy"], top1=side["top1"], hedge_rate=side["hedge_rate"],
                    n_distinct=side["n_distinct"], meta=(c["control"] if is_ctrl else c["uncertain"]).get("meta", {}))

    with open(out_dir / "prompts.jsonl", "w", encoding="utf-8") as fu, open(out_dir / "controls.jsonl", "w", encoding="utf-8") as fc:
        for i, (c, u, k) in enumerate(pairs):
            fu.write(json.dumps(rec(c, u, c["twin_id"] + "_u", False, split[i]), ensure_ascii=False) + "\n")
            fc.write(json.dumps(rec(c, k, c["twin_id"] + "_c", True, split[i]), ensure_ascii=False) + "\n")
    with open(out_dir / "consistently_wrong.jsonl", "w", encoding="utf-8") as f:
        for c, u, k in cwrong:
            f.write(json.dumps(rec(c, u, c["twin_id"] + "_cw", False, "working"), ensure_ascii=False) + "\n")
    with open(out_dir / "all_measured.jsonl", "w", encoding="utf-8") as f:
        for c, u, k in measured:
            keep = ("slick", "frac_correct", "entropy", "hedge_rate", "n_distinct", "n_tokens", "n_tokens_total", "question", "gold", "greedy", "samples", "free")
            f.write(json.dumps(dict(twin_id=c["twin_id"], uncertain={x: u[x] for x in keep}, control={x: k[x] for x in keep}), ensure_ascii=False) + "\n")  # full record so failed candidates can be re-judged
    report = dict(category=args.category, model=args.model_id, precision=getattr(model, "cn_precision", None), n_candidates=len(cands),
                  n_pairs=len(pairs), n_working=n_work, n_held_out=len(pairs) - n_work, n_consistently_wrong=len(cwrong), fails=fails,
                  gate=dict(entropy_gap=args.entropy_gap, hedge_gap=args.hedge_gap, require_hedge=args.require_hedge, max_len_diff=args.max_len_diff),
                  mean_entropy_uncertain=float(np.mean([u["entropy"] for _, u, _ in pairs])) if pairs else None,
                  mean_entropy_control=float(np.mean([k["entropy"] for _, _, k in pairs])) if pairs else None,
                  mean_hedge_uncertain=float(np.mean([u["hedge_rate"] for _, u, _ in pairs])) if pairs else None,
                  mean_hedge_control=float(np.mean([k["hedge_rate"] for _, _, k in pairs])) if pairs else None,
                  entropy_gap_all_candidates=float(np.mean([u["entropy"] - k["entropy"] for _, u, k in measured])) if measured else None)
    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    write_provenance(out_dir / "prompts.jsonl", build_provenance(model, script="scripts/gate_twins.py", candidates=args.candidates, **{k: v for k, v in vars(args).items() if k not in ("model_id", "candidates")}))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
