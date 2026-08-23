"""
Type B (familiarity) twins from PopQA, labeled the UE-literature way and gated on the model.

Pipeline (one model at a time; run once per model):
  1. Load PopQA (akariasai/PopQA), group by relation template (`prop`).
  2. Candidate pool per relation: subjects in the bottom --lo-quantile of popularity (uncertain
     candidates) and the top --hi-quantile (control candidates), answer-token type matched by
     relation.
  3. Label by SAMPLED CORRECTNESS under the final prompt (chat template + prefill "The answer is"):
     greedy + --n-samples at T=0.7, exact-match against `possible_answers` aliases (normalized).
     SliCK classes: HighlyKnown (greedy correct and >= 0.9 of samples), Unknown (0 correct).
  4. Pair within relation: Unknown subject <-> HighlyKnown subject, token length of the question
     within --max-len-diff.
  5. Gate: first-token entropy gap >= --entropy-gap and hedging gap >= --hedge-gap over
     --n-free free-generation samples (24 tokens, no prefill beyond the assistant header).
     Also classify the uncertain item as dispersed (>= 2 distinct sampled answers) vs
     consistently-wrong (one wrong answer in >= 8/10 samples) — the latter is kept as a separate
     negative-control file.
  6. Write data/familiarity/{prompts,controls}.jsonl in the repo schema (+ twin_id, slick_class,
     gate values, provenance), plus data/familiarity/consistently_wrong.jsonl and
     data/familiarity/gate_report.json.

Usage (from any CWD; needs the model):
  python scripts/build_familiarity_twins.py --n-per-relation 40 --overwrite
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json
import random
import re
import string

import numpy as np
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from behavioral_test import HEDGE_RE, generate_batch
from shared.model_utils import compute_entropy, get_next_token_probs
from shared.prompt_format import build_completion_prompt, format_chat_prompt, seeded_shuffle
from shared.provenance import build_provenance, write_provenance

POPQA = "akariasai/PopQA"
PREFILL = "The answer is"


def normalize(s):
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return " ".join(s.split())


def answer_head(answer):
    """The short-answer part of a generation: text before the first sentence break or clause marker, so that an
    elaboration ('England. London is ... United Kingdom') cannot match a gold alias by accident."""
    head = re.split(r"[.;\n]|\s+(?:however|but|although|which|who|that)\b|,\s", answer.strip(), maxsplit=1)[0]
    return head if head.strip() else answer


def is_correct(answer, aliases, strict=True):
    """Alias match on the answer head: exact, prefix, or whole-word containment (judge-audited 2026-08-24:
    the old whole-answer substring rule produced 7/81 false positives on elaborated answers)."""
    a = normalize(answer_head(answer) if strict else answer)
    if not a:
        return False
    for g in aliases:
        g = normalize(g)
        if g and (a == g or a.startswith(g + " ") or re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", a)):
            return True
    return False


@torch.no_grad()
def sample_answers(model, tokenizer, prompt, n, temperature=0.7, max_new_tokens=12):
    enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(model.device)
    enc.pop("token_type_ids", None)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                         num_return_sequences=n, pad_token_id=tokenizer.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True).split("\n")[0].strip() for t in new]


@torch.no_grad()
def greedy_answer(model, tokenizer, prompt, max_new_tokens=12):
    return generate_batch(model, tokenizer, [prompt], max_new_tokens)[0].split("\n")[0].strip()


def slick_class(greedy_ok, frac_ok):
    if greedy_ok and frac_ok >= 0.9:
        return "HighlyKnown"
    if frac_ok == 0.0 and not greedy_ok:
        return "Unknown"
    return "Maybe"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--n-per-relation", type=int, default=40, help="candidates per arm per relation before labeling")
    ap.add_argument("--relations", default=None, help="comma list of PopQA `prop` values (default: all 16)")
    ap.add_argument("--lo-quantile", type=float, default=0.10)
    ap.add_argument("--hi-quantile", type=float, default=0.90)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--n-free", type=int, default=3)
    ap.add_argument("--entropy-gap", type=float, default=0.5)
    ap.add_argument("--hedge-gap", type=float, default=0.3)
    ap.add_argument("--max-len-diff", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/familiarity")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap total candidates")
    args = ap.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    guard_output(out_dir / "prompts.jsonl", args.overwrite)
    rng = random.Random(args.seed)

    from datasets import load_dataset
    ds = load_dataset(POPQA, split="test")
    rows = [dict(r) for r in ds]
    props = sorted({r["prop"] for r in rows})
    if args.relations:
        props = [p for p in props if p in set(args.relations.split(","))]
    model, tokenizer = load_model_from_args(args)
    precision = getattr(model, "cn_precision", None)

    # ---- candidate pools by popularity ----
    cands = []
    for prop in props:
        items = [r for r in rows if r["prop"] == prop and r.get("s_pop") is not None]
        if len(items) < 20:
            continue
        pops = np.array([r["s_pop"] for r in items], float)
        lo = [r for r in items if r["s_pop"] <= np.quantile(pops, args.lo_quantile)]
        hi = [r for r in items if r["s_pop"] >= np.quantile(pops, args.hi_quantile)]
        rng.shuffle(lo); rng.shuffle(hi)
        for r in lo[: args.n_per_relation]:
            cands.append(dict(r, arm="uncertain"))
        for r in hi[: args.n_per_relation]:
            cands.append(dict(r, arm="control"))
    if args.limit:
        cands = cands[: args.limit]
    print(f"{len(cands)} candidates over {len(props)} relations; labeling with greedy + {args.n_samples} samples ...")

    # ---- label by sampled correctness; measure entropy and hedging ----
    labeled = []
    for i, r in enumerate(cands):
        q = r["question"].strip()
        aliases = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) else list(r["possible_answers"])
        chat = build_completion_prompt(tokenizer, q, cloze_suffix=PREFILL, ensure_question_mark=True)["chat_formatted_prompt"]
        g = greedy_answer(model, tokenizer, chat)
        samples = sample_answers(model, tokenizer, chat, args.n_samples)
        oks = [is_correct(s, aliases) for s in samples]
        frac = float(np.mean(oks))
        cls = slick_class(is_correct(g, aliases), frac)
        probs = get_next_token_probs(model, tokenizer, chat)
        H = compute_entropy(probs)
        free_prompt = format_chat_prompt(tokenizer, q if q.endswith("?") else q + "?")
        free = sample_answers(model, tokenizer, free_prompt, args.n_free, temperature=0.7, max_new_tokens=24)
        hedge = float(np.mean([bool(HEDGE_RE.search(f)) for f in free]))
        distinct = len({normalize(s) for s in samples if normalize(s)})
        labeled.append(dict(r, q=q, aliases=aliases, chat=chat, greedy=g, samples=samples, frac_correct=frac,
                            slick=cls, entropy=H, top1=float(probs.max()), hedge_rate=hedge, free=free,
                            n_distinct=distinct, n_q_tokens=len(tokenizer(q, add_special_tokens=False)["input_ids"])))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cands)} labeled")

    # ---- pair within relation and gate ----
    pairs, cwrong, fails = [], [], {"no_unknown_or_known": 0, "len": 0, "entropy": 0, "hedge": 0}
    for prop in props:
        unk = [x for x in labeled if x["prop"] == prop and x["arm"] == "uncertain" and x["slick"] == "Unknown"]
        kn = [x for x in labeled if x["prop"] == prop and x["arm"] == "control" and x["slick"] == "HighlyKnown"]
        if not unk or not kn:
            fails["no_unknown_or_known"] += len(unk) + len(kn)
            continue
        kn_pool = list(kn)
        for u in unk:
            if u["n_distinct"] <= 1 and u["frac_correct"] == 0.0:
                cwrong.append(u)  # consistently wrong: separate negative control
                continue
            best = None
            for c in kn_pool:
                if abs(c["n_q_tokens"] - u["n_q_tokens"]) <= args.max_len_diff:
                    best = c; break
            if best is None:
                fails["len"] += 1; continue
            if u["entropy"] - best["entropy"] < args.entropy_gap:
                fails["entropy"] += 1; continue
            if u["hedge_rate"] - best["hedge_rate"] < args.hedge_gap:
                fails["hedge"] += 1; continue
            kn_pool.remove(best)
            pairs.append((u, best))
    print(f"pairs passing the gate: {len(pairs)} | consistently-wrong uncertain items: {len(cwrong)} | fails: {fails}")

    # ---- split and write in the repo schema ----
    idx = seeded_shuffle(list(range(len(pairs))), args.seed)
    n_work = int(0.7 * len(pairs))
    split = {i: ("working" if k < n_work else "held_out") for k, i in enumerate(idx)}
    out_dir.mkdir(parents=True, exist_ok=True)

    def rec(x, pid, twin, is_ctrl, spl):
        return dict(prompt_id=pid, category="familiarity", raw_prompt=x["q"], chat_formatted_prompt=x["chat"],
                    source_dataset=f"{POPQA} (test) familiarity twins, SliCK-labeled, gated", split=spl, is_control=is_ctrl,
                    twin_id=twin, relation=x["prop"], subject=x["subj"], s_pop=x["s_pop"], gold=x["aliases"],
                    slick_class=x["slick"], frac_correct=x["frac_correct"], greedy=x["greedy"], samples=x["samples"],
                    entropy=x["entropy"], top1=x["top1"], hedge_rate=x["hedge_rate"], n_distinct=x["n_distinct"])

    with open(out_dir / "prompts.jsonl", "w", encoding="utf-8") as fu, open(out_dir / "controls.jsonl", "w", encoding="utf-8") as fc:
        for i, (u, c) in enumerate(pairs):
            t = f"fam_{i:04d}"
            fu.write(json.dumps(rec(u, t + "_u", t, False, split[i]), ensure_ascii=False) + "\n")
            fc.write(json.dumps(rec(c, t + "_c", t, True, split[i]), ensure_ascii=False) + "\n")
    with open(out_dir / "consistently_wrong.jsonl", "w", encoding="utf-8") as f:
        for i, u in enumerate(cwrong):
            f.write(json.dumps(rec(u, f"famcw_{i:04d}", f"famcw_{i:04d}", False, "working"), ensure_ascii=False) + "\n")
    report = dict(model=args.model_id, precision=precision, n_candidates=len(cands), n_relations=len(props),
                  slick_counts={k: sum(1 for x in labeled if x["slick"] == k) for k in ("HighlyKnown", "Maybe", "Unknown")},
                  n_pairs=len(pairs), n_working=n_work, n_held_out=len(pairs) - n_work, n_consistently_wrong=len(cwrong),
                  fails=fails, gate=dict(entropy_gap=args.entropy_gap, hedge_gap=args.hedge_gap, max_len_diff=args.max_len_diff,
                                         n_samples=args.n_samples, n_free=args.n_free),
                  mean_entropy_uncertain=float(np.mean([u["entropy"] for u, _ in pairs])) if pairs else None,
                  mean_entropy_control=float(np.mean([c["entropy"] for _, c in pairs])) if pairs else None,
                  mean_hedge_uncertain=float(np.mean([u["hedge_rate"] for u, _ in pairs])) if pairs else None,
                  mean_hedge_control=float(np.mean([c["hedge_rate"] for _, c in pairs])) if pairs else None,
                  pairs_per_relation={p: sum(1 for u, _ in pairs if u["prop"] == p) for p in props})
    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    write_provenance(out_dir / "prompts.jsonl", build_provenance(model, script="scripts/build_familiarity_twins.py", source=POPQA, **{k: v for k, v in vars(args).items() if k != "model_id"}))
    print(json.dumps({k: v for k, v in report.items() if k != "pairs_per_relation"}, indent=1))
    print("pairs per relation:", report["pairs_per_relation"])


if __name__ == "__main__":
    main()
