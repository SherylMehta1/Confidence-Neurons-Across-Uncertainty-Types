"""
LLM-judge audit of the deterministic alias grader used by the gate (UNCERTAINTY_DEFINITION.md section 3).

The judge never replaces the grader: it reads only saved (question, gold aliases, model answer) triples and
gives a second opinion. We report agreement overall, per side (control / uncertain), per grader verdict,
and -- the confound that matters -- per hedged / not-hedged answer, plus Cohen's kappa. Items are a seeded
stratified sample over grader verdict x side x hedged, drawn from the gated files
(data/<category>/{prompts,controls,consistently_wrong}.jsonl, which carry the greedy answer and the samples).

Usage:
  python scripts/judge_audit.py --categories familiarity,conflict,aleatoric --judge Qwen/Qwen2.5-32B-Instruct
  python scripts/judge_audit.py --categories familiarity --data-root results/second_model/qwen25_7b_instruct/data \\
         --out-dir results/second_model/qwen25_7b_instruct/results
Outputs: <out-dir>/judge_audit_<category>.csv and judge_audit_<category>_summary.txt (+ provenance).
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json
import random
from collections import defaultdict

import pandas as pd
import torch

from _common import REPO_ROOT, guard_output
from build_familiarity_twins import is_correct
from shared.model_utils import load_model
from shared.provenance import build_provenance, write_provenance

SYSTEM = ("You grade short answers to factual questions. You are given the question, a list of acceptable reference answers "
          "(aliases of the same fact), and a candidate answer produced by a model. Decide whether the candidate answer states "
          "the same fact as the reference. Paraphrases, extra detail and different formatting are fine. A candidate that "
          "names a different entity, number or date is INCORRECT. A candidate that refuses, hedges without giving the fact, or "
          "says the information is unavailable is INCORRECT. Answer with exactly one word: CORRECT or INCORRECT.")
LABELS = ("CORRECT", "INCORRECT")


def load_items(data_root, category):
    recs = []
    for fname, side in (("prompts.jsonl", "uncertain"), ("controls.jsonl", "control"), ("consistently_wrong.jsonl", "consistently_wrong")):
        p = data_root / category / fname
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); r["_side"] = side; recs.append(r)
    items = []
    for r in recs:
        gold = r.get("gold")
        if not gold:
            continue  # no reference answer -> nothing to grade (aleatoric uncertain side)
        answers = [("greedy", r.get("greedy"))] + [(f"sample{i}", a) for i, a in enumerate(r.get("samples") or [])]
        for kind, a in answers:
            if a is None:
                continue
            items.append(dict(prompt_id=r["prompt_id"], side=r["_side"], kind=kind, question=r["raw_prompt"], gold=list(gold), answer=str(a),
                              grader=bool(is_correct(str(a), gold)), hedged=float(r.get("hedge_rate", 0.0)) >= 0.5, slick=r.get("slick_class")))
    return items


def stratified_sample(items, n_per_stratum, seed):
    by = defaultdict(list)
    for it in items:
        by[(it["grader"], it["side"], it["hedged"])].append(it)
    rng = random.Random(seed); out = []
    for key in sorted(by):
        pool = by[key]; rng.shuffle(pool); out += pool[:n_per_stratum]
    return out, {str(k): len(v) for k, v in by.items()}


def judge_prompt(tokenizer, it):
    user = (f"Question: {it['question']}\nReference answers: {' | '.join(it['gold'][:12])}\nCandidate answer: {it['answer']}\n\n"
            "Is the candidate answer CORRECT or INCORRECT?")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def run_judge(model, tokenizer, items, batch_size=16):
    """Single forward pass per batch: compare the next-token log-probs of the two label words (no sampling, no parsing)."""
    label_ids = [tokenizer(f" {w}", add_special_tokens=False)["input_ids"][0] for w in LABELS]
    label_ids_nospace = [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in LABELS]
    print("label tokenization:", {w: (tokenizer.tokenize(f" {w}"), tokenizer.tokenize(w)) for w in LABELS})
    ids_c, ids_i = {label_ids[0], label_ids_nospace[0]}, {label_ids[1], label_ids_nospace[1]}
    assert ids_c.isdisjoint(ids_i), f"label first-tokens collide: {ids_c} vs {ids_i}; pick different label words for this tokenizer"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    verdicts, margins = [], []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        enc = tokenizer([judge_prompt(tokenizer, it) for it in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        enc.pop("token_type_ids", None)
        logits = model(**enc, use_cache=False).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        score = torch.stack([torch.logsumexp(torch.stack([lp[:, a], lp[:, b]], 0), 0) for a, b in zip(label_ids, label_ids_nospace)], 1)
        m = (score[:, 0] - score[:, 1]).tolist()
        verdicts += [x > 0 for x in m]; margins += m
        if (i // batch_size) % 10 == 0:
            print(f"  judged {min(i + batch_size, len(items))}/{len(items)}")
    return verdicts, margins


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def summarize(df, category, judge_id, strata_sizes):
    lines = [f"Judge audit -- {category}; judge {judge_id}; {len(df)} graded items (stratified by grader verdict x side x hedged)",
             f"overall agreement {df.agree.mean():.3f}  kappa {kappa(df.grader.tolist(), df.judge.tolist()):.3f}", ""]
    for name, col in (("side", "side"), ("grader verdict", "grader"), ("hedged answer", "hedged"), ("answer kind", "kind_group")):
        lines.append(f"by {name}:")
        for k, g in df.groupby(col):
            lines.append(f"  {str(k):<18} n={len(g):4d}  agreement {g.agree.mean():.3f}  grader-says-correct {g.grader.mean():.2f}  judge-says-correct {g.judge.mean():.2f}")
        lines.append("")
    fn = df[(df.grader == False) & (df.judge == True)]; fp = df[(df.grader == True) & (df.judge == False)]
    lines.append(f"grader false negatives (judge CORRECT, grader INCORRECT): {len(fn)} / {int((df.grader == False).sum())} grader-negatives")
    lines.append(f"grader false positives (judge INCORRECT, grader CORRECT): {len(fp)} / {int((df.grader == True).sum())} grader-positives")
    if len(df[df.hedged]) and len(df[~df.hedged]):
        lines.append(f"disagreement rate hedged {1 - df[df.hedged].agree.mean():.3f} vs not hedged {1 - df[~df.hedged].agree.mean():.3f}"
                     "  <- if hedged is higher, grader error correlates with the behavioral readout")
    lines.append(""); lines.append("examples of grader false negatives:")
    for _, r in fn.head(8).iterrows():
        lines.append(f"  Q: {r.question[:70]} | gold: {r.gold[:60]} | answer: {r.answer[:60]}")
    lines.append("examples of grader false positives:")
    for _, r in fp.head(8).iterrows():
        lines.append(f"  Q: {r.question[:70]} | gold: {r.gold[:60]} | answer: {r.answer[:60]}")
    lines.append(""); lines.append("population sizes per stratum (grader, side, hedged): " + json.dumps(strata_sizes))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", default="familiarity")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--n-per-stratum", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    data_root, out_dir = REPO_ROOT / args.data_root, REPO_ROOT / args.out_dir
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    for c in cats:
        guard_output(out_dir / f"judge_audit_{c}.csv", args.overwrite)

    model, tokenizer = load_model(model_id=args.judge)
    model.eval()
    for c in cats:
        items = load_items(data_root, c)
        sample, sizes = stratified_sample(items, args.n_per_stratum, args.seed)
        print(f"[{c}] {len(items)} gradable answers; {len(sample)} sampled over {len(sizes)} strata")
        if not sample:
            continue
        verdicts, margins = run_judge(model, tokenizer, sample, args.batch_size)
        df = pd.DataFrame(sample)
        df["gold"] = df["gold"].apply(lambda g: " | ".join(g)); df["judge"] = verdicts; df["judge_margin"] = margins
        df["agree"] = df.grader == df.judge; df["kind_group"] = df.kind.where(df.kind == "greedy", "sample")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"judge_audit_{c}.csv", index=False)
        summary = summarize(df, c, args.judge, sizes)
        (out_dir / f"judge_audit_{c}_summary.txt").write_text(summary + "\n", encoding="utf-8")
        write_provenance(out_dir / f"judge_audit_{c}.csv", build_provenance(model, script="scripts/judge_audit.py", judge=args.judge, category=c,
                                                                           n_per_stratum=args.n_per_stratum, seed=args.seed, data_root=args.data_root))
        print(summary)


if __name__ == "__main__":
    main()
