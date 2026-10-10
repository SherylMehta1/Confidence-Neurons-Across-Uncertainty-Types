"""E3 lens split (PREREG e3_lens_split.md): derived from jlens_cascade.py; adds hedge_term and answer_term columns.

Jacobian-lens trajectories of the hedge readout on the familiarity pool, for one model (jlens_cascade_criterion.md).

For each of the 2,637 PopQA candidates (data/familiarity_v3/all_measured.jsonl): build the model's prompt (chat
template + "The answer is" prefill, or the base-model prompt "Question: {q}\nAnswer: The answer is"), label the
item by SliCK on THAT prompt (greedy + 10 samples, the builder's grader) unless --labels-pool supplies labels made
on the identical prompt, then decode the last position of every layer through the model's prefitted Jacobian lens
and record the hedge-vs-answer log-odds, the lens entropy and the rank of " unknown".

BOS handling: jlens adds a BOS itself (force_bos), so the leading BOS text that chat templates emit is stripped
before jlens tokenizes. Before any item is processed, the final-layer logits jlens returns are checked against a
plain forward pass on the same text with exactly one BOS (max |delta logit| < 0.5 on 5 prompts); failure aborts.

Resumable: labels.jsonl and lens.csv are appended per item; finished items are skipped on restart.

  python scripts/jlens_cascade.py --model-id unsloth/gemma-2-2b --format base \
      --lens-file gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt --out-dir results/jlens_cascade/gemma-2-2b
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import csv
import json

import numpy as np
import torch

from _common import REPO_ROOT, add_model_args, load_model_from_args
from build_familiarity_twins import PREFILL, greedy_answer, is_correct, sample_answers, slick_class
from circuit_common import Readout
from direction_bridge import token_ids
from shared.prompt_format import build_completion_prompt
from shared.provenance import build_provenance, write_provenance

BASE_TEMPLATE = "Question: {q}\nAnswer: " + PREFILL


def item_key(r):
    return f"{r['prop']}::{r['subj']}"


def build_prompt(tokenizer, q, fmt):
    """Full prompt text as the model sees it (with BOS text where the tokenizer has one)."""
    if fmt == "chat":
        return build_completion_prompt(tokenizer, q, cloze_suffix=PREFILL, ensure_question_mark=True)["chat_formatted_prompt"]
    qq = q.strip() if q.strip().endswith("?") else q.strip() + "?"
    bos = tokenizer.bos_token if getattr(tokenizer, "bos_token_id", None) is not None else ""
    return (bos or "") + BASE_TEMPLATE.format(q=qq)


def strip_bos(tokenizer, text):
    bos = tokenizer.bos_token if getattr(tokenizer, "bos_token_id", None) is not None else None
    return text[len(bos):] if bos and text.startswith(bos) else text


@torch.no_grad()
def bos_check(model, tokenizer, lens, wrapped, prompts):
    worst = 0.0
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(model.device)
        ref = model(**enc).logits[0, -1].float()
        _, ml, _ = lens.apply(wrapped, strip_bos(tokenizer, p), positions=[-1])
        v = (ml[0] if ml.dim() > 1 else ml).float().to(ref.device)
        worst = max(worst, (v - ref).abs().max().item())
    return worst


def main():
    import jlens

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--format", choices=["chat", "base"], required=True)
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", required=True)
    ap.add_argument("--labels-pool", default=None, help="data/<pool> whose SliCK labels were made on the identical prompt")
    ap.add_argument("--items", default="data/familiarity_v3/all_measured.jsonl")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in open(REPO_ROOT / args.items, encoding="utf-8") if l.strip()]
    items = [r for r in items if r.get("subj")][: args.limit]
    model, tokenizer = load_model_from_args(args)
    model.eval()
    prompts = {item_key(r): build_prompt(tokenizer, r["q"], args.format) for r in items}

    # ---- labels ----
    lab_path = out / "labels.jsonl"
    labels = {}
    if lab_path.exists():
        for l in open(lab_path, encoding="utf-8"):
            if l.strip():
                x = json.loads(l); labels[x["key"]] = x
    if args.labels_pool and not labels:
        # Reuse labels for every item the pool scored on the identical prompt; any differing prompt aborts.
        # Items the pool never scored (e.g. Llama's pool holds 36 candidates a later run did not draw) are
        # labelled below with the same procedure.
        pool = {item_key(r): r for r in map(json.loads, open(REPO_ROOT / "data" / args.labels_pool / "all_measured.jsonl", encoding="utf-8")) if r.get("subj")}
        shared = [k for k in prompts if k in pool]
        n_same = sum(pool[k]["chat"] == prompts[k] for k in shared)
        if n_same != len(shared):
            raise SystemExit(f"--labels-pool prompts differ from this run's prompts ({n_same}/{len(shared)} identical); refusing to reuse labels")
        with open(lab_path, "w", encoding="utf-8") as f:
            for k in shared:
                x = dict(key=k, slick=pool[k]["slick"], frac_correct=pool[k]["frac_correct"], greedy=pool[k]["greedy"], source=args.labels_pool)
                labels[k] = x; f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"labels reused from {args.labels_pool} for {n_same} items (all prompts identical); "
              f"{len(prompts) - len(shared)} items not in the pool are labelled here")
    if True:
        todo = [r for r in items if item_key(r) not in labels]
        print(f"labelling {len(todo)} items ({len(labels)} already done)")
        with open(lab_path, "a", encoding="utf-8") as f:
            for i, r in enumerate(todo):
                k, p = item_key(r), prompts[item_key(r)]
                aliases = r["aliases"] if isinstance(r["aliases"], list) else json.loads(r["aliases"])
                g = greedy_answer(model, tokenizer, p)
                s = sample_answers(model, tokenizer, p, args.n_samples)
                frac = float(np.mean([is_correct(a, aliases) for a in s]))
                x = dict(key=k, slick=slick_class(is_correct(g, aliases), frac), frac_correct=frac, greedy=g, source="this run")
                labels[k] = x; f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
                if (i + 1) % 200 == 0:
                    print(f"  labelled {i + 1}/{len(todo)}")

    # ---- lens ----
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    worst = bos_check(model, tokenizer, lens, wrapped, [prompts[item_key(r)] for r in items[:5]])
    print(f"BOS/consistency check: max |jlens final logits - plain forward| = {worst:.4f}")
    (out / "bos_check.txt").write_text(f"{worst:.6f}\n", encoding="utf-8")
    if worst >= 0.5:
        raise SystemExit("consistency check FAILED (>= 0.5): not analysing this model")
    ro = Readout(tokenizer)
    unk = token_ids(tokenizer, [" unknown"])[0]
    lens_path = out / "lens.csv"
    done = set()
    if lens_path.exists():
        with open(lens_path, encoding="utf-8") as f:
            done = {row["key"] for row in csv.DictReader(f)}
    new = not lens_path.exists()
    with open(lens_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["key", "layer", "logodds", "hedge_term", "answer_term", "entropy"])
        todo = [r for r in items if item_key(r) not in done]
        print(f"lens pass over {len(todo)} items ({len(done)} already done)")
        for i, r in enumerate(todo):
            k = item_key(r)
            ll, ml, _ = lens.apply(wrapped, strip_bos(tokenizer, prompts[k]), positions=[-1])
            seq = sorted(ll.items()) + [(-1, ml)]
            for l, lg in seq:
                v = (lg[0] if lg.dim() > 1 else lg).float()
                lp = torch.log_softmax(v, -1)
                w.writerow([k, int(l), f"{ro.logodds(v).item():.5f}", f"{torch.logsumexp(lp[ro.hedge_ids], -1).item():.5f}",
                            f"{torch.logsumexp(lp[ro.answer_ids], -1).item():.5f}", f"{ro.entropy(v).item():.5f}"])
            if (i + 1) % 200 == 0:
                f.flush(); print(f"  lens {i + 1}/{len(todo)}")
    write_provenance(lens_path, build_provenance(model, script="scripts/e3_lens_split.py", lens=f"{args.lens_repo}/{args.lens_file}",
                                                 format=args.format, labels_pool=args.labels_pool, n_items=len(items), bos_check=worst))
    print("E3_DONE")


if __name__ == "__main__":
    main()
