"""
screen_templates.py -- Person B, lack-of-knowledge. Step 1 of the live pipeline.

    python person_B_lack_of_knowledge/screen_templates.py [--n-per-side 15] [--seed 42]
                                                          [--allow-nf4] [--min-gap 0.15]

WHY: NEC's 78 question templates were built for free-text evaluation, not
single-next-token cloze completion. For most of them, top-1 probability on
"The answer is" measures phrasing convergence, not knowledge -- the
fabricated-entity and real-entity versions come out equally peaked. This
script measures EVERY template on a seeded, matched batch of unanswerable
(fabricated) and answerable (real) entities and writes the full table to

    person_B_lack_of_knowledge/template_screen_results.json

which rebuild_lack_of_knowledge_whitelisted.py reads to derive the template
whitelist (gap = answerable_mean_top1 - unanswerable_mean_top1 >= min_gap).

PRECISION: this is a measurement that decides the dataset, so it must be run
on the unquantized bf16 model (load_model(quantize=False), the default). NF4
is refused unless --allow-nf4 is passed, and the precision is recorded in
the output file either way.

Needs GPU + gated Llama-3.1-8B-Instruct. Tokenizer-only steps live in
nec_templates.py.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from person_B_lack_of_knowledge.nec_templates import (  # noqa: E402
    TEMPLATE_TO_CATEGORY, UNKNOWNBENCH_COMMIT, load_and_classify,
)

RESULTS_PATH = REPO_ROOT / "person_B_lack_of_knowledge" / "template_screen_results.json"


def _model_precision(model) -> str:
    prec = getattr(model, "cn_precision", None)
    if prec:
        return prec
    qc = getattr(getattr(model, "config", None), "quantization_config", None)
    if qc is not None:
        return "nf4" if getattr(qc, "load_in_4bit", False) else "quantized"
    return "unknown"


def screen_all_templates(model, tokenizer, n_per_side: int = 15, seed: int = 42, verbose: bool = True):
    """
    For every (category, template) pair with >= 2 items per side, sample up
    to n_per_side unanswerable + n_per_side answerable prompts (seeded),
    measure top-1 under the position-fixed cloze prompt, and return a list
    of per-template dicts sorted by gap descending.
    """
    from shared.model_utils import get_next_token_probs, compute_top1_prob
    from shared.prompt_format import build_completion_prompt

    rng = random.Random(seed)
    unans, ans, _unmatched = load_and_classify()
    by_u, by_a = defaultdict(list), defaultdict(list)
    for r in unans:
        by_u[r["template"]].append(r["prompt"])
    for r in ans:
        by_a[r["template"]].append(r["prompt"])

    def top1(prompt):
        formatted = build_completion_prompt(tokenizer, prompt)
        return compute_top1_prob(get_next_token_probs(model, tokenizer, formatted["chat_formatted_prompt"]))

    results = []
    all_templates = sorted(set(by_u) | set(by_a))
    for i, template in enumerate(all_templates):
        u_pool, a_pool = by_u.get(template, []), by_a.get(template, [])
        if len(u_pool) < 2 or len(a_pool) < 2:
            results.append({"template": template, "nec_category": TEMPLATE_TO_CATEGORY.get(template),
                            "n_unanswerable_pool": len(u_pool), "n_answerable_pool": len(a_pool),
                            "skipped": "too few items per side"})
            continue
        u_sample = rng.sample(u_pool, min(n_per_side, len(u_pool)))
        a_sample = rng.sample(a_pool, min(n_per_side, len(a_pool)))
        u_top1s = [top1(p) for p in u_sample]
        a_top1s = [top1(p) for p in a_sample]
        u_mean = sum(u_top1s) / len(u_top1s)
        a_mean = sum(a_top1s) / len(a_top1s)
        results.append({
            "template": template, "nec_category": TEMPLATE_TO_CATEGORY.get(template),
            "n_unanswerable_pool": len(u_pool), "n_answerable_pool": len(a_pool),
            "n_unanswerable_measured": len(u_top1s), "n_answerable_measured": len(a_top1s),
            "unanswerable_mean_top1": u_mean, "answerable_mean_top1": a_mean,
            "gap": a_mean - u_mean,
            "unanswerable_top1s": u_top1s, "answerable_top1s": a_top1s,
        })
        if verbose:
            print(f"[{i+1}/{len(all_templates)}] gap={a_mean-u_mean:+.3f}  "
                  f"(unans={u_mean:.3f}, ans={a_mean:.3f})  {template}")

    results.sort(key=lambda r: -r.get("gap", float("-inf")))
    return results


def whitelist_from_results(results, min_gap: float = 0.15, max_unanswerable_mean: float = 0.5):
    """Templates whose fabricated-entity mean is clearly lower than the
    real-entity mean, and whose fabricated mean is not already peaked."""
    return [r["template"] for r in results
            if "gap" in r and r["gap"] >= min_gap and r["unanswerable_mean_top1"] <= max_unanswerable_mean]


def save_results(results, model, n_per_side, seed, min_gap, path=RESULTS_PATH):
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": getattr(model, "cn_model_id", None),
        "precision": _model_precision(model),
        "unknownbench_commit": UNKNOWNBENCH_COMMIT,
        "n_per_side": n_per_side, "seed": seed,
        "whitelist_rule": {"min_gap": min_gap, "max_unanswerable_mean_top1": 0.5},
        "whitelist": whitelist_from_results(results, min_gap),
        "results": results,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} template rows -> {path}")
    print(f"{len(payload['whitelist'])} templates pass gap >= {min_gap}:")
    for t in payload["whitelist"]:
        print(f"  {t}")
    return payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-side", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-gap", type=float, default=0.15)
    p.add_argument("--allow-nf4", action="store_true",
                   help="allow a 4-bit model (development only; results are NOT usable for the dataset)")
    p.add_argument("--quantize", action="store_true", help="load the model in NF4 (requires --allow-nf4)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from shared.model_utils import load_model

    if args.quantize and not args.allow_nf4:
        sys.exit("Refusing to screen templates with an NF4 model. Pass --allow-nf4 to override "
                 "(development only).")
    model, tokenizer = load_model(quantize=args.quantize)
    prec = _model_precision(model)
    if prec != "bf16":
        if not args.allow_nf4:
            sys.exit(f"Model precision is {prec!r}, not bf16 -- refusing. Pass --allow-nf4 to override.")
        print("\n" + "!" * 78 + f"\nWARNING: screening with a {prec} model. The whitelist derived from this run "
              "is for pipeline development only and must NOT be used to build committed data.\n" + "!" * 78 + "\n")

    results = screen_all_templates(model, tokenizer, n_per_side=args.n_per_side, seed=args.seed)
    save_results(results, model, args.n_per_side, args.seed, args.min_gap)
    print("\nNEXT: python person_B_lack_of_knowledge/rebuild_lack_of_knowledge_whitelisted.py")
    return results


if __name__ == "__main__":
    main()
