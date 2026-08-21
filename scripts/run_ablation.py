"""
scripts/run_ablation.py -- mean-ablation causal test for a candidate file on
one or more categories (uncertain prompts + matched controls, all splits),
writing the v4 results schema (RESULTS_SCHEMA.md) incrementally with resume.

  python scripts/run_ablation.py --candidates candidate_neurons.json \
      --categories all --baseline general --out-dir results/ablation_v4 --precision bf16

--baseline general          : mean from shared.baselines (60 documented prompts;
                              --general-baseline-file overrides; --baseline-format raw|chat)
--baseline pooled_controls  : mean from the category's working-split uncertain + control prompts
--baseline category_working : the category's working-split uncertain prompts only (legacy)
--control-neurons N         : also run N random non-candidate neurons from the candidate
                              layer range (listed in provenance as control_neurons)
"""

import argparse
import random

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe: make _common importable
from _common import (REPO_ROOT, add_model_args, data_paths, load_category, load_model_from_args,
                     parse_categories)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--candidates", default=str(REPO_ROOT / "candidate_neurons.json"))
    ap.add_argument("--categories", default="all")
    ap.add_argument("--baseline", choices=("general", "pooled_controls", "category_working"), default="general")
    ap.add_argument("--general-baseline-file", default=None,
                    help=".jsonl (chat_formatted_prompt) or .txt (one prompt per line)")
    ap.add_argument("--baseline-format", choices=("raw", "chat"), default="raw")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "ablation_v4"))
    ap.add_argument("--positions", choices=("last", "all"), default="last")
    ap.add_argument("--control-neurons", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    from shared.detection import load_candidate_neurons
    from shared.run_ablation_pipeline import run_category
    from shared.baselines import general_baseline_prompts, load_baseline_file

    candidates = load_candidate_neurons(args.candidates)
    model, tokenizer = load_model_from_args(args)

    baseline = None
    mean_source = None
    if args.baseline == "general":
        baseline = (load_baseline_file(args.general_baseline_file) if args.general_baseline_file
                    else general_baseline_prompts(tokenizer, args.baseline_format))
    else:
        mean_source = args.baseline

    control_neurons = []
    if args.control_neurons:
        rng = random.Random(args.seed)
        layers = sorted({c["layer"] for c in candidates})
        taken = {(c["layer"], c["neuron_idx"]) for c in candidates}
        while len(control_neurons) < args.control_neurons:
            pair = (rng.choice(layers), rng.randrange(model.config.intermediate_size))
            if pair not in taken:
                taken.add(pair)
                control_neurons.append(pair)
        print(f"control neurons: {control_neurons}")

    for cat in parse_categories(args.categories):
        prompts, controls = load_category(cat)
        print(f"[{cat}] {len(prompts)} prompts, {len(controls)} controls, {len(candidates)} candidates")
        run_category(
            model, tokenizer, candidates, prompts, controls, cat,
            baseline_prompts_for_mean=baseline, mean_source=mean_source,
            out_dir=args.out_dir, overwrite=args.overwrite, resume=not args.no_resume,
            control_neurons=control_neurons, positions=args.positions,
            candidates_path=args.candidates, data_paths=[str(p) for p in data_paths(cat)], seed=args.seed,
        )


if __name__ == "__main__":
    main()
