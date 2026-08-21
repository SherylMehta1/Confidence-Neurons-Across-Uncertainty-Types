"""
scripts/frozen_norm.py -- frozen-RMSNorm counterfactual for specific neurons:
per prompt, clean entropy, full mean-ablation entropy, and entropy under
mean-ablation with the final norm's scale frozen at its clean value. The gap
between the two ablations is the share of the effect that flows through the
normalization denominator (Stolfo et al.).

  python scripts/frozen_norm.py --neurons L31_N2477,L30_N3533 --categories all \
      --out results/frozen_norm_v4.csv

The mean uses the general baseline (shared.baselines) unless --baseline is
given (pooled_controls | category_working).
"""

import argparse

import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe: make _common importable
from _common import (REPO_ROOT, add_model_args, data_paths, guard_output, load_category,
                     load_model_from_args, parse_categories, parse_neurons)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--categories", default="all")
    ap.add_argument("--splits", default="working,held_out")
    ap.add_argument("--include-controls", action="store_true")
    ap.add_argument("--baseline", choices=("general", "pooled_controls", "category_working"), default="general")
    ap.add_argument("--baseline-format", choices=("raw", "chat"), default="raw")
    ap.add_argument("--positions", choices=("last", "all"), default="last")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "frozen_norm_v4.csv"))
    args = ap.parse_args(argv)
    out = guard_output(args.out, args.overwrite)

    from shared.ablation import (compute_mean_activations, frozen_norm_ablate_and_get_probs,
                                 mean_ablate_and_get_probs, get_probs_and_activation)
    from shared.model_utils import compute_entropy, compute_top1_prob
    from shared.baselines import general_baseline_prompts, GENERAL_BASELINE_SHA256
    from shared.run_ablation_pipeline import resolve_mean_prompts
    from shared.provenance import build_provenance, sha256_prompts, write_provenance, data_file_hashes

    neurons = parse_neurons(args.neurons)
    splits = set(args.splits.split(","))
    model, tokenizer = load_model_from_args(args)
    cats = parse_categories(args.categories)

    rows, files, mean_info = [], [], {}
    for cat in cats:
        prompts, controls = load_category(cat)
        files += list(data_paths(cat))
        if args.baseline == "general":
            src, mean_prompts = "general_baseline", general_baseline_prompts(tokenizer, args.baseline_format)
        else:
            src, mean_prompts = resolve_mean_prompts(prompts, controls, None, args.baseline)
        means = compute_mean_activations(model, tokenizer, neurons, mean_prompts)
        mean_info[cat] = {"mean_source": src, "baseline_prompt_sha256": sha256_prompts(mean_prompts),
                          "means": {f"L{l}_N{n}": v for (l, n), v in means.items()}}
        recs = [(r, False) for r in prompts] + ([(r, True) for r in controls] if args.include_controls else [])
        recs = [(r, c) for r, c in recs if r["split"] in splits]
        for (l, n) in neurons:
            for i, (r, is_ctrl) in enumerate(recs):
                text = r["chat_formatted_prompt"]
                clean, act = get_probs_and_activation(model, tokenizer, text, l, n)
                full = mean_ablate_and_get_probs(model, tokenizer, text, l, n, means[(l, n)], args.positions)
                frozen = frozen_norm_ablate_and_get_probs(model, tokenizer, text, l, n, means[(l, n)], args.positions)
                e0, e1, e2 = compute_entropy(clean), compute_entropy(full), compute_entropy(frozen)
                rows.append({
                    "neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n, "category": cat,
                    "prompt_id": r["prompt_id"], "split": r["split"], "is_control": is_ctrl,
                    "orig_entropy": e0, "ablated_entropy": e1, "entropy_shift": e1 - e0,
                    "frozen_norm_ablated_entropy": e2, "shift_under_frozen_norm": e2 - e0,
                    "norm_pathway_share": ((e1 - e2) / (e1 - e0)) if abs(e1 - e0) > 1e-9 else float("nan"),
                    "orig_top1_prob": compute_top1_prob(clean), "ablated_top1_prob": compute_top1_prob(full),
                    "frozen_norm_top1_prob": compute_top1_prob(frozen),
                    "orig_activation": act, "mean_val": means[(l, n)], "mean_source": src,
                    "precision": model.cn_precision,
                })
            print(f"[{cat}] L{l}_N{n}: {len(recs)} prompts done")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    write_provenance(out, build_provenance(
        model, neurons=[f"L{l}_N{n}" for l, n in neurons], categories=cats, positions=args.positions,
        baseline=args.baseline, general_baseline_sha256=GENERAL_BASELINE_SHA256, means=mean_info,
        data_file_sha256s=data_file_hashes(files)))
    print(df.groupby(["category", "neuron_id"])[["entropy_shift", "shift_under_frozen_norm"]].mean())
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
