"""
scripts/dose_response.py -- activation dose-response: clamp each neuron to
mean + k*sigma for a list of k (sigma = std of the neuron's last-token
activation over the baseline prompts) and record the per-prompt entropy at
every level. One row per (neuron, category, prompt, level).

  python scripts/dose_response.py --neurons L31_N2477 --values -2,-1,0,1,2,4 \
      --categories all --out results/dose_response_v4.csv
"""

import argparse

import numpy as np
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
    ap.add_argument("--values", default="-2,-1,-0.5,0,0.5,1,2,4", help="sigma multipliers")
    ap.add_argument("--categories", default="all")
    ap.add_argument("--splits", default="working,held_out")
    ap.add_argument("--include-controls", action="store_true")
    ap.add_argument("--baseline-format", choices=("raw", "chat"), default="raw")
    ap.add_argument("--positions", choices=("last", "all"), default="last")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "dose_response_v4.csv"))
    args = ap.parse_args(argv)
    out = guard_output(args.out, args.overwrite)

    from shared.ablation import activation_sweep_and_get_probs, get_probs_and_activation
    from shared.detection import capture_intermediate_activations
    from shared.model_utils import compute_entropy, compute_top1_prob
    from shared.baselines import general_baseline_prompts, GENERAL_BASELINE_SHA256
    from shared.provenance import build_provenance, write_provenance, data_file_hashes

    neurons = parse_neurons(args.neurons)
    multipliers = [float(v) for v in args.values.split(",")]
    splits = set(args.splits.split(","))
    model, tokenizer = load_model_from_args(args)

    # baseline mean and sigma per neuron (one pass per baseline prompt)
    baseline = general_baseline_prompts(tokenizer, args.baseline_format)
    layers = sorted({l for l, _ in neurons})
    acts = {k: [] for k in neurons}
    for p in baseline:
        cap = capture_intermediate_activations(model, tokenizer, p, layers)
        for l, n in neurons:
            acts[(l, n)].append(float(cap[l][n]))
    stats = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acts.items()}
    print("baseline mean/sigma:", {f"L{l}_N{n}": stats[(l, n)] for l, n in neurons})

    rows, files = [], []
    cats = parse_categories(args.categories)
    for cat in cats:
        prompts, controls = load_category(cat)
        files += list(data_paths(cat))
        recs = [(r, False) for r in prompts] + ([(r, True) for r in controls] if args.include_controls else [])
        recs = [(r, c) for r, c in recs if r["split"] in splits]
        for (l, n) in neurons:
            mu, sd = stats[(l, n)]
            levels = [mu + k * sd for k in multipliers]
            for r, is_ctrl in recs:
                text = r["chat_formatted_prompt"]
                clean, act = get_probs_and_activation(model, tokenizer, text, l, n)
                e0 = compute_entropy(clean)
                for k, v, probs in zip(multipliers, levels,
                                       activation_sweep_and_get_probs(model, tokenizer, text, l, n, levels, args.positions)):
                    e = compute_entropy(probs)
                    rows.append({
                        "neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n, "category": cat,
                        "prompt_id": r["prompt_id"], "split": r["split"], "is_control": is_ctrl,
                        "sigma_multiplier": k, "clamp_value": v, "baseline_mean": mu, "baseline_sigma": sd,
                        "orig_activation": act, "orig_entropy": e0, "clamped_entropy": e,
                        "entropy_shift": e - e0, "orig_top1_prob": compute_top1_prob(clean),
                        "clamped_top1_prob": compute_top1_prob(probs), "precision": model.cn_precision,
                    })
            print(f"[{cat}] L{l}_N{n}: {len(recs)} prompts x {len(levels)} levels done")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    write_provenance(out, build_provenance(
        model, neurons=[f"L{l}_N{n}" for l, n in neurons], categories=cats, sigma_multipliers=multipliers,
        positions=args.positions, baseline_prompt_sha256=GENERAL_BASELINE_SHA256,
        n_baseline_prompts=len(baseline), baseline_format=args.baseline_format,
        baseline_stats={f"L{l}_N{n}": {"mean": stats[(l, n)][0], "sigma": stats[(l, n)][1]} for l, n in neurons},
        data_file_sha256s=data_file_hashes(files)))
    print(df.groupby(["neuron_id", "sigma_multiplier"])["entropy_shift"].mean())
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
