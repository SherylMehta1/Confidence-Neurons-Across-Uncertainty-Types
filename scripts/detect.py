"""
scripts/detect.py -- split-half-validated candidate-neuron detection on the
working splits of data/<category>/prompts.jsonl AND controls.jsonl for all
three categories, stratified by category (uncertain and control prompts of a
category count as one stratum; pass --stratify-controls to stratify by
category x is_control).

  python scripts/detect.py --layer-range 20-31 --top-k-per-half 60 \
      --min-abs-corr 0.5 --seed 42 --out candidate_neurons_v4.json

Writes <out>, the sibling provenance JSON, and the full per-half correlation
distribution (--distribution-out, default full_correlation_distribution_v4.json).
"""

import argparse

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe: make _common importable
from _common import (REPO_ROOT, add_model_args, guard_output, load_category, load_model_from_args,
                     parse_categories, parse_layer_range)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--categories", default="all")
    ap.add_argument("--layer-range", default="20-31")
    ap.add_argument("--top-k-per-half", type=int, default=60)
    ap.add_argument("--top-k-final", type=int, default=None, help="default: keep all split-half survivors")
    ap.add_argument("--min-abs-corr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-controls", action="store_true", help="use uncertain prompts only")
    ap.add_argument("--stratify-controls", action="store_true")
    ap.add_argument("--no-stratify", action="store_true", help="legacy unstratified split")
    ap.add_argument("--out", default=str(REPO_ROOT / "candidate_neurons_v4.json"))
    ap.add_argument("--distribution-out", default=str(REPO_ROOT / "full_correlation_distribution_v4.json"))
    args = ap.parse_args(argv)

    out = guard_output(args.out, args.overwrite)
    dist_out = guard_output(args.distribution_out, args.overwrite)

    prompts, labels, files = [], [], []
    for cat in parse_categories(args.categories):
        unc, ctrl = load_category(cat)
        pp, cp = REPO_ROOT / "data" / cat / "prompts.jsonl", REPO_ROOT / "data" / cat / "controls.jsonl"
        files.append(pp)
        recs = [(r, False) for r in unc if r["split"] == "working"]
        if not args.no_controls:
            recs += [(r, True) for r in ctrl if r["split"] == "working"]
            files.append(cp)
        for r, is_ctrl in recs:
            prompts.append(r["chat_formatted_prompt"])
            labels.append(f"{cat}{'_control' if (args.stratify_controls and is_ctrl) else ''}")
    print(f"Baseline pool: {len(prompts)} working-split prompts "
          f"({', '.join(f'{l}:{labels.count(l)}' for l in sorted(set(labels)))})")

    model, tokenizer = load_model_from_args(args)

    from shared.detection import detect_candidate_neurons_split_half, save_candidate_neurons_v2
    from shared.provenance import data_file_hashes
    candidates, dist = detect_candidate_neurons_split_half(
        model, tokenizer, prompts, layer_range=parse_layer_range(args.layer_range),
        top_k_per_half=args.top_k_per_half, top_k_final=args.top_k_final, seed=args.seed,
        min_abs_corr=args.min_abs_corr, stratify_by=None if args.no_stratify else labels,
    )
    save_candidate_neurons_v2(candidates, dist, prompts, args.seed, path=out, distribution_path=dist_out,
                              extra_provenance={"data_file_sha256s": data_file_hashes(files),
                                                "categories": args.categories,
                                                "include_controls": not args.no_controls})
    for c in candidates:
        print(f"  {c['neuron_id']}: r_a={c['detection_correlation_half_a']:+.3f} "
              f"r_b={c['detection_correlation_half_b']:+.3f}")


if __name__ == "__main__":
    main()
