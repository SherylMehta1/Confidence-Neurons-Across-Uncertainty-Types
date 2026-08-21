"""
scripts/mechanism_check.py -- logit-lens mechanism check for a candidate file:
per neuron, the fp32 weights-only direct_effect_score (max |W_U (gamma*w_out)|),
the top-k most promoted and most suppressed tokens, and the gamma-folded
output-weight norm.

  python scripts/mechanism_check.py --candidates candidate_neurons.json [--k 10]

Writes results/mechanism_check_<first 12 hex of candidate-file sha256>.json
plus its provenance sibling.
"""

import argparse
import json

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe: make _common importable
from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--candidates", default=str(REPO_ROOT / "candidate_neurons.json"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args(argv)

    from shared.detection import load_candidate_neurons
    from shared.logit_lens import direct_effect_score, top_direct_effect_tokens, neuron_output_vector
    from shared.provenance import build_provenance, sha256_file, write_provenance

    cand_hash = sha256_file(args.candidates)
    out = guard_output(f"{args.out_dir}/mechanism_check_{cand_hash[:12]}.json", args.overwrite)
    candidates = load_candidate_neurons(args.candidates)
    model, tokenizer = load_model_from_args(args)

    results = []
    for c in candidates:
        l, n = c["layer"], c["neuron_idx"]
        entry = {
            "neuron_id": c["neuron_id"], "layer": l, "neuron_idx": n,
            "direct_effect_score": direct_effect_score(model, l, n),
            "w_tilde_norm": float(neuron_output_vector(model, l, n).norm()),
            "top_tokens": top_direct_effect_tokens(model, tokenizer, l, n, k=args.k, largest=True),
            "bottom_tokens": top_direct_effect_tokens(model, tokenizer, l, n, k=args.k, largest=False),
        }
        for key in ("detection_correlation_half_a", "detection_correlation_half_b", "detection_correlation"):
            if key in c:
                entry[key] = c[key]
        results.append(entry)
        print(f"{c['neuron_id']}: DE={entry['direct_effect_score']:.4f} "
              f"top={[t for t, _ in entry['top_tokens'][:5]]}")

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    write_provenance(out, build_provenance(model, candidate_file_sha256=cand_hash, candidate_file=args.candidates,
                                           k=args.k, computation="fp32 weights-only logit lens, gamma folded"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
