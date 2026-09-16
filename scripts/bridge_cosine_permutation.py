"""Label-permutation null for cos(familiarity direction, hedging direction), per layer.

bridge_cosine_bootstrap.py asks how much the cosine wobbles under resampling, and for the
cosine of two noisily estimated directions that answer is biased low (resampling adds
independent noise to each direction; orthogonal noise shrinks cosines). This asks the
complementary question: if the hedger/confabulator labels carried no information, how
large a cosine would a hedging direction built from them reach anyway?

The familiarity direction is held fixed at its full-sample estimate. On the hedging side,
labels are swapped within each matched pair independently with probability 1/2 -- every
permuted split still has one "hedger" and one "confabulator" per pair, so the matching is
never broken -- the hedging direction is recomputed, and its cosine with the familiarity
direction recorded. The observed cosine is compared to that null: one-sided p for positive
alignment, plus null mean/sd and upper percentiles.

Writes cos_permutation.csv and cos_permutation.json into --out-dir, alongside (never over)
the bootstrap files. Refuses to overwrite without --overwrite.

  python scripts/bridge_cosine_permutation.py --category familiarity_v2 \
      --hedging-category uu_hedge_confab --layer-range 20-31 --out-dir results/bridge_familiarity_v3
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args, parse_layer_range
from bridge_cosine_bootstrap import load_working
from direction_bridge import diff_of_means, last_token_residuals
from shared.provenance import build_provenance, write_provenance


def pair_swap_masks(work, rng):
    """One relabeling: per pair, flip which member is called the hedger with prob 1/2.
    Returns (pos, neg) boolean masks over rows of `work`."""
    by_pair = {}
    for i, r in enumerate(work):
        by_pair.setdefault(r["twin_id"], {})["c" if r["is_control"] else "h"] = i
    pos = np.zeros(len(work), dtype=bool)
    for members in by_pair.values():
        h, c = members["h"], members["c"]
        if rng.random() < 0.5:
            h, c = c, h
        pos[h] = True
    return pos, ~pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity_v2")
    ap.add_argument("--hedging-category", default="uu_hedge_confab")
    ap.add_argument("--layer-range", default="20-31")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/bridge_familiarity_v3")
    args = ap.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_csv = out_dir / "cos_permutation.csv"
    guard_output(out_csv, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    layers = parse_layer_range(args.layer_range)
    fam_work = load_working(args.category)
    hed_work = load_working(args.hedging_category)
    print(f"[{args.category}] {len(fam_work)} working prompts; [{args.hedging_category}] {len(hed_work)} working prompts")
    Xf = {l: t.cpu() for l, t in last_token_residuals(model, tokenizer, [r["chat_formatted_prompt"] for r in fam_work], layers).items()}
    Xh = {l: t.cpu() for l, t in last_token_residuals(model, tokenizer, [r["chat_formatted_prompt"] for r in hed_work], layers).items()}
    f_ctrl = np.array([r["is_control"] for r in fam_work])
    h_ctrl = np.array([r["is_control"] for r in hed_work])

    fam = {l: diff_of_means(Xf[l], ~f_ctrl, f_ctrl) for l in layers}
    observed = {l: float(fam[l] @ diff_of_means(Xh[l], ~h_ctrl, h_ctrl)) for l in layers}

    rng = np.random.default_rng(args.seed)
    null = {l: np.empty(args.n_perm) for l in layers}
    for k in range(args.n_perm):
        pos, neg = pair_swap_masks(hed_work, rng)
        for l in layers:
            null[l][k] = float(fam[l] @ diff_of_means(Xh[l], pos, neg))
        if (k + 1) % 2000 == 0:
            print(f"  {k + 1}/{args.n_perm} permutations")

    rows = []
    for l in layers:
        n = null[l]
        p_one = (np.sum(n >= observed[l]) + 1) / (args.n_perm + 1)
        p_two = (np.sum(np.abs(n) >= abs(observed[l])) + 1) / (args.n_perm + 1)
        rows.append(dict(layer=l, cos=observed[l], null_mean=float(n.mean()), null_sd=float(n.std(ddof=1)),
                         null_p95=float(np.percentile(n, 95)), null_p99=float(np.percentile(n, 99)),
                         null_max=float(n.max()), p_one_sided=float(p_one), p_two_sided=float(p_two)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("layer,cos,null_mean,null_sd,null_p95,null_p99,null_max,p_one_sided,p_two_sided\n")
        for r in rows:
            f.write(",".join(str(r[k]) if k == "layer" else f"{r[k]:.6g}" for k in
                             ("layer", "cos", "null_mean", "null_sd", "null_p95", "null_p99", "null_max", "p_one_sided", "p_two_sided")) + "\n")
    summary = dict(category=args.category, hedging_category=args.hedging_category, n_perm=args.n_perm, seed=args.seed,
                   permutation="within-pair label swap, p=1/2 per pair, familiarity direction held fixed",
                   n_pairs_hedging=len({r["twin_id"] for r in hed_work}), per_layer=rows)
    (out_dir / "cos_permutation.json").write_text(json.dumps(summary, indent=2))
    write_provenance(out_csv, build_provenance(model, script="scripts/bridge_cosine_permutation.py", **{
        k: v for k, v in vars(args).items() if k != "model_id"}))
    print("\ncos(familiarity, hedging) vs within-pair label-permutation null:")
    for r in rows:
        print(f"  L{r['layer']}: observed {r['cos']:+.3f} | null {r['null_mean']:+.3f} +- {r['null_sd']:.3f}, "
              f"p95 {r['null_p95']:+.3f}, max {r['null_max']:+.3f} | p = {r['p_one_sided']:.2g}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
