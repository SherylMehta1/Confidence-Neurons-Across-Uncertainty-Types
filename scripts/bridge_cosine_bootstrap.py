"""Pair-level bootstrap CI on cos(familiarity direction, hedging direction), per layer.

direction_bridge.py reports a point estimate of the cosine between the familiarity
direction (uncertain minus control, from --category) and the hedging direction
(hedger minus confabulator, from --hedging-category). Its saved .npz holds only the two
final unit vectors, not the per-prompt residuals, so the CI has to recompute residuals.
Both directions are then re-estimated on B bootstrap resamples drawn at the pair level
(twin_id: a resampled pair contributes both of its members), and the cosine recomputed.

Same percentile-bootstrap estimator as fig_dissociation.py / stats_upgrades.boot_ci.
Writes cos_bootstrap.csv (per layer: point, ci_lo, ci_hi) and cos_bootstrap.json into
--out-dir. Refuses to overwrite without --overwrite.

  python scripts/bridge_cosine_bootstrap.py --category familiarity_v2 \
      --hedging-category uu_hedge_confab --layer-range 20-31 --out-dir results/bridge_familiarity_v3
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_category, load_model_from_args, parse_layer_range
from direction_bridge import diff_of_means, last_token_residuals
from shared.provenance import build_provenance, write_provenance


def load_working(category):
    prompts, ctrls = load_category(category)
    recs = [dict(r, is_control=False) for r in prompts] + [dict(r, is_control=True) for r in ctrls]
    return [r for r in recs if r["split"] == "working"]


def pair_resample(work, rng):
    """Row indices for one bootstrap resample of pairs (twin_id), with replacement."""
    by_pair = {}
    for i, r in enumerate(work):
        by_pair.setdefault(r["twin_id"], []).append(i)
    pairs = list(by_pair)
    pick = rng.integers(0, len(pairs), len(pairs))
    return np.array([i for k in pick for i in by_pair[pairs[k]]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity_v2")
    ap.add_argument("--hedging-category", default="uu_hedge_confab")
    ap.add_argument("--layer-range", default="20-31")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/bridge_familiarity_v3")
    args = ap.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_csv = out_dir / "cos_bootstrap.csv"
    guard_output(out_csv, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    layers = parse_layer_range(args.layer_range)

    fam_work = load_working(args.category)
    hed_work = load_working(args.hedging_category)
    print(f"[{args.category}] {len(fam_work)} working prompts; [{args.hedging_category}] {len(hed_work)} working prompts")
    Xf = {l: t.cpu() for l, t in last_token_residuals(model, tokenizer, [r["chat_formatted_prompt"] for r in fam_work], layers).items()}
    Xh = {l: t.cpu() for l, t in last_token_residuals(model, tokenizer, [r["chat_formatted_prompt"] for r in hed_work], layers).items()}
    f_ctrl = np.array([r["is_control"] for r in fam_work]); f_unc = ~f_ctrl
    h_ctrl = np.array([r["is_control"] for r in hed_work]); h_pos = ~h_ctrl  # hedgers are the prompts side

    def cosines(fi, hi):
        out = {}
        for l in layers:
            fam = diff_of_means(Xf[l][fi], f_unc[fi], f_ctrl[fi])
            hed = diff_of_means(Xh[l][hi], h_pos[hi], h_ctrl[hi])
            out[l] = float(fam @ hed)
        return out

    point = cosines(np.arange(len(fam_work)), np.arange(len(hed_work)))
    rng = np.random.default_rng(args.seed)
    boots = {l: np.empty(args.n_boot) for l in layers}
    for b in range(args.n_boot):
        c = cosines(pair_resample(fam_work, rng), pair_resample(hed_work, rng))
        for l in layers:
            boots[l][b] = c[l]
        if (b + 1) % 1000 == 0:
            print(f"  {b + 1}/{args.n_boot} resamples")

    rows = []
    for l in layers:
        lo, hi = np.percentile(boots[l], [2.5, 97.5])
        rows.append(dict(layer=l, cos=point[l], ci_lo=float(lo), ci_hi=float(hi), boot_mean=float(boots[l].mean())))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("layer,cos,ci_lo,ci_hi,boot_mean\n")
        for r in rows:
            f.write(f"{r['layer']},{r['cos']:.6f},{r['ci_lo']:.6f},{r['ci_hi']:.6f},{r['boot_mean']:.6f}\n")
    summary = dict(category=args.category, hedging_category=args.hedging_category, n_boot=args.n_boot, seed=args.seed,
                   n_pairs_familiarity=len({r["twin_id"] for r in fam_work}),
                   n_pairs_hedging=len({r["twin_id"] for r in hed_work}), per_layer=rows)
    (out_dir / "cos_bootstrap.json").write_text(json.dumps(summary, indent=2))
    write_provenance(out_csv, build_provenance(model, script="scripts/bridge_cosine_bootstrap.py", **{
        k: v for k, v in vars(args).items() if k != "model_id"}))
    print("\ncos(familiarity, hedging) with pair-level bootstrap 95% CI:")
    for r in rows:
        print(f"  L{r['layer']}: {r['cos']:+.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
