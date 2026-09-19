"""
Causal timing: at which layer is the uncertain-vs-control difference causally sufficient to move the readout?

The Jacobian-lens trajectory asks where the hedge decision becomes linearly readable; this asks the
causal version with the same residual-stream patching used everywhere else in the circuit pipeline
(circuit_common.Patch, kind='resid', all aligned positions, held-out pairs, both directions).

Two cumulation modes, swept over a candidate layer l:
  up_to  patch the residual stream at layers 0..l from the source twin and let the model finish on
         its own from l+1. recovery(l) rises from the null floor to ~1; the transition layer is where
         the decision is already fixed in the stream -- the causal counterpart of "readable by l".
  from   patch layers l..L-1 (the literal "from l onward"). Because the last position is among the
         patched positions and layer L-1's output at that position is exactly what lm_head reads,
         this reproduces the source's logits for every l and recovery is ~1 throughout. Kept as a
         mode so the degeneracy is shown, not asserted.

The source twin's residual stream is captured once at all layers per pair and direction; each l is
then one patched forward pass (L passes per pair per direction; L=32 -> ~4 min on 60 pairs).

Writes <out-dir>/causal_timing_<mode>.csv (per pair, layer, direction: logodds_rec, entropy_rec),
causal_timing_<mode>_summary.csv (per layer, direction: means with percentile-bootstrap 95% CIs over
pairs, 10k resamples, seed 0) and causal_timing_<mode>_summary.txt. Refuses to overwrite without
--overwrite.

  python scripts/circuit_causal_timing.py --category familiarity_v2 --mode up_to --out-dir results/circuit_familiarity_v2
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse

import numpy as np
import pandas as pd

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from circuit_common import Readout, encode_pair, load_pairs, recovery, reverse_map, run_capture, run_patched
from circuit_faithfulness import all_position_pairs
from shared.provenance import build_provenance, write_provenance

B, SEED = 10_000, 0


def boot_ci(v, rng):
    v = np.asarray(v, dtype=float)
    idx = rng.integers(0, len(v), (B, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])


def layer_set(mode, l, L):
    return list(range(0, l + 1)) if mode == "up_to" else list(range(l, L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity_v2")
    ap.add_argument("--out-dir", default="results/circuit_familiarity_v2")
    ap.add_argument("--mode", choices=("up_to", "from"), default="up_to")
    ap.add_argument("--layers", default=None, help="comma list of layers to sweep (default: every layer)")
    ap.add_argument("--limit", type=int, default=60, help="held-out pairs")
    args = ap.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_csv = out_dir / f"causal_timing_{args.mode}.csv"
    guard_output(out_csv, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    ro = Readout(tokenizer)
    L = model.config.num_hidden_layers
    sweep = [int(x) for x in args.layers.split(",")] if args.layers else list(range(L))
    pairs, _ = load_pairs(args.category, args.limit, "held_out")
    if not pairs:
        pairs, _ = load_pairs(args.category, args.limit, None); print("WARNING: no held-out split; using all pairs")
    print(f"[{args.category}] mode={args.mode}; {len(pairs)} held-out pairs; sweep {sweep[0]}..{sweep[-1]} ({len(sweep)} layers)")

    rows = []
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pm = all_position_pairs(al)
        lg_u, src_u = run_capture(model, enc_u, list(range(L)), "resid")
        lg_c, src_c = run_capture(model, enc_c, list(range(L)), "resid")
        lo_u, H_u, lo_c, H_c = ro.logodds(lg_u).item(), ro.entropy(lg_u).item(), ro.logodds(lg_c).item(), ro.entropy(lg_c).item()
        for direc, enc_t, src, pmap, tgt, s in (("control_to_uncertain", enc_u, src_c, pm, (lo_u, H_u), (lo_c, H_c)),
                                              ("uncertain_to_control", enc_c, src_u, reverse_map(pm), (lo_c, H_c), (lo_u, H_u))):
            for l in sweep:
                edits = [(j, "resid", pmap, src[j], None) for j in layer_set(args.mode, l, L)]
                lg = run_patched(model, enc_t, edits)
                rows.append(dict(pair=k, layer=l, direction=direc,
                                 logodds_rec=recovery(ro.logodds(lg).item(), tgt[0], s[0]),
                                 entropy_rec=recovery(ro.entropy(lg).item(), tgt[1], s[1])))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    rng = np.random.default_rng(SEED)
    summ = []
    for direc in ("control_to_uncertain", "uncertain_to_control"):
        for l in sweep:
            g = df[(df.direction == direc) & (df.layer == l)]
            lo, hi = boot_ci(g.logodds_rec, rng)
            elo, ehi = boot_ci(g.entropy_rec, rng)
            summ.append(dict(layer=l, direction=direc, logodds_rec_mean=float(g.logodds_rec.mean()), ci_lo=float(lo), ci_hi=float(hi),
                             entropy_rec_mean=float(g.entropy_rec.mean()), entropy_ci_lo=float(elo), entropy_ci_hi=float(ehi), n_pairs=len(g)))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out_dir / f"causal_timing_{args.mode}_summary.csv", index=False)

    lines = [f"Causal timing -- {args.category}; mode={args.mode} ({'resid patched at layers 0..l' if args.mode == 'up_to' else 'resid patched at layers l..L-1'}); {len(pairs)} held-out pairs; all aligned positions"]
    for direc in ("control_to_uncertain", "uncertain_to_control"):
        s = sdf[sdf.direction == direc].set_index("layer")
        lines.append(f"  {direc}: log-odds recovery by layer: " + " ".join(f"L{l}:{s.loc[l].logodds_rec_mean:+.2f}" for l in sweep))
        above = [l for l in sweep if s.loc[l].ci_lo > 0.7]
        first = [l for l in sweep if s.loc[l].logodds_rec_mean > 0.7]
        lines.append(f"    first layer with mean > 0.7: {('L%d' % first[0]) if first else 'none'}; "
                     f"first with CI lower bound > 0.7: {('L%d' % above[0]) if above else 'none'}")
        lines.append(f"    entropy recovery by layer: " + " ".join(f"L{l}:{s.loc[l].entropy_rec_mean:+.2f}" for l in sweep))
    summary = "\n".join(lines); print(summary)
    (out_dir / f"causal_timing_{args.mode}_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_csv, build_provenance(model, script="scripts/circuit_causal_timing.py", category=args.category,
                                               mode=args.mode, layers=sweep, n_pairs=len(pairs)))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
