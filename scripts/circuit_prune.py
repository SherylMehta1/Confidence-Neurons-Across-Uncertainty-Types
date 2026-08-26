"""
Decider experiment 2 -- greedy pruning of the faithfulness component set (minimality curve).

Takes the 20-head + 100-neuron set of an arm, scores each component's leave-one-out marginal contribution
to the control->uncertain hedge log-odds recovery on a subset of held-out pairs, then evaluates NESTED
subsets (top-k components by marginal) on the held-out pairs. Output: recovery-vs-size curve and the
smallest subset with recovery > 0.7 -- turning the single (20+100) point of the faithfulness test into a
curve, and answering "sufficient or minimal?".

Outputs: <out-dir>/prune_marginals.csv, prune_curve.csv, prune_summary.txt.
Usage: python scripts/circuit_prune.py --category familiarity [--n-heads 20 --n-neurons 100]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse

import numpy as np
import pandas as pd

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from circuit_common import Readout, encode_pair, load_pairs, position_map, recovery, run_capture
from circuit_faithfulness import all_position_pairs, component_set, patch_all
from shared.provenance import build_provenance, write_provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--circuit-dir", default=None)
    ap.add_argument("--n-heads", type=int, default=20)
    ap.add_argument("--n-neurons", type=int, default=100)
    ap.add_argument("--loo-pairs", type=int, default=25, help="held-out pairs used for leave-one-out scoring")
    ap.add_argument("--eval-pairs", type=int, default=60, help="held-out pairs for the nested-subset curve")
    ap.add_argument("--sizes", default="5,10,20,40,60,80,100,120")
    args = ap.parse_args()
    circ = REPO_ROOT / (args.circuit_dir or f"results/circuit_{args.category}")
    guard_output(circ / "prune_curve.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    ro = Readout(tokenizer)
    heads, neurons = component_set(circ, args.n_heads, args.n_neurons)
    comps = [("head", h) for h in heads] + [("neuron", n) for n in neurons]
    pairs, _ = load_pairs(args.category, args.eval_pairs, "held_out")
    loo_pairs = pairs[: args.loo_pairs]
    print(f"[{args.category}] {len(comps)} components; LOO on {len(loo_pairs)} pairs, curve on {len(pairs)} pairs")

    encs = []
    for u, c in pairs:
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        lg_u = run_capture(model, enc_u, [], "resid")[0]; lg_c = run_capture(model, enc_c, [], "resid")[0]
        encs.append((enc_u, enc_c, all_position_pairs(al), ro.logodds(lg_u).item(), ro.logodds(lg_c).item()))

    def rec_set(hs, ns, subset):
        vals = []
        for enc_u, enc_c, pm, lo_u, lo_c in subset:
            lo, _ = patch_all(model, ro, enc_u, enc_c, pm, hs, ns)
            vals.append(recovery(lo, lo_u, lo_c))
        return float(np.nanmean(vals))

    full = rec_set(heads, neurons, encs[: len(loo_pairs)])
    print(f"  full-set recovery on LOO subset: {full:+.3f}")
    marg = []
    for i, (kind, comp) in enumerate(comps):
        hs = [h for j, (k2, h) in enumerate(comps) if k2 == "head" and j != i]
        ns = [n for j, (k2, n) in enumerate(comps) if k2 == "neuron" and j != i]
        r = rec_set(hs, ns, encs[: len(loo_pairs)])
        marg.append(dict(kind=kind, layer=comp[0], index=comp[1], drop_recovery=r, marginal=full - r))
        if (i + 1) % 20 == 0:
            print(f"  LOO {i + 1}/{len(comps)}")
    mdf = pd.DataFrame(marg).sort_values("marginal", ascending=False)
    circ.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(circ / "prune_marginals.csv", index=False)

    order = list(mdf[["kind", "layer", "index"]].itertuples(index=False, name=None))
    curve = []
    for k in [int(x) for x in args.sizes.split(",") if x.strip()]:
        top = order[:k]
        hs = [(l, i) for kd, l, i in top if kd == "head"]; ns = [(l, i) for kd, l, i in top if kd == "neuron"]
        r = rec_set(hs, ns, encs)
        curve.append(dict(k=k, n_heads=len(hs), n_neurons=len(ns), recovery=r))
        print(f"  top-{k}: recovery {r:+.3f} ({len(hs)} heads, {len(ns)} neurons)")
    cdf = pd.DataFrame(curve); cdf.to_csv(circ / "prune_curve.csv", index=False)
    passing = cdf[cdf.recovery > 0.7]
    lines = [f"Greedy pruning -- {args.category}; {len(comps)} components ranked by LOO marginal on {len(loo_pairs)} pairs; curve on {len(pairs)} held-out pairs",
             "  " + " ".join(f"k={int(r.k)}:{r.recovery:+.2f}" for r in cdf.itertuples()),
             f"  smallest subset with recovery > 0.7: {int(passing.k.min())} components" if len(passing) else "  no subset reaches 0.7 (full set included)",
             f"  top-10 by marginal: " + ", ".join(f"{r.kind[0].upper()}L{int(r.layer)}_{int(r.index)}({r.marginal:+.03f})" for r in mdf.head(10).itertuples())]
    summary = "\n".join(lines); print(summary)
    (circ / "prune_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(circ / "prune_curve.csv", build_provenance(model, script="scripts/circuit_prune.py", category=args.category, n_components=len(comps), loo_pairs=len(loo_pairs), eval_pairs=len(pairs)))


if __name__ == "__main__":
    main()
