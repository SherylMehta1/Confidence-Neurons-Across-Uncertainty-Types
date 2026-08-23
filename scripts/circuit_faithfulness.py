"""
Circuit step 5 -- faithfulness of the found set, and step 6 -- overlap between arms.

Faithfulness: take the verified heads (head_verification.csv, top --n-heads by recovery) and the top --n-neurons
neurons (neuron_attribution.csv) of one arm and patch ALL of them jointly from the control twin into the uncertain
prompt (all aligned positions) on HELD-OUT pairs; report the recovered fraction of the hedge log-odds gap and of the
entropy gap. Null: --n-random random sets with the same number of heads and neurons. The reverse direction
(uncertain -> control) is reported too. A set that recovers > 0.7 with < 2% of components is a circuit.

Overlap (optional, --compare <other circuit dir>): Jaccard overlap of the verified head sets and of the top-neuron
sets between two arms, and the Spearman correlation of head recoveries over the union.

Outputs: <out-dir>/faithfulness.csv, faithfulness_summary.txt (+ overlap section when --compare is given).
Usage: python scripts/circuit_faithfulness.py --category familiarity --n-heads 20 --n-neurons 100 [--compare results/circuit_conflict]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import random

import numpy as np
import pandas as pd

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from circuit_common import Readout, encode_pair, load_pairs, position_map, recovery, reverse_map, run_capture, run_patched
from shared.provenance import build_provenance, write_provenance


def all_position_pairs(al):
    return position_map(al, "prefix") + position_map(al, "entity") + position_map(al, "suffix")


def component_set(circ_dir, n_heads, n_neurons):
    heads = pd.read_csv(circ_dir / "head_verification.csv").sort_values("rec_control_to_uncertain", ascending=False).head(n_heads)
    neurons = pd.read_csv(circ_dir / "neuron_attribution.csv").head(n_neurons)
    return [(int(r["layer"]), int(r["head"])) for _, r in heads.iterrows()], [(int(r.layer), int(r.neuron)) for _, r in neurons.iterrows()]


def patch_all(model, ro, enc_t, enc_s, pmap, heads, neurons, forward_dir=True):
    layers_h = sorted({l for l, _ in heads}); layers_n = sorted({l for l, _ in neurons})
    _, src_h = run_capture(model, enc_s, layers_h, "heads") if layers_h else (None, {})
    _, src_n = run_capture(model, enc_s, layers_n, "neurons") if layers_n else (None, {})
    edits = [(l, "heads", pmap, src_h[l], h) for l, h in heads] + [(l, "neurons", pmap, src_n[l], j) for l, j in neurons]
    lg = run_patched(model, enc_t, edits)
    return ro.logodds(lg).item(), ro.entropy(lg).item()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--circuit-dir", default=None)
    ap.add_argument("--n-heads", type=int, default=20)
    ap.add_argument("--n-neurons", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--limit", type=int, default=60, help="held-out pairs")
    ap.add_argument("--compare", default=None, help="another circuit dir for overlap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    circ = REPO_ROOT / (args.circuit_dir or f"results/circuit_{args.category}")
    guard_output(circ / "faithfulness.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    ro = Readout(tokenizer); rng = random.Random(args.seed)
    heads, neurons = component_set(circ, args.n_heads, args.n_neurons)
    pairs, how = load_pairs(args.category, args.limit, "held_out")
    if not pairs:
        pairs, how = load_pairs(args.category, args.limit, None); print("WARNING: no held-out split; using all pairs")
    H, L, D = model.config.num_attention_heads, model.config.num_hidden_layers, model.config.intermediate_size
    n_comp, total = len(heads) + len(neurons), L * H + L * D
    frac_heads, frac_neurons = len(heads) / (L * H), len(neurons) / (L * D)  # sparsity per component type
    sparse = frac_heads < 0.02 and frac_neurons < 0.02
    print(f"[{args.category}] {len(heads)} heads ({100 * frac_heads:.2f}% of heads) + {len(neurons)} neurons ({100 * frac_neurons:.3f}% of neurons); {len(pairs)} held-out pairs")

    rows = []
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pm = all_position_pairs(al)
        lg_u, lg_c = run_capture(model, enc_u, [], "resid")[0], run_capture(model, enc_c, [], "resid")[0]
        lo_u, H_u, lo_c, H_c = ro.logodds(lg_u).item(), ro.entropy(lg_u).item(), ro.logodds(lg_c).item(), ro.entropy(lg_c).item()
        lo, Hh = patch_all(model, ro, enc_u, enc_c, pm, heads, neurons)
        rows.append(dict(pair=k, set="circuit", direction="control_to_uncertain", logodds_rec=recovery(lo, lo_u, lo_c), entropy_rec=recovery(Hh, H_u, H_c)))
        lo, Hh = patch_all(model, ro, enc_c, enc_u, reverse_map(pm), heads, neurons)
        rows.append(dict(pair=k, set="circuit", direction="uncertain_to_control", logodds_rec=recovery(lo, lo_c, lo_u), entropy_rec=recovery(Hh, H_c, H_u)))
        for r_i in range(args.n_random):
            rh = list({(rng.randrange(L), rng.randrange(H)) for _ in heads}); rn = list({(rng.randrange(L), rng.randrange(D)) for _ in neurons})
            lo, Hh = patch_all(model, ro, enc_u, enc_c, pm, rh, rn)
            rows.append(dict(pair=k, set=f"random{r_i}", direction="control_to_uncertain", logodds_rec=recovery(lo, lo_u, lo_c), entropy_rec=recovery(Hh, H_u, H_c)))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs")
    df = pd.DataFrame(rows); df.to_csv(circ / "faithfulness.csv", index=False)
    circ_cu = df[(df.set == "circuit") & (df.direction == "control_to_uncertain")]; circ_uc = df[(df.set == "circuit") & (df.direction == "uncertain_to_control")]
    rnd = df[df.set.str.startswith("random")].groupby("set").logodds_rec.mean()
    lines = [f"Faithfulness -- {args.category}; {len(heads)} heads ({100 * frac_heads:.2f}% of all heads) + {len(neurons)} neurons ({100 * frac_neurons:.3f}% of all MLP neurons); {len(pairs)} held-out pairs",
             f"  control -> uncertain: log-odds recovery {circ_cu.logodds_rec.mean():+.3f}, entropy recovery {circ_cu.entropy_rec.mean():+.3f}",
             f"  uncertain -> control: log-odds recovery {circ_uc.logodds_rec.mean():+.3f}, entropy recovery {circ_uc.entropy_rec.mean():+.3f}",
             f"  random sets (same size, c->u): log-odds recovery {rnd.mean():+.3f} +- {rnd.std():.3f} over {len(rnd)} sets",
             f"  verdict: {'CIRCUIT (>0.7 recovery, <2% of heads and of neurons)' if circ_cu.logodds_rec.mean() > 0.7 and sparse else 'NOT a sufficient circuit at this size'}"]
    if args.compare:
        other = REPO_ROOT / args.compare
        oh, on = component_set(other, args.n_heads, args.n_neurons)
        a = pd.read_csv(circ / "head_verification.csv"); b = pd.read_csv(other / "head_verification.csv")
        m = a.merge(b, on=["layer", "head"], suffixes=("_a", "_b"))
        rho = m[["rec_control_to_uncertain_a", "rec_control_to_uncertain_b"]].corr(method="spearman").iloc[0, 1] if len(m) > 2 else float("nan")
        lines += ["", f"Overlap with {args.compare}: heads Jaccard {jaccard(heads, oh):.2f} ({len(set(heads) & set(oh))} shared of {len(heads)}), "
                  f"neurons Jaccard {jaccard(neurons, on):.2f} ({len(set(neurons) & set(on))} shared), head-recovery Spearman over {len(m)} common verified heads {rho:+.2f}",
                  "  shared heads: " + ", ".join(f"L{l}H{h}" for l, h in sorted(set(heads) & set(oh))),
                  "  shared neuron layers: " + ", ".join(f"L{l}" for l in sorted({l for l, _ in set(neurons) & set(on)}))]
    summary = "\n".join(lines); print(summary)
    (circ / "faithfulness_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(circ / "faithfulness.csv", build_provenance(model, script="scripts/circuit_faithfulness.py", category=args.category, n_heads=len(heads), n_neurons=len(neurons), n_pairs=len(pairs), compare=args.compare))


if __name__ == "__main__":
    main()
