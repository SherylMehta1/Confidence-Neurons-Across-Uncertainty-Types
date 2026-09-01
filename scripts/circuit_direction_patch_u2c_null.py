"""
Decider experiment 1 -- rank-1 direction patching vs the component circuit.

At each layer of --layers, replace ONLY the projection of the residual stream onto that layer's uncertainty
direction (diff of means at the last position, computed on WORKING pairs) with the source twin's projection,
at all aligned positions, and measure held-out recovery of the hedge log-odds and entropy with exactly the
faithfulness protocol (both directions, size-matched nulls: random unit directions).

If rank-1 recovery ~= the 20-head+100-neuron set's recovery, the mechanism is a low-rank directional channel
that the components implement; if it falls short, the component set carries more than the direction.

Outputs: <out-dir>/direction_patch.csv, direction_patch_summary.txt (+ provenance).
Usage: python scripts/circuit_direction_patch.py --category familiarity --layers 13-19 [--limit 60]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args, parse_layer_range
from circuit_common import Readout, component_module, encode_pair, load_pairs, position_map, recovery, reverse_map, run_capture
from shared.provenance import build_provenance, write_provenance


def all_position_pairs(al):
    return position_map(al, "prefix") + position_map(al, "entity") + position_map(al, "suffix")


class Rank1Patch:
    """Replace h[t]·d with h_src[s]·d along unit direction d at the given layers/positions (resid stream)."""

    def __init__(self, model, layers, dirs, pmap, src_resid):
        self.model, self.layers, self.dirs, self.pmap, self.src = model, layers, dirs, pmap, src_resid
        self.handles = []

    def __enter__(self):
        for l in self.layers:
            self.handles.append(component_module(self.model, l, "resid").register_forward_hook(self._hook(l)))
        return self

    def _hook(self, l):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            d = self.dirs[l].to(h.device, torch.float32)
            for t, s in self.pmap:
                v = h[:, t, :].float()
                sv = self.src[l][:, s, :].to(h.device).float()
                v = v - (v @ d)[:, None] * d + (sv @ d)[:, None] * d
                h[:, t, :] = v.to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def __exit__(self, *exc):
        for x in self.handles:
            x.remove()
        self.handles = []


@torch.no_grad()
def rank1_forward(model, ro, enc_t, enc_s, layers, dirs, pmap):
    _, src = run_capture(model, enc_s, layers, "resid")
    with Rank1Patch(model, layers, dirs, pmap, src):
        lg = model(**enc_t, use_cache=False).logits[0, -1]
    return ro.logodds(lg).item(), ro.entropy(lg).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--layers", default="13-19", help="layers whose direction component is swapped")
    ap.add_argument("--limit", type=int, default=60, help="held-out pairs")
    ap.add_argument("--n-random", type=int, default=10, help="random unit directions as the null")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    guard_output(out_dir / "direction_patch_u2c_null.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    ro = Readout(tokenizer)
    layers = parse_layer_range(args.layers)
    gen = torch.Generator().manual_seed(args.seed)
    d_model = model.config.hidden_size

    work, _ = load_pairs(args.category, None, "working")
    held, how = load_pairs(args.category, args.limit, "held_out")
    if not held:
        held, how = load_pairs(args.category, args.limit, None); print("WARNING: no held-out split; using all pairs")
    print(f"[{args.category}] directions from {len(work)} working pairs; evaluated on {len(held)} held-out pairs; layers {layers}")

    # per-layer unit directions from the working pairs (last position, uncertain - control)
    X = defaultdict(list)
    for u, c in work:
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        _, ru = run_capture(model, enc_u, layers, "resid"); _, rc = run_capture(model, enc_c, layers, "resid")
        for l in layers:
            X[l].append((ru[l][0, -1].float().cpu(), rc[l][0, -1].float().cpu()))
    dirs = {}
    for l in layers:
        d = torch.stack([a for a, _ in X[l]]).mean(0) - torch.stack([b for _, b in X[l]]).mean(0)
        dirs[l] = d / (d.norm() + 1e-8)

    rand_dirs = []
    for _ in range(args.n_random):
        rd = {l: torch.nn.functional.normalize(torch.randn(d_model, generator=gen), dim=0) for l in layers}
        rand_dirs.append(rd)

    rows = []
    for k, (u, c) in enumerate(held):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pm = all_position_pairs(al)
        lg_u = run_capture(model, enc_u, [], "resid")[0]; lg_c = run_capture(model, enc_c, [], "resid")[0]
        lo_u, H_u = ro.logodds(lg_u).item(), ro.entropy(lg_u).item()
        lo_c, H_c = ro.logodds(lg_c).item(), ro.entropy(lg_c).item()
        lo, Hh = rank1_forward(model, ro, enc_u, enc_c, layers, dirs, pm)
        rows.append(dict(pair=k, set="direction", direction="control_to_uncertain", logodds_rec=recovery(lo, lo_u, lo_c), entropy_rec=recovery(Hh, H_u, H_c)))
        lo, Hh = rank1_forward(model, ro, enc_c, enc_u, layers, dirs, reverse_map(pm))
        rows.append(dict(pair=k, set="direction", direction="uncertain_to_control", logodds_rec=recovery(lo, lo_c, lo_u), entropy_rec=recovery(Hh, H_c, H_u)))
        for r_i, rd in enumerate(rand_dirs):
            lo, Hh = rank1_forward(model, ro, enc_u, enc_c, layers, rd, pm)
            rows.append(dict(pair=k, set=f"random{r_i}", direction="control_to_uncertain", logodds_rec=recovery(lo, lo_u, lo_c), entropy_rec=recovery(Hh, H_u, H_c)))
            lo, Hh = rank1_forward(model, ro, enc_c, enc_u, layers, rd, reverse_map(pm))
            rows.append(dict(pair=k, set=f"random{r_i}", direction="uncertain_to_control", logodds_rec=recovery(lo, lo_c, lo_u), entropy_rec=recovery(Hh, H_c, H_u)))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(held)} pairs")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "direction_patch_u2c_null.csv", index=False)

    cu = df[(df.set == "direction") & (df.direction == "control_to_uncertain")]
    uc = df[(df.set == "direction") & (df.direction == "uncertain_to_control")]
    rnd = df[df.set.str.startswith("random") & (df.direction == "control_to_uncertain")].groupby("set").logodds_rec.mean()
    rnd_uc = df[df.set.str.startswith("random") & (df.direction == "uncertain_to_control")].groupby("set").logodds_rec.mean()
    se = lambda g: 1.96 * g.std() / max(1, np.sqrt(len(g)))
    lines = [f"Rank-1 direction patching -- {args.category}; layers {layers[0]}-{layers[-1]}; ONE dimension per layer swapped; {len(held)} held-out pairs",
             f"  control -> uncertain: log-odds {cu.logodds_rec.mean():+.3f} [{cu.logodds_rec.mean()-se(cu.logodds_rec):+.3f},{cu.logodds_rec.mean()+se(cu.logodds_rec):+.3f}], entropy {cu.entropy_rec.mean():+.3f}",
             f"  uncertain -> control: log-odds {uc.logodds_rec.mean():+.3f} [{uc.logodds_rec.mean()-se(uc.logodds_rec):+.3f},{uc.logodds_rec.mean()+se(uc.logodds_rec):+.3f}], entropy {uc.entropy_rec.mean():+.3f}",
             f"  random unit directions (c->u): {rnd.mean():+.3f} +- {rnd.std():.3f} over {len(rnd)}",
             f"  random unit directions (u->c): {rnd_uc.mean():+.3f} +- {rnd_uc.std():.3f} over {len(rnd_uc)}; max {rnd_uc.max():+.3f}",
             "  compare with results/circuit_<arm>/faithfulness_summary.txt (the 20-head + 100-neuron set)"]
    summary = "\n".join(lines); print(summary)
    (out_dir / "direction_patch_u2c_null_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_dir / "direction_patch_u2c_null.csv", build_provenance(model, script="scripts/circuit_direction_patch_u2c_null.py", category=args.category, layers=layers, n_pairs=len(held), n_random=args.n_random))


if __name__ == "__main__":
    main()
