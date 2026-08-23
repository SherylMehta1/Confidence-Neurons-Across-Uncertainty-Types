"""
Circuit step 1 -- position x layer map of twin activation patching.

For each twin pair and each layer, the residual stream after the block is patched from the control twin into the
uncertain prompt (and the reverse) at ONE position group at a time: the entity span, each suffix token
(question tail + prefill, right-aligned), and the last token. Recovery of the hedge log-odds and of the entropy
is averaged over pairs. This shows where the "unknown" state is computed (entity positions, early layers) and
where it reaches the final position (suffix / last token, later layers).

Outputs: <out>/position_map.csv (pair x layer x group x direction), position_map_summary.txt, provenance.
Usage: python scripts/circuit_position_map.py --category familiarity [--layers 0-31] [--limit 60]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse

import pandas as pd

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args, parse_layer_range
from circuit_common import Readout, encode_pair, load_pairs, position_map, recovery, reverse_map, run_capture, run_patched
from shared.provenance import build_provenance, write_provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="working | held_out | (all)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    guard_output(out_dir / "position_map.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    layers = parse_layer_range(args.layers) if args.layers else list(range(model.config.num_hidden_layers))
    pairs, how = load_pairs(args.category, args.limit, args.split)
    ro = Readout(tokenizer)
    print(f"[{args.category}] {len(pairs)} pairs ({how}); layers {layers[0]}-{layers[-1]}")

    rows = []
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        lg_u, act_u = run_capture(model, enc_u, layers, "resid")
        lg_c, act_c = run_capture(model, enc_c, layers, "resid")
        base = dict(u=(ro.logodds(lg_u).item(), ro.entropy(lg_u).item()), c=(ro.logodds(lg_c).item(), ro.entropy(lg_c).item()))
        groups = [("entity", position_map(al, "entity")), ("last", position_map(al, "last"))]
        groups += [(f"suffix-{j}", [pm]) for j, pm in enumerate(position_map(al, "suffix")) if j > 0]  # suffix-0 == last
        if al["prefix"]:
            groups.append(("prefix", position_map(al, "prefix")))
        for gname, pmap in groups:
            if not pmap:
                continue
            for l in layers:
                lg = run_patched(model, enc_u, [(l, "resid", pmap, act_c[l], None)])          # control -> uncertain
                lo, H = ro.logodds(lg).item(), ro.entropy(lg).item()
                rows.append(dict(pair=k, uncertain_id=u["prompt_id"], split=u.get("split"), layer=l, group=gname, n_pos=len(pmap), direction="control_to_uncertain",
                                 logodds_rec=recovery(lo, base["u"][0], base["c"][0]), entropy_rec=recovery(H, base["u"][1], base["c"][1])))
                lg = run_patched(model, enc_c, [(l, "resid", reverse_map(pmap), act_u[l], None)])  # uncertain -> control
                lo, H = ro.logodds(lg).item(), ro.entropy(lg).item()
                rows.append(dict(pair=k, uncertain_id=u["prompt_id"], split=u.get("split"), layer=l, group=gname, n_pos=len(pmap), direction="uncertain_to_control",
                                 logodds_rec=recovery(lo, base["c"][0], base["u"][0]), entropy_rec=recovery(H, base["c"][1], base["u"][1])))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "position_map.csv", index=False)

    lines = [f"Position x layer map -- {args.category}; {len(pairs)} pairs; layers {layers[0]}-{layers[-1]}", ""]
    for d, g in df.groupby("direction"):
        lines.append(f"[{d}] mean log-odds recovery by group (rows) x layer (cols):")
        piv = g.groupby(["group", "layer"]).logodds_rec.mean().unstack("layer")
        order = ["entity"] + sorted([x for x in piv.index if x.startswith("suffix")], key=lambda s: -int(s.split("-")[1])) + [x for x in ("last", "prefix") if x in piv.index]
        for gname in order:
            if gname in piv.index:
                lines.append(f"  {gname:<10}" + " ".join(f"{v:+.2f}" for v in piv.loc[gname].values))
        ent = g[g.group == "entity"].groupby("layer").logodds_rec.mean(); last = g[g.group == "last"].groupby("layer").logodds_rec.mean()
        f_ent = next((l for l, v in ent.items() if v >= 0.5), None); f_last = next((l for l, v in last.items() if v >= 0.5), None)
        peak = f"entity peak {ent.max():.2f} at L{ent.idxmax()}" if len(ent) else "no differing span between twins (entity group empty)"
        lines.append(f"  first layer with >=0.5 recovery: entity span {f_ent}, last token {f_last}; {peak}")
        lines.append("")
    summary = "\n".join(lines); print(summary)
    (out_dir / "position_map_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_dir / "position_map.csv", build_provenance(model, script="scripts/circuit_position_map.py", category=args.category, layers=layers, n_pairs=len(pairs), pairing=how))


if __name__ == "__main__":
    main()
