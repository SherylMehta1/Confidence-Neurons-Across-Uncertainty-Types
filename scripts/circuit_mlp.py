"""
Circuit step 4 -- MLP blocks and neurons.

(a) Per-layer MLP-output patching at the last position (and at all aligned positions) from the control twin into
    the uncertain prompt: which MLP blocks write the state that the readout uses.
(b) Neuron attribution inside --window layers: score = (control activation - uncertain activation) x
    d[hedge log-odds]/d[activation] on the down_proj input at the last position (and summed over all positions),
    per pair, averaged. Ranked neurons are verified by patching the top-N jointly (N in --verify-sizes) from the
    control twin, with recovery of the hedge log-odds; random N-sets from the same window give the null.
(c) Readers vs writers: for each top neuron, the cosine of its output weight with the per-layer familiarity
    direction (writer of the direction) and the correlation of its activation with the projection of the residual
    stream onto the direction at the previous layer (reader of it).

Outputs under <out-dir>: mlp_patching.csv, neuron_attribution.csv, neuron_verification.csv, mlp_summary.txt.
Usage: python scripts/circuit_mlp.py --category familiarity --window 12-26 [--limit 80]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args, parse_layer_range
from circuit_common import Readout, attribution, encode_pair, load_pairs, position_map, recovery, run_capture, run_patched
from shared.provenance import build_provenance, write_provenance


def all_position_pairs(al):
    return position_map(al, "prefix") + position_map(al, "entity") + position_map(al, "suffix")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--layers", default=None, help="layers for MLP-output patching (default all)")
    ap.add_argument("--window", default="12-26", help="layers for neuron attribution")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--verify-sizes", default="10,30,100,300")
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    guard_output(out_dir / "neuron_verification.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    layers = parse_layer_range(args.layers) if args.layers else list(range(model.config.num_hidden_layers))
    window = parse_layer_range(args.window)
    pairs, how = load_pairs(args.category, args.limit, "working")
    ro = Readout(tokenizer); d_mlp = model.config.intermediate_size
    print(f"[{args.category}] {len(pairs)} working pairs ({how}); MLP patching layers {layers[0]}-{layers[-1]}; neuron window {window[0]}-{window[-1]}")

    # ---- (a) MLP-output patching
    rows, cache = [], []
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        lg_u, mu = run_capture(model, enc_u, layers, "mlp"); lg_c, mc = run_capture(model, enc_c, layers, "mlp")
        lo_u, lo_c = ro.logodds(lg_u).item(), ro.logodds(lg_c).item()
        last, allp = position_map(al, "last"), all_position_pairs(al)
        for l in layers:
            for gname, pmap in (("last", last), ("all", allp)):
                lo = ro.logodds(run_patched(model, enc_u, [(l, "mlp", pmap, mc[l], None)])).item()
                rows.append(dict(pair=k, layer=l, positions=gname, rec=recovery(lo, lo_u, lo_c)))
        cache.append((enc_u, enc_c, al, lo_u, lo_c))
        if (k + 1) % 10 == 0:
            print(f"  mlp patching {k + 1}/{len(pairs)}")
    mlp = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp.to_csv(out_dir / "mlp_patching.csv", index=False)

    # ---- (b) neuron attribution in the window
    acc_last, acc_all = defaultdict(float), defaultdict(float)
    for k, (enc_u, enc_c, al, lo_u, lo_c) in enumerate(cache):
        _, src = run_capture(model, enc_c, window, "neurons")
        attr = attribution(model, enc_u, window, "neurons", ro, src, all_position_pairs(al))
        for l, m in attr.items():  # [seq_u, d_mlp]
            acc_last[l] += m[-1].numpy(); acc_all[l] += m.sum(0).numpy()
        if (k + 1) % 10 == 0:
            print(f"  neuron attribution {k + 1}/{len(cache)}")
    n = len(cache)
    nrows = [dict(neuron_id=f"L{l}_N{j}", layer=l, neuron=j, score_last=float(acc_last[l][j] / n), score_all=float(acc_all[l][j] / n)) for l in window for j in range(d_mlp)]
    neu = pd.DataFrame(nrows); neu["abs_all"] = neu.score_all.abs()
    neu = neu.sort_values("abs_all", ascending=False)
    neu.head(2000).to_csv(out_dir / "neuron_attribution.csv", index=False)

    # ---- (b') verification: patch top-N neurons jointly (all positions), random N-sets as null
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.verify_sizes.split(",") if x.strip()]
    ver = []
    def patch_set(neurons, enc_u, enc_c, al):
        _, src = run_capture(model, enc_c, sorted({l for l, _ in neurons}), "neurons")
        pm = all_position_pairs(al)
        edits = [(l, "neurons", pm, src[l], j) for l, j in neurons]
        return ro.logodds(run_patched(model, enc_u, edits)).item()
    for N in sizes:
        top = [(int(r.layer), int(r.neuron)) for _, r in neu.head(N).iterrows()]
        sub = cache[: min(30, len(cache))]  # same pairs for the top set and for every random set
        recs = [recovery(patch_set(top, eu, ec, al), lu, lc) for (eu, ec, al, lu, lc) in sub]
        nulls = []
        for _ in range(args.n_random):
            rnd = list({(rng.choice(window), rng.randrange(d_mlp)) for _ in range(N)})
            nulls.append(float(np.nanmean([recovery(patch_set(rnd, eu, ec, al), lu, lc) for (eu, ec, al, lu, lc) in sub])))
        ver.append(dict(n_neurons=N, rec_top=float(np.nanmean(recs)), rec_random_mean=float(np.mean(nulls)), rec_random_sd=float(np.std(nulls)),
                        frac_of_window=N / (len(window) * d_mlp)))
        print(f"  top-{N} neurons: recovery {ver[-1]['rec_top']:+.3f} vs random {ver[-1]['rec_random_mean']:+.3f} +- {ver[-1]['rec_random_sd']:.3f}")
    verdf = pd.DataFrame(ver); verdf.to_csv(out_dir / "neuron_verification.csv", index=False)

    # ---- (c) readers vs writers for the top 50
    dirs = {}
    X = defaultdict(list)
    for (enc_u, enc_c, al, lu, lc) in cache:
        _, ru = run_capture(model, enc_u, window, "resid"); _, rc = run_capture(model, enc_c, window, "resid")
        for l in window:
            X[l].append((ru[l][0, -1].float().cpu(), rc[l][0, -1].float().cpu()))
    for l in window:
        d = torch.stack([a for a, _ in X[l]]).mean(0) - torch.stack([b for _, b in X[l]]).mean(0); dirs[l] = d / (d.norm() + 1e-8)
    rw = []
    for _, r in neu.head(50).iterrows():
        l, j = int(r.layer), int(r.neuron)
        w = model.model.layers[l].mlp.down_proj.weight.detach().float()[:, j].cpu()
        cos_w = float((w @ dirs[l]) / (w.norm() * dirs[l].norm() + 1e-8))
        prev = l - 1 if l - 1 in dirs else l
        acts, projs = [], []
        for i, (enc_u, enc_c, al, lu, lc) in enumerate(cache[:40]):
            _, a = run_capture(model, enc_u, [l], "neurons"); acts.append(float(a[l][0, -1, j]))
            projs.append(float(X[prev][i][0] @ dirs[prev]))
        corr = float(np.corrcoef(acts, projs)[0, 1]) if np.std(acts) > 0 and np.std(projs) > 0 else float("nan")
        rw.append(dict(neuron_id=r.neuron_id, score_all=float(r.score_all), cos_out_with_direction=cos_w, corr_act_with_prev_direction=corr))
    rwdf = pd.DataFrame(rw); rwdf.to_csv(out_dir / "neuron_roles.csv", index=False)

    lines = [f"MLP / neurons -- {args.category}; {len(pairs)} working pairs", "",
             "MLP-output patching, mean recovery of hedge log-odds (control -> uncertain):"]
    for gname, g in mlp.groupby("positions"):
        rec = g.groupby("layer").rec.mean(); lines.append(f"  {gname:<5}: " + " ".join(f"L{l}:{v:+.2f}" for l, v in rec.items()))
    lines += ["", "Top neurons by |attribution| (all positions):"]
    for _, r in rwdf.head(15).iterrows():
        lines.append(f"  {r.neuron_id:<11} attr {r.score_all:+.4f} | cos(w_out, familiarity dir) {r.cos_out_with_direction:+.2f} | corr(act, prev-layer projection) {r.corr_act_with_prev_direction:+.2f}")
    lines += ["", "Joint patching of top-N neurons vs random N-sets:"]
    for _, r in verdf.iterrows():
        lines.append(f"  N={int(r.n_neurons):>4} ({100 * r.frac_of_window:.2f}% of window): top {r.rec_top:+.3f} | random {r.rec_random_mean:+.3f} +- {r.rec_random_sd:.3f}")
    summary = "\n".join(lines); print(summary)
    (out_dir / "mlp_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_dir / "neuron_verification.csv", build_provenance(model, script="scripts/circuit_mlp.py", category=args.category, window=window, n_pairs=len(pairs), verify_sizes=sizes))


if __name__ == "__main__":
    main()
