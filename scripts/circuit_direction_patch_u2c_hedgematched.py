"""Stretch null: random unit directions whose projections onto the hedge-token unembedding rows
match the real per-layer uncertainty direction's projections. Runs uncertain_to_control (injection) only.
Copy of circuit_direction_patch.py's setup; writes direction_patch_u2c_hedgematched.csv."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args, parse_layer_range
from circuit_common import Readout, encode_pair, load_pairs, recovery, reverse_map, run_capture
from circuit_direction_patch import all_position_pairs, rank1_forward
from direction_bridge import HEDGE_TOKENS, token_ids
from shared.provenance import build_provenance, write_provenance


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--category", default="conflict")
    ap.add_argument("--layers", default="11-21")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    out_csv = out_dir / "direction_patch_u2c_hedgematched.csv"
    guard_output(out_csv, args.overwrite)
    model, tokenizer = load_model_from_args(args)
    ro = Readout(tokenizer)
    layers = parse_layer_range(args.layers)
    gen = torch.Generator().manual_seed(args.seed)
    d_model = model.config.hidden_size
    work, _ = load_pairs(args.category, None, "working")
    held, how = load_pairs(args.category, args.limit, "held_out")
    print(f"[{args.category}] directions from {len(work)} working pairs; evaluated on {len(held)} held-out pairs; layers {layers}")
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
    # hedge-token unembedding rows -> orthonormal basis Q of their span
    hedge_ids = token_ids(tokenizer, HEDGE_TOKENS)
    W = model.lm_head.weight[hedge_ids].detach().float().cpu()  # (5, d)
    Q, _ = torch.linalg.qr(W.T)  # (d, 5)
    print("hedge ids", hedge_ids, "W shape", tuple(W.shape), "Q shape", tuple(Q.shape))
    rand_dirs, diag = [], []
    for r_i in range(args.n_random):
        rd = {}
        for l in layers:
            d_real = dirs[l]
            r = torch.nn.functional.normalize(torch.randn(d_model, generator=gen), dim=0)
            in_span_real = Q @ (Q.T @ d_real)
            r_perp = r - Q @ (Q.T @ r)
            a = torch.sqrt(torch.clamp(1.0 - in_span_real.norm() ** 2, min=0.0)) / (r_perp.norm() + 1e-8)
            rr = a * r_perp + in_span_real
            rd[l] = rr / (rr.norm() + 1e-8)
            diag.append(dict(rand=r_i, layer=l, norm=rd[l].norm().item(), cos_real=(rd[l] @ d_real).item(),
                             max_abs_proj_err=(W @ rd[l] - W @ d_real).abs().max().item(),
                             span_frac_real=(in_span_real.norm() ** 2).item()))
        rand_dirs.append(rd)
    dd = pd.DataFrame(diag)
    print("projection-match diagnostics: max |W r - W d| =", dd.max_abs_proj_err.max(), "; cos(r, d_real) mean", dd.cos_real.mean(), "; ||d_real|| fraction in hedge span mean", dd.span_frac_real.mean())
    rows = []
    for k, (u, c) in enumerate(held):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pm = all_position_pairs(al)
        lg_u = run_capture(model, enc_u, [], "resid")[0]; lg_c = run_capture(model, enc_c, [], "resid")[0]
        lo_u, H_u = ro.logodds(lg_u).item(), ro.entropy(lg_u).item()
        lo_c, H_c = ro.logodds(lg_c).item(), ro.entropy(lg_c).item()
        lo, Hh = rank1_forward(model, ro, enc_c, enc_u, layers, dirs, reverse_map(pm))
        rows.append(dict(pair=k, set="direction", direction="uncertain_to_control", logodds_rec=recovery(lo, lo_c, lo_u), entropy_rec=recovery(Hh, H_c, H_u)))
        for r_i, rd in enumerate(rand_dirs):
            lo, Hh = rank1_forward(model, ro, enc_c, enc_u, layers, rd, reverse_map(pm))
            rows.append(dict(pair=k, set=f"hedgematched{r_i}", direction="uncertain_to_control", logodds_rec=recovery(lo, lo_c, lo_u), entropy_rec=recovery(Hh, H_c, H_u)))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(held)} pairs")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    dd.to_csv(out_dir / "direction_patch_u2c_hedgematched_diag.csv", index=False)
    uc = df[df.set == "direction"]
    rnd = df[df.set.str.startswith("hedgematched")].groupby("set")[["logodds_rec", "entropy_rec"]].mean()
    print(f"direction u->c: log-odds {uc.logodds_rec.mean():+.3f} entropy {uc.entropy_rec.mean():+.3f}")
    print(f"hedge-matched random u->c: log-odds {rnd.logodds_rec.mean():+.3f} +- {rnd.logodds_rec.std():.3f} (max {rnd.logodds_rec.max():+.3f}); entropy {rnd.entropy_rec.mean():+.3f}")
    print(rnd)
    write_provenance(out_csv, build_provenance(model, script="scripts/circuit_direction_patch_u2c_hedgematched.py", category=args.category, layers=layers, n_pairs=len(held), n_random=args.n_random))


if __name__ == "__main__":
    main()
