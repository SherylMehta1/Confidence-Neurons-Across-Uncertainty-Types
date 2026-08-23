"""
Circuit steps 2 + 3 -- attention heads.

Step 2 (attribution -> verification): attribution patching scores every head at every position for every twin pair
(score = sum over the head's dims of (control activation - uncertain activation) x d[hedge log-odds]/d[activation],
evaluated on the uncertain run). Heads are ranked by mean |score| summed over positions; the top --verify heads are
then verified by REAL activation patching of that head's output (all aligned positions) from the control twin into
the uncertain prompt, reporting the mean recovery of the hedge log-odds (and the reverse direction).

Step 3 (what the heads do): for every verified head, the mean attention weight from the last position to the entity
span (uncertain vs control prompts), and the projection of the head's output at the last position onto the
per-layer familiarity direction (diff of means of the residual stream at the last position, uncertain - control,
computed here on the working split).

Outputs under <out-dir>: head_attribution.csv (layer, head, position group, mean score, mean |score|),
head_verification.csv (ranked heads with recovery), head_analysis.csv, heads_summary.txt, provenance.
Usage: python scripts/circuit_heads.py --category familiarity [--limit 80] [--verify 30]
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
from circuit_common import Capture, Readout, attribution, encode_pair, head_dim, load_pairs, position_map, recovery, reverse_map, run_capture, run_patched
from shared.provenance import build_provenance, write_provenance


def group_of(al, pos, side):
    """Position group name for a target/source position."""
    L = al["len_u"] if side == "u" else al["len_c"]
    s, e = al["ent_u"] if side == "u" else al["ent_c"]
    if pos == L - 1:
        return "last"
    if s <= pos < e:
        return "entity"
    if pos < al["prefix"]:
        return "prefix"
    return "suffix"


def all_position_pairs(al):
    return position_map(al, "prefix") + position_map(al, "entity") + position_map(al, "suffix")


@torch.no_grad()
def attention_to_entity(model, enc, layer, head, al, side):
    """Mean attention weight from the last position to the entity span for one head (eager attention path)."""
    out = model(**enc, use_cache=False, output_attentions=True)
    if out.attentions is None or out.attentions[layer] is None:
        return float("nan")
    A = out.attentions[layer][0, head]  # [q, k]
    s, e = al["ent_u"] if side == "u" else al["ent_c"]
    return float(A[-1, s:e].sum().item()) if e > s else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--limit", type=int, default=80, help="pairs used for attribution and verification")
    ap.add_argument("--verify", type=int, default=30, help="top heads to verify by real patching")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-attn-patterns", action="store_true", help="skip step 3 attention patterns (needs eager attention)")
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    guard_output(out_dir / "head_verification.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    layers = parse_layer_range(args.layers) if args.layers else list(range(model.config.num_hidden_layers))
    pairs, how = load_pairs(args.category, args.limit, "working")
    ro = Readout(tokenizer); H = model.config.num_attention_heads
    print(f"[{args.category}] {len(pairs)} working pairs ({how}); {len(layers)} layers x {H} heads")

    # ---- step 2a: attribution
    scores = defaultdict(list)  # (layer, head, group) -> list of per-pair scores
    cache = []
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pmap = all_position_pairs(al)
        _, src = run_capture(model, enc_c, layers, "heads")
        attr = attribution(model, enc_u, layers, "heads", ro, src, pmap)
        per_pair = defaultdict(float)  # (layer, head, group) -> score summed over the group's positions, this pair
        for l, m in attr.items():  # m: [seq_u, n_heads]
            for t, s in pmap:
                g = group_of(al, t, "u")
                for h in range(H):
                    per_pair[(l, h, g)] += float(m[t, h])
        for key, v in per_pair.items():
            scores[key].append(v)
        cache.append((enc_u, enc_c, al, pmap))
        if (k + 1) % 10 == 0:
            print(f"  attribution {k + 1}/{len(pairs)}")
    rows = [dict(layer=l, head=h, group=g, mean_score=float(np.mean(v)), mean_abs=float(np.mean(np.abs(v))), n=len(v)) for (l, h, g), v in scores.items()]
    att = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    att.to_csv(out_dir / "head_attribution.csv", index=False)
    tot = att.groupby(["layer", "head"]).agg(score=("mean_score", "sum"), abs_score=("mean_abs", "sum")).reset_index().sort_values("abs_score", ascending=False)
    top = tot.head(args.verify)

    # ---- step 2b: verification by real patching (head output at all aligned positions)
    ver = []
    for _, r in top.iterrows():
        l, h = int(r["layer"]), int(r["head"])
        recs_cu, recs_uc = [], []
        for (enc_u, enc_c, al, pmap) in cache:
            lg_u, act_u = run_capture(model, enc_u, [l], "heads")
            lg_c, act_c = run_capture(model, enc_c, [l], "heads")
            lo_u, lo_c = ro.logodds(lg_u).item(), ro.logodds(lg_c).item()
            lo_p = ro.logodds(run_patched(model, enc_u, [(l, "heads", pmap, act_c[l], h)])).item()
            lo_q = ro.logodds(run_patched(model, enc_c, [(l, "heads", reverse_map(pmap), act_u[l], h)])).item()
            recs_cu.append(recovery(lo_p, lo_u, lo_c)); recs_uc.append(recovery(lo_q, lo_c, lo_u))
        ver.append(dict(layer=l, head=h, attr_score=float(r.score), attr_abs=float(r.abs_score), rec_control_to_uncertain=float(np.nanmean(recs_cu)),
                        rec_uncertain_to_control=float(np.nanmean(recs_uc)), n_pairs=len(cache)))
        print(f"  verified L{l}H{h}: attr {r.score:+.3f} | recovery c->u {ver[-1]['rec_control_to_uncertain']:+.3f}, u->c {ver[-1]['rec_uncertain_to_control']:+.3f}")
    verdf = pd.DataFrame(ver).sort_values("rec_control_to_uncertain", ascending=False)
    verdf.to_csv(out_dir / "head_verification.csv", index=False)

    # ---- step 3: what the heads do
    if not args.no_attn_patterns:  # sdpa returns no attention weights; switch to eager for the pattern readout
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        else:
            model.config._attn_implementation = "eager"
    ana = []
    # per-layer familiarity direction at the last position
    dirs = {}
    X = defaultdict(list)
    for (enc_u, enc_c, al, pmap) in cache:
        _, ru = run_capture(model, enc_u, layers, "resid"); _, rc = run_capture(model, enc_c, layers, "resid")
        for l in layers:
            X[l].append((ru[l][0, -1].float().cpu(), rc[l][0, -1].float().cpu()))
    for l in layers:
        d = torch.stack([a for a, _ in X[l]]).mean(0) - torch.stack([b for _, b in X[l]]).mean(0)
        dirs[l] = d / (d.norm() + 1e-8)
    hd = head_dim(model)
    for _, r in verdf.iterrows():
        l, h = int(r["layer"]), int(r["head"])
        Wo = model.model.layers[l].self_attn.o_proj.weight.detach().float()  # [d_model, H*hd]
        proj_u, proj_c, att_u, att_c = [], [], [], []
        for (enc_u, enc_c, al, pmap) in cache[: min(40, len(cache))]:
            _, au = run_capture(model, enc_u, [l], "heads"); _, ac = run_capture(model, enc_c, [l], "heads")
            hu = (Wo[:, h * hd:(h + 1) * hd] @ au[l][0, -1, h * hd:(h + 1) * hd].float().to(Wo.device)).cpu()
            hc = (Wo[:, h * hd:(h + 1) * hd] @ ac[l][0, -1, h * hd:(h + 1) * hd].float().to(Wo.device)).cpu()
            proj_u.append(float(hu @ dirs[l])); proj_c.append(float(hc @ dirs[l]))
            if not args.no_attn_patterns:
                att_u.append(attention_to_entity(model, enc_u, l, h, al, "u")); att_c.append(attention_to_entity(model, enc_c, l, h, al, "c"))
        ana.append(dict(layer=l, head=h, proj_dir_uncertain=float(np.mean(proj_u)), proj_dir_control=float(np.mean(proj_c)),
                        proj_dir_diff=float(np.mean(proj_u) - np.mean(proj_c)), attn_to_entity_uncertain=float(np.nanmean(att_u)) if att_u else float("nan"),
                        attn_to_entity_control=float(np.nanmean(att_c)) if att_c else float("nan"), rec_control_to_uncertain=float(r.rec_control_to_uncertain)))
    anadf = pd.DataFrame(ana)
    anadf.to_csv(out_dir / "head_analysis.csv", index=False)

    lines = [f"Heads -- {args.category}; {len(pairs)} working pairs; attribution over {len(layers)} layers x {H} heads; top {args.verify} verified", "",
             "Rank by verified recovery (control -> uncertain) of the hedge log-odds:"]
    for _, r in verdf.head(15).iterrows():
        a = anadf[(anadf["layer"] == r["layer"]) & (anadf["head"] == r["head"])].iloc[0]
        lines.append(f"  L{int(r['layer']):>2}H{int(r['head']):<2} rec c->u {r.rec_control_to_uncertain:+.3f}  u->c {r.rec_uncertain_to_control:+.3f} | attr {r.attr_score:+.3f} | "
                     f"attn->entity U {a.attn_to_entity_uncertain:.2f} C {a.attn_to_entity_control:.2f} | proj on familiarity dir U {a.proj_dir_uncertain:+.2f} C {a.proj_dir_control:+.2f}")
    byg = att.groupby("group").mean_abs.sum()
    lines += ["", "attribution mass by position group: " + ", ".join(f"{g}: {v:.3f}" for g, v in byg.items()),
              f"sum of verified recoveries over top {args.verify} heads (c->u): {verdf.rec_control_to_uncertain.clip(lower=0).sum():.2f}"]
    summary = "\n".join(lines); print(summary)
    (out_dir / "heads_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_dir / "head_verification.csv", build_provenance(model, script="scripts/circuit_heads.py", category=args.category, layers=layers, n_pairs=len(pairs), verify=args.verify))


if __name__ == "__main__":
    main()
