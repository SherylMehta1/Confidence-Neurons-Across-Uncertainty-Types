"""
E2: neuron <-> direction bridge.

Two residual-stream directions at the last (prefilled) position, per layer:
  familiarity : mean(uncertain prompts) - mean(matched control prompts)      [Ferrando-style, diff of means]
  hedging     : mean(prompts whose clean generation hedged) - mean(did not)  [Ji-style, from behavioral_bf16.csv],
                computed within the UNCERTAIN arm only so it is not a proxy for familiarity.

Then, for every MLP neuron in --layer-range, cosine(gamma * w_out, direction) -- i.e. how much
each neuron WRITES into each direction -- with percentile ranks against all scanned neurons,
flags for the named key neurons, and the cosine between the two directions per layer.

Optional causal sanity check (--steer): add alpha * direction (alpha in sigma units of the
projection) at the last position of the layer where the direction is taken and measure the
hedge-vs-answer log-odds and entropy on held-out uncertain and control prompts.

Outputs: results/direction_bridge_neurons.csv (per neuron), results/direction_bridge_summary.txt,
         results/direction_bridge_directions.npz (the directions, per layer)
Usage: python scripts/direction_bridge.py --category lack_of_knowledge --layer-range 20-31 --key-neurons L31_N11541,L31_N6772,L31_N2477,L30_N1457
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_category, load_model_from_args, parse_layer_range, parse_neurons
from shared.model_utils import tokenize_prompt
from shared.provenance import build_provenance, write_provenance

HEDGE_TOKENS = [" not", " I", " unknown", " unclear", " no"]
ANSWER_TOKENS = [" a", " the", " that", " in", " Yes", " yes"]


@torch.no_grad()
def last_token_residuals(model, tokenizer, prompts, layers, verbose=True):
    """{layer: [n_prompts, d] fp32} residual stream AFTER block `layer` at the last position."""
    out = {l: [] for l in layers}
    for i, p in enumerate(prompts):
        enc = tokenize_prompt(tokenizer, p, device=model.device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states  # len = n_layers + 1
        for l in layers:
            out[l].append(hs[l + 1][0, -1].float().cpu())
        if verbose and (i + 1) % 50 == 0:
            print(f"  residuals {i + 1}/{len(prompts)}")
    return {l: torch.stack(v) for l, v in out.items()}


def diff_of_means(X, mask_a, mask_b):
    d = X[mask_a].mean(0) - X[mask_b].mean(0)
    return d / (d.norm() + 1e-12)


@torch.no_grad()
def neuron_cosines(model, layers, direction, device):
    """cosine(gamma * w_out, direction) for every neuron in `layers`."""
    gamma = model.model.norm.weight.detach().to(device, torch.float32)
    d = direction.to(device) / direction.norm()
    rows = []
    for l in layers:
        W = model.model.layers[l].mlp.down_proj.weight
        if hasattr(W, "quant_state"):
            import bitsandbytes.functional as bnb_F
            W = bnb_F.dequantize_4bit(W, W.quant_state)
        Wt = gamma[:, None] * W.detach().to(device, torch.float32)
        cos = (d @ Wt) / (Wt.norm(dim=0) + 1e-12)
        rows.append(pd.DataFrame({"neuron_id": [f"L{l}_N{i}" for i in range(Wt.shape[1])], "layer": l,
                                  "neuron_idx": range(Wt.shape[1]), "cos": cos.cpu().numpy()}))
    return pd.concat(rows, ignore_index=True)


def token_ids(tokenizer, toks):
    ids = []
    for t in toks:
        enc = tokenizer(t, add_special_tokens=False)["input_ids"]
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


@torch.no_grad()
def steer_readout(model, tokenizer, prompts, layer, direction, alphas, hedge_ids, answer_ids, sigma):
    """Add alpha*sigma*direction at the last position of block `layer`'s output; return per-prompt
    hedge-vs-answer log-odds and entropy for each alpha."""
    block = model.model.layers[layer]
    state = {"v": None}
    dvec = direction.to(model.device, torch.float32)

    def hook(module, args, output):
        if state["v"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = h[:, -1, :] + (state["v"] * dvec).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = block.register_forward_hook(hook)
    rows = []
    try:
        for p in prompts:
            enc = tokenize_prompt(tokenizer, p["chat_formatted_prompt"], device=model.device)
            for a in alphas:
                state["v"] = a * sigma
                logits = model(**enc, use_cache=False).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)
                probs = lp.exp()
                rows.append(dict(prompt_id=p["prompt_id"], is_control=p["is_control"], alpha=a,
                                 hedge_logodds=(torch.logsumexp(lp[hedge_ids], 0) - torch.logsumexp(lp[answer_ids], 0)).item(),
                                 entropy=-(torch.xlogy(probs, probs)).sum().item(), top1=tokenizer.decode(int(logits.argmax()))))
    finally:
        handle.remove()
        state["v"] = None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="lack_of_knowledge")
    ap.add_argument("--layer-range", default="20-31")
    ap.add_argument("--direction-layer", type=int, default=None, help="layer whose directions are used for neuron cosines (default: last scanned)")
    ap.add_argument("--behavioral", default="results/behavioral_bf16.csv")
    ap.add_argument("--key-neurons", default="L31_N11541,L31_N6772,L31_N2477,L30_N1457")
    ap.add_argument("--steer", action="store_true", help="also run the steering sanity check on held-out prompts")
    ap.add_argument("--alphas", default="-3,-1.5,0,1.5,3")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_csv = out_dir / "direction_bridge_neurons.csv"
    guard_output(out_csv, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    layers = parse_layer_range(args.layer_range)
    dl = args.direction_layer if args.direction_layer is not None else layers[-1]
    prompts, ctrls = load_category(args.category)
    recs = [dict(r, is_control=False) for r in prompts] + [dict(r, is_control=True) for r in ctrls]
    work = [r for r in recs if r["split"] == "working"]
    held = [r for r in recs if r["split"] == "held_out"]
    print(f"[{args.category}] {len(work)} working prompts for directions; {len(held)} held-out for steering")

    X = last_token_residuals(model, tokenizer, [r["chat_formatted_prompt"] for r in work], layers)
    is_ctrl = np.array([r["is_control"] for r in work])
    if all("hedge_rate" in r for r in work):  # gated twin sets carry a per-prompt hedging rate from the gate's free generations
        hedged = np.array([float(r["hedge_rate"]) >= 0.5 for r in work])
        print("hedged labels from the gate's hedge_rate (>= 0.5)")
    else:
        beh = pd.read_csv(REPO_ROOT / args.behavioral)
        clean = beh[beh.condition == "clean"].set_index("prompt_id")
        hedged = np.array([bool(clean.hedged.get(r["prompt_id"], False)) for r in work])
    unc = ~is_ctrl
    if hedged[unc].sum() < 5 or (~hedged[unc]).sum() < 5:
        print("WARNING: too few hedged/non-hedged uncertain prompts for a hedging direction")

    fam, hed, cos_fh = {}, {}, {}
    for l in layers:
        fam[l] = diff_of_means(X[l], unc, is_ctrl)
        hed[l] = diff_of_means(X[l], unc & hedged, unc & ~hedged)
        cos_fh[l] = float(fam[l] @ hed[l])
    np.savez(out_dir / "direction_bridge_directions.npz", **{f"familiarity_L{l}": fam[l].numpy() for l in layers},
             **{f"hedging_L{l}": hed[l].numpy() for l in layers})

    device = model.get_output_embeddings().weight.device
    cf = neuron_cosines(model, layers, fam[dl], device).rename(columns={"cos": "cos_familiarity"})
    ch = neuron_cosines(model, layers, hed[dl], device).rename(columns={"cos": "cos_hedging"})
    df = cf.merge(ch, on=["neuron_id", "layer", "neuron_idx"])
    for c in ("cos_familiarity", "cos_hedging"):
        df[f"{c}_pctile"] = df[c].abs().rank(pct=True) * 100
    key = [f"L{l}_N{n}" for l, n in parse_neurons(args.key_neurons)]
    df["is_key"] = df.neuron_id.isin(key)
    df.to_csv(out_csv, index=False)

    lines = [f"Direction bridge -- {args.category}; directions at layer {dl}; neurons scanned in layers {layers}",
             "cos(familiarity, hedging) per layer: " + ", ".join(f"L{l}:{cos_fh[l]:+.2f}" for l in layers), "",
             "Key neurons (cosine of gamma*w_out with each direction; percentile of |cos| among all scanned neurons):"]
    for r in df[df.is_key].itertuples():
        lines.append(f"  {r.neuron_id:<11} familiarity {r.cos_familiarity:+.3f} ({r.cos_familiarity_pctile:5.1f} pct)   hedging {r.cos_hedging:+.3f} ({r.cos_hedging_pctile:5.1f} pct)")
    for c in ("cos_familiarity", "cos_hedging"):
        top = df.reindex(df[c].abs().sort_values(ascending=False).index).head(10)
        lines.append(f"\nTop-10 |{c}| neurons: " + ", ".join(f"{r.neuron_id}({getattr(r, c):+.2f})" for r in top.itertuples()))

    if args.steer:
        hedge_ids, answer_ids = token_ids(tokenizer, HEDGE_TOKENS), token_ids(tokenizer, ANSWER_TOKENS)
        alphas = [float(a) for a in args.alphas.split(",")]
        for name, d in (("familiarity", fam[dl]), ("hedging", hed[dl])):
            proj = (X[dl] @ d)
            sigma = float(proj.std())
            s = steer_readout(model, tokenizer, held, dl, d, alphas, hedge_ids, answer_ids, sigma)
            s["direction"] = name
            s.to_csv(out_dir / f"direction_bridge_steer_{name}.csv", index=False)
            lines.append(f"\nSteering along {name} at L{dl} (alpha in sigma units of the projection, held-out prompts):")
            for arm, g in s.groupby("is_control"):
                lines.append(f"  {'control' if arm else 'uncertain':<9} " + "  ".join(
                    f"a={a:+g}: logodds {gg.hedge_logodds.mean():+.2f} H {gg.entropy.mean():.2f}" for a, gg in g.groupby("alpha")))
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "direction_bridge_summary.txt").write_text(summary + "\n")
    write_provenance(out_csv, build_provenance(model, script="scripts/direction_bridge.py", category=args.category,
                                               layers=layers, direction_layer=dl, key_neurons=key, behavioral=args.behavioral,
                                               steer=args.steer, hedge_tokens=HEDGE_TOKENS, answer_tokens=ANSWER_TOKENS))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
