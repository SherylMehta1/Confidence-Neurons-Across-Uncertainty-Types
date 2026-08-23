"""
E5: activation patching between uncertain / control TWINS at the last (prefilled) position,
per layer -- no ablation reference to choose, so the signed-shift reference problem of
mean-ablation does not arise.

Pairs: for each uncertain prompt, a control prompt with the same `template` field (lack-of-
knowledge data carries it); greedy one-to-one matching within template, unmatched prompts
dropped (count reported). For paired categories (contradictory_context) pairs are by index.

For each pair and each layer L, the residual stream AFTER block L at the last position is
copied from the SOURCE run into the TARGET run (both directions: control->uncertain and
uncertain->control). Readout: hedge-vs-answer log-odds at the prefilled position
(logsumexp over HEDGE_TOKENS minus logsumexp over ANSWER_TOKENS) and entropy.
Recovery(L) = (patched - target) / (source - target), averaged over pairs (clipped to [-1, 2]).

Outputs: results/twin_patching.csv (per pair x layer x direction), results/twin_patching_summary.txt
Usage: python scripts/twin_patching.py --category lack_of_knowledge [--layers 0-31]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_category, load_model_from_args, parse_layer_range
from direction_bridge import ANSWER_TOKENS, HEDGE_TOKENS, token_ids
from shared.model_utils import tokenize_prompt
from shared.provenance import build_provenance, write_provenance


def make_pairs(prompts, ctrls):
    if all(("twin_id" in r) for r in prompts + ctrls):  # gated twin sets carry explicit pair ids
        by_id = {c["twin_id"]: c for c in ctrls}
        pairs = [(p, by_id[p["twin_id"]]) for p in prompts if p["twin_id"] in by_id]
        return pairs, "twin_id"
    if all(("template" in r) for r in prompts + ctrls):
        by_t = defaultdict(list)
        for c in ctrls:
            by_t[c["template"]].append(c)
        pairs = []
        for p in prompts:
            pool = by_t.get(p["template"])
            if pool:
                pairs.append((p, pool.pop(0)))
        return pairs, "template"
    n = min(len(prompts), len(ctrls))
    return list(zip(prompts[:n], ctrls[:n])), "index"


@torch.no_grad()
def readout(logits, hedge_ids, answer_ids):
    lp = torch.log_softmax(logits.float(), -1)
    p = lp.exp()
    return (torch.logsumexp(lp[hedge_ids], 0) - torch.logsumexp(lp[answer_ids], 0)).item(), -(torch.xlogy(p, p)).sum().item()


@torch.no_grad()
def run_with_cache(model, enc, layers):
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.logits[0, -1], {l: out.hidden_states[l + 1][0, -1].detach().clone() for l in layers}


@torch.no_grad()
def run_patched(model, enc, layer, vec):
    block = model.model.layers[layer]

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = vec.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = block.register_forward_hook(hook)
    try:
        return model(**enc, use_cache=False).logits[0, -1]
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="lack_of_knowledge")
    ap.add_argument("--layers", default=None, help="e.g. 0-31 (default: all)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/twin_patching.csv")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    guard_output(out, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    layers = parse_layer_range(args.layers) if args.layers else list(range(model.config.num_hidden_layers))
    prompts, ctrls = load_category(args.category)
    pairs, how = make_pairs(prompts, ctrls)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[{args.category}] {len(pairs)} twin pairs (matched by {how}; {len(prompts) - len(pairs)} uncertain prompts unmatched)")
    hedge_ids, answer_ids = token_ids(tokenizer, HEDGE_TOKENS), token_ids(tokenizer, ANSWER_TOKENS)

    rows = []
    for k, (u, c) in enumerate(pairs):
        enc_u = tokenize_prompt(tokenizer, u["chat_formatted_prompt"], device=model.device)
        enc_c = tokenize_prompt(tokenizer, c["chat_formatted_prompt"], device=model.device)
        lg_u, cache_u = run_with_cache(model, enc_u, layers)
        lg_c, cache_c = run_with_cache(model, enc_c, layers)
        lo_u, H_u = readout(lg_u, hedge_ids, answer_ids)
        lo_c, H_c = readout(lg_c, hedge_ids, answer_ids)
        for l in layers:
            lo_p, H_p = readout(run_patched(model, enc_u, l, cache_c[l]), hedge_ids, answer_ids)   # control -> uncertain
            lo_q, H_q = readout(run_patched(model, enc_c, l, cache_u[l]), hedge_ids, answer_ids)   # uncertain -> control
            rows.append(dict(pair=k, uncertain_id=u["prompt_id"], control_id=c["prompt_id"], split=u["split"], layer=l,
                             direction="control_to_uncertain", logodds_target=lo_u, logodds_source=lo_c, logodds_patched=lo_p,
                             entropy_target=H_u, entropy_source=H_c, entropy_patched=H_p))
            rows.append(dict(pair=k, uncertain_id=u["prompt_id"], control_id=c["prompt_id"], split=u["split"], layer=l,
                             direction="uncertain_to_control", logodds_target=lo_c, logodds_source=lo_u, logodds_patched=lo_q,
                             entropy_target=H_c, entropy_source=H_u, entropy_patched=H_q))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs")
    df = pd.DataFrame(rows)
    for m in ("logodds", "entropy"):
        den = (df[f"{m}_source"] - df[f"{m}_target"])
        df[f"{m}_recovery"] = ((df[f"{m}_patched"] - df[f"{m}_target"]) / den.where(den.abs() > 1e-6)).clip(-1, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    lines = [f"Twin activation patching -- {args.category}; {len(pairs)} pairs matched by {how}; last position; layers {layers[0]}-{layers[-1]}",
             f"clean hedge log-odds: uncertain {df[df.direction=='control_to_uncertain'].logodds_target.mean():+.2f}, control {df[df.direction=='uncertain_to_control'].logodds_target.mean():+.2f}", ""]
    for d, g in df.groupby("direction"):
        rec = g.groupby("layer")[["logodds_recovery", "entropy_recovery"]].mean()
        best = rec.logodds_recovery.idxmax()
        lines.append(f"{d}: peak log-odds recovery {rec.logodds_recovery.max():.2f} at layer {best}; "
                     + "recovery by layer: " + " ".join(f"L{l}:{v:.2f}" for l, v in rec.logodds_recovery.items()))
    summary = "\n".join(lines)
    print(summary)
    out.with_name(out.stem + "_summary.txt").write_text(summary + "\n")
    write_provenance(out, build_provenance(model, script="scripts/twin_patching.py", category=args.category, layers=layers,
                                           n_pairs=len(pairs), pairing=how, hedge_tokens=HEDGE_TOKENS, answer_tokens=ANSWER_TOKENS))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
