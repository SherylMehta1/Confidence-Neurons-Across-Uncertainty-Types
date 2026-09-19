"""
Leave-one-token-out, GPU half: cache the last-token log-probs the hedge/answer readout reads.

Readout.logodds() reduces the full-vocab last-token logits to one scalar with a fixed 5-token hedge
set and 6-token answer set (direction_bridge.HEDGE_TOKENS / ANSWER_TOKENS), then recovery() clips
and circuit_faithfulness.py / circuit_direction_patch.py discard everything upstream. To ask whether
the headline recovery numbers depend on any single hedge token (" I" in particular) without paying
for a forward pass per variant, this script reruns the same patching once and stores, for every
forward pass those scripts make, the log-probs at exactly the hedge-union-answer token ids. Any
subset log-odds is then a two-logsumexp CPU recomputation (loto_recompute.py).

Cells cached, on the same held-out pairs, sets and seeds as the committed CSVs:
  circuit   -- the top --n-heads + --n-neurons set, both directions, plus --n-random-sets random
               size-matched sets (control_to_uncertain), exactly as circuit_faithfulness.py
  direction -- the per-layer rank-1 directions from the working pairs over --layers, both directions,
               plus --n-random-dirs random unit directions (control_to_uncertain), exactly as
               circuit_direction_patch.py
With the same seeds the full-set variant must reproduce faithfulness.csv and direction_patch.csv per
pair; loto_recompute.py checks that.

Writes <out-dir>/loto_cache.csv (one row per forward pass: cell, pair, set, direction, condition
in {baseline_u, baseline_c, patched}, lp_<token id> ...) and loto_cache_tokens.json (token -> id).
Refuses to overwrite without --overwrite.

  python scripts/circuit_loto_cache.py --category familiarity_v2 --out-dir results/circuit_familiarity_v2
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json
import random
from collections import defaultdict

import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args, parse_layer_range
from circuit_common import Readout, encode_pair, load_pairs, reverse_map, run_capture, run_patched
from circuit_direction_patch import Rank1Patch
from circuit_faithfulness import all_position_pairs, component_set
from direction_bridge import ANSWER_TOKENS, HEDGE_TOKENS
from shared.provenance import build_provenance, write_provenance


def token_map(tokenizer, toks):
    """token string -> first token id, keeping the strings (token_ids() drops them)."""
    out = {}
    for t in toks:
        enc = tokenizer(t, add_special_tokens=False)["input_ids"]
        if enc:
            out[t] = int(enc[0])
    return out


def patch_all_logits(model, enc_t, enc_s, pmap, heads, neurons):
    layers_h = sorted({l for l, _ in heads}); layers_n = sorted({l for l, _ in neurons})
    _, src_h = run_capture(model, enc_s, layers_h, "heads") if layers_h else (None, {})
    _, src_n = run_capture(model, enc_s, layers_n, "neurons") if layers_n else (None, {})
    edits = [(l, "heads", pmap, src_h[l], h) for l, h in heads] + [(l, "neurons", pmap, src_n[l], j) for l, j in neurons]
    return run_patched(model, enc_t, edits)


@torch.no_grad()
def rank1_logits(model, enc_t, enc_s, layers, dirs, pmap):
    _, src = run_capture(model, enc_s, layers, "resid")
    with Rank1Patch(model, layers, dirs, pmap, src):
        return model(**enc_t, use_cache=False).logits[0, -1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity_v2")
    ap.add_argument("--out-dir", default="results/circuit_familiarity_v2")
    ap.add_argument("--limit", type=int, default=60, help="held-out pairs (circuit_faithfulness/direction_patch default)")
    ap.add_argument("--n-heads", type=int, default=20)
    ap.add_argument("--n-neurons", type=int, default=100)
    ap.add_argument("--n-random-sets", type=int, default=30, help="random component sets; 30 matches the committed faithfulness.csv")
    ap.add_argument("--layers", default="13-19", help="rank-1 direction layers")
    ap.add_argument("--n-random-dirs", type=int, default=10, help="random unit directions; 10 matches direction_patch.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_csv = out_dir / "loto_cache.csv"
    guard_output(out_csv, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    ro = Readout(tokenizer)
    hmap, amap = token_map(tokenizer, HEDGE_TOKENS), token_map(tokenizer, ANSWER_TOKENS)
    ids = sorted(set(hmap.values()) | set(amap.values()))
    assert set(ro.hedge_ids) | set(ro.answer_ids) == set(ids), "cached ids must be exactly what Readout reads"
    cols = [f"lp_{i}" for i in ids]

    def lp_row(lg):
        lp = torch.log_softmax(lg.float(), -1)
        return {c: float(lp[i]) for c, i in zip(cols, ids)}

    pairs, _ = load_pairs(args.category, args.limit, "held_out")
    if not pairs:
        pairs, _ = load_pairs(args.category, args.limit, None); print("WARNING: no held-out split; using all pairs")
    L, H, D = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.intermediate_size
    rows = []

    # ---- circuit cell: same set, pairs, rng draw order and seed as circuit_faithfulness.py ----
    heads, neurons = component_set(out_dir, args.n_heads, args.n_neurons)
    rng = random.Random(args.seed)
    print(f"[circuit] {len(heads)} heads + {len(neurons)} neurons; {len(pairs)} held-out pairs; {args.n_random_sets} random sets")
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pm = all_position_pairs(al)
        lg_u, lg_c = run_capture(model, enc_u, [], "resid")[0], run_capture(model, enc_c, [], "resid")[0]
        rows.append(dict(cell="circuit", pair=k, set="baseline", direction="", condition="baseline_u", **lp_row(lg_u)))
        rows.append(dict(cell="circuit", pair=k, set="baseline", direction="", condition="baseline_c", **lp_row(lg_c)))
        rows.append(dict(cell="circuit", pair=k, set="circuit", direction="control_to_uncertain", condition="patched",
                         **lp_row(patch_all_logits(model, enc_u, enc_c, pm, heads, neurons))))
        rows.append(dict(cell="circuit", pair=k, set="circuit", direction="uncertain_to_control", condition="patched",
                         **lp_row(patch_all_logits(model, enc_c, enc_u, reverse_map(pm), heads, neurons))))
        for r_i in range(args.n_random_sets):
            rh = list({(rng.randrange(L), rng.randrange(H)) for _ in heads}); rn = list({(rng.randrange(L), rng.randrange(D)) for _ in neurons})
            rows.append(dict(cell="circuit", pair=k, set=f"random{r_i}", direction="control_to_uncertain", condition="patched",
                             **lp_row(patch_all_logits(model, enc_u, enc_c, pm, rh, rn))))
        if (k + 1) % 10 == 0:
            print(f"  [circuit] {k + 1}/{len(pairs)} pairs")

    # ---- direction cell: same directions, pairs, generator seed as circuit_direction_patch.py ----
    layers = parse_layer_range(args.layers)
    gen = torch.Generator().manual_seed(args.seed)
    work, _ = load_pairs(args.category, None, "working")
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
    rand_dirs = [{l: torch.nn.functional.normalize(torch.randn(model.config.hidden_size, generator=gen), dim=0) for l in layers}
                 for _ in range(args.n_random_dirs)]
    print(f"[direction] layers {layers}; directions from {len(work)} working pairs; {args.n_random_dirs} random unit directions")
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pm = all_position_pairs(al)
        lg_u, lg_c = run_capture(model, enc_u, [], "resid")[0], run_capture(model, enc_c, [], "resid")[0]
        rows.append(dict(cell="direction", pair=k, set="baseline", direction="", condition="baseline_u", **lp_row(lg_u)))
        rows.append(dict(cell="direction", pair=k, set="baseline", direction="", condition="baseline_c", **lp_row(lg_c)))
        rows.append(dict(cell="direction", pair=k, set="direction", direction="control_to_uncertain", condition="patched",
                         **lp_row(rank1_logits(model, enc_u, enc_c, layers, dirs, pm))))
        rows.append(dict(cell="direction", pair=k, set="direction", direction="uncertain_to_control", condition="patched",
                         **lp_row(rank1_logits(model, enc_c, enc_u, layers, dirs, reverse_map(pm)))))
        for r_i, rd in enumerate(rand_dirs):
            rows.append(dict(cell="direction", pair=k, set=f"random{r_i}", direction="control_to_uncertain", condition="patched",
                             **lp_row(rank1_logits(model, enc_u, enc_c, layers, rd, pm))))
        if (k + 1) % 10 == 0:
            print(f"  [direction] {k + 1}/{len(pairs)} pairs")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    (out_dir / "loto_cache_tokens.json").write_text(json.dumps(dict(hedge=hmap, answer=amap, ids=ids), indent=2))
    write_provenance(out_csv, build_provenance(model, script="scripts/circuit_loto_cache.py", category=args.category,
                                               n_pairs=len(pairs), n_heads=len(heads), n_neurons=len(neurons),
                                               n_random_sets=args.n_random_sets, layers=layers, n_random_dirs=args.n_random_dirs,
                                               hedge_tokens=hmap, answer_tokens=amap))
    print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
