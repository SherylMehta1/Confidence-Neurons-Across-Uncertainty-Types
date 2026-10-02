"""
L5 / step 9: behavioral validation of a discovered circuit.

Every other circuit result in this project is a LOGIT claim: patch the circuit's heads and
neurons from a control twin into an uncertain twin's single forward pass and read the
hedge-vs-answer log-odds at the next token (circuit_faithfulness.py). This script asks the
separate question a reader asks immediately: if you patch the circuit and let the model KEEP
TALKING, does what it actually says change -- hedge instead of confabulate, or vice versa --
or does the probability shift wash out over the next couple of dozen tokens?

WHAT IS PATCHED, AND WHY ONLY THERE
Patching is defined by aligning positions between two fixed prompts (align()/position_map()).
Once generation starts, the two twins' continuations diverge token by token and there is no
principled source position to patch a generated position against. So the patch is applied to
the PROMPT positions during the single prefill forward pass -- the same forward pass
circuit_faithfulness.py scores -- and then the hooks come off and decoding continues greedily
and unpatched for the remaining --max-new-tokens. This is a one-shot "decision point" test:
does flipping the circuit's state where the decision is made change the token actually emitted
and the continuation that follows from it. Note the patch still propagates past the prefill
through the KV cache, because patching a neuron's down_proj input or a head's o_proj input at
a prompt position changes that position's residual stream and hence the keys/values that
layers above it write for that position.

This is deliberately a weaker intervention than behavioral_test.py's clamping, which applies at
every generated position -- but clamping to a fixed scalar is well defined at any position,
whereas transplanting a specific twin's activation is not.

READOUT
HEDGE_RE (imported from behavioral_test.py, unchanged) on the generated continuation, per pair,
per direction, for: the clean continuation, the circuit-patched continuation, and each random
size-matched component set's patched continuation.

DIRECTIONS (same naming and semantics as circuit_faithfulness.py)
  control_to_uncertain : control's components into the UNCERTAIN prompt -> hedging should go DOWN
  uncertain_to_control : uncertain's components into the CONTROL prompt -> hedging should go UP
The two expected signs are opposite; the summary scores each direction against its own
expected sign rather than pooling them.

Usage:
  python scripts/circuit_behavioral.py --category familiarity_v3 \
      --circuit-dir results/circuit_familiarity_v3 --n-heads 20 --n-neurons 100 \
      --limit 141 --max-new-tokens 24 --n-random 10
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import random

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from behavioral_test import HEDGE_RE
from circuit_common import Patch, encode_pair, load_pairs, position_map, reverse_map, run_capture
from circuit_faithfulness import all_position_pairs, component_set
from shared.provenance import build_provenance, write_provenance

DIRECTIONS = ("control_to_uncertain", "uncertain_to_control")
EXPECTED_SIGN = {"control_to_uncertain": -1, "uncertain_to_control": +1}  # of (patched - clean) hedging


@torch.no_grad()
def greedy_continue(model, tokenizer, enc, max_new_tokens, edits=None):
    """Greedy-decode `max_new_tokens` from `enc`. If `edits` is given, the PREFILL pass only runs
    inside Patch(model, edits); the hooks are removed before any further step. Returns the decoded
    continuation string."""
    ctx = Patch(model, edits) if edits else None
    if ctx is not None:
        ctx.__enter__()
    try:
        out = model(**enc, use_cache=True)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)  # hooks off: every step after prefill is unpatched
    cache = out.past_key_values
    nxt = out.logits[0, -1].argmax().view(1, 1)
    gen = [nxt.item()]
    attn = enc["attention_mask"]
    for _ in range(max_new_tokens - 1):
        if nxt.item() == tokenizer.eos_token_id:
            break
        attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
        step = model(input_ids=nxt, attention_mask=attn, past_key_values=cache, use_cache=True)
        cache = step.past_key_values
        nxt = step.logits[0, -1].argmax().view(1, 1)
        gen.append(nxt.item())
    return tokenizer.decode(gen, skip_special_tokens=True)


def build_edits(model, enc_s, pmap, heads, neurons):
    """The same `edits` list circuit_faithfulness.patch_all() builds: source activations captured
    from the source prompt, mapped onto the target's aligned positions."""
    layers_h = sorted({l for l, _ in heads})
    layers_n = sorted({l for l, _ in neurons})
    src_h = run_capture(model, enc_s, layers_h, "heads")[1] if layers_h else {}
    src_n = run_capture(model, enc_s, layers_n, "neurons")[1] if layers_n else {}
    return ([(l, "heads", pmap, src_h[l], h) for l, h in heads]
            + [(l, "neurons", pmap, src_n[l], j) for l, j in neurons])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity_v3")
    ap.add_argument("--circuit-dir", default=None)
    ap.add_argument("--n-heads", type=int, default=20)
    ap.add_argument("--n-neurons", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--limit", type=int, default=141, help="held-out pairs")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--eval-split", default="held_out", choices=["held_out", "working"],
                    help="split to evaluate on. held_out is the only confirmatory setting; "
                         "'working' is IN-SAMPLE and writes to behavioral_working.* so it can "
                         "never overwrite a held-out result.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    circ = REPO_ROOT / (args.circuit_dir or f"results/circuit_{args.category}")
    tag = "" if args.eval_split == "held_out" else f"_{args.eval_split}"
    out_csv = _Path(args.out) if args.out else circ / f"behavioral{tag}.csv"
    out_csv = guard_output(out_csv, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    rng = random.Random(args.seed)
    heads, neurons = component_set(circ, args.n_heads, args.n_neurons)
    pairs, _ = load_pairs(args.category, args.limit, args.eval_split)
    if not pairs:
        raise SystemExit(f"no pairs in split {args.eval_split} for {args.category}")
    if args.eval_split != "held_out":
        print(f"NOTE: {args.eval_split} split -- IN-SAMPLE, exploratory only; writing {out_csv.name}")
    L, H, D = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.intermediate_size
    print(f"[{args.category}] {len(heads)} heads + {len(neurons)} neurons; {len(pairs)} {args.eval_split} pairs; "
          f"{args.max_new_tokens} new tokens; {args.n_random} random sets per direction")

    rows = []
    for k, (u, c) in enumerate(pairs):
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        pm = all_position_pairs(al)
        # target = the prompt we generate from; source = the twin we transplant from
        setup = {"control_to_uncertain": (enc_u, enc_c, pm),
                 "uncertain_to_control": (enc_c, enc_u, reverse_map(pm))}
        clean = {d: greedy_continue(model, tokenizer, setup[d][0], args.max_new_tokens) for d in DIRECTIONS}
        for d in DIRECTIONS:
            enc_t, enc_s, pmap = setup[d]
            sets = [("circuit", heads, neurons)]
            for r_i in range(args.n_random):
                rh = list({(rng.randrange(L), rng.randrange(H)) for _ in heads})
                rn = list({(rng.randrange(L), rng.randrange(D)) for _ in neurons})
                sets.append((f"random{r_i}", rh, rn))
            for name, hh, nn in sets:
                edits = build_edits(model, enc_s, pmap, hh, nn)
                g = greedy_continue(model, tokenizer, enc_t, args.max_new_tokens, edits=edits)
                rows.append(dict(pair=k, set=name, direction=d,
                                 hedged_clean=bool(HEDGE_RE.search(clean[d])),
                                 hedged_patched=bool(HEDGE_RE.search(g)),
                                 gen_clean=clean[d], gen_patched=g))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    lines = [f"Behavioral validation -- {args.category} (eval split: {args.eval_split})",
             f"  {len(heads)} heads + {len(neurons)} neurons; {len(pairs)} pairs; "
             f"{args.max_new_tokens} greedy tokens; patch applied to prompt positions at prefill only",
             f"  readout: HEDGE_RE on the generated continuation", ""]
    for d in DIRECTIONS:
        sub = df[df.direction == d]
        cir = sub[sub.set == "circuit"]
        rnd = sub[sub.set.str.startswith("random")]
        cl = cir.hedged_clean.mean()
        cp = cir.hedged_patched.mean()
        per_set = rnd.groupby("set").hedged_patched.mean()
        sgn = EXPECTED_SIGN[d]
        lines += [f"  {d}  (expected sign of delta: {'-' if sgn < 0 else '+'})",
                  f"    clean hedge rate            {cl:.3f}",
                  f"    circuit-patched hedge rate  {cp:.3f}   delta {cp - cl:+.3f}  "
                  f"(expected direction: {'YES' if sgn * (cp - cl) > 0 else 'NO'})",
                  f"    random-set patched rate     {per_set.mean():.3f} +- {per_set.std():.3f} "
                  f"over {len(per_set)} sets   delta {per_set.mean() - cl:+.3f}",
                  f"    circuit delta minus random-set delta: {(cp - cl) - (per_set.mean() - cl):+.3f}"]
        # pair-level flip analysis, restricted to pairs whose twins actually differ behaviourally
        wide = sub.pivot_table(index="pair", columns="set", values="hedged_patched", aggfunc="first")
        cleanmap = cir.set_index("pair").hedged_clean
        other = "uncertain_to_control" if d == "control_to_uncertain" else "control_to_uncertain"
        src_clean = df[(df.direction == other) & (df.set == "circuit")].set_index("pair").hedged_clean
        disc = [p for p in cleanmap.index if bool(cleanmap[p]) != bool(src_clean[p])]
        if disc:
            flip_c = np.mean([bool(wide.loc[p, "circuit"]) == bool(src_clean[p]) for p in disc])
            rcols = [c for c in wide.columns if c.startswith("random")]
            flip_r = np.mean([[bool(wide.loc[p, rc]) == bool(src_clean[p]) for p in disc] for rc in rcols])
            lines += [f"    discordant pairs (twins differ in clean hedging): {len(disc)}/{len(cleanmap)}",
                      f"    flip-to-source-behaviour rate   circuit {flip_c:.3f}   random {flip_r:.3f}   "
                      f"margin {flip_c - flip_r:+.3f}"]
        else:
            lines += ["    discordant pairs: 0 -- flip analysis not defined"]
        lines += [""]
    summary = "\n".join(lines)
    print(summary)
    out_csv.with_name(out_csv.stem + "_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_csv, build_provenance(
        model, script="scripts/circuit_behavioral.py", category=args.category, eval_split=args.eval_split,
        n_heads=len(heads), n_neurons=len(neurons), n_pairs=len(pairs), n_random=args.n_random,
        max_new_tokens=args.max_new_tokens, patch_scope="prompt positions, prefill pass only",
        hedge_regex=HEDGE_RE.pattern, seed=args.seed))


if __name__ == "__main__":
    main()
