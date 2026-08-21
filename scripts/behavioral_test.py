"""
Behavioral test: does clamping a neuron change what the model SAYS, not just its
next-token entropy?

For each prompt (uncertain + matched control) we greedily generate `--max-new-tokens`
tokens under several conditions of one neuron's activation, clamped at the last
position of the prefill and at every generated position:
  clean   : no intervention
  mean    : clamp to the neuron's mean activation over the pooled working prompts
            (uncertain + control) -- in-distribution mean-ablation
  plus2   : clamp to mean + 2 sigma
  minus2  : clamp to mean - 2 sigma

Readouts per generation:
  hedged           : abstention / hedging language present (HEDGE_RE)
  changed_vs_clean : generated text differs from the clean generation
  edit_ratio       : difflib similarity to the clean generation (1 = identical)
  first_token      : first generated token (for answer-flip analysis)

Summary (printed + <out>_summary.csv): per neuron x condition x arm, hedge rate,
delta vs clean with a paired exact test on discordant prompts (McNemar-style),
change rate, and the uncertain-minus-control interaction of the hedge delta.

Usage (from any CWD):
  python scripts/behavioral_test.py --neurons L31_N11541,L31_N6772,L31_N2477 \
      --categories lack_of_knowledge --control-neurons 2 --out results/behavioral_bf16.csv
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import difflib
import random
import re

import numpy as np
import pandas as pd
import torch
from scipy import stats

from _common import (REPO_ROOT, add_model_args, guard_output, load_category, load_model_from_args,
                     parse_categories, parse_neurons)
from shared.detection import capture_intermediate_activations
from shared.provenance import build_provenance, write_provenance

HEDGE_RE = re.compile(
    r"\b(i(?:'m| am)? not (?:sure|certain|aware|familiar)|i (?:do not|don't) (?:know|have)|"
    r"not (?:known|clear|available|possible|provided|specified|listed|documented|recognized|found)|"
    r"unknown|no (?:information|record|records|data|evidence|such)|unable to|"
    r"(?:cannot|can't|could not|couldn't) (?:determine|find|verify|provide|identify|locate|confirm)|"
    r"there is no|does not (?:exist|appear|seem)|doesn't exist|fictional|not a real|made[- ]up|"
    r"unclear|uncertain|not familiar|i have no|no widely|hypothetical|unverified)\b",
    re.IGNORECASE,
)
CONDITIONS = ("clean", "mean", "plus2", "minus2")


def activation_stats(model, tokenizer, prompts, neurons, verbose=True):
    """Per-neuron (mean, sigma) of the last-token activation over `prompts`."""
    layers = sorted({l for l, _ in neurons})
    acc = {n: [] for n in neurons}
    for i, p in enumerate(prompts):
        cap = capture_intermediate_activations(model, tokenizer, p, layers)
        for l, n in neurons:
            acc[(l, n)].append(float(cap[l][n]))
        if verbose and (i + 1) % 50 == 0:
            print(f"  activation stats: {i + 1}/{len(prompts)}")
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in acc.items()}


class ClampHook:
    """Forward pre-hook on down_proj clamping one neuron at the LAST position of
    whatever sequence passes through -- the final prompt token during prefill
    (left padding keeps it aligned across the batch) and every step during decode."""

    def __init__(self, layer_idx, neuron_idx, value):
        self.layer_idx, self.neuron_idx, self.value = layer_idx, neuron_idx, value

    def __call__(self, module, args):
        x = args[0].clone()
        x[:, -1, self.neuron_idx] = self.value
        return (x,) + tuple(args[1:])


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, hook=None, layer_idx=None):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    enc.pop("token_type_ids", None)  # some fast tokenizers emit it; generate() rejects it
    handle = None
    if hook is not None:
        handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    finally:
        if handle is not None:
            handle.remove()
    new = out[:, enc["input_ids"].shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True) for t in new]


def pick_control_neurons(candidates, n, seed, intermediate_size):
    rng = random.Random(seed)
    taken = set(candidates)
    layers = sorted({l for l, _ in candidates}) or [31]
    out = []
    while len(out) < n:
        pair = (rng.choice(layers), rng.randrange(intermediate_size))
        if pair not in taken:
            taken.add(pair)
            out.append(pair)
    return out


def summarize(df):
    rows = []
    clean = df[df.condition == "clean"].set_index("prompt_id")
    for (nid, cond), g in df[df.condition != "clean"].groupby(["neuron_id", "condition"]):
        deltas = {}
        for arm, gg in g.groupby("is_control"):
            c = clean.loc[gg.prompt_id]
            h1, h0 = gg.hedged.to_numpy(bool), c.hedged.to_numpy(bool)
            b, cc = int((h1 & ~h0).sum()), int((~h1 & h0).sum())  # discordant pairs
            p = stats.binomtest(b, b + cc, 0.5).pvalue if (b + cc) else 1.0
            arm_name = "control" if arm else "uncertain"
            deltas[arm_name] = h1.mean() - h0.mean()
            rows.append(dict(neuron_id=nid, condition=cond, arm=arm_name, n=len(gg),
                             clamp_value=gg.clamp_value.iloc[0],
                             hedge_rate_clean=h0.mean(), hedge_rate=h1.mean(),
                             hedge_delta=h1.mean() - h0.mean(), n_gained_hedge=b, n_lost_hedge=cc,
                             hedge_delta_p=p, changed_rate=gg.changed_vs_clean.mean(),
                             mean_edit_ratio=gg.edit_ratio.mean(),
                             first_token_flip_rate=(gg.first_token.to_numpy() != c.first_token.to_numpy()).mean()))
        if {"uncertain", "control"} <= set(deltas):
            for r in rows[-2:]:
                r["hedge_delta_interaction"] = deltas["uncertain"] - deltas["control"]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--neurons", required=True, help="comma list, e.g. L31_N11541,L31_N6772")
    ap.add_argument("--categories", default="lack_of_knowledge")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--control-neurons", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="debug: use only the first N prompts per arm")
    ap.add_argument("--out", default="results/behavioral_bf16.csv")
    args = ap.parse_args()

    out = REPO_ROOT / args.out
    guard_output(out, args.overwrite)
    model, tokenizer = load_model_from_args(args)
    neurons = parse_neurons(args.neurons)
    controls = pick_control_neurons(neurons, args.control_neurons, args.seed, model.config.intermediate_size)
    all_neurons = neurons + controls
    conditions = [c for c in args.conditions.split(",") if c]
    rows = []
    for cat in parse_categories(args.categories):
        prompts, ctrls = load_category(cat)
        recs = [dict(r, is_control=False) for r in prompts] + [dict(r, is_control=True) for r in ctrls]
        if args.limit:
            recs = [r for r in recs if not r["is_control"]][: args.limit] + [r for r in recs if r["is_control"]][: args.limit]
        pooled = [r["chat_formatted_prompt"] for r in recs if r["split"] == "working"]
        print(f"[{cat}] {len(recs)} prompts; activation stats over {len(pooled)} working prompts for {len(all_neurons)} neurons")
        st = activation_stats(model, tokenizer, pooled, all_neurons)
        texts = [r["chat_formatted_prompt"] for r in recs]
        clean_gen = []
        for i in range(0, len(texts), args.batch_size):
            clean_gen += generate_batch(model, tokenizer, texts[i:i + args.batch_size], args.max_new_tokens)
        for r, g in zip(recs, clean_gen):
            rows.append(dict(neuron_id="clean", category=cat, condition="clean", prompt_id=r["prompt_id"],
                             split=r["split"], is_control=r["is_control"], clamp_value=np.nan,
                             generated=g, hedged=bool(HEDGE_RE.search(g)), first_token=g.strip().split(" ")[0] if g.strip() else "",
                             changed_vs_clean=False, edit_ratio=1.0))
        for (l, n) in all_neurons:
            nid = f"L{l}_N{n}"
            mu, sd = st[(l, n)]
            for cond in conditions:
                if cond == "clean":
                    continue
                value = {"mean": mu, "plus2": mu + 2 * sd, "minus2": mu - 2 * sd}[cond]
                gen = []
                for i in range(0, len(texts), args.batch_size):
                    gen += generate_batch(model, tokenizer, texts[i:i + args.batch_size], args.max_new_tokens,
                                          hook=ClampHook(l, n, value), layer_idx=l)
                for r, g, g0 in zip(recs, gen, clean_gen):
                    rows.append(dict(neuron_id=nid, category=cat, condition=cond, prompt_id=r["prompt_id"],
                                     split=r["split"], is_control=r["is_control"], clamp_value=value,
                                     generated=g, hedged=bool(HEDGE_RE.search(g)),
                                     first_token=g.strip().split(" ")[0] if g.strip() else "",
                                     changed_vs_clean=(g != g0),
                                     edit_ratio=difflib.SequenceMatcher(None, g0, g).ratio()))
                print(f"  {nid} {cond} (clamp {value:+.3f}) done")
    df = pd.DataFrame(rows)
    df["is_candidate"] = df.neuron_id.isin([f"L{l}_N{n}" for l, n in neurons] + ["clean"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summ = summarize(df)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    write_provenance(out, build_provenance(
        model, script="scripts/behavioral_test.py", model_id=getattr(args, "model_id", None),
        neurons=[f"L{l}_N{n}" for l, n in neurons], control_neurons=[f"L{l}_N{n}" for l, n in controls],
        conditions=conditions, max_new_tokens=args.max_new_tokens, hedge_regex=HEDGE_RE.pattern,
        activation_stats={f"L{l}_N{n}": st[(l, n)] for l, n in all_neurons}, seed=args.seed,
        categories=args.categories, limit=args.limit))
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"wrote {out} ({len(df)} rows) and {out.with_name(out.stem + '_summary.csv')}")


if __name__ == "__main__":
    main()
