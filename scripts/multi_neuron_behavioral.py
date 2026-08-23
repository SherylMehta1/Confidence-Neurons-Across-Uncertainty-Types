"""
E4: multi-neuron behavioral test with two readouts -- hedging in generated text AND verbalized
confidence ("how sure are you, 0-100?").

Named neuron SETS are clamped SIMULTANEOUSLY (one pre-hook per layer, all set members at the
last position of every generated token): each set at its members' means (ablation), and at
mean +/- 2 sigma. Sets are given as --sets name=path.json[:k] (candidate JSON files, optionally
truncated to the first k) or name=L31_N11541,L31_N6772 (explicit lists).

Readouts per prompt and condition:
  hedged            : HEDGE_RE on the 24-token greedy answer (as in behavioral_test.py)
  verbal_confidence : the model is then asked, in a new user turn, "How confident are you in
                      that answer, from 0 to 100? Reply with a number only." -> parsed integer
                      (NaN if none); the clamp stays active during this second generation.

Summary: per set x condition x arm: hedge rate and delta vs clean with a paired exact test,
mean verbal confidence and delta vs clean with a paired Wilcoxon, and the uncertain-minus-control
interaction of each delta.

Usage:
  python scripts/multi_neuron_behavioral.py --sets freq=results/candidates_frequency_weights.json \
      entropy=results/candidates_entropy_weights.json key=L31_N11541,L31_N6772 --control-sets 2
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from scipy import stats

from _common import REPO_ROOT, add_model_args, guard_output, load_category, load_model_from_args, parse_categories, parse_neurons
from behavioral_test import HEDGE_RE, activation_stats, generate_batch
from shared.detection import load_candidate_neurons
from shared.provenance import build_provenance, write_provenance

CONF_QUESTION = "How confident are you in that answer, from 0 to 100? Reply with a number only."
NUM_RE = re.compile(r"\b(100|\d{1,2})\b")


class MultiClampHook:
    """Clamp several neurons of ONE layer at the last position."""

    def __init__(self, idx_to_value):
        self.idx = torch.tensor(sorted(idx_to_value))
        self.val = torch.tensor([idx_to_value[i] for i in sorted(idx_to_value)])

    def __call__(self, module, args):
        x = args[0].clone()
        x[:, -1, self.idx.to(x.device)] = self.val.to(x.device, x.dtype)
        return (x,) + tuple(args[1:])


def register_set(model, neurons, values):
    """neurons: list of (layer, idx); values: {(layer, idx): clamp}. Returns hook handles."""
    by_layer = {}
    for (l, i) in neurons:
        by_layer.setdefault(l, {})[i] = float(values[(l, i)])
    return [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(MultiClampHook(m)) for l, m in by_layer.items()]


def parse_sets(specs):
    sets = {}
    for spec in specs:
        name, val = spec.split("=", 1)
        if val.endswith(".json") or ".json:" in val:
            path, _, k = val.partition(":")
            cands = load_candidate_neurons(REPO_ROOT / path)
            if k:
                cands = cands[: int(k)]
            sets[name] = [(c["layer"], c["neuron_idx"]) for c in cands]
        else:
            sets[name] = parse_neurons(val)
    return sets


@torch.no_grad()
def confidence_followup(model, tokenizer, chat_prompts, answers, max_new_tokens=6, batch_size=16):
    """Append the generated answer + a confidence question as a new turn; return parsed numbers."""
    followups = []
    for p, a in zip(chat_prompts, answers):
        followups.append(p + a.rstrip() + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + CONF_QUESTION
                         + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    outs = []
    for i in range(0, len(followups), batch_size):
        outs += generate_batch(model, tokenizer, followups[i:i + batch_size], max_new_tokens)
    nums = []
    for o in outs:
        m = NUM_RE.search(o)
        nums.append(float(m.group(1)) if m else np.nan)
    return outs, nums


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--sets", nargs="+", required=True, help="name=path.json[:k] or name=L31_N1,L30_N2")
    ap.add_argument("--control-sets", type=int, default=2, help="random neuron sets matched in size to the largest set")
    ap.add_argument("--categories", default="lack_of_knowledge")
    ap.add_argument("--conditions", default="mean,plus2,minus2")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/multi_neuron_behavioral.csv")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    guard_output(out, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    sets = parse_sets(args.sets)
    rng = random.Random(args.seed)
    taken = {n for s in sets.values() for n in s}
    layers_pool = sorted({l for s in sets.values() for l, _ in s})
    size = max(len(s) for s in sets.values())
    for c in range(args.control_sets):
        ctrl = []
        while len(ctrl) < size:
            p = (rng.choice(layers_pool), rng.randrange(model.config.intermediate_size))
            if p not in taken:
                taken.add(p); ctrl.append(p)
        sets[f"random_ctrl_{c}"] = ctrl
    all_neurons = sorted(taken)
    conditions = [c for c in args.conditions.split(",") if c]
    rows = []
    for cat in parse_categories(args.categories):
        prompts, ctrls = load_category(cat)
        recs = [dict(r, is_control=False) for r in prompts] + [dict(r, is_control=True) for r in ctrls]
        if args.limit:
            recs = [r for r in recs if not r["is_control"]][: args.limit] + [r for r in recs if r["is_control"]][: args.limit]
        pooled = [r["chat_formatted_prompt"] for r in recs if r["split"] == "working"]
        print(f"[{cat}] {len(recs)} prompts; {len(sets)} sets; stats for {len(all_neurons)} neurons")
        st = activation_stats(model, tokenizer, pooled, all_neurons)
        texts = [r["chat_formatted_prompt"] for r in recs]

        def run_condition(set_name, cond, handles):
            try:
                gen = []
                for i in range(0, len(texts), args.batch_size):
                    gen += generate_batch(model, tokenizer, texts[i:i + args.batch_size], args.max_new_tokens)
                conf_txt, conf = confidence_followup(model, tokenizer, texts, gen, batch_size=args.batch_size)
            finally:
                for h in handles:
                    h.remove()
            for r, g, ct, cn in zip(recs, gen, conf_txt, conf):
                rows.append(dict(set=set_name, condition=cond, category=cat, prompt_id=r["prompt_id"], split=r["split"],
                                 is_control=r["is_control"], generated=g, hedged=bool(HEDGE_RE.search(g)),
                                 confidence_text=ct, verbal_confidence=cn))
            print(f"  {set_name:<16} {cond:<7} hedge(unc) {np.mean([x['hedged'] for x in rows[-len(recs):] if not x['is_control']]):.3f}"
                  f"  conf(unc) {np.nanmean([x['verbal_confidence'] for x in rows[-len(recs):] if not x['is_control']]):.1f}")

        run_condition("clean", "clean", [])
        for name, neurons in sets.items():
            for cond in conditions:
                k = {"mean": 0.0, "plus2": 2.0, "minus2": -2.0}[cond]
                values = {n: st[n][0] + k * st[n][1] for n in neurons}
                run_condition(name, cond, register_set(model, neurons, values))
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    clean = df[df.set == "clean"].set_index("prompt_id")
    srows = []
    for (name, cond), g in df[df.set != "clean"].groupby(["set", "condition"]):
        deltas = {}
        for arm, gg in g.groupby("is_control"):
            c = clean.loc[gg.prompt_id]
            h1, h0 = gg.hedged.to_numpy(bool), c.hedged.to_numpy(bool)
            b, cc = int((h1 & ~h0).sum()), int((~h1 & h0).sum())
            p_h = stats.binomtest(b, b + cc, 0.5).pvalue if (b + cc) else 1.0
            v1, v0 = gg.verbal_confidence.to_numpy(float), c.verbal_confidence.to_numpy(float)
            ok = ~np.isnan(v1) & ~np.isnan(v0)
            p_v = stats.wilcoxon(v1[ok], v0[ok]).pvalue if ok.sum() > 5 and (v1[ok] != v0[ok]).any() else 1.0
            arm_name = "control" if arm else "uncertain"
            deltas[arm_name] = (h1.mean() - h0.mean(), np.nanmean(v1) - np.nanmean(v0))
            srows.append(dict(set=name, condition=cond, arm=arm_name, n=len(gg), n_neurons=len(sets[name]),
                              hedge_rate_clean=h0.mean(), hedge_rate=h1.mean(), hedge_delta=h1.mean() - h0.mean(),
                              hedge_gained=b, hedge_lost=cc, hedge_delta_p=p_h,
                              conf_clean=np.nanmean(v0), conf=np.nanmean(v1), conf_delta=np.nanmean(v1) - np.nanmean(v0),
                              conf_delta_p=p_v, conf_parsed_frac=float(np.mean(~np.isnan(v1)))))
        if {"uncertain", "control"} <= set(deltas):
            for r in srows[-2:]:
                r["hedge_interaction"] = deltas["uncertain"][0] - deltas["control"][0]
                r["conf_interaction"] = deltas["uncertain"][1] - deltas["control"][1]
    summ = pd.DataFrame(srows)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    write_provenance(out, build_provenance(model, script="scripts/multi_neuron_behavioral.py",
                                           sets={k: [f"L{l}_N{n}" for l, n in v] for k, v in sets.items()},
                                           conditions=conditions, confidence_question=CONF_QUESTION, seed=args.seed))
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
