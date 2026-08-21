"""
Token-frequency neurons (Stolfo et al. 2024, the second confidence-regulation class).

A token-frequency neuron's direct logit contribution is proportional to each token's
log unigram frequency: activating it slides the output distribution toward (or away
from) the unigram distribution -- a distributional confidence mechanism, distinct from
the temperature-like entropy neurons.

Weight-space score per neuron: Pearson correlation over the vocabulary between the
direct-logit vector d = W_U . (gamma * w_out) (fp32, RMSNorm gamma folded) and the
log unigram frequency of each token, plus the slope of d on log-frequency. Reported
with percentile ranks against N random same-layer neurons.

Unigram frequencies are estimated by tokenizing a WikiText-103 slice with the model's
own tokenizer (cached to results/unigram_logfreq.npy + .json provenance). Tokens never
seen get add-one smoothing.

Usage (from any CWD):
  python analysis/token_frequency_neurons.py --candidates results/candidate_neurons_bf16.json,candidate_neurons.json \
      --out results/token_frequency_neurons [--n-docs 20000] [--n-random 1000]
"""
import sys as _sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(REPO_ROOT))
import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch

from shared.detection import load_candidate_neurons
from shared.provenance import build_provenance, write_provenance


def unigram_logfreq(tokenizer, n_docs, cache_dir, verbose=True):
    """log p(token) over the vocab from a WikiText-103 slice; cached by tokenizer+n_docs."""
    key = hashlib.sha256(f"{tokenizer.name_or_path}|{n_docs}|{len(tokenizer)}".encode()).hexdigest()[:12]
    cache = _Path(cache_dir) / f"unigram_logfreq_{key}.npy"
    if cache.exists():
        return np.load(cache), cache
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    vocab = len(tokenizer)
    counts = np.zeros(vocab, dtype=np.int64)
    seen = 0
    for rec in ds:
        t = rec["text"].strip()
        if len(t) < 40:
            continue
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        counts += np.bincount(ids, minlength=vocab)[:vocab]
        seen += 1
        if verbose and seen % 5000 == 0:
            print(f"  unigram: {seen}/{n_docs} docs, {counts.sum():,} tokens")
        if seen >= n_docs:
            break
    logfreq = np.log((counts + 1.0) / (counts.sum() + vocab))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, logfreq)
    json.dump({"source": "Salesforce/wikitext wikitext-103-raw-v1 train", "n_docs": seen,
               "n_tokens": int(counts.sum()), "vocab": vocab, "tokenizer": tokenizer.name_or_path},
              open(cache.with_suffix(".json"), "w"), indent=2)
    return logfreq, cache


def direct_logits(model, layer, idx, unembed, gamma):
    w = model.model.layers[layer].mlp.down_proj.weight
    if hasattr(w, "quant_state"):
        import bitsandbytes.functional as bnb_F
        w = bnb_F.dequantize_4bit(w, w.quant_state)
    v = gamma * w[:, idx].detach().to(unembed.device, torch.float32)
    return unembed @ v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, help="comma list of candidate JSON files")
    ap.add_argument("--out", default="results/token_frequency_neurons")
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    csv_path = out.with_suffix(".csv")
    if csv_path.exists() and not args.overwrite:
        raise SystemExit(f"{csv_path} exists; pass --overwrite")

    from shared.model_utils import load_model
    model, tokenizer = load_model(model_id=args.model_id, quantize=args.quantize)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unembed = model.get_output_embeddings().weight.detach().to(device, torch.float32)
    gamma = model.model.norm.weight.detach().to(device, torch.float32)
    vocab_rows = unembed.shape[0]

    logfreq, cache = unigram_logfreq(tokenizer, args.n_docs, REPO_ROOT / "results")
    lf = torch.tensor(logfreq[:vocab_rows], dtype=torch.float32, device=device)
    lf_c = lf - lf.mean()
    lf_var = (lf_c ** 2).sum()

    cands = {}
    for f in args.candidates.split(","):
        f = f.strip()
        for c in load_candidate_neurons(REPO_ROOT / f):
            cands.setdefault((c["layer"], c["neuron_idx"]), set()).add(_Path(f).stem)
    rng = random.Random(args.seed)
    layers = sorted({l for l, _ in cands})
    rand = set()
    while len(rand) < args.n_random:
        p = (rng.choice(layers), rng.randrange(model.config.intermediate_size))
        if p not in cands:
            rand.add(p)

    rows = []
    todo = [(l, n, "candidate", "|".join(sorted(s))) for (l, n), s in sorted(cands.items())] + \
           [(l, n, "random", "") for l, n in sorted(rand)]
    for i, (l, n, kind, src) in enumerate(todo):
        d = direct_logits(model, l, n, unembed, gamma)
        d_c = d - d.mean()
        corr = ((d_c * lf_c).sum() / torch.sqrt((d_c ** 2).sum() * lf_var)).item()
        slope = ((d_c * lf_c).sum() / lf_var).item()
        rows.append(dict(neuron_id=f"L{l}_N{n}", layer=l, neuron_idx=n, kind=kind, sets=src,
                         freq_corr=corr, freq_slope=slope, direct_logit_std=d.std().item()))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(todo)} neurons")
    df = pd.DataFrame(rows)
    r = df[df.kind == "random"]
    df["abs_freq_corr_pctile"] = df.freq_corr.abs().apply(lambda v: (r.freq_corr.abs() < v).mean() * 100)
    df.to_csv(csv_path, index=False)

    c = df[df.kind == "candidate"].sort_values("abs_freq_corr_pctile", ascending=False)
    lines = [f"Token-frequency neuron score: candidates vs {len(r)} random neurons (layers {layers}, seed {args.seed})",
             f"random null |corr|: median {r.freq_corr.abs().median():.3f}, 95th pct {r.freq_corr.abs().quantile(.95):.3f}, 99th {r.freq_corr.abs().quantile(.99):.3f}", "",
             f"{'neuron':<12} {'freq_corr':>9} {'pct':>6}  {'slope':>8}  verdict  sets"]
    for x in c.itertuples():
        verdict = "FREQ-NEURON-LIKE" if x.abs_freq_corr_pctile >= 99 else ("elevated" if x.abs_freq_corr_pctile >= 95 else "-")
        lines.append(f"{x.neuron_id:<12} {x.freq_corr:+9.3f} {x.abs_freq_corr_pctile:6.1f}  {x.freq_slope:+8.4f}  {verdict:<16} {x.sets}")
    summary = "\n".join(lines)
    print(summary)
    out.with_name(out.name + "_summary.txt").write_text(summary + "\n")
    write_provenance(csv_path, build_provenance(model, script="analysis/token_frequency_neurons.py",
                                                unigram_cache=str(cache), n_docs=args.n_docs,
                                                candidates=args.candidates, n_random=args.n_random, seed=args.seed))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
