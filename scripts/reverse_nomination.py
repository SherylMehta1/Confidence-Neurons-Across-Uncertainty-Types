"""
Reverse nomination: find the model's entropy neurons and token-frequency neurons FROM THE
WEIGHTS (Stolfo et al. 2024; Du et al. 2025 procedure), independent of any activation
correlation, so they can then be tested causally with the same ablation / frozen-norm /
dose-response / frequency scripts as the correlation-selected candidates.

Weight scan over --layer-range (default: last 8 layers), every MLP neuron:
  w_tilde      = gamma * down_proj[:, i]        (final RMSNorm gamma folded)
  w_norm       = ||w_tilde||
  logit_var    = Var_vocab(W_U w_tilde) / ||w_tilde||^2
  nullfrac_k64 = fraction of ||w_tilde||^2 in the bottom-64 right-singular directions of W_U
  freq_corr    = corr_vocab(W_U w_tilde, log unigram frequency)

Nomination rules:
  entropy neurons   (Du et al.): among neurons in the top --norm-quantile (default 25%) of
                    w_norm within the scanned layers, the --k lowest logit_var
  frequency neurons: the --k largest |freq_corr| not already nominated

Outputs: results/reverse_nomination_scan.csv.gz (all scanned neurons), two candidate files in
the {"provenance","candidates"} format consumed by scripts/run_ablation.py etc.:
  results/candidates_entropy_weights.json, results/candidates_frequency_weights.json

Usage: python scripts/reverse_nomination.py [--layer-range 24-31] [--k 20] [--norm-quantile 0.75]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import glob
import json

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args, parse_layer_range
from shared.provenance import build_provenance, write_provenance


def align_logfreq(logfreq, vocab_rows):
    """Align a unigram log-frequency vector to the unembedding's row count: truncate if the
    tokenizer has more entries than the matrix, pad with the minimum (unseen-token) value if the
    matrix is larger (padded vocab, e.g. Qwen: 152064 rows vs 151665 tokens)."""
    import numpy as _np
    lf = _np.asarray(logfreq, dtype=_np.float64)
    if len(lf) >= vocab_rows:
        return lf[:vocab_rows]
    return _np.concatenate([lf, _np.full(vocab_rows - len(lf), lf.min())])

@torch.no_grad()
def scan_layers(model, layers, logu=None, k_null=64, chunk=2048, verbose=True):
    """Per-neuron weight statistics for every MLP neuron in `layers`. Returns a DataFrame."""
    device = model.get_output_embeddings().weight.device
    W_U = model.get_output_embeddings().weight.detach().to(device, torch.float32)  # [V, d]
    gamma = model.model.norm.weight.detach().to(device, torch.float32)
    V = W_U.shape[0]
    _, _, Vh = torch.linalg.svd(W_U, full_matrices=False)  # Vh: [d, d], rows = right-singular dirs
    null_basis = Vh[-k_null:]  # [k_null, d]
    if logu is not None:
        lf = torch.tensor(align_logfreq(logu, V), dtype=torch.float32, device=device)
        lf_c = lf - lf.mean()
        lf_norm = torch.sqrt((lf_c ** 2).sum())
    rows = []
    for layer in layers:
        W = model.model.layers[layer].mlp.down_proj.weight
        if hasattr(W, "quant_state"):
            import bitsandbytes.functional as bnb_F
            W = bnb_F.dequantize_4bit(W, W.quant_state)
        W = W.detach().to(device, torch.float32)  # [d, n]
        Wt = gamma[:, None] * W                    # gamma folded
        n = Wt.shape[1]
        norm2 = (Wt ** 2).sum(0)                   # [n]
        nullfrac = ((null_basis @ Wt) ** 2).sum(0) / norm2
        for s in range(0, n, chunk):
            blk = Wt[:, s:s + chunk]               # [d, c]
            L = W_U @ blk                          # [V, c]
            lv = L.var(dim=0, unbiased=True) / norm2[s:s + chunk]
            if logu is not None:
                Lc = L - L.mean(0, keepdim=True)
                fc = (Lc * lf_c[:, None]).sum(0) / (torch.sqrt((Lc ** 2).sum(0)) * lf_norm)
            else:
                fc = torch.full((blk.shape[1],), float("nan"), device=device)
            for j in range(blk.shape[1]):
                i = s + j
                rows.append(dict(neuron_id=f"L{layer}_N{i}", layer=layer, neuron_idx=i,
                                 w_norm=float(norm2[i].sqrt()), logit_var=float(lv[j]),
                                 nullfrac_k64=float(nullfrac[i]), freq_corr=float(fc[j])))
            del L
        if verbose:
            print(f"  scanned layer {layer}: {n} neurons")
    return pd.DataFrame(rows)


def nominate(df, k=20, norm_quantile=0.75):
    """Du/Stolfo entropy rule: top-norm pool, then lowest LogitVar; frequency: largest |freq_corr|."""
    pool = df[df.w_norm >= df.w_norm.quantile(norm_quantile)]
    entropy = pool.nsmallest(k, "logit_var").copy()
    entropy["nomination"] = "entropy_weights"
    rest = df[~df.neuron_id.isin(entropy.neuron_id)].copy()
    rest["abs_fc"] = rest.freq_corr.abs()
    freq = rest.nlargest(k, "abs_fc").drop(columns="abs_fc").copy()
    freq["nomination"] = "frequency_weights"
    return entropy, freq


def to_candidates(df, prov):
    return {"provenance": prov, "candidates": [
        dict(neuron_id=r.neuron_id, layer=int(r.layer), neuron_idx=int(r.neuron_idx),
             w_norm=float(r.w_norm), logit_var=float(r.logit_var), nullfrac_k64=float(r.nullfrac_k64),
             freq_corr=float(r.freq_corr), nomination=r.nomination) for r in df.itertuples()]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--layer-range", default=None, help="e.g. 24-31 (default: last 8 layers)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--norm-quantile", type=float, default=0.75)
    ap.add_argument("--unigram", default=None, help="unigram_logfreq_*.npy (default: newest in results/)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    scan_path = out_dir / "reverse_nomination_scan.csv.gz"
    guard_output(scan_path, args.overwrite)

    model, tokenizer = load_model_from_args(args)
    n_layers = model.config.num_hidden_layers
    layers = parse_layer_range(args.layer_range) if args.layer_range else list(range(n_layers - 8, n_layers))
    unis = sorted(glob.glob(str(REPO_ROOT / "results/unigram_logfreq_*.npy")))
    logu = np.load(args.unigram or unis[-1]) if (args.unigram or unis) else None
    if logu is None:
        print("WARNING: no unigram table found; freq_corr will be NaN (run analysis/token_frequency_neurons.py first)")

    df = scan_layers(model, layers, logu)
    df.to_csv(scan_path, index=False)
    entropy, freq = nominate(df, args.k, args.norm_quantile)
    prov = build_provenance(model, script="scripts/reverse_nomination.py", layers=layers, k=args.k,
                            norm_quantile=args.norm_quantile, n_scanned=len(df), unigram=str(args.unigram or (unis[-1] if unis else None)))
    for name, d in (("entropy", entropy), ("frequency", freq)):
        p = out_dir / f"candidates_{name}_weights.json"
        p.write_text(json.dumps(to_candidates(d, dict(prov, nomination=f"{name}_weights")), indent=2))
        print(f"wrote {p}")
    write_provenance(scan_path, prov)
    pd.set_option("display.width", 200)
    print("\nENTROPY-NEURON nominations (top-norm pool, lowest LogitVar):")
    print(entropy[["neuron_id", "w_norm", "logit_var", "nullfrac_k64", "freq_corr"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nFREQUENCY-NEURON nominations (largest |freq_corr|):")
    print(freq[["neuron_id", "w_norm", "logit_var", "nullfrac_k64", "freq_corr"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nscan: {len(df)} neurons over layers {layers}; w_norm quantile cut {df.w_norm.quantile(args.norm_quantile):.3f}")


if __name__ == "__main__":
    main()
