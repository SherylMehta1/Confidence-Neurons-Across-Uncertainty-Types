"""
Stolfo et al. (2024) weight-space entropy-neuron criteria for a candidate file,
against (1) a random same-layer-range null and (2) a weight-norm-matched null.

Stolfo et al. identify entropy neurons FROM THE WEIGHTS: an entropy neuron has
a large output-weight norm whose direction composes mostly with the null space
of the unembedding -- it cannot push specific tokens, so its effect flows
through the final normalization denominator. Per neuron, with the final
RMSNorm gamma folded in (w_tilde = gamma * w_out):

  w_norm               ||w_tilde||
  logit_var            Var_vocab(W_U w_tilde) / ||w_tilde||^2          (LOW for entropy neurons)
  nullfrac_k{16,64,256} ||P_bottom-k w_tilde||^2 / ||w_tilde||^2, P = projection onto the
                       bottom-k right-singular directions of W_U       (HIGH for entropy neurons)
  nullfrac_bottom10pct the notebook's variant: bottom 10% of singular directions
                       (norm ratio, NOT squared -- kept as in the notebook for comparability)

Nulls:
  random  : n_random neurons drawn uniformly from the candidate layers (seed),
            excluding candidates.
  matched : for EACH candidate, n_matched random same-layer neurons whose
            ||w_tilde|| lies within +/-10% of the candidate's. Percentiles
            against this null ask "given a neuron with this big an output
            vector, is its direction unusually null-space aligned?"

Outputs: <out>.csv (one row per neuron, kind in {candidate, random, matched},
matched rows carry `matched_to`), <out>_summary.txt, <out>.provenance.json.

Usage (from anywhere):
  python analysis/stolfo_criteria.py --candidates candidate_neurons.json \
      --out analysis/stolfo_criteria_v4 --n-random 1000 --seed 42 \
      --model-id meta-llama/Llama-3.1-8B-Instruct [--quantize]
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.detection import load_candidate_neurons  # noqa: E402
from shared.logit_lens import _dequantized_down_proj_weight  # noqa: E402
from shared.provenance import build_provenance, sha256_file, write_provenance  # noqa: E402

NULL_KS = (16, 64, 256)
MATCH_TOL = 0.10


def _layer_folded_weights(model, layer, gamma, device):
    """gamma-folded down_proj columns for a whole layer: [hidden, intermediate] fp32."""
    W = _dequantized_down_proj_weight(model, layer).detach().to(device, torch.float32)
    return gamma[:, None] * W


def _criteria(w_tilde, unembed, Vh, null_ks, n_bottom10):
    norm2 = torch.dot(w_tilde, w_tilde)
    logits = unembed @ w_tilde
    coeffs = Vh @ w_tilde  # coordinates in the right-singular basis (rows of Vh ordered by decreasing S)
    row = {
        "w_norm": float(norm2.sqrt()),
        "logit_var": float(logits.var() / norm2),
    }
    for k in null_ks:
        k_eff = min(k, coeffs.numel())
        row[f"nullfrac_k{k}"] = float((coeffs[-k_eff:] ** 2).sum() / norm2)
    # notebook variant: ||proj||/||w|| over the bottom 10% directions (norm ratio)
    row["nullfrac_bottom10pct"] = float(coeffs[-n_bottom10:].norm() / (norm2.sqrt() + 1e-10))
    return row


def analyze(model, candidates, out_csv, n_random=1000, n_matched=50, seed=42, null_ks=NULL_KS,
            candidates_path=None, verbose=True):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = model.get_output_embeddings().weight.device
    unembed = model.get_output_embeddings().weight.detach().to(device, torch.float32)
    gamma = model.model.norm.weight.detach().to(device, torch.float32)
    null_ks = tuple(k for k in null_ks)

    if verbose:
        print(f"SVD of unembedding {tuple(unembed.shape)} on {device}...")
    _, S, Vh = torch.linalg.svd(unembed, full_matrices=False)  # Vh: [hidden, hidden]
    n_bottom10 = max(1, int(len(S) * 0.10))
    if verbose:
        print(f"  singular values: max {S[0]:.3f}, min {S[-1]:.5f}; bottom-10% = {n_bottom10} dirs")

    cand_pairs = [(int(c["layer"]), int(c["neuron_idx"])) for c in candidates]
    cand_set = set(cand_pairs)
    layers = sorted({l for l, _ in cand_pairs})
    intermediate = model.config.intermediate_size
    rng = random.Random(seed)

    # random null
    random_pairs = []
    seen = set()
    while len(random_pairs) < n_random:
        pair = (rng.choice(layers), rng.randrange(intermediate))
        if pair not in cand_set and pair not in seen:
            seen.add(pair)
            random_pairs.append(pair)

    # per-layer folded weights + norms (for matching)
    folded = {l: _layer_folded_weights(model, l, gamma, device) for l in layers}
    norms = {l: folded[l].norm(dim=0) for l in layers}

    rows = []
    for l, n in cand_pairs:
        rows.append({"neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n, "kind": "candidate",
                     "matched_to": None, **_criteria(folded[l][:, n], unembed, Vh, null_ks, n_bottom10)})
    for i, (l, n) in enumerate(random_pairs):
        rows.append({"neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n, "kind": "random",
                     "matched_to": None, **_criteria(folded[l][:, n], unembed, Vh, null_ks, n_bottom10)})
        if verbose and (i + 1) % 200 == 0:
            print(f"  random null {i + 1}/{n_random}")

    # norm-matched null
    n_eligible = {}
    for l, n in cand_pairs:
        target = float(norms[l][n])
        lo, hi = target * (1 - MATCH_TOL), target * (1 + MATCH_TOL)
        elig = [int(j) for j in torch.nonzero((norms[l] >= lo) & (norms[l] <= hi)).flatten().tolist()
                if (l, int(j)) not in cand_set]
        n_eligible[f"L{l}_N{n}"] = len(elig)
        pick = rng.sample(elig, min(n_matched, len(elig))) if elig else []
        for j in pick:
            rows.append({"neuron_id": f"L{l}_N{j}", "layer": l, "neuron_idx": j, "kind": "matched",
                         "matched_to": f"L{l}_N{n}", **_criteria(folded[l][:, j], unembed, Vh, null_ks, n_bottom10)})

    df = pd.DataFrame(rows)
    metrics = ["w_norm", "logit_var", "nullfrac_bottom10pct"] + [f"nullfrac_k{k}" for k in null_ks]
    rand = df[df.kind == "random"]
    for m in metrics:
        df[f"{m}_pctile_random"] = df[m].apply(lambda v: float((rand[m] < v).mean() * 100))
        df[f"{m}_pctile_matched"] = np.nan
    for l, n in cand_pairs:
        nid = f"L{l}_N{n}"
        matched = df[(df.kind == "matched") & (df.matched_to == nid)]
        ci = df.index[df.neuron_id.eq(nid) & df.kind.eq("candidate")]
        for m in metrics:
            if len(matched):
                df.loc[ci, f"{m}_pctile_matched"] = float((matched[m] < df.loc[ci, m].iloc[0]).mean() * 100)
    df.to_csv(out_csv, index=False)

    # summary
    kmid = f"nullfrac_k{null_ks[min(1, len(null_ks) - 1)]}"
    lines = [f"Stolfo weight criteria -- {len(cand_pairs)} candidates vs {len(rand)} random neurons "
             f"(layers {layers}, seed {seed}) and a +/-{int(MATCH_TOL * 100)}% norm-matched null "
             f"(up to {n_matched} per candidate)", "",
             f"{'neuron':<12} {'w_norm':>8} {'pR':>5} | {'logit_var':>10} {'pR':>5} {'pM':>5} | "
             f"{kmid:>13} {'pR':>5} {'pM':>5} | {'b10%':>7} {'pR':>5} {'pM':>5} | nM   verdict"]
    for _, r in df[df.kind == "candidate"].iterrows():
        is_entropy = (r["w_norm_pctile_random"] > 90 and r["logit_var_pctile_random"] < 10
                      and r[f"{kmid}_pctile_random"] > 90)
        lines.append(
            f"{r.neuron_id:<12} {r.w_norm:8.3f} {r.w_norm_pctile_random:5.1f} | "
            f"{r.logit_var:10.5f} {r.logit_var_pctile_random:5.1f} {r.logit_var_pctile_matched:5.1f} | "
            f"{r[kmid]:13.5f} {r[f'{kmid}_pctile_random']:5.1f} {r[f'{kmid}_pctile_matched']:5.1f} | "
            f"{r.nullfrac_bottom10pct:7.4f} {r.nullfrac_bottom10pct_pctile_random:5.1f} "
            f"{r.nullfrac_bottom10pct_pctile_matched:5.1f} | {n_eligible[r.neuron_id]:<4d} "
            f"{'ENTROPY-NEURON-LIKE' if is_entropy else '-'}")
    lines += ["", "pR = percentile vs random null; pM = percentile vs norm-matched null; "
              "nM = number of eligible norm-matched neurons in that layer.",
              "Verdict rule (random null): w_norm > p90, logit_var < p10, "
              f"{kmid} > p90."]
    summary = "\n".join(lines)
    if verbose:
        print(summary)
    summary_path = out_csv.with_name(out_csv.stem + "_summary.txt")
    summary_path.write_text(summary + "\n")

    prov = build_provenance(
        model, seed=seed, n_random=n_random, n_matched=n_matched, match_tolerance=MATCH_TOL,
        null_ks=list(null_ks), n_bottom10pct_dirs=n_bottom10, layer_range=layers,
        candidate_file_sha256=sha256_file(candidates_path) if candidates_path else None,
        candidate_neurons=[f"L{l}_N{n}" for l, n in cand_pairs],
        n_eligible_matched=n_eligible, outputs=[str(out_csv), str(summary_path)],
    )
    write_provenance(out_csv, prov)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=str(REPO_ROOT / "candidate_neurons.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "stolfo_criteria_v4"),
                    help="output stem (writes <out>.csv, <out>_summary.txt, <out>.provenance.json)")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--n-matched", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-id", default=os.environ.get("CN_MODEL_ID", os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")))
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out_csv = Path(args.out).with_suffix(".csv")
    if out_csv.exists() and not args.overwrite:
        raise SystemExit(f"{out_csv} exists; pass --overwrite")

    from shared.model_utils import load_model
    model, _ = load_model(model_id=args.model_id, quantize=args.quantize)
    candidates = load_candidate_neurons(args.candidates)
    analyze(model, candidates, out_csv, n_random=args.n_random, n_matched=args.n_matched,
            seed=args.seed, candidates_path=args.candidates)


if __name__ == "__main__":
    main()
