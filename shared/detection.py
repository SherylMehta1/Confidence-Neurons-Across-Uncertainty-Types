"""
shared/detection_v2.py -- Phase 3 "Redo detection properly" (shared, all three)

Adds on top of shared/detection.py (unchanged, still used for the actual
activation-capture hooks):

1. LARGER, DOCUMENTED BASELINE: detect_candidate_neurons_v2 takes an
   explicit, named baseline prompt set (recommend >= 300 prompts, pooled
   across a mix of clearly-confident and clearly-uncertain prompts spanning
   all three categories' NEW, position-fixed data -- not just one category)
   instead of whatever was used before. The old candidate_neurons.json's
   provenance block already records a baseline count/hash/seed (Phase 1
   fix) -- check that value before assuming the old baseline was big enough;
   if you don't have it handy, err on the side of a bigger one here.

2. SPLIT-HALF VALIDATION: runs the correlation-based detection independently
   on two random halves of the baseline set, and keeps only neurons whose
   |correlation| clears a threshold AND lands in the top-K of BOTH halves --
   i.e. a neuron only counts as "detected" if it looks like a real, stable
   correlate of entropy on data it wasn't picked using. This directly
   addresses the original audit's "selection effects in detection" finding
   (top-15-of-157k, no split-half check, textbook winner's curse -- e.g.
   L30_N5509: top detection correlation 0.759, zero causal effect anywhere).

3. FULL CORRELATION DISTRIBUTION SAVED, not just the top-K: written to
   full_correlation_distribution.json so anyone can later ask "how unusual
   is this neuron's correlation, really?" without re-running detection.

This supersedes the FROZEN candidate_neurons.json from Phase 2 -- running
this produces a NEW candidate set, which needs Phase 3 (mechanism check)
and Phase 4 (ablation) re-run on top of it, on the new/fixed per-category
data, for all three people. Treat this as a hard reset of the "frozen"
artifact, with the reset itself now reproducible and documented.
"""

import hashlib
import json
import numpy as np

from shared.detection import capture_intermediate_activations
from shared.model_utils import get_next_token_probs, compute_entropy


def _correlate_all_neurons(model, tokenizer, prompts, layer_range):
    """Same core loop as shared.detection.detect_candidate_neurons, factored
    out so it can be called twice (once per half) without duplicating the
    activation-capture logic."""
    entropies = np.array([
        compute_entropy(get_next_token_probs(model, tokenizer, p))
        for p in prompts
    ])

    layer_range = list(layer_range)
    intermediate_size = model.config.intermediate_size
    acts_by_layer = {
        l: np.zeros((len(prompts), intermediate_size), dtype=np.float32)
        for l in layer_range
    }

    for i, prompt in enumerate(prompts):
        captured = capture_intermediate_activations(model, tokenizer, prompt, layer_range)
        for l in layer_range:
            acts_by_layer[l][i, :] = captured[l]
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(prompts)} prompts processed")

    corr = {}
    for layer_idx in layer_range:
        layer_acts = acts_by_layer[layer_idx]
        for neuron_idx in range(intermediate_size):
            col = layer_acts[:, neuron_idx]
            if np.std(col) < 1e-8:
                continue
            r = np.corrcoef(col, entropies)[0, 1]
            if not np.isnan(r):
                corr[(layer_idx, neuron_idx)] = float(r)
    return corr


def detect_candidate_neurons_split_half(
    model, tokenizer, baseline_prompts, layer_range,
    top_k_per_half: int = 60, top_k_final: int = 15, seed: int = 42,
):
    """
    Splits baseline_prompts in half, runs full-neuron correlation detection
    on each half independently, and keeps only neurons that appear in the
    top `top_k_per_half` of BOTH halves (by |correlation|), ranked in the
    final list by the MINIMUM of the two halves' |correlation| (a neuron
    that's merely lucky in one half and mediocre in the other should not
    outrank one that's consistently strong in both).

    Returns (candidates, full_distributions) where full_distributions is
    {"half_a": {...}, "half_b": {...}} keyed by "L{layer}_N{neuron}" for
    every neuron with nonzero variance in that half -- save this alongside
    the candidate list (see save_candidate_neurons_v2 below).
    """
    rng = np.random.default_rng(seed)
    prompts = list(baseline_prompts)
    idx = np.arange(len(prompts))
    rng.shuffle(idx)
    mid = len(idx) // 2
    half_a = [prompts[i] for i in idx[:mid]]
    half_b = [prompts[i] for i in idx[mid:]]
    print(f"Split {len(prompts)} baseline prompts into {len(half_a)} / {len(half_b)}")

    print("Running detection on half A...")
    corr_a = _correlate_all_neurons(model, tokenizer, half_a, layer_range)
    print("Running detection on half B...")
    corr_b = _correlate_all_neurons(model, tokenizer, half_b, layer_range)

    top_a = set(sorted(corr_a, key=lambda k: -abs(corr_a[k]))[:top_k_per_half])
    top_b = set(sorted(corr_b, key=lambda k: -abs(corr_b[k]))[:top_k_per_half])
    stable = top_a & top_b
    print(f"{len(top_a)} in top-{top_k_per_half} of half A, {len(top_b)} of half B, "
          f"{len(stable)} in BOTH (split-half-stable)")

    if len(stable) < top_k_final:
        print(f"WARNING: only {len(stable)} neurons survived split-half agreement, "
              f"fewer than the requested top_k_final={top_k_final}. This is itself "
              f"a meaningful result (per the project's own scope/limitations "
              f"language) -- do not lower top_k_per_half just to backfill the count.")

    ranked = sorted(stable, key=lambda k: -min(abs(corr_a[k]), abs(corr_b[k])))
    final = ranked[:top_k_final]

    candidates = [
        {
            "neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n,
            "detection_correlation_half_a": corr_a[(l, n)],
            "detection_correlation_half_b": corr_b[(l, n)],
            "detection_correlation_min_abs": min(abs(corr_a[(l, n)]), abs(corr_b[(l, n)])),
        }
        for (l, n) in final
    ]

    full_distributions = {
        "half_a": {f"L{l}_N{n}": r for (l, n), r in corr_a.items()},
        "half_b": {f"L{l}_N{n}": r for (l, n), r in corr_b.items()},
    }
    return candidates, full_distributions


def save_candidate_neurons_v2(
    candidates, full_distributions, baseline_prompts, seed,
    path="candidate_neurons.json",
    distribution_path="full_correlation_distribution.json",
):
    prompt_blob = json.dumps(sorted(baseline_prompts)).encode("utf-8")
    prompt_hash = hashlib.sha256(prompt_blob).hexdigest()[:16]

    payload = {
        "provenance": {
            "n_baseline_prompts": len(baseline_prompts),
            "baseline_prompt_hash": prompt_hash,
            "seed": seed,
            "method": "split_half_validated",
        },
        "candidates": candidates,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(distribution_path, "w") as f:
        json.dump(full_distributions, f, indent=2)
    print(f"Saved {len(candidates)} split-half-validated candidates to {path}")
    print(f"Saved full correlation distribution ({len(full_distributions['half_a'])} + "
          f"{len(full_distributions['half_b'])} entries) to {distribution_path}")
