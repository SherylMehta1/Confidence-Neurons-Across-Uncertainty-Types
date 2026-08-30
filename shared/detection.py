"""
shared/detection.py -- the single detection module: activation capture hooks,
split-half-validated activation/entropy correlation, candidate-file I/O.

(Historically split across shared/detection.py (old, single-shot top-k) and
shared/detection_v2.py / old_detection.py; everything now lives here.
shared/old_detection.py is a deprecated re-export shim.)

Method
------
detect_candidate_neurons_split_half() splits the baseline prompts into two
halves (seeded; optionally stratified by category), computes, independently on
each half, the Pearson correlation between every MLP neuron's last-token
activation (down_proj input) and the fp32 next-token entropy, and keeps only
neurons that are in the top-`top_k_per_half` by |r| of BOTH halves (and, if
`min_abs_corr` is set, clear that threshold in both). Survivors are ranked by
the minimum of the two halves' |r|. `top_k_final=None` keeps every survivor.

Historical note (the "rescue"): the first run of this method used
top_k_final=15 and silently truncated the 17 split-half-stable neurons to 15;
shared/rescue_untruncated_candidates.py re-derived the full 17 from
full_correlation_distribution.json. With top_k_final=None the truncation no
longer happens and that script has been removed. To reproduce the original
(non-stratified) split, call with stratify_by=None -- the shuffle is the same
numpy default_rng(seed) permutation as before.

Every neuron's correlation in each half is returned/saved
(full_correlation_distribution.json) so "how unusual is this r?" can be
answered without re-running.
"""

import json
import warnings

import numpy as np
import torch

from shared import model_utils as _mu
from shared.model_utils import get_next_token_probs, compute_entropy
from shared.provenance import build_provenance, sha256_prompts, write_provenance


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

def capture_intermediate_activations(model, tokenizer, prompt, layer_indices, position=-1):
    """Capture the MLP intermediate activation (down_proj's INPUT, i.e.
    act_fn(gate) * up) for every layer in layer_indices, in ONE forward pass,
    at token `position` (default: last). Returns {layer_idx: float32 numpy
    array [intermediate_size]}. Hooks are removed in a finally block so an
    exception can never leave a stale hook attached."""
    captured = {}
    handles = []

    def make_pre_hook(layer_idx):
        def hook_fn(module, args):
            captured[layer_idx] = args[0].detach()[0, position, :].to(torch.float32).cpu().numpy()
        return hook_fn

    try:
        for layer_idx in layer_indices:
            down_proj = model.model.layers[layer_idx].mlp.down_proj
            handles.append(down_proj.register_forward_pre_hook(make_pre_hook(layer_idx)))

        inputs = _mu.tokenize_prompt(tokenizer, prompt, device=model.device)
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return captured


def get_neuron_activation(model, tokenizer, prompt, layer_idx, neuron_idx):
    """Last-token activation of one neuron (one forward pass)."""
    captured = capture_intermediate_activations(model, tokenizer, prompt, [layer_idx])
    return float(captured[layer_idx][neuron_idx])


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def _collect_activations_and_entropy(model, tokenizer, prompts, layer_range, verbose=True):
    """One forward pass per prompt (with capture hooks on every layer in
    layer_range) gives both the activations and the entropy."""
    layer_range = list(layer_range)
    intermediate_size = model.config.intermediate_size
    acts_by_layer = {
        l: np.zeros((len(prompts), intermediate_size), dtype=np.float32) for l in layer_range
    }
    entropies = np.zeros(len(prompts), dtype=np.float64)

    for i, prompt in enumerate(prompts):
        captured = capture_intermediate_activations(model, tokenizer, prompt, layer_range)
        for l in layer_range:
            acts_by_layer[l][i, :] = captured[l]
        # entropy from a second (hook-free) pass: cheap relative to the hook
        # bookkeeping and keeps the entropy measurement identical to ablation's.
        entropies[i] = compute_entropy(get_next_token_probs(model, tokenizer, prompt))
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(prompts)} prompts processed")
    return acts_by_layer, entropies


def correlate_layer(acts, entropies, zero_var_eps=1e-8):
    """Vectorized Pearson r of every column of acts [n, d] with entropies [n].
    Returns (r [d] with NaN for zero-variance columns, n_dropped)."""
    X = np.asarray(acts, dtype=np.float64)
    e = np.asarray(entropies, dtype=np.float64)
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0)
    ec = e - e.mean()
    e_sd = ec.std()
    keep = sd > zero_var_eps
    r = np.full(X.shape[1], np.nan)
    if e_sd > zero_var_eps and keep.any():
        Z = Xc[:, keep] / sd[keep]
        r[keep] = (Z.T @ (ec / e_sd)) / n
    return r, int((~keep).sum())


def _correlate_all_neurons(model, tokenizer, prompts, layer_range, verbose=True):
    """{(layer, neuron): r} for every finite-r neuron in layer_range, plus a
    per-layer count of dropped zero-variance neurons."""
    layer_range = list(layer_range)
    acts_by_layer, entropies = _collect_activations_and_entropy(
        model, tokenizer, prompts, layer_range, verbose=verbose)
    corr, dropped = {}, {}
    for layer_idx in layer_range:
        r, n_dropped = correlate_layer(acts_by_layer[layer_idx], entropies)
        dropped[layer_idx] = n_dropped
        for neuron_idx in np.flatnonzero(np.isfinite(r)):
            corr[(layer_idx, int(neuron_idx))] = float(r[neuron_idx])
    total_dropped = sum(dropped.values())
    if verbose:
        print(f"    dropped {total_dropped} zero-variance neurons "
              f"({', '.join(f'L{l}:{d}' for l, d in dropped.items() if d)})")
    return corr, dropped


# ---------------------------------------------------------------------------
# Split-half detection
# ---------------------------------------------------------------------------

def split_half_indices(n, seed, stratify_by=None):
    """Seeded split of range(n) into two halves. Without stratify_by this is
    the legacy permutation (numpy default_rng(seed).shuffle, first floor(n/2)
    to A). With stratify_by (list of labels, length n) each label group is
    shuffled and split in half; the extra item of odd-sized groups alternates
    between A and B so the totals stay balanced."""
    rng = np.random.default_rng(seed)
    if stratify_by is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        mid = n // 2
        return idx[:mid].tolist(), idx[mid:].tolist()

    if len(stratify_by) != n:
        raise ValueError(f"stratify_by has {len(stratify_by)} labels for {n} prompts")
    labels = list(stratify_by)
    a, b = [], []
    extra_to_a = True
    for lab in sorted(set(labels), key=str):
        grp = np.array([i for i, l in enumerate(labels) if l == lab])
        rng.shuffle(grp)
        half = len(grp) // 2
        if len(grp) % 2 == 1:
            half += 1 if extra_to_a else 0
            extra_to_a = not extra_to_a
        a += grp[:half].tolist()
        b += grp[half:].tolist()
    return a, b


def _top_set(corr, top_k, min_abs_corr):
    keys = sorted(corr, key=lambda k: -abs(corr[k]))
    if min_abs_corr is not None:
        keys = [k for k in keys if abs(corr[k]) >= min_abs_corr]
    return set(keys[:top_k])


def detect_candidate_neurons_split_half(
    model, tokenizer, baseline_prompts, layer_range,
    top_k_per_half=60, top_k_final=None, seed=42,
    min_abs_corr=None, stratify_by=None, verbose=True,
):
    """
    Split-half-validated detection (see module docstring).

    Returns (candidates, full_distributions):
      candidates: list of dicts {neuron_id, layer, neuron_idx,
          detection_correlation_half_a, detection_correlation_half_b,
          detection_correlation_min_abs}, ranked by min |r|.
      full_distributions: {"half_a": {"L{l}_N{n}": r}, "half_b": {...},
          "provenance": {...}} -- provenance holds the contract fields
          (seed, layer_range, top_k_per_half, top_k_final, min_abs_corr,
          baseline_prompt_sha256, n_baseline_prompts, split sizes, dropped
          zero-variance counts, stratified flag, model/version info).
    """
    prompts = list(baseline_prompts)
    layer_range = list(layer_range)
    idx_a, idx_b = split_half_indices(len(prompts), seed, stratify_by)
    half_a = [prompts[i] for i in idx_a]
    half_b = [prompts[i] for i in idx_b]
    if verbose:
        print(f"Split {len(prompts)} baseline prompts into {len(half_a)} / {len(half_b)}"
              f"{' (stratified)' if stratify_by is not None else ''}")
        print("Running detection on half A...")
    corr_a, dropped_a = _correlate_all_neurons(model, tokenizer, half_a, layer_range, verbose)
    if verbose:
        print("Running detection on half B...")
    corr_b, dropped_b = _correlate_all_neurons(model, tokenizer, half_b, layer_range, verbose)

    top_a = _top_set(corr_a, top_k_per_half, min_abs_corr)
    top_b = _top_set(corr_b, top_k_per_half, min_abs_corr)
    stable = top_a & top_b
    if verbose:
        print(f"{len(top_a)} in top-{top_k_per_half} of half A, {len(top_b)} of half B, "
              f"{len(stable)} in BOTH (split-half-stable)"
              f"{f', |r|>={min_abs_corr}' if min_abs_corr is not None else ''}")

    ranked = sorted(stable, key=lambda k: -min(abs(corr_a[k]), abs(corr_b[k])))
    if top_k_final is not None:
        if len(stable) > top_k_final and verbose:
            print(f"WARNING: truncating {len(stable)} stable neurons to top_k_final={top_k_final}; "
                  f"pass top_k_final=None to keep all split-half survivors.")
        ranked = ranked[:top_k_final]

    candidates = [
        {
            "neuron_id": f"L{l}_N{n}", "layer": int(l), "neuron_idx": int(n),
            "detection_correlation_half_a": corr_a[(l, n)],
            "detection_correlation_half_b": corr_b[(l, n)],
            "detection_correlation_min_abs": min(abs(corr_a[(l, n)]), abs(corr_b[(l, n)])),
        }
        for (l, n) in ranked
    ]

    strat_counts = None
    if stratify_by is not None:
        strat_counts = {
            "half_a": {str(k): int(sum(1 for i in idx_a if stratify_by[i] == k)) for k in set(stratify_by)},
            "half_b": {str(k): int(sum(1 for i in idx_b if stratify_by[i] == k)) for k in set(stratify_by)},
        }
    provenance = build_provenance(
        model,
        method="split_half_validated",
        seed=seed, layer_range=[int(l) for l in layer_range],
        top_k_per_half=top_k_per_half, top_k_final=top_k_final, min_abs_corr=min_abs_corr,
        baseline_prompt_sha256=sha256_prompts(prompts), n_baseline_prompts=len(prompts),
        n_half_a=len(half_a), n_half_b=len(half_b), stratified=stratify_by is not None,
        stratum_counts=strat_counts,
        dropped_zero_variance={"half_a": {str(k): v for k, v in dropped_a.items()},
                               "half_b": {str(k): v for k, v in dropped_b.items()}},
        n_stable=len(stable), n_candidates=len(candidates),
    )
    full_distributions = {
        "half_a": {f"L{l}_N{n}": r for (l, n), r in corr_a.items()},
        "half_b": {f"L{l}_N{n}": r for (l, n), r in corr_b.items()},
        "provenance": provenance,
    }
    return candidates, full_distributions


# ---------------------------------------------------------------------------
# Candidate file I/O
# ---------------------------------------------------------------------------

def save_candidate_neurons_v2(
    candidates, full_distributions, baseline_prompts, seed,
    path="candidate_neurons.json",
    distribution_path="full_correlation_distribution.json",
    extra_provenance=None,
):
    """Write {"provenance": ..., "candidates": [...]} to `path`, the full
    per-half correlation dicts to `distribution_path`, and a sibling
    <path>.provenance.json. Keeps the legacy keys (n_baseline_prompts,
    baseline_prompt_hash = first 16 hex of the sha256, seed, method)."""
    prov = dict(full_distributions.get("provenance") or build_provenance(None))
    prov.update({
        "n_baseline_prompts": len(baseline_prompts),
        "baseline_prompt_sha256": sha256_prompts(list(baseline_prompts)),
        "baseline_prompt_hash": sha256_prompts(list(baseline_prompts))[:16],
        "seed": seed,
        "method": prov.get("method", "split_half_validated"),
    })
    if extra_provenance:
        prov.update(extra_provenance)

    with open(path, "w") as f:
        json.dump({"provenance": prov, "candidates": candidates}, f, indent=2, default=str)
    with open(distribution_path, "w") as f:
        json.dump({"half_a": full_distributions["half_a"], "half_b": full_distributions["half_b"],
                   "provenance": prov}, f, indent=2, default=str)
    write_provenance(path, prov)
    print(f"Saved {len(candidates)} split-half-validated candidates to {path}")
    print(f"Saved full correlation distribution ({len(full_distributions['half_a'])} + "
          f"{len(full_distributions['half_b'])} entries) to {distribution_path}")


def _normalize_candidate(item):
    if isinstance(item, dict):
        d = dict(item)
        if "layer" not in d or "neuron_idx" not in d:
            if "neuron_id" in d:
                l, n = d["neuron_id"].split("_")
                d.setdefault("layer", int(l[1:]))
                d.setdefault("neuron_idx", int(n[1:]))
            else:
                raise ValueError(f"Candidate dict without layer/neuron_idx: {item}")
        d["layer"], d["neuron_idx"] = int(d["layer"]), int(d["neuron_idx"])
        d.setdefault("neuron_id", f"L{d['layer']}_N{d['neuron_idx']}")
        return d
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        layer, neuron_idx = int(item[0]), int(item[1])
        d = {"neuron_id": f"L{layer}_N{neuron_idx}", "layer": layer, "neuron_idx": neuron_idx}
        if len(item) >= 3:
            d["detection_correlation"] = float(item[2])
        return d
    raise ValueError(f"Unexpected candidate format: {item!r}")


def load_candidate_neurons(path="candidate_neurons.json"):
    """
    Returns a list of dicts with at least neuron_id, layer, neuron_idx (plus
    whatever correlation fields the file carries). Accepts all three
    historical formats:
      1. raw list of [layer, neuron_idx(, corr)] lists,
      2. list of dicts,
      3. {"provenance" | "metadata": {...}, "candidates" | "neurons": [...]}.
    """
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items = None
        for key in ("candidates", "neurons"):
            if key in raw:
                items = raw[key]
                break
        if items is None:
            raise ValueError(f"{path}: dict without 'candidates'/'neurons' key")
    else:
        items = raw
    return [_normalize_candidate(it) for it in items]


def load_candidate_provenance(path="candidate_neurons.json"):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw.get("provenance") or raw.get("metadata") or {}
    return {}


# ---------------------------------------------------------------------------
# Legacy single-shot detection (kept for reference / old notebooks)
# ---------------------------------------------------------------------------

def detect_candidate_neurons(model, tokenizer, baseline_prompts, layer_range, top_k=15):
    """DEPRECATED single-shot top-k detection (no split-half validation)."""
    warnings.warn("detect_candidate_neurons is deprecated; use detect_candidate_neurons_split_half",
                  DeprecationWarning, stacklevel=2)
    corr, _ = _correlate_all_neurons(model, tokenizer, list(baseline_prompts), layer_range)
    results = sorted(((l, n, r) for (l, n), r in corr.items()), key=lambda x: -abs(x[2]))
    return results[:top_k]


def save_candidate_neurons(results, baseline_prompts, seed, path="candidate_neurons.json"):
    """Legacy saver for detect_candidate_neurons output (list of tuples)."""
    prompts = list(baseline_prompts)
    prov = build_provenance(None, n_baseline_prompts=len(prompts),
                            baseline_prompt_sha256=sha256_prompts(prompts),
                            baseline_prompt_hash=sha256_prompts(prompts)[:16], seed=seed,
                            method="single_shot_topk")
    payload = {
        "provenance": prov,
        "candidates": [
            {"neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n, "detection_correlation": r}
            for l, n, r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    write_provenance(path, prov)
