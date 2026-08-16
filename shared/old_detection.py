"""
Shared Tool 3 + Phase 2: extract neuron activations via raw PyTorch hooks,
run candidate-neuron detection.

Fixes applied (Phase 1, post-audit):
- add_special_tokens=False on tokenization (matches model_utils.py fix,
  avoids double-BOS).
- Hooks now removed via try/finally, so a mid-run exception can't leave a
  stale hook attached and silently contaminate later "unablated" measurements.
- save_candidate_neurons now stores baseline-prompt count, a hash of the
  baseline prompt list, and the random seed used, for provenance/reproducibility.
"""

import hashlib
import json
import numpy as np
import torch

from shared.model_utils import get_next_token_probs, compute_entropy


def capture_intermediate_activations(model, tokenizer, prompt, layer_indices):
    """Captures the MLP intermediate activation (down_proj's INPUT) for every
    layer in layer_indices, in ONE forward pass, at the last token position."""
    captured = {}
    handles = []

    def make_pre_hook(layer_idx):
        def hook_fn(module, args):
            captured[layer_idx] = args[0].detach()[0, -1, :].to(torch.float32).cpu().numpy()
        return hook_fn

    try:
        for layer_idx in layer_indices:
            down_proj = model.model.layers[layer_idx].mlp.down_proj
            handles.append(down_proj.register_forward_pre_hook(make_pre_hook(layer_idx)))

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        # try/finally guarantees hooks are removed even if the forward pass
        # raises -- prevents a crashed run from leaving hooks attached to
        # contaminate subsequent, supposedly-unmodified forward passes.
        for h in handles:
            h.remove()

    return captured


def get_neuron_activation(model, tokenizer, prompt, layer_idx, neuron_idx):
    captured = capture_intermediate_activations(model, tokenizer, prompt, [layer_idx])
    return float(captured[layer_idx][neuron_idx])


def detect_candidate_neurons(model, tokenizer, baseline_prompts, layer_range, top_k=15):
    entropies = np.array([
        compute_entropy(get_next_token_probs(model, tokenizer, p))
        for p in baseline_prompts
    ])

    layer_range = list(layer_range)
    intermediate_size = model.config.intermediate_size

    acts_by_layer = {
        l: np.zeros((len(baseline_prompts), intermediate_size), dtype=np.float32)
        for l in layer_range
    }

    for i, prompt in enumerate(baseline_prompts):
        captured = capture_intermediate_activations(model, tokenizer, prompt, layer_range)
        for l in layer_range:
            acts_by_layer[l][i, :] = captured[l]
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(baseline_prompts)} baseline prompts")

    results = []
    for layer_idx in layer_range:
        layer_acts = acts_by_layer[layer_idx]
        for neuron_idx in range(intermediate_size):
            col = layer_acts[:, neuron_idx]
            if np.std(col) < 1e-8:
                continue
            corr = np.corrcoef(col, entropies)[0, 1]
            if not np.isnan(corr):
                results.append((layer_idx, neuron_idx, float(corr)))

    results.sort(key=lambda x: -abs(x[2]))
    return results[:top_k]


def save_candidate_neurons(results, baseline_prompts, seed, path="candidate_neurons.json"):
    """
    Now records provenance: baseline prompt count, a hash of the exact
    baseline prompt list, and the seed used -- so the candidate list is
    independently auditable/reproducible (audit: "Detection provenance").
    """
    prompt_blob = json.dumps(sorted(baseline_prompts)).encode("utf-8")
    prompt_hash = hashlib.sha256(prompt_blob).hexdigest()[:16]

    payload = {
        "provenance": {
            "n_baseline_prompts": len(baseline_prompts),
            "baseline_prompt_hash": prompt_hash,
            "seed": seed,
        },
        "candidates": [
            {"neuron_id": f"L{layer}_N{neuron}", "layer": layer, "neuron_idx": neuron, "detection_correlation": corr}
            for layer, neuron, corr in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(payload['candidates'])} candidate neurons to {path} (hash={prompt_hash}, seed={seed})")


def load_candidate_neurons(path="candidate_neurons.json"):
    """
    Always returns a list of dicts with keys: neuron_id, layer, neuron_idx,
    detection_correlation. Handles the new provenance-wrapped format, the
    old flat-dict-list format, and the oldest raw-tuple-list format.
    """
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "candidates" in raw:
        return raw["candidates"]

    normalized = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, (list, tuple)):
            layer, neuron_idx, corr = item[0], item[1], item[2]
            normalized.append({
                "neuron_id": f"L{layer}_N{neuron_idx}",
                "layer": layer, "neuron_idx": neuron_idx,
                "detection_correlation": corr,
            })
        else:
            raise ValueError(f"Unexpected candidate format: {item}")
    return normalized