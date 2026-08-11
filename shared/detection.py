"""
Shared Tool 3 + Phase 2: extract neuron activations via raw PyTorch hooks,
and run the candidate-neuron detection scan.

Reference: Gurnee et al. (2024), "Universal Neurons in GPT-2" for the activation
extraction methodology; Stolfo et al. (2024) for the correlation-based detection approach.

NOTE: "neuron" here means one unit of the MLP's INTERMEDIATE activation space
(the down_proj's INPUT, size intermediate_size) -- NOT the mlp module's overall
OUTPUT (size hidden_size). This matches the indexing used in logit_lens.py's
down_proj.weight[:, neuron_idx]. An earlier version of this file hooked the
whole mlp module's output instead, which silently used a different neuron
space than logit_lens.py -- fixed here.

BUILT AND RUN ONCE, TOGETHER. Output (candidate_neurons.json) is then frozen and shared.
"""

import json
import numpy as np
import torch

from shared.model_utils import get_next_token_probs, compute_entropy


def capture_intermediate_activations(model, tokenizer, prompt, layer_indices):
    """
    Captures the MLP intermediate activation (down_proj's INPUT) for every
    layer in layer_indices, in ONE forward pass, at the last token position.
    """
    captured = {}
    handles = []

    def make_pre_hook(layer_idx):
        def hook_fn(module, args):
            captured[layer_idx] = args[0].detach()[0, -1, :].to(torch.float32).cpu().numpy()
        return hook_fn

    for layer_idx in layer_indices:
        down_proj = model.model.layers[layer_idx].mlp.down_proj
        handles.append(down_proj.register_forward_pre_hook(make_pre_hook(layer_idx)))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return captured


def get_neuron_activation(model, tokenizer, prompt, layer_idx, neuron_idx):
    """Tool 3: single-neuron convenience wrapper (correct intermediate-space indexing)."""
    captured = capture_intermediate_activations(model, tokenizer, prompt, [layer_idx])
    return float(captured[layer_idx][neuron_idx])


def detect_candidate_neurons(model, tokenizer, baseline_prompts, layer_range, top_k=15):
    """
    Phase 2: for every neuron (intermediate-space) in layer_range, correlate its
    activation with entropy across baseline_prompts (general-purpose, NOT
    category-specific). One forward pass per prompt, all target layers captured
    simultaneously -- not one forward pass per layer.
    """
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


def save_candidate_neurons(results, path="candidate_neurons.json"):
    payload = [
        {"neuron_id": f"L{layer}_N{neuron}", "layer": layer, "neuron_idx": neuron, "detection_correlation": corr}
        for layer, neuron, corr in results
    ]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(payload)} candidate neurons to {path}")


def load_candidate_neurons(path="candidate_neurons.json"):
    """
    Always returns a list of dicts with keys: neuron_id, layer, neuron_idx,
    detection_correlation -- regardless of whether the file on disk was saved
    as dicts (via save_candidate_neurons) or as raw [layer, neuron, corr] lists.
    """
    with open(path) as f:
        raw = json.load(f)

    normalized = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, (list, tuple)):
            layer, neuron_idx, corr = item[0], item[1], item[2]
            normalized.append({
                "neuron_id": f"L{layer}_N{neuron_idx}",
                "layer": layer,
                "neuron_idx": neuron_idx,
                "detection_correlation": corr,
            })
        else:
            raise ValueError(f"Unexpected candidate format: {item}")
    return normalized


if __name__ == "__main__":
    from shared.model_utils import load_model

    model, tokenizer = load_model()

    baseline_prompts = [
        "The weather today is", "My favorite hobby is", "The history of Rome began",
    ]

    num_layers = model.config.num_hidden_layers
    late_layers = range(int(num_layers * 0.66), num_layers)

    results = detect_candidate_neurons(model, tokenizer, baseline_prompts, late_layers, top_k=15)
    save_candidate_neurons(results)