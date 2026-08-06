"""
Shared Tool 3 + Phase 2: extract neuron activations via raw PyTorch hooks,
and run the candidate-neuron detection scan.

Reference: Gurnee et al. (2024), "Universal Neurons in GPT-2" for the activation
extraction methodology; Stolfo et al. (2024) for the correlation-based detection approach.

BUILT AND RUN ONCE, TOGETHER. Output (candidate_neurons.json) is then frozen and shared.
"""

import json
import numpy as np
import torch

from shared.model_utils import get_next_token_probs, compute_entropy


def get_neuron_activation(model, tokenizer, prompt, layer_idx, neuron_idx):
    """Tool 3: hook into one MLP layer, capture one neuron's activation at the last token position."""
    activations = {}

    def hook_fn(module, input, output):
        activations["val"] = output.detach()

    handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()  # always remove hooks when done

    return activations["val"][0, -1, neuron_idx].item()


def detect_candidate_neurons(model, tokenizer, baseline_prompts, layer_range, top_k=15):
    """
    Phase 2: for every neuron in layer_range, correlate its activation with entropy
    across baseline_prompts (a general-purpose prompt set, NOT category-specific).

    layer_range should be restricted to late layers (roughly the last third of the
    model) per prior work (Stolfo et al., Context Copying Modulation) rather than
    scanning every layer — this keeps the search bounded and principled, not a blind scan.

    Returns the top_k (layer, neuron, correlation) tuples by absolute correlation.
    """
    # Precompute entropy for each baseline prompt once
    entropies = []
    for prompt in baseline_prompts:
        probs = get_next_token_probs(model, tokenizer, prompt)
        entropies.append(compute_entropy(probs))

    hidden_dim = model.config.hidden_size
    results = []

    for layer_idx in layer_range:
        # Capture the WHOLE layer's activations once per prompt (much faster than
        # re-running the model once per neuron)
        layer_acts = []  # shape: [n_prompts, hidden_dim]
        for prompt in baseline_prompts:
            activations = {}

            def hook_fn(module, input, output):
                activations["val"] = output.detach()

            handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            handle.remove()
            layer_acts.append(activations["val"][0, -1, :].cpu().numpy())

        layer_acts = np.array(layer_acts)  # [n_prompts, hidden_dim]

        for neuron_idx in range(hidden_dim):
            acts_for_neuron = layer_acts[:, neuron_idx]
            corr = np.corrcoef(acts_for_neuron, entropies)[0, 1]
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
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    from shared.model_utils import load_model

    model, tokenizer = load_model()

    # Replace with a real general-purpose baseline prompt set (a few hundred generic
    # sentences) before running for real — this is just a placeholder.
    baseline_prompts = [
        "The weather today is",
        "My favorite hobby is",
        "The history of Rome began",
    ]

    # Restrict to late layers per Stolfo et al. — adjust range once you've read their
    # paper's specifics on where they found effects in comparable-sized models.
    num_layers = model.config.num_hidden_layers
    late_layers = range(int(num_layers * 0.66), num_layers)

    results = detect_candidate_neurons(model, tokenizer, baseline_prompts, late_layers, top_k=15)
    save_candidate_neurons(results)
