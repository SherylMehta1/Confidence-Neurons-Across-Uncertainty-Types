"""
Shared Tool 4: mean-ablation, the causal test used in Phase 4.

Reference: Menta et al. (2025), "Transformers Don't Need LayerNorm at Inference Time" —
use THEIR protocol: replace a neuron's activation with its dataset average, not zero.
Zero-ablation is a different (and less validated, for this purpose) intervention.

BUILT TOGETHER. Each person calls these same functions on their own category's prompts.
"""

import torch
import numpy as np

from shared.model_utils import get_next_token_probs, compute_entropy, compute_top1_prob
from shared.detection import get_neuron_activation


def compute_mean_activation(model, tokenizer, baseline_prompts, layer_idx, neuron_idx):
    """Compute a neuron's average activation across a baseline prompt set.

    baseline_prompts here should be a reasonably large, general prompt set (per Menta
    et al., ~1000 sequences in the original protocol — scale down for a first pass,
    but keep it general-purpose, not category-specific, so the "mean" represents the
    neuron's typical/default behavior).
    """
    vals = [
        get_neuron_activation(model, tokenizer, p, layer_idx, neuron_idx)
        for p in baseline_prompts
    ]
    return float(np.mean(vals))


def mean_ablate_and_get_probs(model, tokenizer, prompt, layer_idx, neuron_idx, mean_val):
    """
    Force a neuron to mean_val for this forward pass, return the resulting next-token probs.

    NOTE: hooks down_proj's INPUT (intermediate space, size intermediate_size),
    matching the neuron space used in detection.py and logit_lens.py. An earlier
    version hooked the mlp module's OUTPUT (hidden_size space) -- that no longer
    matches candidate_neurons.json's neuron_idx values (which now range up to
    intermediate_size-1, e.g. 14335 for Llama-3.1-8B) and would throw an
    out-of-bounds error.
    """
    down_proj = model.model.layers[layer_idx].mlp.down_proj

    def hook_fn(module, args):
        modified = args[0].clone()
        modified[:, :, neuron_idx] = mean_val
        return (modified,) + args[1:]

    handle = down_proj.register_forward_pre_hook(hook_fn)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    handle.remove()

    logits = outputs.logits[0, -1, :]
    return torch.nn.functional.softmax(logits, dim=-1)


def run_ablation_experiment(model, tokenizer, prompts, layer_idx, neuron_idx, mean_val, category, split="working"):
    """
    Run the core Phase 4 causal test: for each prompt, compare original vs. ablated
    entropy/top-1 probability for one candidate neuron.

    Returns a list of dicts matching RESULTS_SCHEMA.md — ready to write to results.csv.
    """
    rows = []
    for prompt_record in prompts:
        prompt_text = prompt_record["chat_formatted_prompt"]
        prompt_id = prompt_record["prompt_id"]

        orig_probs = get_next_token_probs(model, tokenizer, prompt_text)
        orig_entropy = compute_entropy(orig_probs)
        orig_top1 = compute_top1_prob(orig_probs)

        ablated_probs = mean_ablate_and_get_probs(model, tokenizer, prompt_text, layer_idx, neuron_idx, mean_val)
        ablated_entropy = compute_entropy(ablated_probs)
        ablated_top1 = compute_top1_prob(ablated_probs)

        rows.append({
            "neuron_id": f"L{layer_idx}_N{neuron_idx}",
            "layer": layer_idx,
            "neuron_idx": neuron_idx,
            "category": category,
            "prompt_id": prompt_id,
            "orig_entropy": orig_entropy,
            "ablated_entropy": ablated_entropy,
            "entropy_shift": ablated_entropy - orig_entropy,
            "orig_top1_prob": orig_top1,
            "ablated_top1_prob": ablated_top1,
            "split": split,
        })
    return rows
