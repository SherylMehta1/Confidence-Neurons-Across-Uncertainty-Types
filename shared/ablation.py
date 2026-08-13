"""
Shared Tool 4: mean-ablation, the causal test used in Phase 4.

Fixes applied (Phase 1, post-audit):
- Ablation now clamps ONLY the final token position, matching the position
  used to compute the mean in compute_mean_activation() and the position
  used in detection's correlation criterion. Previously the clamp applied
  to every position while the mean was computed from the last position only
  -- a mismatched intervention, not true mean-ablation (audit finding).
- Hooks removed via try/finally.
- run_ablation_experiment now also computes and emits direct_effect_score,
  restoring the 12-column schema RESULTS_SCHEMA.md requires.
"""

import torch
import numpy as np

from shared.model_utils import get_next_token_probs, compute_entropy, compute_top1_prob
from shared.detection import get_neuron_activation
from shared.logit_lens import direct_effect_score


def compute_mean_activation(model, tokenizer, baseline_prompts, layer_idx, neuron_idx):
    """Mean activation at the FINAL token position, across baseline prompts.
    This position must match where the ablation clamp is applied below."""
    vals = [
        get_neuron_activation(model, tokenizer, p, layer_idx, neuron_idx)
        for p in baseline_prompts
    ]
    return float(np.mean(vals))


def mean_ablate_and_get_probs(model, tokenizer, prompt, layer_idx, neuron_idx, mean_val):
    """
    Force a neuron to mean_val, ONLY at the final token position (matching
    compute_mean_activation's measurement position and detection.py's
    correlation criterion), for this forward pass.
    """
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    handle = None

    def hook_fn(module, args):
        modified = args[0].clone()
        modified[:, -1, neuron_idx] = mean_val   # last position ONLY -- not [:, :, neuron_idx]
        return (modified,) + args[1:]

    try:
        handle = down_proj.register_forward_pre_hook(hook_fn)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        if handle is not None:
            handle.remove()

    logits = outputs.logits[0, -1, :].float()
    return torch.nn.functional.softmax(logits, dim=-1)


def run_ablation_experiment(model, tokenizer, prompts, layer_idx, neuron_idx, mean_val, category, split="working"):
    """
    Core Phase 4 causal test. Now also computes direct_effect_score once per
    neuron (constant across prompts, since it depends only on model weights
    -- see logit_lens.py) and includes it in every row, restoring the full
    12-column schema.
    """
    de_score = direct_effect_score(model, layer_idx, neuron_idx)

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
            "direct_effect_score": de_score,
            "split": split,
        })
    return rows