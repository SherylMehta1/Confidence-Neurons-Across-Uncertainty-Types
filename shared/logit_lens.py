"""
Shared Tool 5: logit lens / direct-effect decomposition, used in Phase 3 (mechanism check).

Reference: Stolfo et al. (2024) for the null-space / indirect-effect finding;
nostalgebraist (2020), "interpreting GPT: the logit lens" for the underlying technique.

This tells you whether a candidate neuron pushes a SPECIFIC token directly (large,
sensible top tokens from direct_effect_logits) or works indirectly through the
normalization pathway (small/meaningless direct effect, but large mean-ablation
effect on entropy from ablation.py) — this is what tests H4.
"""

import torch


def direct_effect_logits(model, layer_idx, neuron_idx):
    """
    Project one neuron's output weight vector directly through the model's unembedding
    matrix, bypassing all later layers, to see what token it would predict "on its own."
    """
    unembed = model.get_output_embeddings().weight  # [vocab_size, hidden_dim]
    # down_proj maps the MLP's intermediate activations back into the residual stream;
    # column neuron_idx is this neuron's individual contribution direction
    neuron_output_vector = model.model.layers[layer_idx].mlp.down_proj.weight[:, neuron_idx]
    direct_logits = unembed @ neuron_output_vector
    return direct_logits


def top_direct_effect_tokens(model, tokenizer, layer_idx, neuron_idx, k=5):
    """Convenience: return the top-k tokens this neuron pushes toward directly."""
    logits = direct_effect_logits(model, layer_idx, neuron_idx)
    top = torch.topk(logits, k)
    return [(tokenizer.decode(idx), val.item()) for val, idx in zip(top.values, top.indices)]


def direct_effect_score(model, layer_idx, neuron_idx):
    """
    A single summary number for direct effect magnitude: the max absolute logit value
    a neuron produces via the direct pathway. Small = likely indirect/null-space mechanism.
    Large = likely a direct, token-specific effect.
    """
    logits = direct_effect_logits(model, layer_idx, neuron_idx)
    return torch.max(torch.abs(logits)).item()
