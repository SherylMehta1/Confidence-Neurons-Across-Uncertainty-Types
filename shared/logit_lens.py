"""
Shared Tool 5: logit lens / direct-effect decomposition, used in Phase 3.

Fix applied (Phase 1, post-audit):
- Now folds in the final RMSNorm's gamma (weight) parameter before
  projecting through the unembedding matrix. The previous version projected
  the raw down_proj output-weight vector directly, ignoring that every
  residual-stream vector passes through RMSNorm scaling before reaching the
  unembedding in the real forward pass -- omitting this distorted the
  direction of the projection, damaging the very direct-vs-indirect
  decomposition (H4) this tool exists to test.
"""

import torch
import bitsandbytes.functional as bnb_F


def direct_effect_logits(model, layer_idx, neuron_idx):
    """
    Project one neuron's output weight vector -- RMSNorm-folded -- directly
    through the model's unembedding matrix, bypassing all later layers.
    """
    unembed = model.get_output_embeddings().weight  # [vocab_size, hidden_dim]
    down_proj = model.model.layers[layer_idx].mlp.down_proj

    if hasattr(down_proj.weight, "quant_state"):
        weight = bnb_F.dequantize_4bit(down_proj.weight, down_proj.weight.quant_state)
    else:
        weight = down_proj.weight

    neuron_output_vector = weight[:, neuron_idx].to(unembed.dtype)

    # Fold in the final RMSNorm's gamma parameter -- this is the fix.
    # model.model.norm is Llama's final RMSNorm before the LM head.
    rmsnorm_gamma = model.model.norm.weight.to(unembed.dtype)
    folded_vector = rmsnorm_gamma * neuron_output_vector

    direct_logits = unembed @ folded_vector
    return direct_logits


def top_direct_effect_tokens(model, tokenizer, layer_idx, neuron_idx, k=5):
    logits = direct_effect_logits(model, layer_idx, neuron_idx)
    top = torch.topk(logits, k)
    return [(tokenizer.decode(idx), val.item()) for val, idx in zip(top.values, top.indices)]


def direct_effect_score(model, layer_idx, neuron_idx):
    logits = direct_effect_logits(model, layer_idx, neuron_idx)
    return torch.max(torch.abs(logits)).item()