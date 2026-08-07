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
import bitsandbytes.functional as bnb_F


def direct_effect_logits(model, layer_idx, neuron_idx):
    """
    Project one neuron's output weight vector directly through the model's unembedding
    matrix, bypassing all later layers, to see what token it would predict "on its own."

    Handles both quantized (4-bit) and full-precision weight tensors.
    """
    unembed = model.get_output_embeddings().weight
    down_proj = model.model.layers[layer_idx].mlp.down_proj

    if hasattr(down_proj.weight, "quant_state"):
        # 4-bit quantized weight — dequantize this one matrix on demand
        weight = bnb_F.dequantize_4bit(down_proj.weight, down_proj.weight.quant_state)
    else:
        # full precision — use as-is
        weight = down_proj.weight

    neuron_output_vector = weight[:, neuron_idx]
    direct_logits = unembed @ neuron_output_vector.to(unembed.dtype)
    return direct_logits


def top_direct_effect_tokens(model, tokenizer, layer_idx, neuron_idx, k=5):
    logits = direct_effect_logits(model, layer_idx, neuron_idx)
    top = torch.topk(logits, k)
    return [(tokenizer.decode(idx), val.item()) for val, idx in zip(top.values, top.indices)]


def direct_effect_score(model, layer_idx, neuron_idx):
    logits = direct_effect_logits(model, layer_idx, neuron_idx)
    return torch.max(torch.abs(logits)).item()