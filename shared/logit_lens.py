"""
Shared Tool 5: logit lens / direct-effect decomposition.

A neuron's direct (weights-only) effect on the logits is the projection of its
down_proj output column, with the final RMSNorm's gamma folded in
(w_tilde = gamma * w_out, because every residual vector is scaled by gamma
before the unembedding), through the unembedding matrix W_U:

    direct_logits = W_U @ (gamma * w_out)          [vocab]

computed in fp32 on the unembedding's device. This is a pure function of the
weights -- it ignores the actual RMSNorm denominator and the activation on a
given prompt -- so it is CONSTANT per neuron.

direct_effect_score(model, layer, neuron) = max_vocab |direct_logits|: the
largest direct logit push (per unit activation) on any token. It is the
weights-only, constant-per-neuron number stored in the results CSV.
direct_effect_on_token() returns the fp32 direct logit for one token so
callers can build per-prompt variants (e.g. multiply by orig_activation /
clean RMS to get the actual contribution to the top-1 token's logit).
"""

import torch


def _dequantized_down_proj_weight(model, layer_idx):
    """down_proj.weight as a dense tensor (dequantizes bitsandbytes 4-bit lazily)."""
    w = model.model.layers[layer_idx].mlp.down_proj.weight
    if hasattr(w, "quant_state"):
        import bitsandbytes.functional as bnb_F  # only needed for quantized models
        w = bnb_F.dequantize_4bit(w, w.quant_state)
    return w


def neuron_output_vector(model, layer_idx, neuron_idx, fold_gamma=True):
    """fp32 w_out (optionally gamma-folded) on the unembedding's device."""
    unembed = model.get_output_embeddings().weight
    w = _dequantized_down_proj_weight(model, layer_idx)[:, neuron_idx].detach()
    w = w.to(unembed.device, torch.float32)
    if fold_gamma:
        w = model.model.norm.weight.detach().to(unembed.device, torch.float32) * w
    return w


def direct_effect_logits(model, layer_idx, neuron_idx):
    """fp32 [vocab] direct logits of the gamma-folded output vector."""
    unembed = model.get_output_embeddings().weight.detach()
    folded = neuron_output_vector(model, layer_idx, neuron_idx)
    with torch.no_grad():
        return unembed.to(torch.float32) @ folded


def top_direct_effect_tokens(model, tokenizer, layer_idx, neuron_idx, k=5, largest=True):
    logits = direct_effect_logits(model, layer_idx, neuron_idx)
    top = torch.topk(logits, k, largest=largest)
    return [(tokenizer.decode([int(idx)]), float(val)) for val, idx in zip(top.values, top.indices)]


def direct_effect_score(model, layer_idx, neuron_idx):
    """max over the vocabulary of |direct logit| -- weights-only, fp32,
    constant per neuron (see module docstring)."""
    return float(torch.max(torch.abs(direct_effect_logits(model, layer_idx, neuron_idx))).item())


def direct_effect_on_token(model, layer_idx, neuron_idx, token_id):
    """fp32 direct logit of one token: W_U[token_id] . (gamma * w_out)."""
    unembed = model.get_output_embeddings().weight.detach()
    folded = neuron_output_vector(model, layer_idx, neuron_idx)
    with torch.no_grad():
        return float(torch.dot(unembed[int(token_id)].to(torch.float32), folded).item())
