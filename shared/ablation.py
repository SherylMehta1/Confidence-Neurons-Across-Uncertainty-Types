"""
Shared Tool 4: mean-ablation (the causal test), frozen-RMSNorm counterfactual,
and activation dose-response sweeps.

Conventions:
- The clamp is applied ONLY at the final token position (positions="last"),
  matching where the mean is estimated (compute_mean_activation*) and where
  detection correlated activations with entropy. positions="all" reproduces
  the historical every-position protocol for comparison only.
- Every forward pass tokenizes through shared.model_utils.tokenize_prompt
  (exactly one BOS) and runs with use_cache=False; probabilities are fp32.
- All hooks / monkeypatches are removed in try/finally.
- run_ablation_experiment emits the v4 results schema (RESULT_COLUMNS).
"""

import torch
import torch.nn.functional as F
import numpy as np

from shared import model_utils as _mu
from shared.model_utils import compute_entropy, compute_top1_prob
from shared.detection import get_neuron_activation, capture_intermediate_activations
from shared.logit_lens import direct_effect_score as _direct_effect_score
from shared.provenance import model_precision

# Results CSV schema v4 (RESULTS_SCHEMA.md). Order is the contract.
RESULT_COLUMNS = [
    "neuron_id", "layer", "neuron_idx", "category", "prompt_id",
    "orig_entropy", "ablated_entropy", "entropy_shift",
    "orig_top1_prob", "ablated_top1_prob", "direct_effect_score", "split",
    "is_control", "orig_activation", "mean_val", "mean_source", "precision",
]
MEAN_SOURCES = ("general_baseline", "category_working", "pooled_controls")


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------

def compute_mean_activation(model, tokenizer, baseline_prompts, layer_idx, neuron_idx):
    """Mean last-token activation of ONE neuron across baseline prompts
    (one forward pass per prompt). Prefer compute_mean_activations for many neurons."""
    vals = [get_neuron_activation(model, tokenizer, p, layer_idx, neuron_idx) for p in baseline_prompts]
    return float(np.mean(vals))


def compute_mean_activations(model, tokenizer, neurons, baseline_prompts, verbose=True):
    """Mean last-token activation for MANY (layer, neuron_idx) pairs with ONE
    forward pass per baseline prompt (hooks on every needed layer).
    Returns {(layer, neuron_idx): mean}."""
    neurons = [(int(l), int(n)) for l, n in neurons]
    layers = sorted({l for l, _ in neurons})
    sums = {k: 0.0 for k in neurons}
    baseline_prompts = list(baseline_prompts)
    if not baseline_prompts:
        raise ValueError("compute_mean_activations: empty baseline prompt list")
    for i, prompt in enumerate(baseline_prompts):
        captured = capture_intermediate_activations(model, tokenizer, prompt, layers)
        for l, n in neurons:
            sums[(l, n)] += float(captured[l][n])
        if verbose and (i + 1) % 20 == 0:
            print(f"  baseline means: {i + 1}/{len(baseline_prompts)} prompts")
    return {k: s / len(baseline_prompts) for k, s in sums.items()}


# ---------------------------------------------------------------------------
# Core forward passes
# ---------------------------------------------------------------------------

def _clamp_slice(positions):
    if positions == "last":
        return slice(-1, None)
    if positions == "all":
        return slice(None)
    raise ValueError(f"positions must be 'last' or 'all', got {positions!r}")


def _probs_from(outputs):
    return F.softmax(outputs.logits[0, -1, :].float(), dim=-1)


def get_probs_and_activation(model, tokenizer, prompt, layer_idx, neuron_idx):
    """Unablated pass returning (fp32 probs, the neuron's last-token
    activation) -- the activation is captured by a pre-hook on down_proj
    during the SAME forward pass, so no extra pass is needed."""
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    box = {}

    def capture(module, args):
        box["act"] = float(args[0][0, -1, neuron_idx].detach().float().item())

    handle = down_proj.register_forward_pre_hook(capture)
    try:
        inputs = _mu.tokenize_prompt(tokenizer, prompt, device=model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return _probs_from(outputs), box["act"]


def mean_ablate_and_get_probs(model, tokenizer, prompt, layer_idx, neuron_idx, mean_val,
                              positions="last"):
    """Force one neuron to mean_val (at the final position by default) for one
    forward pass; returns fp32 next-token probs."""
    return activation_sweep_and_get_probs(
        model, tokenizer, prompt, layer_idx, neuron_idx, [mean_val], positions=positions)[0]


def activation_sweep_and_get_probs(model, tokenizer, prompt, layer_idx, neuron_idx, values,
                                   positions="last"):
    """Dose-response: clamp the neuron to each value in `values` in turn (one
    forward pass per value, a single hook registration, tokenized once) and
    return the list of fp32 next-token prob tensors, in the same order."""
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    sl = _clamp_slice(positions)
    state = {"value": None}

    def hook_fn(module, args):
        modified = args[0].clone()
        modified[:, sl, neuron_idx] = state["value"]
        return (modified,) + tuple(args[1:])

    inputs = _mu.tokenize_prompt(tokenizer, prompt, device=model.device)
    handle = down_proj.register_forward_pre_hook(hook_fn)
    out = []
    try:
        with torch.no_grad():
            for v in values:
                state["value"] = float(v)
                out.append(_probs_from(model(**inputs, use_cache=False)))
    finally:
        handle.remove()
    return out


def frozen_norm_ablate_and_get_probs(model, tokenizer, prompt, layer_idx, neuron_idx, mean_val,
                                     positions="last"):
    """
    Frozen-RMSNorm counterfactual: mean-ablate the neuron but keep the FINAL
    RMSNorm's per-position scale fixed at its clean-run value, so only the
    neuron's direct (logit-direction) effect survives and the entropy-via-
    normalization pathway is blocked (Stolfo et al.'s decomposition).

    Pass 1 (clean): a forward PRE-hook on model.model.norm records
        inv_rms[pos] = 1 / sqrt(mean(x[pos]^2) + eps)   (fp32, like LlamaRMSNorm).
    Pass 2 (ablated): down_proj pre-hook clamps the neuron; model.model.norm's
        forward is replaced so output = weight * (x_ablated * clean_inv_rms),
        same positions and tokenization. The replacement mirrors
        LlamaRMSNorm.forward (fp32 math, cast back to the input dtype, then
        multiply by weight). Everything is restored in try/finally.
    Returns fp32 probs.
    """
    norm = model.model.norm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    inputs = _mu.tokenize_prompt(tokenizer, prompt, device=model.device)
    captured = {}

    def capture_pre(module, args):
        x = args[0].float()
        captured["inv_rms"] = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).detach()

    h = norm.register_forward_pre_hook(capture_pre)
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        h.remove()
    inv_rms = captured["inv_rms"]

    down_proj = model.model.layers[layer_idx].mlp.down_proj
    sl = _clamp_slice(positions)

    def ablate_pre(module, args):
        modified = args[0].clone()
        modified[:, sl, neuron_idx] = float(mean_val)
        return (modified,) + tuple(args[1:])

    def frozen_forward(hidden_states):
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32) * inv_rms
        return norm.weight * x.to(input_dtype)

    had_instance_forward = "forward" in norm.__dict__
    saved_forward = norm.__dict__.get("forward")
    ablate_handle = down_proj.register_forward_pre_hook(ablate_pre)
    try:
        norm.forward = frozen_forward
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        ablate_handle.remove()
        if had_instance_forward:
            norm.forward = saved_forward
        else:
            norm.__dict__.pop("forward", None)
    return _probs_from(outputs)


# ---------------------------------------------------------------------------
# Experiment rows
# ---------------------------------------------------------------------------

def run_ablation_experiment(model, tokenizer, prompts, layer_idx, neuron_idx, mean_val, category,
                            split="working", is_control=False, mean_source="general_baseline",
                            direct_effect_score=None, precision=None, positions="last",
                            verbose=False):
    """
    Mean-ablate one neuron on each prompt record (dicts with prompt_id and
    chat_formatted_prompt) and return one v4-schema row per prompt
    (RESULT_COLUMNS). orig_activation is captured during the original pass.
    direct_effect_score (weights-only, constant per neuron) is computed here
    if not supplied. `split` must stay in {working, held_out}; use is_control
    for matched-control prompts.
    """
    if split not in ("working", "held_out"):
        raise ValueError(f"split must be 'working' or 'held_out' (use is_control for controls), got {split!r}")
    if mean_source not in MEAN_SOURCES:
        raise ValueError(f"mean_source must be one of {MEAN_SOURCES}, got {mean_source!r}")
    if direct_effect_score is None:
        direct_effect_score = _direct_effect_score(model, layer_idx, neuron_idx)
    if precision is None:
        precision = model_precision(model)
    mean_val = float(mean_val)

    rows = []
    for i, rec in enumerate(prompts):
        prompt_text = rec["chat_formatted_prompt"]
        orig_probs, orig_act = get_probs_and_activation(model, tokenizer, prompt_text, layer_idx, neuron_idx)
        orig_entropy = compute_entropy(orig_probs)
        orig_top1 = compute_top1_prob(orig_probs)

        ablated_probs = mean_ablate_and_get_probs(model, tokenizer, prompt_text, layer_idx, neuron_idx,
                                                  mean_val, positions=positions)
        ablated_entropy = compute_entropy(ablated_probs)
        ablated_top1 = compute_top1_prob(ablated_probs)

        rows.append({
            "neuron_id": f"L{layer_idx}_N{neuron_idx}",
            "layer": int(layer_idx),
            "neuron_idx": int(neuron_idx),
            "category": category,
            "prompt_id": rec["prompt_id"],
            "orig_entropy": orig_entropy,
            "ablated_entropy": ablated_entropy,
            "entropy_shift": ablated_entropy - orig_entropy,
            "orig_top1_prob": orig_top1,
            "ablated_top1_prob": ablated_top1,
            "direct_effect_score": float(direct_effect_score),
            "split": split,
            "is_control": bool(is_control),
            "orig_activation": orig_act,
            "mean_val": mean_val,
            "mean_source": mean_source,
            "precision": precision,
        })
        if verbose and (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(prompts)} prompts")
    return rows
