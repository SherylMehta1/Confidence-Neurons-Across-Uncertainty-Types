"""
Shared Tools 1-2: load the model, tokenize, run forward passes, measure entropy.

Key points:
- tokenize_prompt() is the ONLY tokenization entry point used by detection and
  ablation. It adds BOS exactly once: if the prompt already starts with the
  tokenizer's BOS token (as every chat-template-formatted prompt does) nothing
  is added; if it does not (raw baseline sentences), BOS is prepended. This
  fixes both the historical double-BOS (add_special_tokens=True on templated
  prompts) and the no-BOS (add_special_tokens=False on raw prompts) bugs.
  Tests monkeypatch this single function to bypass real tokenizers.
- Probabilities/entropy are computed in fp32, upcast from the model's native
  precision, so bf16 quantization does not floor the measured effects.
- load_model() tags the model with `cn_precision` ("bf16" | "nf4") so every
  artifact can record which precision produced it.
"""

import torch
import torch.nn.functional as F

from shared.prompt_format import format_chat_prompt  # noqa: F401  (re-exported)


DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def resolve_model_id(model_id=None):
    """Model id precedence: explicit arg > CN_MODEL_ID env var > gated default.
    Set CN_MODEL_ID=unsloth/Meta-Llama-3.1-8B-Instruct (ungated mirror of the same
    weights) on machines without an HF token."""
    import os
    return model_id or os.environ.get("CN_MODEL_ID") or DEFAULT_MODEL_ID


def load_model(model_id=None, quantize=False, dtype=None,
               device_map="auto", **kwargs):
    """
    quantize=False (default): unquantized weights in `dtype` (default bf16).
        Required for any measurement reported as a real result.
    quantize=True: 4-bit NF4 via bitsandbytes (imported lazily, so the package
        is only needed when actually quantizing). Pipeline development only.

    `dtype` is passed as `dtype=` to from_pretrained, falling back to the
    pre-v5 `torch_dtype=` spelling if the installed transformers rejects it.
    Sets model.cn_precision ("nf4" | "bf16" | "fp16" | "fp32") and
    model.cn_model_id for provenance.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = resolve_model_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = dtype or torch.bfloat16
    if device_map is not None:
        try:
            import accelerate  # noqa: F401  (device_map needs it)
        except ImportError:
            import warnings
            warnings.warn("accelerate not installed; loading without device_map (single device)")
            device_map = None

    if quantize:
        from transformers import BitsAndBytesConfig  # needs bitsandbytes at load time
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map=device_map, **kwargs,
        )
        precision = "nf4"
    else:
        model = _from_pretrained_with_dtype(AutoModelForCausalLM, model_id, dtype,
                                            device_map=device_map, **kwargs)
        precision = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(dtype, str(dtype))

    model.eval()
    model.cn_precision = precision
    model.cn_model_id = model_id
    return model, tokenizer


def _from_pretrained_with_dtype(cls, model_id, dtype, **kwargs):
    """transformers >=5 renamed torch_dtype -> dtype; 4.x accepts torch_dtype
    (and recent 4.x also accept dtype). Try the new name first."""
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def tokenize_prompt(tokenizer, prompt, device=None):
    """
    Tokenize `prompt` so that the sequence starts with exactly ONE BOS token.
    Returns a dict {"input_ids": LongTensor[1, T], "attention_mask": LongTensor[1, T]}
    on `device` (if given).

    Logic: encode with add_special_tokens=False; if the first id is not
    bos_token_id (and the tokenizer has one), prepend it. Prompts produced by
    format_chat_prompt() already begin with <|begin_of_text|>, so they are
    left alone; raw sentences get BOS added once.
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and (input_ids.shape[1] == 0 or int(input_ids[0, 0]) != bos):
        bos_col = torch.full((input_ids.shape[0], 1), bos, dtype=input_ids.dtype)
        input_ids = torch.cat([bos_col, input_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def forward_logits(model, tokenizer, prompt):
    """One forward pass (no KV cache); returns fp32 next-token logits [vocab]."""
    inputs = tokenize_prompt(tokenizer, prompt, device=model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    return outputs.logits[0, -1, :].float()


def get_next_token_probs(model, tokenizer, prompt):
    """Run a forward pass, return the fp32 probability distribution over the
    next token. Logits are upcast to fp32 BEFORE softmax (bf16 logits
    quantize the distribution coarsely enough to swallow the effect sizes
    being measured)."""
    return F.softmax(forward_logits(model, tokenizer, prompt), dim=-1)


def compute_entropy(probs):
    """Shannon entropy (nats) in fp32 via xlogy, which is exactly 0 where p=0
    (no epsilon, no NaN). Accepts a torch tensor or array-like."""
    p = torch.as_tensor(probs).float()
    p = torch.clamp(p, min=0.0)
    return float(-torch.special.xlogy(p, p).sum().item())


def compute_top1_prob(probs):
    return float(torch.max(torch.as_tensor(probs).float()).item())


if __name__ == "__main__":
    model, tokenizer = load_model(quantize=False)

    confident_prompt = format_chat_prompt(tokenizer, "The capital of France is")
    ambiguous_prompt = format_chat_prompt(tokenizer, "Tomorrow we should go to the")

    p1 = get_next_token_probs(model, tokenizer, confident_prompt)
    p2 = get_next_token_probs(model, tokenizer, ambiguous_prompt)

    e1, e2 = compute_entropy(p1), compute_entropy(p2)
    print(f"Confident prompt entropy: {e1:.6f}")
    print(f"Ambiguous prompt entropy: {e2:.6f}")
    assert e2 > e1, "Sanity check failed."
    print("Sanity check passed.")
