"""
Shared Tools 1-2: load the model, run forward passes, measure entropy.

Fixes applied (Phase 1, post-audit):
- Entropy is now computed in fp32, upcast from the model's native precision,
  before softmax. This removes the bf16 quantization floor that was making
  measured "effects" indistinguishable from measurement noise.
- Tokenization now uses add_special_tokens=False when the input string
  already contains a chat-template-inserted BOS token, avoiding a double-BOS
  on every measured forward pass.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_id="meta-llama/Llama-3.1-8B-Instruct", quantize=True):
    """
    quantize=True: 4-bit loading, fits Kaggle's free 16GB GPUs. Use for
    pipeline development / early testing ONLY.
    quantize=False: full precision (bf16). Required for any measurement
    that will be reported as a real result -- see audit finding on
    quantization-floor confound. Needs ~16GB+ VRAM just for weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )

    model.eval()
    return model, tokenizer


def format_chat_prompt(tokenizer, user_message):
    """Wrap a raw prompt in the Llama-3.1 chat template. This already inserts
    the BOS token -- do NOT add it again at tokenization time (see get_next_token_probs)."""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_next_token_probs(model, tokenizer, prompt):
    """
    Run a forward pass, return the fp32 probability distribution over the next token.

    add_special_tokens=False: the prompt string, if produced by
    format_chat_prompt(), already contains <|begin_of_text|> from the chat
    template. Tokenizing with the default add_special_tokens=True would
    prepend a SECOND BOS token, putting every measurement slightly
    off-distribution (audit: "Double BOS on every measured forward pass").
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Upcast to fp32 BEFORE softmax -- this is the core fix. Computing
    # entropy from bf16/fp16 logits quantizes the probability distribution
    # to a grid coarse enough to swallow the actual effect sizes being measured.
    logits = outputs.logits[0, -1, :].float()
    return F.softmax(logits, dim=-1)


def compute_entropy(probs):
    """
    Shannon entropy of a probability distribution, computed via log_softmax-
    style safe log to avoid the old +1e-10 epsilon (which can underflow and
    produce NaN in fp16 -- no longer relevant now that probs is fp32 from
    get_next_token_probs, but this is written defensively regardless).
    """
    log_probs = torch.log(torch.clamp(probs, min=1e-12))
    return -torch.sum(probs * log_probs).item()


def compute_top1_prob(probs):
    return torch.max(probs).item()


if __name__ == "__main__":
    model, tokenizer = load_model(quantize=False)  # full precision for real smoke test

    confident_prompt = format_chat_prompt(tokenizer, "The capital of France is")
    ambiguous_prompt = format_chat_prompt(tokenizer, "Tomorrow we should go to the")

    p1 = get_next_token_probs(model, tokenizer, confident_prompt)
    p2 = get_next_token_probs(model, tokenizer, ambiguous_prompt)

    e1, e2 = compute_entropy(p1), compute_entropy(p2)
    print(f"Confident prompt entropy: {e1:.6f}")
    print(f"Ambiguous prompt entropy: {e2:.6f}")

    # Phase 1 "done when" check: entropies should NOT be exact multiples of 1/256
    # (the bf16 quantum) any more.
    import fractions
    for e, name in [(e1, "confident"), (e2, "ambiguous")]:
        remainder = (e * 256) % 1
        print(f"{name}: e*256 mod 1 = {remainder:.6f} (should NOT be ~0 if fix worked)")

    assert e2 > e1, "Sanity check failed."
    print("Sanity check passed.")