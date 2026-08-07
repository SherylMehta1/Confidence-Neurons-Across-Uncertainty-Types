"""
Shared Tools 1-2: load the model, run forward passes, measure entropy.

BUILT TOGETHER. Reviewed by all 3 team members before anyone builds on it.
Do not fork a private copy — if something needs to change, change it here and tell the team.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_id="meta-llama/Llama-3.1-8B-Instruct", quantize=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    model.eval()
    return model, tokenizer


def format_chat_prompt(tokenizer, user_message):
    """Wrap a raw prompt in the Llama-3.1 chat template."""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_next_token_probs(model, tokenizer, prompt):
    """Tool 1b: run a forward pass, return the probability distribution over the next token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last position = next-token prediction
    return F.softmax(logits, dim=-1)


def compute_entropy(probs):
    """Tool 2: Shannon entropy of a probability distribution. Higher = more uncertain."""
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()


def compute_top1_prob(probs):
    """Convenience: top-1 token probability, a second confidence proxy alongside entropy."""
    return torch.max(probs).item()


if __name__ == "__main__":
    # Sanity check — run this first. Confident prompt should have lower entropy
    # than an ambiguous one. If this doesn't hold, something's wrong before you go further.
    model, tokenizer = load_model()

    confident_prompt = "The capital of France is"
    ambiguous_prompt = "Tomorrow we should go to the"

    p1 = get_next_token_probs(model, tokenizer, confident_prompt)
    p2 = get_next_token_probs(model, tokenizer, ambiguous_prompt)

    print(f"Confident prompt entropy: {compute_entropy(p1):.3f}")
    print(f"Ambiguous prompt entropy: {compute_entropy(p2):.3f}")
    assert compute_entropy(p2) > compute_entropy(p1), "Sanity check failed — investigate before proceeding."
    print("Sanity check passed.")
