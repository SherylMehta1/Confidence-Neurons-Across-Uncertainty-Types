"""
Shared Tools 1-2: load the model, run forward passes, measure entropy.

BUILT TOGETHER. Reviewed by all 3 team members before anyone builds on it.
Do not fork a private copy — if something needs to change, change it here and tell the team.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_id="meta-llama/Llama-3.1-8B-Instruct", quantize=True):
    """Tool 1: load the model and tokenizer.

    quantize=True uses 4-bit loading (fits in 16GB, good for Kaggle free GPUs).
    Set quantize=False for precision-sensitive Phase 3/4 runs if you have access
    to a 24GB+ GPU — quantization is a known confound for neuron-level measurements.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )

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


def compute_multi_token_entropy(model, tokenizer, prompt, k=5):
    """
    Supplementary metric for prompts where single-token entropy isn't
    meaningful (e.g. open-ended questions where the first generated token
    is mostly grammatical/conversational scaffolding, not a genuine
    confidence signal). Prefer reformatting the prompt into a cloze/
    completion style instead where possible (see convert_to_cloze_prompt()
    in person_B_lack_of_knowledge/preprocess_lack_of_knowledge.py) --
    this function is a documented fallback, not a silent replacement.

    NOTE ON INTERPRETATION: this is noisier than single-token entropy.
    Each step is conditioned on a greedily-picked previous token, so an
    early "wrong turn" skews every later value -- these are not k
    independent samples. The choice of k, and of greedy vs. sampled
    decoding, are hidden hyperparameters this metric introduces that
    single-token entropy does not have. If used, report this explicitly
    as a methodological choice, not hide it.

    Returns dict with mean_entropy, max_entropy, and the raw per-token list.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = inputs["input_ids"]
    entropies = []

    with torch.no_grad():
        for _ in range(k):
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            entropies.append(compute_entropy(probs))
            next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return {
        "mean_entropy": sum(entropies) / len(entropies),
        "max_entropy": max(entropies),
        "per_token_entropy": entropies,
    }


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
