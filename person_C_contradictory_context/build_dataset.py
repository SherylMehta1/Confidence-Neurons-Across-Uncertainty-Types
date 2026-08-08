"""
Person C -- Contradictory Context dataset builder.

Source: CounterFact (azhx/counterfact on HuggingFace), from the ROME paper (Meng et al., 2022).
Construction method follows the "Redefine" contradiction pattern used in Tighidet et al. (2024)
and its replications (Ortu et al.) -- cite the method, not the original repo's code.

IMPORTANT: this script needs the model + tokenizer ALREADY LOADED, because it verifies the
model actually knows the "true" fact before building a contradiction prompt around it.
Run this INSIDE your Kaggle session, after Tool 1-2's sanity check has passed and `model`,
`tokenizer` are already in memory -- do not run this standalone.

Output: data/contradictory_context/prompts.jsonl, matching the shared schema in DATA_SOURCES.md
"""

import json
import random
from datasets import load_dataset

import sys
sys.path.append(".")
from shared.model_utils import get_next_token_probs, compute_top1_prob


def load_counterfact():
    return load_dataset("azhx/counterfact", split="train")


def model_knows_fact(model, tokenizer, prompt_template, subject, threshold=0.1):
    """
    Check the model actually has SOME confidence in a plausible answer for this fact
    before we build a contradiction around it -- a contradiction test is meaningless
    if the model never knew the true fact to begin with.

    threshold is a starting point -- pilot on ~20 examples and adjust based on what
    you see; too high and you'll filter out almost everything, too low and you'll
    include facts the model doesn't really know.
    """
    prompt = prompt_template.format(subject)
    probs = get_next_token_probs(model, tokenizer, prompt)
    return compute_top1_prob(probs) > threshold


def build_contradiction_prompt(row):
    """
    Build the "Redefine" style contradiction prompt: state the false fact (target_new)
    as context, then repeat the original query -- the model must choose between
    copying the context or falling back on parametric knowledge (target_true).
    """
    rw = row["requested_rewrite"]
    subject = rw["subject"]
    template = rw["prompt"]  # e.g. "The mother tongue of {} is"
    target_true = rw["target_true"]["str"]
    target_new = rw["target_new"]["str"]

    base_prompt = template.format(subject)
    context_sentence = f"{base_prompt} {target_new}."
    contradiction_prompt = f"{context_sentence} {base_prompt}"

    return {
        "prompt": contradiction_prompt,
        "subject": subject,
        "correct_object": target_true,
        "wrong_object": target_new,
        "relation_id": rw["relation_id"],
        "case_id": row["case_id"],
    }


def build_contradictory_context_prompts(model, tokenizer, n_target=150, knows_fact_threshold=0.1, seed=42):
    random.seed(seed)
    counterfact = load_counterfact()
    print(f"Loaded {len(counterfact)} CounterFact rows")

    # shuffle indices so we don't always pull from the start of the dataset
    indices = list(range(len(counterfact)))
    random.shuffle(indices)

    filtered = []
    checked = 0
    for idx in indices:
        row = counterfact[idx]
        rw = row["requested_rewrite"]
        checked += 1

        if model_knows_fact(model, tokenizer, rw["prompt"], rw["subject"], threshold=knows_fact_threshold):
            filtered.append(build_contradiction_prompt(row))

        if len(filtered) >= n_target:
            break

    print(f"Checked {checked} rows, kept {len(filtered)} where the model knew the true fact")

    rows = []
    for i, record in enumerate(filtered):
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        rows.append({
            "prompt_id": f"contra_{i:04d}",
            "category": "contradictory_context",
            "raw_prompt": record["prompt"],
            "chat_formatted_prompt": chat_prompt,
            "source_dataset": "CounterFact (azhx/counterfact)",
            "split": None,
            "correct_object": record["correct_object"],
            "wrong_object": record["wrong_object"],
            "case_id": record["case_id"],
        })

    split_idx = int(len(rows) * 0.7)
    for i, r in enumerate(rows):
        r["split"] = "working" if i < split_idx else "held_out"

    return rows


def save_prompts(rows, path="data/contradictory_context/prompts.jsonl"):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows)} prompts to {path}")


if __name__ == "__main__":
    # This block assumes `model` and `tokenizer` already exist in your Kaggle session
    # (i.e. you've already run Tool 1-2's load_model() earlier in the same notebook).
    # If running as a standalone script rather than pasted into an existing session,
    # uncomment the next two lines:
    # from shared.model_utils import load_model
    # model, tokenizer = load_model(quantize=True)

    rows = build_contradictory_context_prompts(model, tokenizer, n_target=150)
    save_prompts(rows)

    print("\nSample prompts (hand-check these):")
    for r in rows[:5]:
        print(f"  [{r['prompt_id']}] {r['raw_prompt']}")
        print(f"    correct: {r['correct_object']}  |  context said: {r['wrong_object']}")
