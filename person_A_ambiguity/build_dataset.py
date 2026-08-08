"""
Person A — Ambiguity dataset builder.

Source: AmbigQA / AmbigNQ (Min et al., 2020)
No model dependency — pure data filtering/reformatting. Can run locally or on Kaggle CPU.

Output: data/ambiguity/prompts.jsonl, matching the shared schema in DATA_SOURCES.md
"""

import json
import random
from datasets import load_dataset

# Llama-3.1 chat template wrapping needs a tokenizer -- only import if you want
# chat_formatted_prompt filled in now. If you don't have the model loaded, you can
# leave chat_formatted_prompt as None here and fill it in later inside the Kaggle
# session where the tokenizer is available (see fill_chat_template.py note at bottom).
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
except Exception:
    TOKENIZER = None
    print("No tokenizer available -- chat_formatted_prompt will be left blank. "
          "Run apply_chat_template later once you have model/tokenizer access.")


def format_chat_prompt(user_message):
    if TOKENIZER is None:
        return None
    messages = [{"role": "user", "content": user_message}]
    return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_ambigqa():
    # "light" config has question + multiple annotated answer sets, good for this purpose
    return load_dataset("sewon/ambig_qa", "light", split="train")


def is_genuinely_ambiguous(item, min_answers=2):
    """
    AmbigQA marks ambiguity via multiple distinct answer sets in annotations.
    An item counts as ambiguous here if it has 2+ distinct answer interpretations.
    """
    annotations = item.get("annotations", {})
    answer_types = annotations.get("type", [])
    # "multipleQAs" type indicates the question was judged genuinely ambiguous
    if "multipleQAs" in answer_types:
        qa_pairs_list = annotations.get("qaPairs", [])
        for qa_pairs in qa_pairs_list:
            if qa_pairs and len(qa_pairs) >= min_answers:
                return True
    return False


def extract_prompt_text(item):
    """Convert the AmbigQA question into a next-token-friendly continuation prompt.

    AmbigQA gives full questions (e.g. "Where did they film that movie?"), which
    work fine as chat prompts directly -- no need to force them into cloze/continuation
    style since we're measuring entropy over the model's generated answer distribution,
    not a single next token in a fill-in-the-blank sense.
    """
    return item["question"]


def build_ambiguity_prompts(n_target=150, seed=42):
    random.seed(seed)
    dataset = load_ambigqa()

    ambiguous_items = [item for item in dataset if is_genuinely_ambiguous(item)]
    print(f"Found {len(ambiguous_items)} genuinely ambiguous items out of {len(dataset)} total")

    if len(ambiguous_items) > n_target:
        ambiguous_items = random.sample(ambiguous_items, n_target)

    rows = []
    for i, item in enumerate(ambiguous_items):
        raw_prompt = extract_prompt_text(item)
        rows.append({
            "prompt_id": f"amb_{i:04d}",
            "category": "ambiguity",
            "raw_prompt": raw_prompt,
            "chat_formatted_prompt": format_chat_prompt(raw_prompt),
            "source_dataset": "AmbigQA",
            "split": None,  # set below
        })

    # 70/30 working/held-out split
    split_idx = int(len(rows) * 0.7)
    random.shuffle(rows)
    for i, r in enumerate(rows):
        r["split"] = "working" if i < split_idx else "held_out"
        r["prompt_id"] = f"amb_{i:04d}"  # renumber after shuffle for clean IDs

    return rows


def save_prompts(rows, path="data/ambiguity/prompts.jsonl"):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows)} prompts to {path}")


if __name__ == "__main__":
    rows = build_ambiguity_prompts(n_target=150)
    save_prompts(rows)

    # quick face-validity print of a few samples -- hand-check these before trusting the set
    print("\nSample prompts (hand-check these):")
    for r in rows[:5]:
        print(f"  [{r['prompt_id']}] {r['raw_prompt']}")
