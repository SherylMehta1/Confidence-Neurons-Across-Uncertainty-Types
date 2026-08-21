"""
Person B — Lack of Knowledge dataset builder.

Source: UnknownBench (Liu et al., 2024), Non-Existent Concepts (NEC) subset
No model dependency -- pure data extraction/reformatting. Can run locally or on Kaggle CPU.

Setup required first (run once, outside this script):
    git clone https://github.com/genglinliu/UnknownBench
Then point NEC_DATA_PATH below at wherever the NEC subset file(s) landed --
check the repo's README for the exact filename/location, it may be a single JSON
or split into separate files per subcategory (animals, countries, food, medicine, etc.)

Output: data/lack_of_knowledge/prompts.jsonl, matching the shared schema in DATA_SOURCES.md
"""

import json
import random
import glob

try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
except Exception:
    TOKENIZER = None
    print("No tokenizer available -- chat_formatted_prompt will be left blank.")


def format_chat_prompt(user_message):
    if TOKENIZER is None:
        return None
    messages = [{"role": "user", "content": user_message}]
    return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Adjust this to wherever you cloned UnknownBench -- check its README for exact paths,
# this is a placeholder pattern assuming JSON files per subcategory under a nec/ folder.
NEC_DATA_GLOB = "UnknownBench/data/nec/*.json"


def load_nec_items(data_glob=NEC_DATA_GLOB):
    """
    Load Non-Existent Concepts items. Each item is expected to have at least:
      - a question about a fabricated entity (the "unanswerable" item)
      - a matched answerable control question (same template, real entity)
    Exact field names depend on UnknownBench's actual file format -- inspect one
    file manually first (e.g. `json.load(open(path))[0]`) and adjust the field
    names below (currently assumed: "question", "answerable", "entity") accordingly.
    """
    all_items = []
    for filepath in glob.glob(data_glob):
        with open(filepath) as f:
            data = json.load(f)
        all_items.extend(data)
    return all_items


def build_lack_of_knowledge_prompts(n_target=150, seed=42):
    random.seed(seed)
    items = load_nec_items()
    print(f"Loaded {len(items)} NEC items")

    if len(items) == 0:
        raise FileNotFoundError(
            f"No files matched {NEC_DATA_GLOB} -- confirm you've cloned UnknownBench "
            "and check its README for the actual NEC data file path/format."
        )

    # Keep only the unanswerable (fabricated-entity) items for THIS category --
    # the answerable controls are useful for validation/comparison but are not
    # themselves "lack of knowledge" prompts.
    unanswerable_items = [item for item in items if item.get("answerable") is False]
    print(f"{len(unanswerable_items)} unanswerable (fabricated-entity) items found")

    if len(unanswerable_items) > n_target:
        unanswerable_items = random.sample(unanswerable_items, n_target)

    rows = []
    for i, item in enumerate(unanswerable_items):
        raw_prompt = item["question"]
        rows.append({
            "prompt_id": f"unk_{i:04d}",
            "category": "lack_of_knowledge",
            "raw_prompt": raw_prompt,
            "chat_formatted_prompt": format_chat_prompt(raw_prompt),
            "source_dataset": "UnknownBench-NEC",
            "split": None,
        })

    split_idx = int(len(rows) * 0.7)
    random.shuffle(rows)
    for i, r in enumerate(rows):
        r["split"] = "working" if i < split_idx else "held_out"
        r["prompt_id"] = f"unk_{i:04d}"

    return rows


def save_prompts(rows, path="data/lack_of_knowledge/prompts.jsonl"):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows)} prompts to {path}")


if __name__ == "__main__":
    rows = build_lack_of_knowledge_prompts(n_target=150)
    save_prompts(rows)

    print("\nSample prompts (hand-check these):")
    for r in rows[:5]:
        print(f"  [{r['prompt_id']}] {r['raw_prompt']}")
