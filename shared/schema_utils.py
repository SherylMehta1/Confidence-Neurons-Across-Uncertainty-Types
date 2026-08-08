"""
schema_utils.py
Shared helpers used by all three category preprocessing scripts, so the
output schema is byte-for-byte identical across Person A / B / C.

Put this file in shared/ alongside model_utils.py etc.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def load_tokenizer():
    """
    Loads the Llama-3.1-8B-Instruct tokenizer. Requires HF gated access
    already granted + a token available (huggingface-cli login, or
    HF_TOKEN env var / Kaggle secret already exported before this runs).
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def format_chat_prompt(tokenizer, raw_prompt: str) -> str:
    """
    Wraps a raw prompt in the Llama-3.1 chat template, ending with the
    generation prompt so the model's next tokens are the completion.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_records(
    raw_prompts: List[str],
    category: str,
    source_dataset: str,
    prefix: str,
    tokenizer,
    n_working: int = 120,
    split_ratio: float = 0.7,
    seed: int = 42,
) -> List[Dict]:
    """
    Turns a list of raw prompt strings into the shared schema, subsamples,
    splits 70/30 working/held-out, and returns the list of record dicts.
    Does NOT write to disk -- call save_records() separately so you can
    inspect/hand-check before committing to a file.
    """
    random.seed(seed)
    pool = list(raw_prompts)
    random.shuffle(pool)

    if len(pool) < n_working:
        raise ValueError(
            f"[{category}] Only {len(pool)} candidate prompts available, "
            f"need at least {n_working}. Loosen your filtering criteria."
        )

    subsample = pool[:n_working]
    n_working_split = int(len(subsample) * split_ratio)

    records = []
    for i, raw_prompt in enumerate(subsample):
        records.append({
            "prompt_id": f"{prefix}_{i:04d}",
            "category": category,
            "raw_prompt": raw_prompt,
            "chat_formatted_prompt": format_chat_prompt(tokenizer, raw_prompt),
            "source_dataset": source_dataset,
            "split": "working" if i < n_working_split else "held_out",
        })
    return records


def save_records(records: List[Dict], out_path: str, hand_check_n: int = 15):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_working = sum(1 for r in records if r["split"] == "working")
    n_held = len(records) - n_working
    print(f"Saved {len(records)} records to {out_path} "
          f"({n_working} working / {n_held} held-out)")

    print(f"\nHand-check these {hand_check_n} items before trusting the label:")
    for r in records[:hand_check_n]:
        print(f"  [{r['prompt_id']}] {r['raw_prompt']}")
