"""
schema_utils.py -- shared helpers for the category preprocessing scripts so the
data schema (DATA_SOURCES.md) is identical across Person A / B / C.

`build_records` is DEPRECATED: it formats prompts with the bare chat template
(no assistant-turn prefill), i.e. the pre-position-fix format in which the
measured token is a fresh assistant-turn opener. Use
shared.prompt_format.build_records_with_formatter instead, which also adds the
`is_control` field. `format_chat_prompt` is re-exported from
shared.prompt_format (single implementation, pinned date_string).
"""

import json
import warnings
from pathlib import Path
from typing import List, Dict

from shared.prompt_format import format_chat_prompt, seeded_shuffle  # noqa: F401

import os
MODEL_NAME = os.environ.get("CN_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")  # CN_MODEL_ID overrides


def load_tokenizer():
    """
    Loads the Llama-3.1-8B-Instruct tokenizer. Requires HF gated access
    already granted + a token available (huggingface-cli login, or
    HF_TOKEN env var / Kaggle secret already exported before this runs).
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


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
    DEPRECATED -- produces the pre-fix format (no assistant-turn prefill, no
    is_control field). Kept only so old preprocessing scripts still import.
    Use shared.prompt_format.build_records_with_formatter.

    Seeded subsample + 70/30 split. Uses a private random.Random(seed), which
    gives exactly the same order as the legacy global random.seed(seed).
    """
    warnings.warn(
        "schema_utils.build_records is deprecated: it produces the pre-position-fix "
        "prompt format. Use shared.prompt_format.build_records_with_formatter.",
        DeprecationWarning, stacklevel=2,
    )
    pool = seeded_shuffle(raw_prompts, seed)

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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
