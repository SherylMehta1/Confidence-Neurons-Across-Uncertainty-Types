"""
preprocess_ambiguity.py -- Person A
Category: ambiguity
Source: AmbigQA (sewon/ambig_qa, "light" config) on HuggingFace.

Filters to genuinely ambiguous questions: annotation type == "multipleQAs"
with 2+ distinct disambiguated answers (uncertainty from underdetermination,
not missing information).

Run on Kaggle (no model-loading dependency for the filtering step itself,
but tokenizer is needed to build chat_formatted_prompt).
"""

import sys
sys.path.append(".")  # run this script from the repo root
from datasets import load_dataset

from shared.schema_utils import load_tokenizer, build_records, save_records

OUTPUT_PATH = "data/ambiguity/prompts.jsonl"


def extract_ambiguous_question(example) -> str | None:
    """
    Returns the raw question string if this example is genuinely ambiguous
    (multipleQAs with 2+ distinct answer sets), else None.
    """
    annotations = example["annotations"]
    ann_types = annotations["type"]

    if "multipleQAs" not in ann_types:
        return None

    idx = ann_types.index("multipleQAs")
    qa_pairs = annotations["qaPairs"][idx]

    # qa_pairs["answer"] is a list of answer-lists, one per disambiguated
    # question. Distinct interpretations = distinct answer sets.
    distinct_answer_sets = {
        tuple(sorted(set(a))) for a in qa_pairs["answer"] if a
    }
    if len(distinct_answer_sets) < 2:
        return None

    question = example["question"].strip()
    if not question.endswith("?"):
        question += "?"
    return question


def main():
    print("Loading AmbigQA (sewon/ambig_qa, light)...")
    ds = load_dataset("sewon/ambig_qa", "light", split="train")

    print(f"Scanning {len(ds)} examples for genuine ambiguity...")
    raw_prompts = []
    for example in ds:
        q = extract_ambiguous_question(example)
        if q is not None:
            raw_prompts.append(q)

    print(f"Found {len(raw_prompts)} ambiguous questions "
          f"(need >= 120 to subsample).")

    print("Loading tokenizer for chat-template formatting...")
    tokenizer = load_tokenizer()

    records = build_records(
        raw_prompts=raw_prompts,
        category="ambiguity",
        source_dataset="AmbigQA",
        prefix="amb",
        tokenizer=tokenizer,
        n_working=120,
        split_ratio=0.7,
    )

    save_records(records, OUTPUT_PATH)


if __name__ == "__main__":
    main()
