"""
build_dataset.py -- Person A, Ambiguity
Simple/first-pass version. See preprocess_ambiguity.py for the full
pipeline with subtype classification and contamination filters --
this file is kept for reference as the original naive filter.

Source: AmbigQA (sewon/ambig_qa, "light" config) on HuggingFace.
"""

import json
import random
from datasets import load_dataset

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


def is_genuinely_ambiguous(item, min_answers=2):
    """
    Naive filter: AmbigQA item has 2+ distinct disambiguated answer sets
    under a 'multipleQAs' annotation. NOTE: this alone is NOT sufficient --
    see preprocess_ambiguity.py's contamination filters and the manual
    review step. This function over-includes annotation artifacts (e.g.
    "last time X happened" style questions where sub-answers are just
    different true historical facts, not genuine alternate readings).
    """
    annotations = item.get("annotations", {})
    ann_types = annotations.get("type", [])
    if "multipleQAs" not in ann_types:
        return False
    idx = ann_types.index("multipleQAs")
    qa_pairs = annotations.get("qaPairs", [])[idx] if idx < len(annotations.get("qaPairs", [])) else None
    if not qa_pairs:
        return False
    answers = qa_pairs.get("answer", [])
    distinct_sets = {tuple(sorted(set(a))) for a in answers if a}
    return len(distinct_sets) >= min_answers
