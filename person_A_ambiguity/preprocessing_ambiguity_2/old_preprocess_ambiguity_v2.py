import sys
sys.path.append(".")
import json
import os
import re

from datasets import load_dataset
from shared.schema_utils import load_tokenizer, build_records, save_records

OUTPUT_PATH = "data/ambiguity/prompts.jsonl"
REVIEW_LOG_PATH = "data/ambiguity/review_log.jsonl"

# --- Stage 2: factoid filter ---------------------------------------------
EXCLUDE_OPEN_ENDED_PATTERNS = (
    r"^how (do|to|does|can|should|did|are|is)\b",
    r"^why\b",
    r"what are the (methods|steps|ways|effects|reasons|benefits|advantages|disadvantages|causes|consequences|implications)",
    r"^describe\b", r"^explain\b", r"^discuss\b",
    r"in what ways", r"to what extent",
)

ACCEPT_FACTOID_PATTERNS = (
    r"^(what|who|where|when|which|name|in which|in what year|how (many|much|old|long|far))\b",
)

def is_factoid_question(question: str) -> bool:
    q = question.strip().lower()
    for pattern in EXCLUDE_OPEN_ENDED_PATTERNS:
        if re.search(pattern, q):
            return False
    for pattern in ACCEPT_FACTOID_PATTERNS:
        if re.search(pattern, q):
            return True
    return False

def looks_malformed(question: str) -> bool:
    q = question.strip().lower()
    question_words = ["who", "what", "when", "where", "which", "how", "why"]
    word_count = q.split()
    n_question_words = sum(1 for w in word_count if w in question_words)
    if n_question_words >= 2:
        return True
    if len(word_count) > 20:
        return True
    return False

def normalize_answer(a: str) -> str:
    if not isinstance(a, str):
        return ""
    a = a.lower().strip()
    a = re.sub(r"[^\w\s]", "", a)
    a = re.sub(r"^(the|a|an)\s+", "", a)
    return a.strip()

def extract_first_string(item) -> str:
    while isinstance(item, (list, tuple)) and len(item) > 0:
        item = item[0]
    return item if isinstance(item, str) else ""

def has_genuinely_distinct_answers(answer_sets: list, min_distinct: int = 2) -> bool:
    normalized_reps = set()
    for group in answer_sets:
        if not group:
            continue
        first_str = extract_first_string(group)
        norm = normalize_answer(first_str)
        if norm:
            normalized_reps.add(norm)
    return len(normalized_reps) >= min_distinct

def convert_to_cloze_prompt(question: str, suffix: str = " The answer is") -> str:
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    return q + suffix

def load_ambigqa_records():
    ds = load_dataset("sewon/ambig_qa", "light", split="train")
    return ds

def extract_question_and_answer_groups(record: dict):
    question = record.get("question")
    annotations = record.get("annotations")
    if question is None or annotations is None:
        return None

    answer_groups = []
    if isinstance(annotations, dict):
        ann_types = annotations.get("type", [])
        qa_pairs_list = annotations.get("qaPairs", [])

        for i, ann_type in enumerate(ann_types):
            if ann_type == "multipleQAs" and i < len(qa_pairs_list):
                qa_pair_dict = qa_pairs_list[i]
                for ans_group in qa_pair_dict.get("answer", []):
                    if ans_group:
                        answer_groups.append(ans_group)

    return question.strip(), answer_groups

def main():
    ds = load_ambigqa_records()
    print(f"Loaded {len(ds)} raw AmbigQA records.")

    review_log = []
    accepted_prompts = []

    for record in ds:
        parsed = extract_question_and_answer_groups(record)
        if parsed is None:
            review_log.append({"question": None, "stage_failed": 0, "reason": "missing question/annotations field"})
            continue
        question, answer_groups = parsed

        if len(answer_groups) < 2:
            review_log.append({"question": question, "stage_failed": 1, "reason": "fewer than 2 answer groups (not ambiguous per AmbigQA)"})
            continue

        if not is_factoid_question(question):
            review_log.append({"question": question, "stage_failed": 2, "reason": "open-ended/explanatory question, no natural short answer"})
            continue

        if looks_malformed(question):
            review_log.append({"question": question, "stage_failed": "2b", "reason": "malformed/run-on: multiple question-word clusters or too long"})
            continue

        if not has_genuinely_distinct_answers(answer_groups, min_distinct=2):
            review_log.append({"question": question, "stage_failed": 3, "reason": "answer groups are near-duplicates after normalization"})
            continue

        review_log.append({"question": question, "stage_failed": None, "reason": "accepted"})
        accepted_prompts.append(convert_to_cloze_prompt(question))

    n_accepted = len(accepted_prompts)
    n_rejected = len(review_log) - n_accepted
    print(f"\nStage summary: {n_accepted} accepted, {n_rejected} rejected.")
    for stage in [0, 1, 2, "2b", 3]:
        n = sum(1 for r in review_log if r["stage_failed"] == stage)
        print(f"  rejected at stage {stage}: {n}")

    os.makedirs(os.path.dirname(REVIEW_LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(REVIEW_LOG_PATH, "w") as f:
        for entry in review_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved full review log ({len(review_log)} entries) to {REVIEW_LOG_PATH}")

    assert n_accepted >= 120, f"Only {n_accepted} items survived all stages, need >= 120 to subsample."

    tokenizer = load_tokenizer()
    records = build_records(
        raw_prompts=accepted_prompts,
        category="ambiguity",
        source_dataset="AmbigQA-light",
        prefix="amb",
        tokenizer=tokenizer,
        n_working=120,
        split_ratio=0.7,
    )
    save_records(records, OUTPUT_PATH)

if __name__ == "__main__":
    main()