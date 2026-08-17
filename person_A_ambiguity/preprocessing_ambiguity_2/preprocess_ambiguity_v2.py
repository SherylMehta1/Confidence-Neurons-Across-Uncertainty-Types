"""
preprocess_ambiguity_v2_FIXED.py -- Person A, Phase 3 (patched)

This is a drop-in replacement for
person_A_ambiguity/preprocessing_ambiguity_2/preprocess_ambiguity_v2.py.

WHAT WAS WRONG WITH THE COMMITTED VERSION (confirmed against the actual data
in data/ambiguity/prompts.jsonl on the repo, commit 7c8e580):

  raw_prompt:            'What was the vote of texas vs johnson case? The answer is'
  chat_formatted_prompt: '...case? The answer is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

`convert_to_cloze_prompt()` baked " The answer is" into the RAW question,
then `shared.schema_utils.build_records()` / `format_chat_prompt()` wrapped
that whole string as the USER turn and applied add_generation_prompt=True.
The chat template closes the user turn on "...The answer is<|eot_id|>" and
opens a FRESH, empty assistant turn -- so the model's first generated token
is a turn-opener ("I", "Based", "The"...), not a continuation of "The
answer is". This is exactly the turn-boundary bug the whole Phase 3
position-fix row was supposed to close out, and it's the reason your
ambiguity Phase 2 numbers were suspect in the first place. It was never
actually applied here -- Person A's rewrite used the old
shared.schema_utils.build_records path instead of
shared.prompt_format.build_completion_prompt /
build_records_with_formatter, which are the only functions in this repo
that do the fix correctly (confirmed correct for Person B and Person C's
committed data).

Also, no data/ambiguity/controls.jsonl was ever produced -- there's no
build_ambiguity_controls() in the committed file at all.

WHAT THIS FILE KEEPS UNCHANGED (it's good work, not the problem):
Stage 1 (multipleQAs / >=2 answer groups), Stage 2 (is_factoid_question),
Stage 2b (looks_malformed), Stage 3 (has_genuinely_distinct_answers), and
the full review_log.jsonl audit trail -- all copied verbatim. This is a
genuinely more reproducible Stage 3 than manual review + approval_tracker
would have been (every one of the 10,036 raw AmbigQA records gets a logged
accept/reject reason, deterministically, from a fixed seed), and it already
resolves the original "ambiguity data contradicts its own review" finding.
No reason to touch it.

TWO FIXES, both isolated to the very end of the pipeline:

1. POSITION FIX: raw_prompt is no longer suffix-baked. The bare question
   goes into build_records_with_formatter(formatter=build_completion_prompt),
   which appends "The answer is" AFTER the chat template's generation
   prompt -- a genuine assistant-turn prefill, same fix already verified
   correct for Person B and C.

2. MATCHED CONTROLS: adds build_ambiguity_controls(), sourced from
   AmbigQA's OTHER annotation type -- "singleAnswer" (the record has one
   agreed-on answer, i.e. NOT ambiguous) -- run through the exact same
   Stage 2 (factoid) + Stage 2b (malformed) filters as the working set,
   with its own review_log so the control set is just as auditable as the
   main one. Stage 1/3 don't apply here (there's no "ambiguity" to check
   for on a set that's controls specifically because it ISN'T ambiguous).

   CAVEAT -- PILOT THIS FIELD ACCESS BEFORE TRUSTING IT AT SCALE, same
   convention as every other new filter in this project: I don't have
   network access to sewon/ambig_qa in this environment, so
   extract_single_answer_questions() below is written from the AmbigQA
   paper/dataset-card schema (annotations.type / annotations.answer as
   parallel per-annotator lists), NOT verified against a live record the
   way everything else in this repo has been. Run
   pilot_check_single_answer_schema() FIRST (prints ~10 raw annotation
   dicts for records with a "singleAnswer" type) and eyeball that
   `annotations["answer"][i]` is actually non-empty and shaped the way the
   code below assumes before trusting build_ambiguity_controls() at scale.

Run inside your Kaggle/RunPod session (tokenizer needed to build; load the
model too if running the induction check right after, recommended):

    exec(open("person_A_ambiguity/preprocessing_ambiguity_2/preprocess_ambiguity_v2.py").read())
    pilot_check_single_answer_schema()        # <-- do this first, read the printed output
    records, control_records = main()
"""

import sys
sys.path.append(".")
import json
import os
import re

from datasets import load_dataset
from shared.schema_utils import load_tokenizer
from shared.prompt_format import build_completion_prompt, build_records_with_formatter

OUTPUT_PATH = "data/ambiguity/prompts.jsonl"
CONTROLS_OUTPUT_PATH = "data/ambiguity/controls.jsonl"
REVIEW_LOG_PATH = "data/ambiguity/review_log.jsonl"
CONTROL_REVIEW_LOG_PATH = "data/ambiguity/control_review_log.jsonl"

# --- unchanged from the committed file ------------------------------------

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
    words = q.split()
    n_question_words = sum(1 for w in words if w in question_words)
    if n_question_words >= 2:
        return True
    if len(words) > 20:
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


def load_ambigqa_records():
    return load_dataset("sewon/ambig_qa", "light", split="train")


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


# --- NEW: singleAnswer extraction for the control set ---------------------

def extract_single_answer_questions(record: dict):
    """
    AmbigQA annotates each record with one or more annotator judgments.
    Where the working-set extractor looks for "multipleQAs" (genuinely
    ambiguous -- annotators disambiguated it into >=2 distinct questions),
    this looks for "singleAnswer" (annotators agreed there's exactly one
    answer, i.e. NOT ambiguous) -- the low-uncertainty twin condition.

    Schema assumption (per the AmbigQA paper/dataset card -- PILOT this,
    see module docstring): annotations["type"] and annotations["answer"]
    are parallel per-annotator lists; annotations["answer"][i] holds the
    accepted answer(s) for the i-th annotation when its type is
    "singleAnswer". We only need the question text here, not the answer
    itself, so we just confirm a non-empty answer entry exists -- that's
    the signal that this annotator judged the question unambiguous.
    """
    question = record.get("question")
    annotations = record.get("annotations")
    if question is None or annotations is None:
        return None

    ann_types = annotations.get("type", [])
    answer_lists = annotations.get("answer", [])
    for i, ann_type in enumerate(ann_types):
        if ann_type == "singleAnswer" and i < len(answer_lists) and answer_lists[i]:
            return question.strip()
    return None


def pilot_check_single_answer_schema(n_print: int = 10):
    """
    Run this BEFORE build_ambiguity_controls(). Prints the raw annotations
    dict for the first n_print records that contain a "singleAnswer" type,
    so you can eyeball that extract_single_answer_questions()'s assumptions
    about the field shapes actually hold for this dataset/version, same as
    you piloted model_knows_fact() and verify_induction_quality() elsewhere
    in this project before trusting them at scale.
    """
    ds = load_ambigqa_records()
    shown = 0
    for record in ds:
        annotations = record.get("annotations")
        if not annotations or "singleAnswer" not in annotations.get("type", []):
            continue
        q = extract_single_answer_questions(record)
        print(f"question: {record.get('question')!r}")
        print(f"  annotations: {annotations}")
        print(f"  extracted -> {q!r}")
        print()
        shown += 1
        if shown >= n_print:
            break
    if shown == 0:
        print("No 'singleAnswer' records found in the first pass of the dataset -- "
              "STOP, the schema assumption above is wrong, do not run "
              "build_ambiguity_controls() until this is fixed.")
    else:
        print(f"Shown {shown} sample(s). Confirm 'extracted' looks like a real "
              f"question string (not None/empty) before trusting the bulk run.")


# --- working set (position-fixed) -----------------------------------------

def build_working_set(tokenizer):
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
        accepted_prompts.append(question)   # <-- FIX: bare question, no suffix baked in

    n_accepted = len(accepted_prompts)
    n_rejected = len(review_log) - n_accepted
    print(f"\nWorking-set stage summary: {n_accepted} accepted, {n_rejected} rejected.")
    for stage in [0, 1, 2, "2b", 3]:
        n = sum(1 for r in review_log if r["stage_failed"] == stage)
        print(f"  rejected at stage {stage}: {n}")

    os.makedirs(os.path.dirname(REVIEW_LOG_PATH), exist_ok=True)
    with open(REVIEW_LOG_PATH, "w") as f:
        for entry in review_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved full review log ({len(review_log)} entries) to {REVIEW_LOG_PATH}")

    assert n_accepted >= 120, f"Only {n_accepted} items survived all stages, need >= 120 to subsample."

    # FIX: build_completion_prompt instead of convert_to_cloze_prompt + build_records --
    # this is what actually applies the assistant-turn prefill correctly.
    records = build_records_with_formatter(
        raw_prompts=accepted_prompts,
        category="ambiguity",
        source_dataset="AmbigQA-light",
        prefix="amb",
        tokenizer=tokenizer,
        formatter=build_completion_prompt,
        n_working=120,
        split_ratio=0.7,
    )
    from shared.schema_utils import save_records
    save_records(records, OUTPUT_PATH)
    return records


# --- NEW: matched controls (position-fixed) --------------------------------

def build_ambiguity_controls(tokenizer):
    ds = load_ambigqa_records()
    print(f"\nScanning for singleAnswer (non-ambiguous, control) records...")

    review_log = []
    accepted_prompts = []

    for record in ds:
        question = extract_single_answer_questions(record)
        if question is None:
            review_log.append({"question": record.get("question"), "stage_failed": 1,
                                "reason": "no singleAnswer annotation on this record"})
            continue

        if not is_factoid_question(question):
            review_log.append({"question": question, "stage_failed": 2,
                                "reason": "open-ended/explanatory question, no natural short answer"})
            continue

        if looks_malformed(question):
            review_log.append({"question": question, "stage_failed": "2b",
                                "reason": "malformed/run-on: multiple question-word clusters or too long"})
            continue

        review_log.append({"question": question, "stage_failed": None, "reason": "accepted"})
        accepted_prompts.append(question)

    n_accepted = len(accepted_prompts)
    n_rejected = len(review_log) - n_accepted
    print(f"Control-set stage summary: {n_accepted} accepted, {n_rejected} rejected.")
    for stage in [1, 2, "2b"]:
        n = sum(1 for r in review_log if r["stage_failed"] == stage)
        print(f"  rejected at stage {stage}: {n}")

    os.makedirs(os.path.dirname(CONTROL_REVIEW_LOG_PATH), exist_ok=True)
    with open(CONTROL_REVIEW_LOG_PATH, "w") as f:
        for entry in review_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved full control review log ({len(review_log)} entries) to {CONTROL_REVIEW_LOG_PATH}")

    assert n_accepted >= 120, (
        f"Only {n_accepted} singleAnswer items survived filtering, need >= 120. "
        f"If this fires, re-check pilot_check_single_answer_schema() output -- "
        f"the extraction may not be finding real singleAnswer records."
    )

    control_records = build_records_with_formatter(
        raw_prompts=accepted_prompts,
        category="ambiguity",
        source_dataset="AmbigQA-light (singleAnswer, matched control)",
        prefix="amb_ctrl",
        tokenizer=tokenizer,
        formatter=build_completion_prompt,
        n_working=120,
        split_ratio=0.7,
        is_control=True,
    )
    os.makedirs(os.path.dirname(CONTROLS_OUTPUT_PATH), exist_ok=True)
    with open(CONTROLS_OUTPUT_PATH, "w") as f:
        for r in control_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(control_records)} control prompts to {CONTROLS_OUTPUT_PATH}")
    return control_records


def main():
    tokenizer = load_tokenizer()
    records = build_working_set(tokenizer)
    control_records = build_ambiguity_controls(tokenizer)

    print("\n--- Sample working prompts (position-fixed) ---")
    for r in records[:5]:
        print(f"  [{r['prompt_id']}] ...{r['chat_formatted_prompt'][-80:]!r}")
    print("\n--- Sample control prompts ---")
    for r in control_records[:5]:
        print(f"  [{r['prompt_id']}] ...{r['chat_formatted_prompt'][-80:]!r}")

    print(
        "\nNEXT: with model+tokenizer loaded, run "
        "shared.prompt_format.verify_induction_quality(model, tokenizer, records[:25]) "
        "and the same on control_records[:25], then diff the two MEANS yourself -- "
        "don't rely on the PASS/FAIL line alone (see the note on why in Person B's "
        "final gate check)."
    )

    return records, control_records


if __name__ == "__main__":
    main()