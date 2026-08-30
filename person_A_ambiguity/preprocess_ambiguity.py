"""
preprocess_ambiguity.py -- Person A, ambiguity category. THE live entry point.

    python person_A_ambiguity/preprocess_ambiguity.py [--overwrite] [--seed 42] [--n 120]
                                                       [--reject-next-event]

Source: AmbigQA "light" config (sewon/ambig_qa), pinned to AMBIGQA_REVISION.
Needs the Llama-3.1-8B-Instruct tokenizer (gated; HF_TOKEN must be set).
No model is needed to build the data.

Pipeline (every raw record gets a logged accept/reject reason):

  Stage 1   multipleQAs annotation with >= 2 answer groups (AmbigQA's own
            "ambiguous" label).
  Stage 2   is_factoid_question: short-answer wh-question, not open-ended.
  Stage 2t  resolvable time references (ported from the v1 script's
            has_resolvable_time_ref / flag_unscoped_present_tense_split and
            extended -- see TIME_REF_PATTERNS / LAST_FIRST_WHEN_PATTERN).
            These are NOT genuinely ambiguous: "last week", "the latest",
            "who currently holds", "when did X last win" each had exactly one
            correct answer when asked; AmbigQA's multiple answers there are an
            annotation artifact (annotators enumerating every historically
            true instance), not alternate readings.
  Stage 2b  looks_malformed: run-on questions (a second wh-word starting a
            clause) or > 20 words.
  Stage 3   has_genuinely_distinct_answers: >= 2 answer groups whose full
            normalized alias sets do not overlap.
  flagged   "next <event>" future-scheduling questions ("when is the next
            world cup") are a flagged subtype: kept by default (they are
            ambiguous about what "now" is) but marked in the review log;
            --reject-next-event turns them into a Stage 2t reject.

Controls (data/ambiguity/controls.jsonl): AmbigQA "singleAnswer" records --
annotators agreed there is exactly one answer -- run through the same Stage
2 / 2t / 2b filters. Records that carry BOTH a singleAnswer and a multipleQAs
annotation with >= 2 distinct groups (annotator disagreement) are excluded
from the controls.

Prompt format: shared.prompt_format.build_completion_prompt -- bare question
in the user turn, "The answer is" PREFILLED into the assistant turn, so the
measured next token is the answer continuation, not a turn-opener.

Outputs:
  data/ambiguity/prompts.jsonl            120 records (84 working / 36 held-out)
  data/ambiguity/controls.jsonl           120 records, is_control = true
  data/ambiguity/review_log.jsonl         one line per raw AmbigQA record
  data/ambiguity/control_review_log.jsonl one line per raw AmbigQA record
Review logs are never silently overwritten: if one exists and --overwrite
is not given, the new log goes to review_log.<timestamp>.jsonl.

Record schema: the 7 shared fields (prompt_id, category, raw_prompt,
chat_formatted_prompt, source_dataset, split, is_control) followed by
source_id (AmbigQA id) and answer_groups (list of alias lists).
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "ambiguity"
OUTPUT_PATH = DATA_DIR / "prompts.jsonl"
CONTROLS_OUTPUT_PATH = DATA_DIR / "controls.jsonl"
REVIEW_LOG_PATH = DATA_DIR / "review_log.jsonl"
CONTROL_REVIEW_LOG_PATH = DATA_DIR / "control_review_log.jsonl"

# sewon/ambig_qa, commit "Convert dataset to Parquet (#1)", 2024-01-09.
AMBIGQA_REVISION = "e969d0132f4dd28c2939d55be34f1788c00ccfe7"
SOURCE_DATASET = f"AmbigQA-light (sewon/ambig_qa@{AMBIGQA_REVISION[:7]})"

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


# --- Stage 2t: resolvable time references ---------------------------------
# Ported from the v1 script (archive/pipelines/A_preprocess_ambiguity_v1.py)
# and extended. All applied to the lower-cased ORIGINAL question.

TIME_REF_PATTERNS = (
    # v1 RESOLVABLE_TIME_PATTERNS, verbatim
    r"\blast week\b", r"\bthis week\b", r"\blast month\b", r"\bthis month\b",
    r"\blast year\b", r"\bthis year\b", r"\brecently\b", r"\byesterday\b",
    r"\btoday\b", r"\bcurrently\b", r"\bright now\b", r"\bat the moment\b",
    r"\bthe latest\b", r"\bthe newest\b",
    # extension: ordinal / recency / present-scope words
    r"\b(last|latest|most recent|first|current|currently|this (year|season))\b",
)
TIME_REF_REGEXES = tuple(re.compile(p) for p in TIME_REF_PATTERNS)

# "when did X last ...", "when was the first time X ...", "when did X first ..."
LAST_FIRST_WHEN_PATTERN = re.compile(
    r"^when (did|was|were|is|are|does|do|will|has|have|had)\b.*\b(last|first)\b"
)

# v1 flag_unscoped_present_tense_split: present-tense starter + every
# disambiguated sub-question scoped by before/after/since/prior to.
PRESENT_TENSE_STARTERS = re.compile(r"^(what is|what does|who is|where is|which is)\b")
BEFORE_AFTER_PATTERN = re.compile(r"\b(before|prior to|since|after)\b", re.IGNORECASE)

# "next <event>" future-scheduling subtype (flagged, kept by default)
NEXT_EVENT_PATTERN = re.compile(r"\b(next|upcoming)\s+[a-z0-9'\-]+")


def has_resolvable_time_ref(question: str):
    """Returns the matched pattern string, or None."""
    text = question.strip().lower()
    for rx in TIME_REF_REGEXES:
        m = rx.search(text)
        if m:
            return m.group(0)
    m = LAST_FIRST_WHEN_PATTERN.search(text)
    if m:
        return "when did ... " + m.group(2)
    return None


def flag_unscoped_present_tense_split(question: str, sub_questions: list) -> bool:
    """Present-tense unscoped question whose sub-questions all split on a
    date (before/after/since). One current stable answer -> not ambiguous."""
    if not PRESENT_TENSE_STARTERS.match(question.strip().lower()):
        return False
    if not sub_questions:
        return False
    return all(BEFORE_AFTER_PATTERN.search(q) for q in sub_questions)


def is_next_event_question(question: str) -> bool:
    return NEXT_EVENT_PATTERN.search(question.strip().lower()) is not None


# --- Stage 2b: malformed / run-on ------------------------------------------

WH_WORDS = {"who", "what", "when", "where", "which", "how", "why"}
CLAUSE_SPLIT = re.compile(r"(?:,\s*|\s+and\s+|\s+or\s+)")


def looks_malformed(question: str) -> bool:
    """A second wh-word counts only when it STARTS a clause (after ', ',
    ' and ', ' or '), so titles like 'who wrote what about love' are not
    penalised. Also rejects > 20 words."""
    q = question.strip().lower().rstrip("?")
    words = q.split()
    if len(words) > 20:
        return True
    clauses = [c.strip() for c in CLAUSE_SPLIT.split(q) if c.strip()]
    n_clause_wh = sum(1 for c in clauses if c.split()[0] in WH_WORDS)
    return n_clause_wh >= 2


# --- Stage 3: distinct answers ---------------------------------------------

def normalize_answer(a: str) -> str:
    if not isinstance(a, str):
        return ""
    a = a.lower().strip()
    a = re.sub(r"[^\w\s]", "", a)
    a = re.sub(r"^(the|a|an)\s+", "", a)
    return a.strip()


def _flatten_strings(item) -> list:
    if isinstance(item, str):
        return [item]
    out = []
    if isinstance(item, (list, tuple)):
        for x in item:
            out.extend(_flatten_strings(x))
    return out


def has_genuinely_distinct_answers(answer_sets: list, min_distinct: int = 2) -> bool:
    """>= min_distinct answer groups whose FULL normalized alias sets are
    pairwise disjoint (not just differing first aliases)."""
    norm_sets = []
    for group in answer_sets:
        s = {normalize_answer(a) for a in _flatten_strings(group)}
        s.discard("")
        if s:
            norm_sets.append(s)
    kept = []
    for s in norm_sets:
        if all(not (s & k) for k in kept):
            kept.append(s)
    return len(kept) >= min_distinct


# --- AmbigQA access ---------------------------------------------------------

def load_ambigqa_records():
    from datasets import load_dataset
    return load_dataset("sewon/ambig_qa", "light", split="train", revision=AMBIGQA_REVISION)


def extract_question_and_answer_groups(record: dict):
    """Returns (question, answer_groups, sub_questions) from every
    multipleQAs annotation, or None if the record lacks the fields."""
    question = record.get("question")
    annotations = record.get("annotations")
    if question is None or annotations is None:
        return None
    answer_groups, sub_questions = [], []
    if isinstance(annotations, dict):
        ann_types = annotations.get("type", [])
        qa_pairs_list = annotations.get("qaPairs", [])
        for i, ann_type in enumerate(ann_types):
            if ann_type == "multipleQAs" and i < len(qa_pairs_list):
                qa_pair_dict = qa_pairs_list[i] or {}
                for ans_group in qa_pair_dict.get("answer", []):
                    if ans_group:
                        answer_groups.append(list(_flatten_strings(ans_group)))
                sub_questions.extend(q for q in qa_pair_dict.get("question", []) if q)
    return question.strip(), answer_groups, sub_questions


def extract_single_answer_questions(record: dict):
    """Returns (question, answer_aliases) if some annotator marked the
    record singleAnswer AND no multipleQAs annotation on the same record
    has >= 2 genuinely distinct groups (mixed-annotation records are
    excluded from the control pool). Otherwise None."""
    question = record.get("question")
    annotations = record.get("annotations")
    if question is None or not isinstance(annotations, dict):
        return None
    ann_types = annotations.get("type", [])
    answer_lists = annotations.get("answer", [])
    single = None
    for i, ann_type in enumerate(ann_types):
        if ann_type == "singleAnswer" and i < len(answer_lists) and answer_lists[i]:
            single = list(_flatten_strings(answer_lists[i]))
            break
    if single is None:
        return None
    parsed = extract_question_and_answer_groups(record)
    if parsed is not None and has_genuinely_distinct_answers(parsed[1], min_distinct=2):
        return None
    return question.strip(), single


def pilot_check_single_answer_schema(n_print: int = 10):
    """Prints raw annotation dicts for the first singleAnswer records so the
    field-shape assumptions in extract_single_answer_questions can be
    eyeballed before a bulk run."""
    ds = load_ambigqa_records()
    shown = 0
    for record in ds:
        annotations = record.get("annotations")
        if not annotations or "singleAnswer" not in annotations.get("type", []):
            continue
        print(f"question: {record.get('question')!r}")
        print(f"  annotations: {annotations}")
        print(f"  extracted -> {extract_single_answer_questions(record)!r}\n")
        shown += 1
        if shown >= n_print:
            break
    if shown == 0:
        print("No 'singleAnswer' records found -- schema assumption is wrong, stop.")


# --- helpers -----------------------------------------------------------------

def write_review_log(entries: list, path: Path, overwrite: bool) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        path = path.with_name(f"{path.stem}.{ts}{path.suffix}")
        print(f"  (existing log kept; writing to {path.name} -- pass --overwrite to replace)")
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"Saved review log ({len(entries)} entries) to {path}")
    return path


def _ensure_qmark(q: str) -> str:
    q = q.strip()
    return q if q.endswith("?") else q + "?"


def _attach_extra_fields(records: list, meta_by_question: dict) -> list:
    for r in records:
        m = meta_by_question.get(r["raw_prompt"], {})
        r["source_id"] = m.get("source_id")
        r["answer_groups"] = m.get("answer_groups", [])
    return records


def _save(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_w = sum(1 for r in records if r["split"] == "working")
    print(f"Saved {len(records)} records to {path} ({n_w} working / {len(records) - n_w} held-out)")


# --- working set -------------------------------------------------------------

def build_working_set(tokenizer, ds, n: int, seed: int, overwrite: bool, reject_next_event: bool):
    from shared.prompt_format import build_completion_prompt, build_records_with_formatter

    print(f"Loaded {len(ds)} raw AmbigQA records.")
    review_log, accepted, meta = [], [], {}

    for record in ds:
        rid = record.get("id")
        parsed = extract_question_and_answer_groups(record)
        if parsed is None:
            review_log.append({"id": rid, "question": None, "stage_failed": 0,
                               "reason": "missing question/annotations field"})
            continue
        question, answer_groups, sub_questions = parsed
        entry = {"id": rid, "question": question}

        if len(answer_groups) < 2:
            review_log.append({**entry, "stage_failed": 1,
                               "reason": "fewer than 2 answer groups (not ambiguous per AmbigQA)"})
            continue
        if not is_factoid_question(question):
            review_log.append({**entry, "stage_failed": 2,
                               "reason": "open-ended/explanatory question, no natural short answer"})
            continue
        tref = has_resolvable_time_ref(question)
        if tref is not None:
            review_log.append({**entry, "stage_failed": "2t",
                               "reason": f"resolvable time reference: {tref!r}"})
            continue
        if flag_unscoped_present_tense_split(question, sub_questions):
            review_log.append({**entry, "stage_failed": "2t",
                               "reason": "unscoped present-tense question, sub-questions split before/after a date"})
            continue
        next_event = is_next_event_question(question)
        if next_event and reject_next_event:
            review_log.append({**entry, "stage_failed": "2t",
                               "reason": "'next <event>' future-scheduling question (--reject-next-event)"})
            continue
        if looks_malformed(question):
            review_log.append({**entry, "stage_failed": "2b",
                               "reason": "malformed/run-on: second wh-clause or too long"})
            continue
        if not has_genuinely_distinct_answers(answer_groups, min_distinct=2):
            review_log.append({**entry, "stage_failed": 3,
                               "reason": "answer groups overlap after normalization"})
            continue
        q_key = _ensure_qmark(question)
        if q_key in meta:
            review_log.append({**entry, "stage_failed": "dup", "reason": "duplicate question text"})
            continue

        review_log.append({**entry, "stage_failed": None, "reason": "accepted",
                           "flag_next_event": next_event})
        accepted.append(question)
        meta[q_key] = {"source_id": rid, "answer_groups": answer_groups}

    n_acc = len(accepted)
    print(f"\nWorking-set stage summary: {n_acc} accepted, {len(review_log) - n_acc} rejected.")
    for stage in [0, 1, 2, "2t", "2b", 3, "dup"]:
        print(f"  rejected at stage {stage}: {sum(1 for r in review_log if r['stage_failed'] == stage)}")
    print(f"  accepted but flagged 'next <event>': "
          f"{sum(1 for r in review_log if r.get('flag_next_event'))}")
    write_review_log(review_log, REVIEW_LOG_PATH, overwrite)

    assert n_acc >= n, f"Only {n_acc} items survived all stages, need >= {n}."

    records = build_records_with_formatter(
        raw_prompts=accepted, category="ambiguity", source_dataset=SOURCE_DATASET,
        prefix="amb", tokenizer=tokenizer, formatter=build_completion_prompt,
        n_working=n, split_ratio=0.7, seed=seed,
    )
    _attach_extra_fields(records, meta)
    _save(records, OUTPUT_PATH)
    return records


# --- controls ------------------------------------------------------------------

def build_ambiguity_controls(tokenizer, ds, n: int, seed: int, overwrite: bool):
    from shared.prompt_format import build_completion_prompt, build_records_with_formatter

    print("\nScanning for singleAnswer (non-ambiguous, control) records...")
    review_log, accepted, meta = [], [], {}

    for record in ds:
        rid = record.get("id")
        extracted = extract_single_answer_questions(record)
        if extracted is None:
            review_log.append({"id": rid, "question": record.get("question"), "stage_failed": 1,
                               "reason": "no singleAnswer annotation, or mixed with a multipleQAs annotation"})
            continue
        question, aliases = extracted
        entry = {"id": rid, "question": question}
        if not is_factoid_question(question):
            review_log.append({**entry, "stage_failed": 2,
                               "reason": "open-ended/explanatory question, no natural short answer"})
            continue
        tref = has_resolvable_time_ref(question)
        if tref is not None:
            review_log.append({**entry, "stage_failed": "2t",
                               "reason": f"resolvable time reference: {tref!r}"})
            continue
        if looks_malformed(question):
            review_log.append({**entry, "stage_failed": "2b",
                               "reason": "malformed/run-on: second wh-clause or too long"})
            continue
        q_key = _ensure_qmark(question)
        if q_key in meta:
            review_log.append({**entry, "stage_failed": "dup", "reason": "duplicate question text"})
            continue
        review_log.append({**entry, "stage_failed": None, "reason": "accepted"})
        accepted.append(question)
        meta[q_key] = {"source_id": rid, "answer_groups": [aliases]}

    n_acc = len(accepted)
    print(f"Control-set stage summary: {n_acc} accepted, {len(review_log) - n_acc} rejected.")
    for stage in [1, 2, "2t", "2b", "dup"]:
        print(f"  rejected at stage {stage}: {sum(1 for r in review_log if r['stage_failed'] == stage)}")
    write_review_log(review_log, CONTROL_REVIEW_LOG_PATH, overwrite)

    assert n_acc >= n, (f"Only {n_acc} singleAnswer items survived, need >= {n}. "
                        f"Re-check pilot_check_single_answer_schema().")

    control_records = build_records_with_formatter(
        raw_prompts=accepted, category="ambiguity",
        source_dataset=SOURCE_DATASET + " (singleAnswer, matched control)",
        prefix="amb_ctrl", tokenizer=tokenizer, formatter=build_completion_prompt,
        n_working=n, split_ratio=0.7, seed=seed, is_control=True,
    )
    _attach_extra_fields(control_records, meta)
    _save(control_records, CONTROLS_OUTPUT_PATH)
    return control_records


# --- CLI -----------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing review logs instead of writing timestamped copies")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=120, help="records per side (working + held-out)")
    p.add_argument("--reject-next-event", action="store_true",
                   help="reject 'next <event>' questions instead of keeping them flagged")
    p.add_argument("--pilot", action="store_true",
                   help="only print singleAnswer schema samples and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.pilot:
        pilot_check_single_answer_schema()
        return None, None

    from shared.schema_utils import load_tokenizer  # transformers import deferred to here
    tokenizer = load_tokenizer()
    ds = load_ambigqa_records()

    records = build_working_set(tokenizer, ds, args.n, args.seed, args.overwrite, args.reject_next_event)
    control_records = build_ambiguity_controls(tokenizer, ds, args.n, args.seed, args.overwrite)

    print("\n--- Sample working prompts ---")
    for r in records[:5]:
        print(f"  [{r['prompt_id']}] ...{r['chat_formatted_prompt'][-80:]!r}")
    print("\n--- Sample control prompts ---")
    for r in control_records[:5]:
        print(f"  [{r['prompt_id']}] ...{r['chat_formatted_prompt'][-80:]!r}")
    print("\nNEXT: with the bf16 model loaded, run "
          "shared.prompt_format.verify_induction_quality on records[:25] and "
          "control_records[:25] and compare the two MEANS directly.")
    return records, control_records


if __name__ == "__main__":
    main()
