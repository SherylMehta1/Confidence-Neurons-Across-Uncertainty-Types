"""
preprocess_ambiguity.py -- Person A
Category: ambiguity
Source: AmbigQA (sewon/ambig_qa, "light" config) on HuggingFace.

IMPORTANT -- READ BEFORE RUNNING:
The naive 'multipleQAs, 2+ distinct answers' filter (see build_dataset.py)
over-includes several distinct contamination patterns. This file documents
every automated filter used to narrow the candidate pool, PLUS the manual
review criteria applied on top -- automated filters alone were NOT
sufficient (verified cases: "current Istanbul airport name", "current
Price Is Right host", "current goal-kick rule" all passed every automated
filter below but are NOT genuinely ambiguous -- present-tense unscoped
questions with one stable current answer, caught only by hand-reading and,
for two of them, a live web search to confirm no further change since the
described split).

Every item in the final 120-item set was individually approved via manual
review against the test below; the automated filters only narrow the pool
enough to make manual review tractable (full raw pool: 10,036 examples ->
2,789 non-temporal ambiguous candidates before any manual review).

THE CORE TEST (applied manually to every kept item):
  Does the fork already exist in the LITERAL question as worded, or does
  it require an unstated qualifier / represent a different question
  entirely? Two sub-questions are competing answers to ONE question only
  if a person answering either one is answering what was actually asked.
"""

import json
import re
import random
from datasets import load_dataset

import sys
sys.path.append(".")
from shared.schema_utils import load_tokenizer, build_records, save_records

OUTPUT_PATH = "data/ambiguity/prompts.jsonl"
AUDIT_PATH = "person_A_ambiguity/subtype_audit.jsonl"


# =========================================================================
# STAGE 1 -- base filter: multipleQAs annotation, 2+ distinct answer sets
# =========================================================================

def extract_ambiguous_question(example):
    """Base AmbigQA filter. Over-includes contamination -- see Stage 2/3."""
    annotations = example["annotations"]
    ann_types = annotations["type"]
    if "multipleQAs" not in ann_types:
        return None
    idx = ann_types.index("multipleQAs")
    qa_pairs = annotations["qaPairs"][idx]
    distinct_answer_sets = {tuple(sorted(set(a))) for a in qa_pairs["answer"] if a}
    if len(distinct_answer_sets) < 2:
        return None
    question = example["question"].strip()
    if not question.endswith("?"):
        question += "?"
    return question


# =========================================================================
# STAGE 2 -- automated contamination pre-filters (narrow the pool before
# manual review; do NOT treat as sufficient on their own)
# =========================================================================

RESOLVABLE_TIME_PATTERNS = [
    r"\blast week\b", r"\bthis week\b", r"\blast month\b", r"\bthis month\b",
    r"\blast year\b", r"\bthis year\b", r"\brecently\b", r"\byesterday\b",
    r"\btoday\b", r"\bcurrently\b", r"\bright now\b", r"\bat the moment\b",
    r"\bthe latest\b", r"\bthe newest\b",
]

def has_resolvable_time_ref(raw_prompt):
    """
    Catches phrases like 'last week' in the ORIGINAL question. These are
    NOT genuinely ambiguous: the question had exactly one correct answer
    at the time it was asked. Sub-answers spanning unrelated past dates
    (e.g. "who hit 4 HRs in one game" -> 2012, 2017 answers for a "last
    week" question) are an AmbigQA annotation artifact (annotators
    enumerating any historically-true instance), not alternate readings.
    """
    text = raw_prompt.lower()
    return any(re.search(p, text) for p in RESOLVABLE_TIME_PATTERNS)


PRESENT_TENSE_STARTERS = re.compile(
    r"^(what is|what does|who is|where is|which is)\b", re.IGNORECASE
)
BEFORE_AFTER_PATTERN = re.compile(r"\b(before|prior to|since|after)\b", re.IGNORECASE)

def flag_unscoped_present_tense_split(item):
    """
    Catches unscoped PRESENT-TENSE questions ('what is the name of X')
    whose sub-questions split cleanly into before-DATE / after-DATE.
    These usually have exactly ONE current stable answer and are false
    positives -- verified cases: Istanbul Airport rename (2019, no
    further change through 2026), goal-kick penalty-area rule (2019
    IFAB change, unchanged through 2026), SAP GUI "current version",
    most-followed Instagram account "as of today".

    IMPORTANT: this flags candidates for extra scrutiny / likely
    rejection -- it does NOT verify whether the fact has changed AGAIN
    since the described split. That still requires an actual search
    before approving (see Person A's review log for two live
    verifications: Istanbul Airport [rejected on review despite
    verification], Destiny USA mall rename [rejected on review despite
    verification] -- both confirmed stable via web search, but the team
    decided present-tense-unscoped-name questions are excluded on
    principle regardless of verification outcome).
    """
    if not PRESENT_TENSE_STARTERS.match(item["raw_prompt"].strip()):
        return False
    sub_qs = item["qa_pairs"].get("question", [])
    if not sub_qs:
        return False
    return all(BEFORE_AFTER_PATTERN.search(q) for q in sub_qs)


ROLE_MODIFIERS = {"deputy", "assistant", "acting", "vice", "interim", "former", "co"}

def extract_role_phrase(question):
    q = question.lower()
    match = re.search(r"the ([a-z\- ]+?)(?: (?:of|in|at|since|from|prior|that|,))", q)
    return match.group(1).strip() if match else None

def has_role_drift(item):
    """
    Catches sub-questions that answer a DIFFERENT role than the original
    question asked about (e.g. original asks 'the editor', a sub-question
    answers 'the deputy editor'). A different role is not a valid
    alternate reading of the original question, regardless of any time
    scoping also present. Verified case: Canberra Times editor question,
    where deputy-editor and editor answers were mixed under one
    'multipleQAs' annotation.
    """
    sub_qs = item["qa_pairs"].get("question", [])
    orig_role = extract_role_phrase(item["raw_prompt"])
    if not orig_role:
        return False
    for sq in sub_qs:
        sub_role = extract_role_phrase(sq)
        if sub_role and sub_role != orig_role and any(mod in sub_role for mod in ROLE_MODIFIERS):
            return True
    return False


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "is",
    "was", "were", "are", "who", "what", "when", "where", "which", "how",
    "does", "did", "do", "that", "this", "with", "by", "from", "as", "it",
    "s", "its", "their", "his", "her",
}

def _clean_tokens(text):
    text = text.lower().strip("?").strip()
    tokens = re.findall(r"[a-z0-9']+", text)
    return [t for t in tokens if t not in STOPWORDS]

def classify_different_attribute(item, n_trailing=2):
    """
    High-confidence AUTO-DROP heuristic: if sub-questions share no common
    trailing content word(s), they very likely ask about a DIFFERENT
    ATTRIBUTE of the entity entirely (e.g. songwriter vs producer vs
    singer; city vs country vs event vs theology) rather than competing
    readings of one question. Verified against a 20-item spot-check of
    this bucket before trusting it at scale -- still treat as "very
    likely drop, not confirmed drop" (false positive/negative rate not
    formally measured).
    """
    sub_qs = item["qa_pairs"].get("question", [])
    trailing = []
    for q in sub_qs:
        tokens = _clean_tokens(q)
        if len(tokens) >= n_trailing:
            trailing.append(set(tokens[-n_trailing:]))
        elif tokens:
            trailing.append(set(tokens))
    if len(trailing) < 2:
        return False
    common = set.intersection(*trailing) if trailing else set()
    return len(common) == 0


# =========================================================================
# STAGE 3 -- manual review test (applied by hand to every candidate that
# survives Stage 2; this is the actual final filter, not the code above)
# =========================================================================
"""
MANUAL REVIEW TEST -- applied to every item that reached final approval:

REJECT if:
  - Sub-questions answer a DIFFERENT ATTRIBUTE of the entity (different
    role, different body part, different data field) -- not competing
    readings of the same question.
  - Sub-questions describe DIFFERENT MILESTONES/EVENTS in a sequence
    (e.g. "officially joined" vs "first season played"; "announced" vs
    "occurred") -- these are a timeline, not a fork.
  - The original question is an ORDINAL LIST question ("X in order",
    "the Nth book in the series") -- every answer is correct for a
    different ordinal, there's no genuine fork.
  - The original question contains a RESOLVABLE relative-time phrase
    ("last week", "last time X happened") where sub-answers are just
    different historically-true instances, not valid alternate readings.
  - The original question is UNSCOPED PRESENT-TENSE ("what is", "who is")
    asking about something with ONE current answer -- even if it split
    historically at some point, if nothing has changed since, there is
    one right answer today. (Team decision: exclude these on principle,
    per items above.)
  - Sub-answers are a COMPOUND/list answer to a multi-part question
    (e.g. "where does the Seine start AND end") -- both parts are
    simultaneously true, not alternatives.

APPROVE if:
  - Two+ genuinely different real-world entities/works share a name or
    description (referential-lexical) -- e.g. two films with the same
    title, two people who held a role at different times with no single
    "default" answer, multiple simultaneous role-holders (e.g. Test vs
    ODI cricket captain, both true AT THE SAME TIME).
  - A word/phrase in the question has multiple defensible senses
    (definitional-scope) -- e.g. "built" spanning "established" vs
    "expanded"; "type" spanning "class" vs "breed".
  - The question turns on WHICH FRAMEWORK/LENS applies (conceptual-
    framework) -- e.g. botanical vs culinary classification, verified
    vs Biblical record, technically-vs-in-practice status.
  - The question has NO single default reading even with full context,
    because it depends on an unstated CONDITIONAL PARAMETER (e.g. pension
    eligibility depends on age; "next World Cup" depends on what "now" is).

Subtype labels used in person_A_ambiguity/subtype_audit.jsonl:
  referential-lexical, definitional-scope, temporal, conceptual-framework,
  de-jure-de-facto, conditional-parameter
"""


# =========================================================================
# Pipeline runner (Stages 1-2 only; Stage 3 was done via interactive
# review, logged to approval_tracker.jsonl -- see finalize step below)
# =========================================================================

def load_and_prefilter(seed=42):
    print("Loading AmbigQA (sewon/ambig_qa, light)...")
    ds = load_dataset("sewon/ambig_qa", "light", split="train")
    print(f"Scanning {len(ds)} examples...")

    candidates = []
    for example in ds:
        q = extract_ambiguous_question(example)
        if q is None:
            continue
        ann = example["annotations"]
        idx = ann["type"].index("multipleQAs")
        candidates.append({
            "raw_prompt": q,
            "qa_pairs": ann["qaPairs"][idx],
        })
    print(f"Stage 1 (base multipleQAs filter): {len(candidates)} candidates")

    # Stage 2 filters -- narrow before manual review, don't auto-approve
    survivors = []
    for item in candidates:
        if has_resolvable_time_ref(item["raw_prompt"]):
            continue
        if flag_unscoped_present_tense_split(item):
            continue
        if has_role_drift(item):
            continue
        if classify_different_attribute(item):
            continue  # high-confidence auto-drop
        survivors.append(item)

    print(f"Stage 2 (automated contamination filters): {len(survivors)} candidates "
          f"remaining for manual review")
    return survivors


def finalize_from_approval_log(tracker_path="approval_tracker.jsonl"):
    """
    Stage 3 result: pulls approved, deduplicated prompts from the manual
    review log and writes the final shared-schema file. This is the
    ACTUAL source of truth for the 120-item working set -- Stages 1-2
    above only produced the pool that was manually reviewed.
    """
    approved_prompts, seen = [], set()
    with open(tracker_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["decision"] == "approved" and entry["raw_prompt"] not in seen:
                approved_prompts.append(entry["raw_prompt"])
                seen.add(entry["raw_prompt"])

    print(f"Loaded {len(approved_prompts)} approved, deduplicated prompts "
          f"from manual review")

    tokenizer = load_tokenizer()
    records = build_records(
        raw_prompts=approved_prompts,
        category="ambiguity",
        source_dataset="AmbigQA",
        prefix="amb",
        tokenizer=tokenizer,
        n_working=len(approved_prompts),
        split_ratio=0.7,
        seed=42,
    )
    save_records(records, OUTPUT_PATH)


if __name__ == "__main__":
    # Stages 1-2: run to regenerate the manual-review candidate pool
    survivors = load_and_prefilter()

    # Stage 3 was done interactively (approval_tracker.jsonl already
    # exists from that session) -- uncomment to regenerate the final
    # schema file from the existing approval log:
    # finalize_from_approval_log()
