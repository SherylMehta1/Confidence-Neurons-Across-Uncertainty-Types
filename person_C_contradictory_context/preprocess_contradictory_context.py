"""
preprocess_contradictory_context.py -- Person C (your category)
Category: contradictory_context
Source: azhx/counterfact on HuggingFace (from the ROME paper, 21.9k rows).

Builds prompts using the "Redefine" pattern (Ortu et al.'s replication of
the CounterFact construction, also used by Context Copying Modulation 2025):

    "Redefine: {base_prompt_with_subject} {target_new}. {base_prompt_with_subject}"

e.g. "Redefine: The official language of Australia is Indonesian.
      The official language of Australia is"

The model has genuine parametric knowledge of target_true, but the
prepended context asserts target_new -- uncertainty here comes from
conflict resolution between the two knowledge sources, not absence or
multiplicity of information.

NOTE: this script tries both the nested ROME-style schema
(record["requested_rewrite"]["prompt"/"target_new"/"target_true"/"subject"])
and a possible flattened schema (top-level "prompt"/"target_new"/etc.),
since the exact column layout of azhx/counterfact specifically wasn't
independently verified before writing this. Run inspect_first_record()
first and delete whichever branch doesn't apply.
"""

import sys
sys.path.append(".")  # run this script from the repo root
from datasets import load_dataset

from shared.schema_utils import load_tokenizer, build_records, save_records

OUTPUT_PATH = "data/contradictory_context/prompts.jsonl"


def inspect_first_record(ds):
    """Run this first to confirm which schema branch below actually applies."""
    print(ds[0])


def get_field(record, *candidates):
    """Try several possible key paths, return the first that resolves."""
    for path in candidates:
        obj = record
        try:
            for key in path:
                obj = obj[key]
            if obj not in (None, ""):
                return obj
        except (KeyError, TypeError, IndexError):
            continue
    return None


def extract_fields(record) -> dict | None:
    """
    Pulls subject / relation_prompt_template / target_true / target_new
    out of either the nested ROME schema or a flattened one.
    relation_prompt_template is expected to contain "{}" for the subject,
    e.g. "The official language of {} is".
    """
    subject = get_field(
        record,
        ("requested_rewrite", "subject"),
        ("subject",),
    )
    prompt_template = get_field(
        record,
        ("requested_rewrite", "prompt"),
        ("prompt",),
    )
    target_true = get_field(
        record,
        ("requested_rewrite", "target_true", "str"),
        ("target_true",),
    )
    target_new = get_field(
        record,
        ("requested_rewrite", "target_new", "str"),
        ("target_new",),
    )

    if not all([subject, prompt_template, target_true, target_new]):
        return None
    if target_true.strip().lower() == target_new.strip().lower():
        return None  # not actually contradictory

    return {
        "subject": subject,
        "prompt_template": prompt_template,  # contains "{}"
        "target_true": target_true.strip(),
        "target_new": target_new.strip(),
    }


def build_redefine_prompt(fields: dict) -> str:
    """
    "Redefine: {base_prompt} {target_new}. {base_prompt}"
    base_prompt = prompt_template with subject filled in, e.g.
    "The official language of Australia is"
    """
    base_prompt = fields["prompt_template"].format(fields["subject"]).strip()
    return f"Redefine: {base_prompt} {fields['target_new']}. {base_prompt}"


def main():
    print("Loading azhx/counterfact ...")
    ds = load_dataset("azhx/counterfact", split="train")

    print("First record (inspect to confirm schema branch used below):")
    inspect_first_record(ds)

    raw_prompts = []
    skipped = 0
    for record in ds:
        fields = extract_fields(record)
        if fields is None:
            skipped += 1
            continue
        raw_prompts.append(build_redefine_prompt(fields))

    print(f"Built {len(raw_prompts)} contradiction prompts "
          f"({skipped} records skipped for missing/matching fields; "
          f"need >= 120 to subsample).")

    tokenizer = load_tokenizer()
    records = build_records(
        raw_prompts=raw_prompts,
        category="contradictory_context",
        source_dataset="CounterFact (azhx/counterfact)",
        prefix="cc",
        tokenizer=tokenizer,
        n_working=120,
        split_ratio=0.7,
    )
    save_records(records, OUTPUT_PATH)


if __name__ == "__main__":
    main()
