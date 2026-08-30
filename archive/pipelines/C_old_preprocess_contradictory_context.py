"""
preprocess_contradictory_context.py -- Person C (contradictory_context)
Source: azhx/counterfact on HuggingFace (ROME paper, ~21.9k rows).

Builds prompts using the "Redefine" pattern (Ortu et al.'s replication of
the CounterFact construction, also used by Context Copying Modulation 2025):

    "Redefine: {base_prompt_with_subject} {target_new}. {base_prompt_with_subject}"

e.g. "Redefine: The official language of Australia is Indonesian.
      The official language of Australia is"

The model has genuine parametric knowledge of target_true, but the
prepended context asserts target_new -- uncertainty here comes from
conflict resolution between the two knowledge sources, not absence or
multiplicity of information.

FIX (2026-08-08): some relations in CounterFact ship with grammatically
incomplete prompt templates (e.g. "{} owner" instead of "{} is owned by",
or "{}, a native Dutch" missing "speaker of"). TEMPLATE_OVERRIDES is
auto-populated at runtime from coastalcph/pararel_patterns, matched by
shared Wikidata relation_id, via build_template_overrides() below.
Confirmed via diagnose_templates.py that ParaRel uses different field
names/placeholder syntax than originally assumed:
  - relation id lives in column "relation", as e.g. "P30.jsonl" (needs
    the ".jsonl" suffix stripped before it matches CounterFact's "P30")
  - template lives in column "template", using "[X]"/"[Y]" placeholders
    instead of CounterFact's "{}" -- converted below, keeping only the
    [X] (subject) slot since the Redefine pattern only fills in subject.
Relations broken in CounterFact with no ParaRel match (currently just
P641, "professionally plays the sport") are dropped rather than
hand-written.
"""

import re
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


def build_template_overrides():
    """
    Scans CounterFact for relations whose own prompt template looks like a
    grammatical fragment (heuristic), then tries to replace each with a
    clean template from ParaRel, matched on shared Wikidata relation_id.

    Returns (overrides: dict[relation_id -> corrected "{}"-style template],
             unresolved: set[relation_id with no ParaRel match -> drop]).
    """
    ds = load_dataset("azhx/counterfact", split="train")

    seen = {}
    for record in ds:
        template = get_field(record, ("requested_rewrite", "prompt"), ("prompt",))
        relation_id = get_field(record, ("requested_rewrite", "relation_id"), ("relation_id",))
        if template and relation_id and relation_id not in seen:
            seen[relation_id] = template

    # heuristic flag: template's tail (after the {} slot) doesn't contain
    # a connector word -> likely a sentence fragment
    suspicious = set()
    for rel_id, tmpl in seen.items():
        tail = tmpl.split("{}")[-1].strip()
        if tail == "" or not re.search(r"\b(is|are|was|were|speaks|of|by|in|for)\b", tail, re.I):
            suspicious.add(rel_id)

    pararel = load_dataset("coastalcph/pararel_patterns", split="train")
    pararel_lookup = {}
    for row in pararel:
        rel_id = row.get("relation")
        if rel_id and rel_id.endswith(".jsonl"):
            rel_id = rel_id[: -len(".jsonl")]
        raw_template = row.get("template")
        if rel_id and raw_template and rel_id not in pararel_lookup:
            # convert ParaRel's [X]/[Y] style to CounterFact's {} style,
            # keep only the [X] (subject) slot -- Redefine only fills that in
            converted = raw_template.replace("[X]", "{}")
            converted = converted.replace(" [Y]", "").replace("[Y]", "")
            converted = converted.strip()
            if converted.endswith("."):
                converted = converted[:-1].strip()
            pararel_lookup[rel_id] = converted

    overrides = {r: pararel_lookup[r] for r in suspicious if r in pararel_lookup}
    unresolved = suspicious - overrides.keys()
    return overrides, unresolved


def extract_fields(record, unresolved: set) -> dict | None:
    """
    Pulls subject / relation_prompt_template / target_true / target_new
    out of either the nested ROME schema or a flattened one.
    relation_prompt_template is expected to contain "{}" for the subject,
    e.g. "The official language of {} is".
    Records whose relation has no clean template available (broken in
    CounterFact, no ParaRel match) are dropped rather than hand-written.
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
    relation_id = get_field(
        record,
        ("requested_rewrite", "relation_id"),
        ("relation_id",),
    )

    if not all([subject, prompt_template, target_true, target_new]):
        return None
    if target_true.strip().lower() == target_new.strip().lower():
        return None  # not actually contradictory
    if relation_id in unresolved:
        return None  # known-broken template, no clean fix available -> drop

    return {
        "subject": subject,
        "prompt_template": prompt_template,  # contains "{}"
        "target_true": target_true.strip(),
        "target_new": target_new.strip(),
        "relation_id": relation_id,
    }


def build_redefine_prompt(fields: dict, overrides: dict) -> str:
    """
    "Redefine: {base_prompt} {target_new}. {base_prompt}"
    base_prompt = prompt_template with subject filled in, e.g.
    "The official language of Australia is"
    Uses a ParaRel-sourced override template if this relation was flagged
    as broken in CounterFact and a replacement was found; otherwise falls
    back to CounterFact's own template.
    """
    template = overrides.get(fields["relation_id"], fields["prompt_template"])
    base_prompt = template.format(fields["subject"]).strip()
    return f"Redefine: {base_prompt} {fields['target_new']}. {base_prompt}"


def main():
    print("Building template overrides from ParaRel...")
    overrides, unresolved = build_template_overrides()
    print(f"{len(overrides)} relations auto-patched from ParaRel; "
          f"{len(unresolved)} unresolved (will be dropped): {sorted(unresolved)}")

    print("\nLoading azhx/counterfact ...")
    ds = load_dataset("azhx/counterfact", split="train")

    print("First record (inspect to confirm schema branch used below):")
    inspect_first_record(ds)

    raw_prompts = []
    skipped = 0
    for record in ds:
        fields = extract_fields(record, unresolved)
        if fields is None:
            skipped += 1
            continue
        raw_prompts.append(build_redefine_prompt(fields, overrides))

    print(f"\nBuilt {len(raw_prompts)} contradiction prompts "
          f"({skipped} records skipped for missing/matching fields or "
          f"unresolved relation; need >= 120 to subsample).")

    tokenizer = load_tokenizer()
    records = build_records(
        raw_prompts=raw_prompts,
        category="contradictory_context",
        source_dataset="CounterFact (azhx/counterfact), templates patched via ParaRel (coastalcph/pararel_patterns)",
        prefix="cc",
        tokenizer=tokenizer,
        n_working=120,
        split_ratio=0.7,
    )
    save_records(records, OUTPUT_PATH)


if __name__ == "__main__":
    main()
