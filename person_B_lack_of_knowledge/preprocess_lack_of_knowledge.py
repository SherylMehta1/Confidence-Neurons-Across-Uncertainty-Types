"""
preprocess_lack_of_knowledge_v2.py -- Person B, Phase 3

Two changes from the Phase-1/2 version (preprocess_lack_of_knowledge.py):

1. POSITION FIX: uses shared.prompt_format.build_completion_prompt instead
   of baking "The answer is" into the raw_prompt text. See
   shared/prompt_format.py's module docstring for why this matters -- this
   is the "Fix lack-of-knowledge position" row in the remediation plan.

2. MATCHED CONTROLS: also builds data/lack_of_knowledge/controls.jsonl from
   the ANSWERABLE half of UnknownBench-NEC (same factoid filter, same
   cloze-prefill format, same subject-matter distribution -- just real
   entities instead of fabricated ones). This is the "genuinely known"
   twin condition: same prompt shape, same source dataset, low uncertainty.
   Without this, a neuron that tracks "this looks like a trivia question"
   is indistinguishable from a neuron that tracks "the model doesn't know
   this" -- the whole point of Phase 3's control-prompt row.

Run from repo root: `python person_B_lack_of_knowledge/preprocess_lack_of_knowledge_v2.py`
Requires model + tokenizer in memory if you want to run verify_induction_quality
inline (recommended) -- otherwise it only needs the tokenizer.
"""

import sys
sys.path.append(".")
import re
from pathlib import Path

# Reuse everything from the Phase-1/2 script except the cloze conversion --
# the UnknownBench loading/parsing/factoid-filter logic is unchanged and
# already correct.
from person_B_lack_of_knowledge.preprocess_lack_of_knowledge import (
    clone_repo, find_nec_files, inspect_structure, is_factoid_question,
    load_nec_records,
)
from shared.schema_utils import load_tokenizer, save_records
from shared.prompt_format import build_records_with_formatter, build_completion_prompt

OUTPUT_PATH = "data/lack_of_knowledge/prompts.jsonl"
CONTROLS_OUTPUT_PATH = "data/lack_of_knowledge/controls.jsonl"


def extract_factoid_questions(records, want_non_existent: bool) -> list[str]:
    """
    Shared extraction logic for both the unanswerable set (the actual
    lack-of-knowledge prompts) and the answerable set (Phase 3 controls).
    """
    questions = []
    for r in records:
        if bool(r.get("_is_non_existent", False)) != want_non_existent:
            continue
        q = r.get("question") or r.get("query") or r.get("prompt") or r.get("text")
        if q is None:
            continue
        q = q.strip()
        if is_factoid_question(q):
            questions.append(q)
    return questions


def main():
    clone_repo()
    files = find_nec_files()
    if not files:
        raise FileNotFoundError("No NEC files found -- see preprocess_lack_of_knowledge.py")

    raw_records = load_nec_records(files)
    print(f"Loaded {len(raw_records)} raw records.")

    unanswerable_qs = extract_factoid_questions(raw_records, want_non_existent=True)
    answerable_qs = extract_factoid_questions(raw_records, want_non_existent=False)
    print(f"{len(unanswerable_qs)} factoid unanswerable (fabricated-entity) questions")
    print(f"{len(answerable_qs)} factoid answerable (real-entity) questions -- Phase 3 controls")

    tokenizer = load_tokenizer()

    # --- Main lack-of-knowledge set (position-fixed) ---
    records = build_records_with_formatter(
        raw_prompts=unanswerable_qs,
        category="lack_of_knowledge",
        source_dataset="UnknownBench-NEC (unanswerable)",
        prefix="lok",
        tokenizer=tokenizer,
        formatter=build_completion_prompt,
        n_working=120,
        split_ratio=0.7,
    )
    save_records(records, OUTPUT_PATH)

    # --- Matched controls: same shape, real entities ---
    control_records = build_records_with_formatter(
        raw_prompts=answerable_qs,
        category="lack_of_knowledge",
        source_dataset="UnknownBench-NEC (answerable, matched control)",
        prefix="lok_ctrl",
        tokenizer=tokenizer,
        formatter=build_completion_prompt,
        n_working=120,
        split_ratio=0.7,  # keep working/held-out split so controls get the same replication check
        is_control=True,
    )
    Path(CONTROLS_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(CONTROLS_OUTPUT_PATH, "w") as f:
        for r in control_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(control_records)} control prompts to {CONTROLS_OUTPUT_PATH}")

    print("\n--- Sample lack-of-knowledge prompts (position-fixed) ---")
    for r in records[:5]:
        print(f"  [{r['prompt_id']}] ...{r['chat_formatted_prompt'][-80:]!r}")
    print("\n--- Sample control prompts ---")
    for r in control_records[:5]:
        print(f"  [{r['prompt_id']}] ...{r['chat_formatted_prompt'][-80:]!r}")

    print(
        "\nNEXT: with model+tokenizer loaded, run "
        "shared.prompt_format.verify_induction_quality(model, tokenizer, records[:25]) "
        "and the same on control_records[:25] -- expect LOW top1 on the unanswerable "
        "set and HIGH top1 (near your control baseline) on the answerable set. If both "
        "look similar, the factoid filter or the fix itself needs another look before "
        "you trust this data."
    )


if __name__ == "__main__":
    main()
