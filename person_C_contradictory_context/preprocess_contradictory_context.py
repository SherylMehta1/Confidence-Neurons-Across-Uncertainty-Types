"""
preprocess_contradictory_context_v2.py -- Person C, Phase 3

Two changes from preprocess_contradictory_context.py:

1. RESTORE MODEL-KNOWS FILTER: the live pipeline (preprocess_contradictory_context.py)
   never checked that the model actually has parametric knowledge of target_true
   before building a contradiction prompt around it -- for an obscure CounterFact
   subject, the prompt then tests "the model never knew this" (lack-of-knowledge)
   dressed up as a contradiction test, not genuine conflict-resolution uncertainty.

   NOTE ON A BUG IN THE FIRST VERSION OF THIS FILE: the check was initially ported
   directly from build_dataset.py's model_knows_fact(), which only tests
   top1_prob > threshold and never looks at WHICH token that is -- true_object was
   accepted as a parameter and never used. That passes a model that's confidently
   wrong just as easily as one that's confidently right, which is not "requiring
   top token = true object" as the remediation plan actually specifies. Fixed
   below: the top-1 token is decoded and compared against true_object's own first
   token before the fact counts as "known." The probability threshold is now a
   secondary, optional strictness knob on top of that match, not a substitute for it.

   NOTE: this makes preprocessing model-dependent -- unlike the current CPU-only
   preprocess_contradictory_context.py, this needs `model` + `tokenizer` already
   loaded (same pattern as build_dataset.py). Run this on Kaggle/RunPod, not locally.

2. MATCHED CONTROLS: same "Redefine" surface structure, but reasserting the TRUE
   fact instead of a false one -- "Redefine: Thailand belongs to the continent of
   Asia. Thailand belongs to the continent of" (context confirms what the model
   already believes, so there's no conflict to resolve). Same subjects/relations
   as the working set where possible, so the control is a genuine twin, not just
   a different sample.

Run from repo root, inside a Kaggle/RunPod session with model+tokenizer already
loaded (same convention as build_dataset.py):
    exec(open("person_C_contradictory_context/preprocess_contradictory_context_v2.py").read())
    build_and_save(model, tokenizer)
"""

import sys
sys.path.append(".")
import json
from datasets import load_dataset

from person_C_contradictory_context.old_preprocess_contradictory_context import (
    build_template_overrides, get_field,
)
from shared.model_utils import get_next_token_probs, compute_top1_prob
from shared.schema_utils import load_tokenizer, build_records

OUTPUT_PATH = "data/contradictory_context/prompts.jsonl"
CONTROLS_OUTPUT_PATH = "data/contradictory_context/controls.jsonl"


def model_knows_fact(
    model, tokenizer, base_prompt: str, true_object: str,
    min_prob: float = None, verbose: bool = False,
) -> bool:
    """
    Parametric-knowledge check, done on base_prompt BEFORE the contradiction
    context is layered on top: requires the model's TOP-1 next token to
    match true_object's own first token -- not just "the model is
    confident about something."

    true_object is often multi-token/multi-word ("United Kingdom", "Marie
    Curie"). Rather than requiring the whole phrase to match in one token
    (which would reject almost everything), this follows the standard
    ROME/CounterFact convention: tokenize true_object on its own (with a
    leading space, matching how it would actually continue the prompt) and
    compare the model's top-1 token against ONLY true_object's first token.
    This is a heuristic, not a proof the model "knows" the full fact --
    pilot on ~20 examples with verbose=True and read the printed
    top-token/true-object pairs before trusting it at scale, same as you'd
    do for any new filter in this project.

    min_prob is now an OPTIONAL secondary gate (default off) on top of the
    match requirement, for cases where you want to additionally require a
    minimum confidence, not a substitute for checking token identity.
    """
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": base_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    probs = get_next_token_probs(model, tokenizer, chat_prompt)
    top1_prob = compute_top1_prob(probs)
    top_token_id = int(probs.argmax().item())
    top_token_str = tokenizer.decode([top_token_id]).strip().lower()

    true_object_token_ids = tokenizer.encode(" " + true_object.strip(), add_special_tokens=False)
    if not true_object_token_ids:
        return False
    true_object_first_token_str = tokenizer.decode([true_object_token_ids[0]]).strip().lower()

    is_match = top_token_str == true_object_first_token_str
    passes_prob_gate = (min_prob is None) or (top1_prob > min_prob)

    if verbose:
        match_flag = "MATCH" if is_match else "no match"
        print(f"    top1={top_token_str!r} (p={top1_prob:.3f})  vs  "
              f"true_object first token={true_object_first_token_str!r}  [{match_flag}]")

    return is_match and passes_prob_gate


def extract_fields_flexible(record, unresolved: set):
    """Same field-extraction logic as preprocess_contradictory_context.py's
    extract_fields(), duplicated here to avoid a fragile cross-import of a
    "private" helper -- keep in sync if that file's schema handling changes."""
    subject = get_field(record, ("requested_rewrite", "subject"), ("subject",))
    prompt_template = get_field(record, ("requested_rewrite", "prompt"), ("prompt",))
    target_true = get_field(record, ("requested_rewrite", "target_true", "str"), ("target_true",))
    target_new = get_field(record, ("requested_rewrite", "target_new", "str"), ("target_new",))
    relation_id = get_field(record, ("requested_rewrite", "relation_id"), ("relation_id",))

    if not all([subject, prompt_template, target_true, target_new]):
        return None
    if target_true.strip().lower() == target_new.strip().lower():
        return None
    if relation_id in unresolved:
        return None

    return {
        "subject": subject, "prompt_template": prompt_template,
        "target_true": target_true.strip(), "target_new": target_new.strip(),
        "relation_id": relation_id,
    }


def build_and_save(model, tokenizer, n_target=120, knows_fact_min_prob=None, verbose_knows_fact=False, seed=42):
    print("Building template overrides from ParaRel...")
    overrides, unresolved = build_template_overrides()
    print(f"{len(overrides)} relations auto-patched; {len(unresolved)} unresolved (dropped)")

    print("\nLoading azhx/counterfact ...")
    ds = load_dataset("azhx/counterfact", split="train")

    contradiction_raw, control_raw = [], []
    checked, knows_fact_count = 0, 0

    for record in ds:
        fields = extract_fields_flexible(record, unresolved)
        if fields is None:
            continue
        template = overrides.get(fields["relation_id"], fields["prompt_template"])
        base_prompt = template.format(fields["subject"]).strip()

        checked += 1
        if not model_knows_fact(model, tokenizer, base_prompt, fields["target_true"],
                                 min_prob=knows_fact_min_prob, verbose=verbose_knows_fact):
            continue
        knows_fact_count += 1

        contradiction_raw.append(
            f"Redefine: {base_prompt} {fields['target_new']}. {base_prompt}"
        )
        control_raw.append(
            f"Redefine: {base_prompt} {fields['target_true']}. {base_prompt}"
        )

        if len(contradiction_raw) >= n_target * 2:  # headroom before subsampling in build_records
            break

    print(f"Checked {checked} rows; model knew the true fact for {knows_fact_count} "
          f"({knows_fact_count/max(checked,1)*100:.1f}%) -- these are the only ones "
          f"used, for both the contradiction set and its matched true-object control.")

    records = build_records(
        raw_prompts=contradiction_raw,
        category="contradictory_context",
        source_dataset="CounterFact + ParaRel, model-knows-filtered",
        prefix="cc",
        tokenizer=tokenizer,
        n_working=n_target,
        split_ratio=0.7,
        seed=seed,
    )
    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} contradiction prompts to {OUTPUT_PATH}")

    control_records = build_records(
        raw_prompts=control_raw,
        category="contradictory_context",
        source_dataset="CounterFact + ParaRel, true-object control (matched)",
        prefix="cc_ctrl",
        tokenizer=tokenizer,
        n_working=n_target,
        split_ratio=0.7,
        seed=seed,  # SAME seed + same underlying subject/relation ordering as
                    # the contradiction set above, so cc_ctrl_0007 and cc_0007
                    # are the SAME subject/relation -- a true matched pair.
    )
    for r in control_records:
        r["is_control"] = True
    with open(CONTROLS_OUTPUT_PATH, "w") as f:
        for r in control_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(control_records)} matched true-object control prompts to {CONTROLS_OUTPUT_PATH}")

    return records, control_records


if __name__ == "__main__":
    print("This script expects `model` and `tokenizer` already loaded in your "
          "session (same convention as build_dataset.py) -- call "
          "build_and_save(model, tokenizer) directly rather than running this "
          "file standalone.")
