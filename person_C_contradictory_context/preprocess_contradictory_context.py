"""
preprocess_contradictory_context.py -- Person C, contradictory-context category.
THE live entry point (needs GPU + the bf16 Llama-3.1-8B-Instruct).

    python person_C_contradictory_context/preprocess_contradictory_context.py
        [--n 120] [--seed 42] [--min-prob 0.0] [--allow-nf4] [--verbose]

Source: CounterFact (azhx/counterfact, ROME) for (subject, relation,
target_true, target_new) plus ParaRel (coastalcph/pararel_patterns) for
replacement templates where CounterFact's own prompt template is a sentence
fragment. Both HF revisions are pinned below. Construction is the "Redefine"
knowledge-conflict pattern of Tighidet et al. (2024):

    user:      Redefine: <base_prompt> <target_new>.
    assistant: <base_prompt>                       <-- prefilled; measured token follows

where base_prompt = template.format(subject), e.g. "Thailand belongs to the
continent of". Controls reassert the TRUE object ("... of Asia.") with the
same prefill, so the only difference is whether the context conflicts with
what the model already knows.

MODEL-KNOWS FILTER (model_knows_fact): a CounterFact row is used only if the
model's top-1 next token at the SAME position as the data -- base_prompt
prefilled in the assistant turn, after a neutral user turn that contains
no fact (KNOWS_FACT_USER_TEXT) -- equals the first token of " " + target_true
(leading space: that is how the object would actually continue the prompt).
--min-prob adds an optional confidence floor on top of the identity match.
Every checked row is logged to data/contradictory_context/knows_fact_log.jsonl.

PRECISION: the knows-fact filter is a model measurement that decides the
dataset, so it must be run in bf16 (load_model(quantize=False)). An NF4 model
is refused unless --allow-nf4 is passed (development only). The precision
is recorded in source_dataset and in data/contradictory_context/provenance.json.

Outputs: data/contradictory_context/{prompts,controls}.jsonl (n records each;
cc_XXXX and cc_ctrl_XXXX share subject/relation 1:1), knows_fact_log.jsonl,
provenance.json. Records carry the 7 shared fields then case_id,
relation_id, subject, target_true, target_new.
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "contradictory_context"
OUTPUT_PATH = DATA_DIR / "prompts.jsonl"
CONTROLS_OUTPUT_PATH = DATA_DIR / "controls.jsonl"
KNOWS_FACT_LOG_PATH = DATA_DIR / "knows_fact_log.jsonl"
PROVENANCE_PATH = DATA_DIR / "provenance.json"

COUNTERFACT_REVISION = "c01c413f856ee38f5c080c9fc5e87aff478e2ff9"   # azhx/counterfact
PARAREL_REVISION = "aadfae52549bb0eb5b6729b27f0d8240d4f55f4f"       # coastalcph/pararel_patterns

# Neutral user turn for the knows-fact check: it must contain no fact, so
# the prefilled base_prompt is completed from parametric memory alone, at
# the same (assistant-prefill) position the treatment/control prompts use.
KNOWS_FACT_USER_TEXT = "Complete the sentence."

# --- template repair ---------------------------------------------------------

CONNECTOR_RE = re.compile(r"\b(is|are|was|were|speaks|of|by|in|for|on|at|to|with|from)\b", re.I)

# Hand-curated CounterFact template tails (text after "{}") known to be
# ungrammatical fragments. Compared after .strip(); see is_fragment_template.
BAD_TAILS = {
    "'s owner", ", from", ", a native", "spoke the language", "originates from",
    "'s capital,", "they understand", "premiered on", "from", ", named after",
}


def is_fragment_template(template: str) -> bool:
    """True if the template's tail (after the subject slot) is not a
    grammatical lead-in to an object: empty, no connector word, a short
    comma-fragment (tail starts with ',' and has <= 2 words), or a known
    bad tail from BAD_TAILS."""
    if "{}" not in template:
        return True
    tail = template.split("{}", 1)[1]
    t = tail.strip()
    if t in BAD_TAILS or tail.rstrip() in BAD_TAILS:
        return True
    if t == "":
        return True
    if t.startswith(",") and len(t.lstrip(",").split()) <= 2:
        return True
    if not CONNECTOR_RE.search(t):
        return True
    return False


def get_field(record, *candidates):
    """Try several key paths, return the first that resolves to a non-empty value."""
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


def _pararel_to_counterfact(raw_template: str) -> str:
    """ParaRel '[X] ... [Y].' -> CounterFact '{} ...' (subject slot only)."""
    converted = raw_template.replace("[X]", "{}")
    converted = converted.replace(" [Y]", "").replace("[Y]", "").strip()
    if converted.endswith("."):
        converted = converted[:-1].strip()
    return converted


def build_template_overrides(ds, pararel):
    """
    Scans every (relation_id, template) pair in CounterFact `ds`. For each
    template that is_fragment_template(), looks for the first ParaRel
    pattern for the same Wikidata relation that is NOT itself a fragment.

    Returns (overrides: {(relation_id, template) -> clean template},
             unresolved: {(relation_id, template)} with no usable ParaRel
             pattern -- records using them are dropped,
             seen: {relation_id -> set(templates)} for reporting).
    """
    seen = {}
    for record in ds:
        template = get_field(record, ("requested_rewrite", "prompt"), ("prompt",))
        relation_id = get_field(record, ("requested_rewrite", "relation_id"), ("relation_id",))
        if template and relation_id:
            seen.setdefault(relation_id, set()).add(template)

    pararel_lookup = {}
    for row in pararel:
        rel_id = row.get("relation")
        if rel_id and rel_id.endswith(".jsonl"):
            rel_id = rel_id[: -len(".jsonl")]
        raw_template = row.get("template")
        if rel_id and raw_template:
            pararel_lookup.setdefault(rel_id, []).append(_pararel_to_counterfact(raw_template))

    overrides, unresolved = {}, set()
    for rel_id, templates in seen.items():
        for tmpl in templates:
            if not is_fragment_template(tmpl):
                continue
            replacement = next((c for c in pararel_lookup.get(rel_id, []) if not is_fragment_template(c)), None)
            if replacement is None:
                unresolved.add((rel_id, tmpl))
            else:
                overrides[(rel_id, tmpl)] = replacement
    return overrides, unresolved, seen


def extract_fields(record, unresolved: set):
    subject = get_field(record, ("requested_rewrite", "subject"), ("subject",))
    prompt_template = get_field(record, ("requested_rewrite", "prompt"), ("prompt",))
    target_true = get_field(record, ("requested_rewrite", "target_true", "str"), ("target_true",))
    target_new = get_field(record, ("requested_rewrite", "target_new", "str"), ("target_new",))
    relation_id = get_field(record, ("requested_rewrite", "relation_id"), ("relation_id",))
    case_id = get_field(record, ("case_id",))
    if not all([subject, prompt_template, target_true, target_new]):
        return None
    if target_true.strip().lower() == target_new.strip().lower():
        return None
    if (relation_id, prompt_template) in unresolved:
        return None
    return {
        "case_id": case_id, "subject": subject, "prompt_template": prompt_template,
        "target_true": target_true.strip(), "target_new": target_new.strip(),
        "relation_id": relation_id,
    }


# --- model-knows filter -------------------------------------------------------

def _model_precision(model) -> str:
    prec = getattr(model, "cn_precision", None)
    if prec:
        return prec
    qc = getattr(getattr(model, "config", None), "quantization_config", None)
    if qc is not None:
        return "nf4" if getattr(qc, "load_in_4bit", False) else "quantized"
    return "unknown"


def model_knows_fact(model, tokenizer, base_prompt: str, target_true: str,
                     min_prob: float = 0.0, user_text: str = KNOWS_FACT_USER_TEXT):
    """
    Returns (passed, info). The check prompt prefills base_prompt into the
    assistant turn after a neutral user turn -- exactly the position the
    treatment/control prompts measure at (minus the Redefine context) --
    and requires top-1 token id == first token id of " " + target_true.
    """
    import torch
    from shared.prompt_format import build_completion_prompt
    from shared.model_utils import tokenize_prompt

    formatted = build_completion_prompt(tokenizer, user_text, cloze_suffix=base_prompt,
                                        ensure_question_mark=False)
    inputs = tokenize_prompt(tokenizer, formatted["chat_formatted_prompt"], device=model.device)
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top1_id = int(probs.argmax().item())
    top1_prob = float(probs[top1_id].item())

    true_ids = tokenizer.encode(" " + target_true.strip(), add_special_tokens=False)
    if not true_ids:
        return False, {"top1_token": tokenizer.decode([top1_id]), "top1_prob": top1_prob, "passed": False}
    true_first = int(true_ids[0])
    true_prob = float(probs[true_first].item())
    passed = (top1_id == true_first) and (top1_prob >= min_prob)
    info = {
        "top1_token": tokenizer.decode([top1_id]), "top1_id": top1_id, "top1_prob": top1_prob,
        "true_first_token": tokenizer.decode([true_first]), "true_first_id": true_first,
        "true_first_prob": true_prob, "passed": passed,
    }
    return passed, info


# --- prompt construction ------------------------------------------------------

def _redefine_formatter(tokenizer, raw_prompt, **kwargs):
    """raw_prompt = 'Redefine: <base_prompt> <target>.|||SPLIT|||<base_prompt>'.
    The Redefine sentence is the user turn; base_prompt is PREFILLED into the
    assistant turn (turn-boundary fix)."""
    from shared.prompt_format import build_completion_prompt
    user_content, continuation = raw_prompt.split("|||SPLIT|||")
    formatted = build_completion_prompt(tokenizer, user_content, cloze_suffix=continuation,
                                        ensure_question_mark=False)
    formatted["raw_prompt"] = f"{user_content} {continuation}"
    return formatted


EXTRA_FIELDS = ("case_id", "relation_id", "subject", "target_true", "target_new")


def build_and_save(model, tokenizer, n_target=120, min_prob=0.0, seed=42, verbose=False,
                   allow_nf4=False, user_text=KNOWS_FACT_USER_TEXT):
    from datasets import load_dataset
    from shared.prompt_format import build_records_with_formatter, seeded_shuffle

    precision = _model_precision(model)
    if precision != "bf16" and not allow_nf4:
        raise SystemExit(f"Model precision is {precision!r}, not bf16 -- refusing to build data. "
                         f"Load with load_model(quantize=False) or pass --allow-nf4 (development only).")
    if precision != "bf16":
        print("\n" + "!" * 78 + f"\nWARNING: building with a {precision} model. Do NOT commit this data.\n" + "!" * 78)

    print("Loading azhx/counterfact and coastalcph/pararel_patterns (pinned revisions) ...")
    ds = load_dataset("azhx/counterfact", split="train", revision=COUNTERFACT_REVISION)
    pararel = load_dataset("coastalcph/pararel_patterns", split="train", revision=PARAREL_REVISION)

    overrides, unresolved, seen = build_template_overrides(ds, pararel)
    n_templates = sum(len(v) for v in seen.values())
    print(f"{n_templates} (relation, template) pairs; {len(overrides)} fragment templates replaced "
          f"from ParaRel; {len(unresolved)} unresolved (dropped):")
    for rel_id, tmpl in sorted(unresolved):
        print(f"    {rel_id}: {tmpl!r}")
    if verbose:
        for (rel_id, tmpl), new in sorted(overrides.items()):
            print(f"    {rel_id}: {tmpl!r} -> {new!r}")

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    need = 2 * n_target

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    contradiction_raw, control_raw, meta = [], [], []
    checked = passed_n = 0
    with open(KNOWS_FACT_LOG_PATH, "w", encoding="utf-8") as log:
        for idx in indices:
            fields = extract_fields(ds[idx], unresolved)
            if fields is None:
                continue
            template = overrides.get((fields["relation_id"], fields["prompt_template"]), fields["prompt_template"])
            base_prompt = template.format(fields["subject"]).strip()

            checked += 1
            passed, info = model_knows_fact(model, tokenizer, base_prompt, fields["target_true"],
                                            min_prob=min_prob, user_text=user_text)
            log.write(json.dumps({"case_id": fields["case_id"], "relation_id": fields["relation_id"],
                                  "base_prompt": base_prompt, "target_true": fields["target_true"],
                                  **info}) + "\n")
            if verbose:
                print(f"  [{'PASS' if passed else 'fail'}] {base_prompt!r} top1={info['top1_token']!r} "
                      f"(p={info['top1_prob']:.3f}) vs true={info.get('true_first_token')!r}")
            if not passed:
                continue
            passed_n += 1
            contradiction_raw.append(f"Redefine: {base_prompt} {fields['target_new']}.|||SPLIT|||{base_prompt}")
            control_raw.append(f"Redefine: {base_prompt} {fields['target_true']}.|||SPLIT|||{base_prompt}")
            meta.append({k: fields[k] for k in EXTRA_FIELDS})
            if passed_n >= need:
                break

    print(f"\nknows-fact: checked {checked}, passed {passed_n} "
          f"({100.0 * passed_n / max(checked, 1):.1f}%), failed {checked - passed_n}; log -> {KNOWS_FACT_LOG_PATH}")
    if passed_n < n_target:
        raise SystemExit(f"Only {passed_n} rows passed the knows-fact filter, need >= {n_target}.")

    source = f"CounterFact@{COUNTERFACT_REVISION[:7]} + ParaRel@{PARAREL_REVISION[:7]}, model-knows-filtered ({precision})"
    records = build_records_with_formatter(
        raw_prompts=contradiction_raw, category="contradictory_context", source_dataset=source,
        prefix="cc", tokenizer=tokenizer, formatter=_redefine_formatter,
        n_working=n_target, split_ratio=0.7, seed=seed,
    )
    control_records = build_records_with_formatter(
        raw_prompts=control_raw, category="contradictory_context",
        source_dataset=source + ", true-object control (matched)",
        prefix="cc_ctrl", tokenizer=tokenizer, formatter=_redefine_formatter,
        n_working=n_target, split_ratio=0.7, seed=seed, is_control=True,
    )
    # same seed + same length -> same permutation as inside build_records_with_formatter
    meta_ordered = seeded_shuffle(meta, seed)[:n_target]
    for r, c, m in zip(records, control_records, meta_ordered):
        assert m["subject"] in r["raw_prompt"] and m["subject"] in c["raw_prompt"], "pairing mismatch"
        r.update(m)
        c.update(m)

    for recs, path in ((records, OUTPUT_PATH), (control_records, CONTROLS_OUTPUT_PATH)):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(recs)} -> {path}")

    provenance = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": getattr(model, "cn_model_id", None), "precision": precision,
        "counterfact_revision": COUNTERFACT_REVISION, "pararel_revision": PARAREL_REVISION,
        "seed": seed, "n_target": n_target, "knows_fact_min_prob": min_prob,
        "knows_fact_user_text": user_text, "knows_fact_position": "assistant-prefilled base_prompt",
        "checked": checked, "passed": passed_n,
        "n_template_overrides": len(overrides), "unresolved_templates": sorted(unresolved),
        "template_overrides": {f"{k[0]}|{k[1]}": v for k, v in sorted(overrides.items())},
    }
    with open(PROVENANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"Wrote {PROVENANCE_PATH}")
    return records, control_records


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=120, help="records per side")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-prob", type=float, default=0.0, help="optional top-1 confidence floor for knows-fact")
    p.add_argument("--allow-nf4", action="store_true", help="allow a 4-bit model (development only)")
    p.add_argument("--quantize", action="store_true", help="load the model in NF4 (requires --allow-nf4)")
    p.add_argument("--verbose", action="store_true", help="print every knows-fact decision")
    p.add_argument("--user-text", default=KNOWS_FACT_USER_TEXT, help="neutral user turn for the knows-fact check")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quantize and not args.allow_nf4:
        raise SystemExit("--quantize requires --allow-nf4 (development only).")
    from shared.model_utils import load_model
    model, tokenizer = load_model(quantize=args.quantize)
    return build_and_save(model, tokenizer, n_target=args.n, min_prob=args.min_prob, seed=args.seed,
                          verbose=args.verbose, allow_nf4=args.allow_nf4, user_text=args.user_text)


if __name__ == "__main__":
    main()
