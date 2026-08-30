# Person C -- Contradictory Context

**Live entry point:** `python person_C_contradictory_context/preprocess_contradictory_context.py`
(GPU, bf16 model required; `--allow-nf4` only for development).
Options: `--n 120`, `--seed 42`, `--min-prob`, `--verbose`, `--user-text`.

## What the category measures
CounterFact facts the model demonstrably knows, contradicted in context using the Tighidet et al.
(2024) "Redefine" pattern:

    user:      Redefine: Thailand belongs to the continent of Europe.
    assistant: Thailand belongs to the continent of          <-- prefilled; next token measured

Uncertainty here is conflict between context and parametric memory, not absence of knowledge.

## Construction
- Sources pinned: `azhx/counterfact@c01c413`, `coastalcph/pararel_patterns@aadfae5`.
- CounterFact templates that are sentence fragments (no connector word, short comma tails, or a
  hand-curated bad-tail list) are replaced per template by the first non-fragment ParaRel pattern
  for the same relation; unresolvable ones are dropped.
- Rows are scanned in `random.Random(seed)` order until `2 * n` pass the knows-fact filter.
- Knows-fact filter: top-1 token at the prefilled position (neutral user turn, `base_prompt`
  prefilled) must equal the first token of `" " + target_true`. All decisions are logged to
  `data/contradictory_context/knows_fact_log.jsonl`; run metadata to `provenance.json`.

## Controls (`controls.jsonl`, `is_control: true`)
Same rows, same prefill, but the Redefine sentence asserts the TRUE object. `cc_0007` and
`cc_ctrl_0007` are the same CounterFact case.

## Extra record fields
`case_id`, `relation_id`, `subject`, `target_true`, `target_new`.

## Known limitations
See `LIMITATIONS.md`: the category shows no clean uncertain-vs-control top-1 gap, and the committed
data were built with the NF4 model and the unprefilled knows-fact position -- regenerate in bf16.
