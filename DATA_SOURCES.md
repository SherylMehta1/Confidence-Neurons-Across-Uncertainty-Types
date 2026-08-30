# Data Sources

| Category | Source | Pinned revision | Live builder |
|---|---|---|---|
| Ambiguity | AmbigQA, `sewon/ambig_qa` ("light" config, HF Hub) -- Min et al. 2020 | `e969d0132f4dd28c2939d55be34f1788c00ccfe7` | `person_A_ambiguity/preprocess_ambiguity.py` |
| Lack of knowledge | UnknownBench NEC subset, `github.com/genglinliu/UnknownBench` (`data/NEC/NEC_{un,}answerable.json`) -- Liu et al. 2024 | commit `7283e4218b9146275d3069306927c3289fad576a` | `person_B_lack_of_knowledge/screen_templates.py` then `rebuild_lack_of_knowledge_whitelisted.py` |
| Contradictory context | CounterFact, `azhx/counterfact` (HF Hub; ROME, Meng et al. 2022) for (subject, relation, true/false object); ParaRel, `coastalcph/pararel_patterns`, for replacement relation templates; prompt pattern is the "Redefine" construction of Tighidet et al. 2024 (the Tighidet code itself is not used) | CounterFact `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`; ParaRel `aadfae52549bb0eb5b6729b27f0d8240d4f55f4f` | `person_C_contradictory_context/preprocess_contradictory_context.py` |

## Files per category

```
data/<category>/prompts.jsonl    uncertain set      120 records, is_control = false
data/<category>/controls.jsonl   matched controls   120 records, is_control = true
```
plus per-category provenance: `data/ambiguity/review_log.jsonl` + `control_review_log.jsonl` (one
accept/reject line per raw AmbigQA record), `person_B_lack_of_knowledge/template_screen_results.json`
+ `template_quota_report.json`, `data/contradictory_context/knows_fact_log.jsonl` + `provenance.json`.

## Common schema

One JSON object per line. The first seven fields are identical across categories; category-specific
fields follow them.

```json
{
  "prompt_id": "amb_0001",
  "category": "ambiguity",
  "raw_prompt": "Who wrote the music for annie get your gun?",
  "chat_formatted_prompt": "<Llama-3.1 chat template: question in the user turn, then 'The answer is' prefilled into the assistant turn>",
  "source_dataset": "AmbigQA-light (sewon/ambig_qa@e969d01)",
  "split": "working",
  "is_control": false
}
```

- `split`: `"working"` or `"held_out"` (70/30, seeded, set once).
- `is_control`: `true` for the matched low-uncertainty twin set in `controls.jsonl`. Control ids use
  the `_ctrl_` prefix (`amb_ctrl_0007`, `lok_ctrl_0007`, `cc_ctrl_0007`). For contradictory context
  `cc_ctrl_NNNN` is the same CounterFact case as `cc_NNNN`.
- `chat_formatted_prompt` already starts with `<|begin_of_text|>` and pins `Today Date: 26 Jul 2024`
  in the system header; tokenize it with `shared.model_utils.tokenize_prompt` (adds BOS only if
  missing). The measured token is the first token after the assistant-turn prefill.

### Category-specific extra fields

| Category | Extra fields |
|---|---|
| Ambiguity | `source_id` (AmbigQA record id), `answer_groups` (list of alias lists; the disambiguated answer groups for the uncertain set, the single agreed answer's aliases for controls) |
| Lack of knowledge | `template` (the NEC question template, e.g. `"What is the capital city of {}?"`), `nec_category` (`animals`/`food`/`countries`/`medicines`/`sports`/`generic`) |
| Contradictory context | `case_id` (CounterFact), `relation_id` (Wikidata P-id), `subject`, `target_true`, `target_new` |

## How controls are built

- **Ambiguity**: AmbigQA `singleAnswer` records (annotators agreed on one answer), same factoid /
  time-reference / malformed filters as the uncertain set; records that also have a `multipleQAs`
  annotation with >= 2 distinct answer groups are excluded.
- **Lack of knowledge**: the answerable (real-entity) NEC items from the same whitelisted templates,
  with identical per-template quotas to the fabricated-entity set.
- **Contradictory context**: the same `Redefine:` construction and prefill, but the context asserts
  the TRUE object, for exactly the rows used in the uncertain set.

## Notes

- Hand-check ~10-15 items per category after any rebuild before trusting the label.
- Model-dependent filtering (B's template screen, C's knows-fact filter) must be run with the bf16
  model; the builders refuse NF4 unless `--allow-nf4` is passed and record precision in their outputs.
