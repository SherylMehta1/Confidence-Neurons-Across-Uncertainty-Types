# E6: entity-level knowledge as a covariate: criterion, written before the run

Written 2026-10-10 ~03:30 UTC, before any E6 data exist. It responds to the round-2 framing referee F1 (Koriat's accessibility account): page-view popularity may stand for how much the model knows about the *entity* in general, not for familiarity. If so, abstention would be a sensible report of low entity knowledge rather than a familiarity heuristic.

Model: unsloth/Meta-Llama-3.1-8B-Instruct, bf16. Items: the 2,673 Llama pool items.

## Reference

The English Wikidata description of each subject. It is fetched once from the Wikidata API using the item's `s_uri`, before any generation, and saved to `results/round3/e6_wikidata.jsonl`. Items without an English description are dropped and counted.

## Entity-knowledge probe (forced, so the model cannot abstain)

- Prompt: the chat template with the user message "What is {subj}? Answer in a few words.", plus the assistant prefix "{subj} is".
- Decoding: greedy and 5 samples (T = 0.7), 24 new tokens each.

## Grading

- The judge is unsloth/Llama-3.3-70B-Instruct-bnb-4bit, greedy, with a fixed prompt. It is given the subject, the Wikidata description and one completion, and labels it:
  - **CORRECT**: it identifies the same entity type and a distinguishing detail consistent with the description, or the same type when the description gives only a type;
  - **WRONG**: a different entity, or a wrong type;
  - **VAGUE**: too generic to judge.
- **entity_known** = the fraction of the 6 completions labelled CORRECT.

## Tests (linear probability models, relation fixed effects, 2,000 item bootstraps, seed 0)

Outcomes:
- (a) two-judge abstention on the 387 items;
- (b) regex abstention on the 2,303 known/unknown items.

Model: `abstain ~ obscure + unknown + entity_known_z`.

- **K1 (main).** The popularity effect SURVIVES ENTITY KNOWLEDGE if the obscure coefficient is > 0 with the CI excluding 0 in both (a) and (b) once entity_known_z is in the model.
  - If it survives in only one, report it as partial.
  - If it survives in neither, report that popularity's effect is accounted for by entity-level knowledge, and narrow the paper's interpretation accordingly.
- **K2.** Report the entity_known_z coefficient and the fraction by which the obscure coefficient shrinks relative to the same model without entity_known_z.
- **K3 (descriptive).** Mean entity_known by popularity × knowledge cell, and the Spearman correlation with log page views.

## Not allowed after seeing data

Changing the prompt, the prefix, the judge, the label definitions, entity_known, the covariates, the outcomes or the bar.
