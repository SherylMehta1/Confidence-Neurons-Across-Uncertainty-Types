# E12: the permissive prompt for Qwen and Mistral: criterion, written before the run

Written before any E12 run, in response to review round 4 (methods F9, framing F5): E9 ran the permissive prompt only for Llama and Gemma. Box 8. Identical to E9 (`e9_permissive_prompt.md`, sha f6459163) in every respect except the models.

## Design

- **Models:** Qwen2.5-7B-Instruct (`Qwen/Qwen2.5-7B-Instruct`) and Mistral-7B-Instruct-v0.3 (`unsloth/mistral-7b-instruct-v0.3`, same weights as `mistralai/Mistral-7B-Instruct-v0.3`).
- **Items:** each model's existing two-judge question set (`results/hedge_measure_v2/items_{qwen,mistral}.jsonl`).
- **Prompt:** the question followed by " If you do not know the answer, say that you do not know."
- **Replies:** 3 replies at T = 0.7, up to 96 tokens.
- **Scoring:** the three E5/E9 judges with the same judge prompt; majority rule (at least 2 of 3 say DONT_KNOW).
- **Reliability:** Fleiss κ per model. MEASURED if κ >= 0.60.

## Tests (per model, only if MEASURED)

Popularity minus knowledge on the majority-rule abstention rate, with 95% CIs from 10,000 within-cell item bootstraps (seed 0).

- **FOLLOWS POPULARITY** if > 0 with the CI excluding 0.
- **FOLLOWS KNOWLEDGE** if < 0 with the CI excluding 0.
- Otherwise **NEITHER**.

Also report every cell rate.

## Licensing

- A family is reported as a permissive-prompt result only if it is MEASURED and its obscure-unknown abstention rate is >= 0.25. Otherwise the paper says its preference cannot be estimated even when permitted.
- The paper may describe the permissive-prompt preference as shared across families only for the families that are MEASURED and FOLLOW POPULARITY.

## Not allowed after seeing data

Changing the prompt wording, the judges, the majority rule, the κ bar, the 0.25 threshold or the items.
