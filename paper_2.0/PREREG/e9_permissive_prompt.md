# E9: a prompt that permits "I don't know" (Llama and Gemma): criterion, written before the run

Written before any E9 run, in response to review round 3 (all referees: Gemma's null sits at a floor; chair item 1 and 9h). Box 3.

## Design

- **Models:** Llama-3.1-8B-Instruct and Gemma-2-9b-it.
- **Items:** each model's existing two-judge item set (`results/hedge_measure_v2/items_{llama,gemma}.jsonl`, up to 100 per popularity x knowledge cell).
- **Prompt:** the chat template with the user message "{question} If you do not know the answer, say that you do not know." (no prefix).
- **Replies:** 3 replies per item, T = 0.7, up to 96 tokens.
- **Scoring:** the same three judges and judge prompt as E5. A reply abstains if at least 2 of 3 say DONT_KNOW.
- **Reliability:** Fleiss κ per model. Abstention counts as MEASURED if κ >= 0.60.

## Tests (per model, only if MEASURED)

Popularity − knowledge on the majority-rule abstention rate, with 95% CIs from 10,000 within-cell item bootstraps (seed 0).
- **FOLLOWS POPULARITY** if > 0 with the CI excluding 0.
- **FOLLOWS KNOWLEDGE** if < 0 with the CI excluding 0.
- Otherwise **NEITHER**.

Also report every cell rate.

## Licensing

- If Gemma is MEASURED here with an obscure-unknown abstention rate >= 0.25, its verdict replaces "cannot be estimated" in the paper, as a permissive-prompt result.
- If its rate stays below 0.25, the paper says that Gemma's preference cannot be estimated even when permitted.

## Not allowed after seeing data

Changing the prompt wording, the judges, the majority rule, the κ bar, the 0.25 threshold or the items.
