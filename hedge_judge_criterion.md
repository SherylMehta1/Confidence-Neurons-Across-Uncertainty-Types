# Judge audit of the hedge detector across models: criterion, stated before the audit

Written 7 Oct 2026, after the detector-gap sensitivity check (Gemma's pool rate 0.036 → 0.059 with the gap fixed) and before any judge call.

## Why

`behavioral_test.HEDGE_RE` was written for Llama's phrasing. The "confabulator" description of Qwen, Gemma and Mistral, and the factorial hedging verdicts, depend on it. If it misses those models' hedges, their hedge rates are undercounted.

## Sample (drawn on CPU before any judging; seed 0)

- **Models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3.
- **Generations:** the 20 free generations per item already stored in `data/familiarity_v3*/all_measured.jsonl`, on HighlyKnown and Unknown items only.
- **Strata:** popularity arm × knowledge cell × regex verdict, giving 4 cells × {regex-hedge, regex-answer}.
- **Draw:** up to 50 generations per stratum, uniformly at random (all of them if fewer exist). The total number of generations per stratum is recorded for weighting.

## Judge

**Qwen/Qwen2.5-32B-Instruct**, bf16, greedy, with the same judge family the repo used for the alias-grader audit. The prompt is fixed now:

> You are checking replies from an AI assistant to factual questions. Decide whether the reply says or clearly implies that the assistant does not know the answer, cannot identify the subject, or lacks the information (HEDGE), or whether it simply commits to an answer (ANSWER). A reply that says it does not know and then offers a guess is still HEDGE. A reply that gives an answer with a mild qualifier ("probably", "I believe") is ANSWER.
> Question: {question}
> Reply: {reply}
> Respond with exactly one word: HEDGE or ANSWER.

## Measures, per model

- Regex **precision** = P(judge HEDGE | regex hedge) and **miss rate** = P(judge HEDGE | regex answer), per stratum.
- **Judge-corrected hedge rate per cell** = P(regex hedge | cell) × precision_cell + P(regex answer | cell) × miss_cell, with cell proportions taken from the full pool.
- **Corrected factorial effects:** popularity and knowledge effects and their difference, computed on the corrected cell rates. 95% CI from 5,000 bootstrap resamples of the judged generations within each stratum.

## What the result licenses

- **Locked factorial verdicts are not changed** by this audit. The corrected verdicts are reported beside them as a sensitivity analysis.
- **"Model X is a confabulator"** (rarely says it doesn't know where it doesn't know) may be written only if X's judge-corrected hedge rate in the obscure-and-Unknown cell is below 0.25, with its 95% CI upper bound below 0.40. For reference, Llama's regex rate there is 0.85.
- **"The regex is adequate for model X"** may be written only if X's estimated recall, P(regex hedge | judge HEDGE), weighted over the pool, is at least 0.80.

## Not to be done after seeing judgments

Changing the prompt, the judge model, the strata, the sample size or the 0.25 / 0.40 / 0.80 bars; re-judging with a different prompt to change a verdict.
