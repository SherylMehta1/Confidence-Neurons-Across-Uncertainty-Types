# E2: the model's own familiarity measure (subject-name surprisal): criterion, written before the run

Written 2026-10-10, before any E2 run, in response to the framing referee (round 2) F1 and chair J2: "familiarity" is a page-view proxy.

Box A (after E1).
- Model for surprisal: unsloth/Meta-Llama-3.1-8B (the base model; exposure is a pretraining property).
- Instruct abstention outcomes come from existing files.

## Measure

`name_surprisal` = the mean per-token surprisal (−log p, nats) of the subject string. It is computed in the fixed context "Here is a name: " followed by the subject, with BOS, over the subject's tokens only. `name_tokens` = the number of subject tokens.

## Items and outcomes

The 2,673 Llama pool items.

Outcomes:
- (a) the two-judge abstention rate on the 387 items, `item_rates.csv` prim;
- (b) the regex abstention rate on the 2,303 known/unknown items, `all_measured.jsonl` hedge_rate.

## Tests

- **S1.** Spearman correlation of name_surprisal with log page views (s_pop) over all items. Descriptive; it is expected to be negative.
- **S2 (main).** Linear probability model:
  - `abstain ~ surprisal_z + unknown + name_tokens_z + relation fixed effects`, on outcome (a) and on outcome (b);
  - 95% CIs from 2,000 item bootstraps, seed 0;
  - the obscure/popular arm is left OUT, so surprisal stands in for familiarity.
  - **TRACKS OWN FAMILIARITY** if the surprisal coefficient > 0 with the CI excluding 0 in both (a) and (b). Licenses: "abstention also tracks how unfamiliar the name is to the model itself, with knowledge held fixed".
- **S3.** The same model with the arm added: does surprisal predict abstention beyond the page-view arm? Report the coefficient and its CI; there is no bar.
- **S4.** Does knowledge still add beyond surprisal? Report the unknown coefficient; no bar.

## Caveat fixed now

Surprisal also reflects spelling and token rarity. Rarity is partly controlled through name_tokens_z, and the paper says that surprisal is a model-side exposure measure, not a clean familiarity variable.

## Not allowed after seeing data

Changing the context string, the base-vs-instruct choice, the covariates, the outcome sets or the bar.
