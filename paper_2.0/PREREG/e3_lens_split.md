# E3: splitting the lens readout into its hedge and answer terms: criterion, written before the run

Written 2026-10-10, before any E3 run, in response to:
- methods referee F8 (rounds 1-2);
- chair J15/I19 (the readout may reflect answer-token rarity);
- numbers audit C1 (window knowledge effects for all four base models, with intervals).

Box B.

## Models, lenses, prompts and labels

The four base models of the lens study, with their official prefitted lenses (as in jlens_cascade_criterion.md, sha c8e04372):
- Llama-3.1-8B
- Gemma-2-9B
- Gemma-2-2B
- OLMo-3-1025-7B

Their instruct partners are run if time allows (descriptive):
- Llama-3.1-8B-Instruct
- Gemma-2-9B-it
- Gemma-2-2B-it

The prompts are the same as in the lens study. Knowledge labels are taken from `results/jlens_cascade/<model>/labels.jsonl`, unchanged.

## Measures (last prompt position, every layer)

- **hedge term** H(ℓ) = logsumexp over the hedge tokens of the lens log-probabilities;
- **answer term** A(ℓ) = logsumexp over the answer tokens;
- these are the same token sets as `circuit_common.Readout`, so the original readout equals H − A.

Popularity (F) and knowledge (K) effects are computed on H, on −A and on H − A, by layer. Window = relative depth [1/3, 2/3], as pre-set. Window means get 95% CIs from 2,000 within-cell item bootstraps, seed 0.

## Tests

- **L1 (main).** For each base model, the popularity effect on H (hedge term alone) in the window.
  - **HEDGE-SIDE** if positive with the CI excluding 0 in at least 3 of the 4 base models. Licenses: "the popularity effect sits in the hedge-token term, not only in answer-token rarity".
  - If it holds in 2 or fewer, say that the effect is carried at least partly by the answer term, and narrow the claim.
- **L2 (required reporting).** Window K and F on H − A, with CIs, for all four base models. This replaces the selective report.
- **L3 (descriptive).** Report H and −A separately for each effect.

## Not allowed after seeing data

Changing the token sets, the window, the models, the bar, or dropping a model.
