# X3: the X2 test on a second model, Gemma-2-9B-it, with the permissive prompt: plan, written before any X3 run

Recorded before running. The hash is self-recorded (`HASHES.txt`); this file and the X3 scripts were committed to git before the pilot. Items the draft marked "proposed" (same question list as X2; the drop times) were adopted as proposed when the run was requested.

## Fixed before the run

| Choice | Value |
|---|---|
| Model | `unsloth/gemma-2-9b-it` (revision fc7d4737cda11c3a19af2b722319e846670b4d89), bf16, vLLM (correct Gemma-2 attention; equal to transformers `eager` in the checks in `results/x2/gemma_judge_diagnosis`) |
| Questions | The X2 list, `results/x2/sample.jsonl`, sha256 9278947a9937bf7d487680a0630fc4c0b16a73cab3d2fd5b6618278f011a9ffc: 4,376 questions, 16 relations, the same 200 pilot questions |
| Popularity | Obscurity = 1 - within-relation rank of page views, as in X2; positive coefficient means more "I don't know" for obscure entities |
| Knowledge score | Gemma's own: share of 10 forced-answer samples ("The answer is", T = 0.7, 12 tokens) graded correct, 0 to 1. The greedy answer is stored and enters one secondary model |
| Free prompt | The paper's permissive prompt (E9): the question followed by " If you do not know the answer, say that you do not know." |
| Abstention, primary | Share of 3 free replies (T = 0.7, 96 tokens) both judges (Qwen2.5-32B-Instruct, Gemma-2-27B-it with correct attention) label DONT_KNOW. The judges see the original question, as in E9. Off-format labels count as not DONT_KNOW |
| Abstention, second measure | Share of 20 short replies (T = 0.7, 24 tokens, first line) to the same permissive prompt that the repo's regex matches |
| Sampling | T = 0.7, top_k = 50 (library default); Gemma's generation_config sets no top_p. Stop at `<eos>` and `<end_of_turn>` |
| Judges | `scripts/x2_run.py judge`, unchanged (vLLM, greedy, 5 tokens; Qwen with its repetition penalty 1.05) |
| Model fit, interval, readouts | `scripts/x2_analysis.py`, unchanged (fractional logit, relation fixed effects, 5,000 whole-relation resamples, seed 0; relations with a constant outcome are dropped from the fit and named; abstention by decile within knowledge bands; predicted abstention at the 90th and 10th percentile of popularity, averaged over all questions) |
| Not done | Ranking popularity against knowledge from standardised coefficients. A second label set with the faulty Gemma-27B attention: no such labels exist for these replies, and the fault is established |

## Decision rules (primary coefficient only)

1. Interval above zero: the effect replicates on a second model. Said with both models and PopQA named.
2. Interval includes zero: not replicated on Gemma-2-9B-it, with how rarely it abstains; claims stay named to Llama. A wide interval that includes zero is this rule, not a pipeline failure.
3. Interval below zero: reported as found, and the claim is rewritten to match.
4. In every case the same table goes in the paper, with the same measures as X2.

## Secondary models (reported, no decision)

As X2: deciles 2 to 9; knowledge in 11 levels; plus the greedy answer; without country, capital and sport; Qwen alone, Gemma alone, regex on the 96-token replies. A secondary model that does not converge (possible when abstention is rare) is reported as not converged and not replaced.

## Pilot (200 questions, seed 1; outputs not used in the analysis)

The full run starts only if all hold (`scripts/x3_pilot_check.py`):

- every question has all outputs; no empty 96-token reply; no missing or empty judge output; at most 1.0% off-format labels per judge;
- greedy correctness equals the stored value for at least 95% of pilot questions in `data/familiarity_v3_gemma` with the same text (40 questions);
- two-judge abstention between 0.05 and 0.95;
- two-judge kappa at least 0.60 (the paper's bar), reported with an interval;
- on the paper's 1,161 Llama replies, Gemma-27B gives the `eager` DONT_KNOW label for at least 95%, and Qwen the stored one for at least 95%.

If the pilot is not clean: stop and report. No prompt, threshold or judge is changed to make it pass; a rule change needs a dated amendment before the full run. Drop rule: not clean by 26 Nov noon IST, or no result by 25 Nov 18:00 IST, and X3 is left out.

## Known before the run

Under the permissive prompt on the paper's 400 Gemma questions (extreme tenths, stored replies, corrected judges), abstention is 0.53 / 0.41 / 0.11 / 0.00 (obscure unknown, obscure known, popular unknown, popular known) and the popularity margin +0.30. The regex finds only part of Gemma's abstentions. 889 of the X2 questions are in Gemma's stored pool. X2's result on Llama is known. The analysis was tested on simulated data only.

## Not allowed after seeing data

Changing the questions, prompts, sampling, judges, label rule, regressors, bands, interval method, draws, seed, pilot rules or decision rules. Any further analysis is marked post hoc.
