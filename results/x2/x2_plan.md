# X2: the popularity effect over the whole popularity range (Llama-3.1-8B-Instruct, PopQA): plan, written before any X2 run

Recorded before running. The hash is self-recorded (`HASHES.txt`); this file and the three X2 scripts were committed to git before the pilot. Nothing in the paper's existing measures, prompts, thresholds or metrics is changed.

## Fixed before the run

| Choice | Value |
|---|---|
| Model | `unsloth/Meta-Llama-3.1-8B-Instruct`, bf16 (as in the paper) |
| Sample | PopQA deduplicated as in `scripts/build_familiarity_twins.py`; up to 30 questions per within-relation popularity decile, so up to 300 per relation, seed 0. All 16 relations give 4,376 questions (four relations have fewer than 300). `sample.jsonl`, sha256 9278947a9937bf7d487680a0630fc4c0b16a73cab3d2fd5b6618278f011a9ffc |
| Popularity | Rank of the subject's page views among all deduplicated PopQA questions of its relation, (average rank - 0.5) / n. The regressor is **obscurity = 1 - rank** (0 = most viewed, 1 = least viewed), so a positive coefficient means more "I don't know" for obscure entities, as in the paper's popularity effect (obscure minus popular) |
| Knowledge score | Share of 10 forced-answer samples ("The answer is", T = 0.7, 12 tokens) graded correct by the repo's grader, 0 to 1. No known/unknown cut-off. The greedy answer is also stored and graded; it enters one secondary model |
| Abstention, primary | Share of 3 free replies (T = 0.7, 96 tokens) that both judges (Qwen2.5-32B-Instruct and Gemma-2-27B-it) label DONT_KNOW. Unparsed labels count as not DONT_KNOW, as in the paper |
| Abstention, second measure | Share of 20 short free replies (T = 0.7, 24 tokens, first line) that the repo's regex matches |
| Sampling settings | T = 0.7, top_p = 0.9 (the model's generation_config) and top_k = 50 (library default), the values the paper's runs resolved to. Judges greedy, 5 tokens; Qwen with repetition penalty 1.05 from its generation_config, as in the paper's runs |
| Engine | vLLM instead of transformers, for speed. Checked against the paper's stored replies and answers in the pilot (below) |
| Model fit | Fractional logistic regression of abstention on obscurity and the knowledge score, with relation fixed effects. Relations whose outcome is constant are dropped from the fit and named |
| Interval | 95% percentile bootstrap resampling whole relations (repeated relations get a larger weight), 5,000 draws (as the paper's relation resampling), seed 0. Draws that do not converge are dropped and counted |
| Primary readout | The obscurity coefficient with its interval, plus abstention by popularity decile within knowledge bands |
| Knowledge bands (figure only) | 0 of 10, 1 to 8 of 10, 9 or 10 of 10 correct |
| Not done | Ranking popularity against knowledge from standardised coefficients |

## Decision rules (primary coefficient only)

1. Interval above zero: the popularity effect holds across the range. Said with the sample size and number of relations, bounded to Llama and PopQA.
2. Interval includes zero: reported as a limit of the extreme-tenths result; the claim stays as the tenths design states it.
3. Interval below zero: reported as found, and the claim is rewritten to match.
4. In all cases the same table goes in the paper, with the paper's measures.

## Secondary models (reported, no decision)

For the two-judge and the regex measure: deciles 2 to 9 only (without the extreme tenths); knowledge score as 11 levels instead of linear; adding the greedy answer; without country, capital and sport. For other measures: Qwen alone, Gemma alone, regex on the 96-token replies.

## Pilot (200 questions, 20 per decile, seed 1; its outputs are not used in the analysis)

The full run starts only if all of these hold (`scripts/x2_pilot_check.py`):

- every question has all outputs; no empty 96-token reply; no missing or empty judge output;
- at most 1.0% unparsed labels per judge. The draft of this plan said "no parse failures"; that was changed before the pilot because the Gemma judge gives off-format labels in every stored run (6 of 1,161, 3 of 1,161, 12 of 1,200) and neither the prompt nor the parser may change;
- greedy correctness equals the stored value for at least 95% of pilot questions already in `data/familiarity_v3` or `data/familiarity_v3_adj` (same question text);
- two-judge abstention between 0.05 and 0.95;
- two-judge kappa at least 0.50, with a question-bootstrap interval (10,000 draws, seed 0) that overlaps the paper's [0.57, 0.67];
- each judge under vLLM gives the stored DONT_KNOW label for at least 95% of the paper's 1,161 Llama replies.

## Known before the run

Existing data already show smaller effects in adjacent bands (E14, 10th-30th against 70th-90th percentile). 901 sampled questions are in `familiarity_v3` and 1,733 in `familiarity_v3_adj`; 129 are among the paper's 387 judged questions. All X2 answers and replies are newly generated. The analysis script was tested on simulated data only.

## Not allowed after seeing data

Changing the sample, prompts, sampling settings, judges, label rule, regressors, bands, interval method, draws, seed, pilot rules or decision rules. Any further analysis is marked post hoc.
