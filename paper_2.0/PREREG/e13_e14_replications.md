# E13 and E14: replications of the Llama popularity preference: criterion, written before the run

Written before any E13 or E14 run, in response to review round 4. Methods F14: the alternative measures rescore the same replies, so they test scoring, not sampling. Methods F6 and the round-4 relation check: the obscure-known cell is concentrated in a few relations, and its relation-clustered interval includes 0. Box 7, run in sequence.

## E13: fresh replies, same questions

- **Items:** the 387 questions of `results/hedge_measure_v2/items_llama.jsonl`.
- **Replies:** 3 new free replies per question (chat template alone, T = 0.7, up to 96 tokens) with torch seed 1 (the original used seed 0).
- **Scoring:** the two original judges (Qwen2.5-32B-Instruct, Gemma-2-27B-it); a reply abstains when both say DONT_KNOW. This is the paper's primary measure.
- **Test:** popularity preference (obscure-known minus popular-unknown abstention), 95% within-cell item bootstrap (10,000, seed 0). **REPLICATES** if > 0 with the interval excluding 0; otherwise **DOES NOT REPLICATE**.

## E14: new questions from adjacent popularity bands

- **Pool:** `build_familiarity_twins.py` with `--lo-quantile 0.30 --hi-quantile 0.70 --n-per-relation 1000 --seed 42`, the same model and labelling rule as the original pool. Every question already in the original pool (bottom and top deciles) is removed. The new arms are therefore the 10th-30th percentile ("obscure-adjacent") and the 70th-90th percentile ("popular-adjacent") of page views within each relation.
- **Sample:** up to 100 questions per popularity × knowledge cell (stratified random, seed 0), with `scripts/explore_make_items.py`.
- **Replies and scoring:** as in the paper: 3 free replies (T = 0.7, 96 tokens, seed 0) and the two original judges, abstaining when both say DONT_KNOW.
- **Test:** popularity preference with a 95% within-cell item bootstrap (10,000, seed 0), plus the relation mix of each cell and a relation-clustered bootstrap (5,000, seed 0).
  - **REPLICATES IN NEW QUESTIONS** if > 0 with the within-cell interval excluding 0, and at least 30 obscure-adjacent known questions.
  - **DOES NOT REPLICATE** if at least 30 and the interval includes 0, or the estimate is <= 0.
  - **UNDERPOWERED** if fewer than 30 obscure-adjacent known questions.
  - The effect is expected to be smaller than in the extreme deciles, because the popularity contrast is narrower; size is reported, not tested.

## Licensing

- E13 REPLICATES: the paper may say the preference replicates with fresh replies.
- E14 REPLICATES: the paper may say the preference holds in new questions from the adjacent popularity bands, with the relation mix stated. Any other E14 outcome is reported as it is, and the paper keeps its extreme-decile scope.

## Not allowed after seeing data

Changing the quantile bands, the exclusion of the original pool, the sampling, the judges, the abstention rule, the bar of 30, or the verdict rules.
