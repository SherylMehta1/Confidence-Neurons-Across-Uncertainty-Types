# X2 amendment 1: the Gemma judge, written after the pilot and before the full run

**What happened.** The pilot was not clean (`pilot_checks.csv`): two-judge kappa 0.906, above the range fixed in `x2_plan.md`, and vLLM-Gemma equal to the stored DONT_KNOW labels on 84.3% of the paper's 1,161 replies (bar 95%). Qwen matched on 100%. All other checks passed.

**Cause** (`gemma_judge_diagnosis/`). Re-running the paper's path (transformers 5.14.1, default `sdpa` attention) reproduces the stored Gemma outputs exactly. Running the same code with `eager` attention gives the same labels as vLLM for all 1,161 replies. Gemma-2 needs attention-logit soft-capping, which transformers applies in `eager` but not in `sdpa`. The stored Gemma labels are therefore the product of an incorrect forward pass. The pilot's failure was in the stored labels, not in vLLM.

**What was seen before this amendment:** the pilot checks, and, on the paper's 387 questions only, the two-judge numbers under correct Gemma labels (kappa 0.91, margin +0.43, unknown-fact gap +0.53). No X2 effect of popularity was computed.

**Changes (decided by the author before the full run):**

1. Primary abstention stays "both judges say DONT_KNOW", with Gemma run correctly (vLLM, equal to `eager`).
2. Every full-run reply is also labelled with the paper's `sdpa` path (`scripts/x2_judge_sdpa.py`), and the whole analysis is repeated with those labels (`scripts/x2_sdpa_compare.py`), reported side by side. It carries no decision.
3. Two pilot rules are replaced: two-judge kappa at least 0.60 (the paper's own bar); vLLM-Gemma equal to `eager` transformers DONT_KNOW labels on at least 95% of the 1,161 replies. Both were known to hold when this was written (0.906; 1.000).

Nothing else changes: sample, prompts, sampling, seeds, regressors, interval, decision rules, `x2_analysis.py`.
