# Direction reuse, version 2 (position-matched, dose-calibrated): criterion, stated before the runs

Written 7 Oct 2026, after version 1 (`direction_reuse_criterion.md`) returned NOT TESTABLE for Llama and Gemma-2B, and before any version 2 run. Version 1 stands and is reported as run. This is a new experiment that fixes a design flaw found in version 1, not a re-analysis of it.

## The flaw it fixes

In version 1 the directions were fitted at the last position of the **prefill** prompt ("… The answer is"), but steering was applied at the last position of the **free-generation** prompt (chat template, no prefill). The vector was therefore added at a position it was not estimated at. Steering at the natural norm moved Llama's hedging only from 0.020 to 0.053.

## Changes from version 1 (everything else identical)

1. **Fit position equals steer position.**
   - Instruct models: directions are fitted at the last position of the free-generation prompt, i.e. `format_chat_prompt(q)`, the prompt that is steered.
   - Base models: at the last position of `Question: {q}\nAnswer:`, so neither model sees the prefill.
   - Labels are unchanged (each model's SliCK labels from the lens study).
2. **Dose calibrated on the instruct model's own direction only.**
   - Doses are k ∈ {1, 2, 4, 8} × ‖d_fam^inst(ℓ*)‖.
   - For each k, own-direction hedging and the mean of 5 random directions are measured at the same k.
   - The analysis dose k* is the smallest k at which own − random ≥ 0.10. If none qualifies, the pair is NOT TESTABLE.
   - The base direction plays no part in choosing k*.
3. **Pairs:** Llama-3.1-8B (base/instruct) and Gemma-2-9B (base/it), the two models whose factorial verdict was REPLICATES. Qwen, Mistral and Gemma-2B are not run in version 2.

## Verdict (per pair, at k*)

Version 1's three conditions, evaluated at k*:
1. cos(d_fam^base, d_fam^inst) at ℓ* exceeds the 99th percentile of the 200-permutation null, recomputed on the version-2 directions;
2. base-direction hedging minus the random mean (at k*) ≥ 0.10;
3. the base-direction effect ≥ 50% of the own-direction effect (each minus "none").

Outcome: REUSED if all three hold, NOT REUSED otherwise, NOT TESTABLE if no k* exists.

- **Double dissociation at k*:** the same rule as version 1 (Δhedge(fam) > Δhedge(know) and ΔH(know) > ΔH(fam), both with 95% CIs excluding 0). Entropy is measured at the prefill prompt, with the vector added at its last position.
- **Coherence (descriptive):** the share of outputs that repeat one token more than 8 times in 24, per condition. Random directions at the same k are the control for degenerate text.

## What may be claimed

- **"Steering along the base model's familiarity direction makes the instruct model say it does not know":** for a pair only if REUSED.
- **"Instruction tuning reuses the base model's familiarity direction":** only if both pairs are REUSED.
- If the version-1 and version-2 verdicts differ, both are reported. The difference is attributed to the position fix only as an interpretation.

## Not to be done after seeing results

Changing the dose grid, the rule for choosing k*, ℓ*, the bars, the prompt set or the pairs; choosing k* by looking at the base direction.
