# X2 results (run 24 Nov 2026, branch arr-v2)

Plan: `x2_plan.md` (committed 45d74be before any run). Amendment: `x2_amendment_1.md` (committed 87fa209 after the pilot, before the full run). Hashes: `HASHES.txt`.
Sample 4,376 PopQA questions (all 16 relations, up to 30 per within-relation popularity decile), Llama-3.1-8B-Instruct, vLLM 0.34.0. Not the 5,000 of the draft: four relations have fewer than 300 questions.

## Result

Coefficient of obscurity (0 = most viewed question of its relation, 1 = least viewed) in a fractional logit with the knowledge score and relation fixed effects; intervals resample whole relations (5,000 draws, seed 0).

| Model | n | Two judges (primary) | Two judges, the paper's faulty Gemma labels | Regex, 20 short replies |
|---|---|---|---|---|
| All deciles | 4,376 | **+3.68 [+2.73, +4.74]** | +2.41 [+1.74, +3.22] | +4.41 [+3.32, +5.68] |
| Deciles 2 to 9 (no extreme tenths) | 3,501 | +3.56 [+2.54, +4.74] | +2.28 [+1.54, +3.23] | +4.15 [+2.99, +5.49] |
| Without country, capital, sport | 3,476 | +3.42 [+2.46, +4.58] | +2.23 [+1.58, +3.02] | +4.10 [+2.99, +5.52] |
| Knowledge score as 11 levels | 4,376 | +3.73 [+2.76, +4.82] | +2.44 [+1.75, +3.27] | +4.44 [+3.34, +5.72] |

Predicted abstention (primary, other covariates as observed): 0.21 at the 90th percentile of popularity, 0.73 at the 10th.
**Decision rule 1 applies:** the interval is above zero, so the popularity effect holds across the range, for Llama and PopQA.
Knowledge also lowers abstention in every model (primary -0.97 [-1.62, -0.42]). The two are not ranked.

Figure: `fig_x2_gradient.pdf` (intervals there resample questions within decile and band, not relations). Decile 1, 9 or 10 of 10 correct, holds 38 questions.

## The Gemma judge in the paper is faulty (found by the pilot)

- The paper's Gemma-2-27B judge used transformers' default `sdpa` attention, which skips Gemma-2's attention-logit soft-capping. Re-running that path reproduces the stored outputs exactly; `eager` and vLLM agree with each other on all 1,161 replies and with the stored labels on 84% (DONT_KNOW or not).
- Example: "I don't have information on the capital of Sanchez Carrion Province." was labelled ANSWER.
- The paper's two-judge numbers on its own 387 questions, same replies (`gemma_judge_diagnosis/paper_numbers_by_gemma_labels.csv`):

| | Paper (sdpa) | Correct (eager) |
|---|---|---|
| kappa with Qwen | 0.62 | 0.91 |
| Abstention OK / OU / PK / PU | 0.43 / 0.48 / 0.00 / 0.13 | 0.61 / 0.71 / 0.00 / 0.18 |
| Margin | +0.30 [+0.20, +0.40] | +0.43 [+0.31, +0.54] |
| Unknown-fact gap | +0.34 [+0.26, +0.43] | +0.53 [+0.42, +0.63] |

- Everything else that uses the Gemma-27B labels inherits this: E13, E14, the other families' kappas (0.30 to 0.47, hence "unmeasured"), E4, E8, E9, E12, E15, E16, the two-judge rows of the robustness table, X0 (A1 to A6, A9), and `existing_gradient.csv` here.
- Gemma-2-9B as a subject model was loaded with the same default and used the same setting; checked afterwards, the effect is negligible (`../rejudge_gemma/README.md`).
- This is post hoc and was seen before the X2 full run; no X2 effect had been computed then.

**Follow-up:** all stored replies were re-judged (`../rejudge_gemma/`) and X0 re-frozen (`../x0_corrected/`). Gemma-2-9B was checked: same setting, negligible effect.

## Pilot

14 pass/fail checks, 12 passed. Failed: two-judge kappa 0.906 (above the planned range) and vLLM-Gemma against stored labels (0.843, bar 0.95). Both trace to the faulty stored labels. Amendment 1 replaced those two rules; both then hold (0.906 against 0.60; 1.000 against 0.95).
The draft's Tuesday-noon drop rule had passed when the run was requested (17:37 IST).
Full run: no empty reply, no missing label; unparsed labels 0 (Qwen), 0 (Gemma), 115 of 13,128 (Gemma sdpa). kappa on the full run 0.913.

## Limits

Knowledge is the repo's string grader on forced answers; some "0 of 10" popular facts are known by meaning (E7). Judges and regex are not validated against people. One model, one dataset; 16 relations resampled. Deciles are ranks within PopQA's relation, not absolute views. vLLM, not transformers, generated the replies (greedy correctness matched stored on 97.5% of 120 pilot questions; mean knowledge score within 0.003).

## Files

`coefficients.csv`, `gradient.csv`, `questions.csv`, `summary.txt` (primary); `paper_sdpa_judge/` (same analysis, faulty labels); `gemma_judge_diagnosis/`; `existing_gradient.csv` (step 1, four-level, stored labels); `pilot_checks.csv`; `sample.jsonl`; `raw/` (every raw reply and label, with provenance); `CHECKSUMS.txt`.
Re-run: `python scripts/x2_analysis.py && python scripts/x2_sdpa_compare.py && python scripts/x2_gemma_diagnosis.py && python scripts/x2_figure.py` (CPU). The analysis was run twice with identical output.
