# X3 results: the X2 test on Gemma-2-9B-it (run 25 Nov 2026, branch arr-v2)

Plan: `x3_plan.md` (commit ce22206, 04:44 UTC, before any run; hashes in `HASHES.txt`). No amendment. Same 4,376 questions as X2, the paper's permissive prompt, vLLM 0.34.0 (correct Gemma-2 attention), judges Qwen2.5-32B and Gemma-2-27B (correct attention). Analysis: the hashed `scripts/x2_analysis.py`, unchanged (`scripts/x3_analysis.py` swaps folders and names only).

## Result

Coefficient of obscurity with Gemma's own knowledge score and relation fixed effects; 95% intervals resample whole relations (5,000 draws, seed 0).

| Model | n | Two judges (primary) | Regex, 20 short replies |
|---|---|---|---|
| All deciles | 4,342 | **+2.99 [+2.35, +3.82]** | +3.16 [+2.50, +3.91] |
| Deciles 2 to 9 | 3,474 | +2.56 [+1.85, +3.49] | +2.75 [+2.11, +3.54] |
| Without country, capital, sport | 3,442 | +3.01 [+2.27, +3.90] | +3.27 [+2.52, +4.06] |
| Knowledge score as 11 levels | 4,342 | +3.03 [+2.38, +3.87] | +3.20 [+2.57, +3.92] |
| Qwen alone / Gemma-27B alone | 4,376 / 4,342 | +3.12 [+2.50, +3.84] / +2.99 [+2.35, +3.82] | |

**Decision rule 1 applies:** the interval is above zero, so the effect replicates on a second model. Claims: Llama-3.1-8B-Instruct and Gemma-2-9B-it on PopQA.
Predicted abstention: 0.15 at the 90th percentile of popularity, 0.47 at the 10th (averaged over the 4,342 fitted questions). Mean abstention 0.31 (two judges), 0.39 (regex).
Knowledge lowers abstention (-1.81 [-2.28, -1.49]). The two are not ranked.
The relation `color` (34 questions) has no abstention at all, so its outcome is constant and it is dropped from the fits, as the plan says. 26 of 5,000 draws of the 11-level two-judge fit and 1 of the regex fit did not converge and are dropped.
Figure: `fig_x3_gradient.pdf` (intervals resample questions within decile and band).

## Pilot and run

Pilot clean, every check (`pilot_checks.csv`): 200 of 200 complete, no empty or off-format output, greedy correctness equal to stored on 40 of 40, abstention 0.35, kappa 0.796 [0.726, 0.862], both judges equal to their reference labels on 1,161 of 1,161.
Full run: 13,128 replies, none empty, 0 off-format labels for either judge, kappa 0.84. The analysis was run twice with identical output. The kappa the draft recalled (0.67) was a Fleiss kappa from the faulty judge.

## Limits

- Sizes are not comparable with X2: the prompt differs (it invites abstaining), as does the model.
- Under the plain prompt Gemma rarely abstains, so the effect is shown only for the permissive prompt.
- One judge (Gemma-2-27B) is from the same family as the tested model; Qwen alone gives the same result (+3.12).
- Knowledge is string-graded forced answers; judges and regex are not validated against people. 16 relations, one dataset. Decile 1 with 9 or 10 of 10 correct holds 47 questions.
- No second label set with the faulty Gemma-27B attention was made (stated in the plan).

## Files

`coefficients.csv`, `gradient.csv`, `questions.csv`, `summary.txt`, `pilot_checks.csv`, `fig_x3_gradient.*`, `paper_insert.tex` (paste-ready, not in any paper, not compiled), `raw/` (every reply and label, with provenance), `CHECKSUMS.txt`.
Re-run: `python scripts/x3_analysis.py && python scripts/x3_figure.py && python scripts/x3_insert.py` (CPU).
