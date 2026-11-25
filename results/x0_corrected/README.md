# X0 with corrected Gemma-27B labels (re-frozen 24 Nov 2026, branch arr-v2)

These numbers supersede `results/x0/`, which is kept unchanged as the record of the paper as printed. Only the Gemma-2-27B judge labels changed (soft-capping on; see `../rejudge_gemma/README.md`); the two-judge rule and every script, seed and draw count are the same.
Rebuild: `sh results/x0_corrected/run_all.sh <python> <empty scratch dir>` (about 2.5 min; needs scipy and torch). It builds a shadow repo with corrected labels, regenerates the repo's summaries there, and runs the X0 scripts on it. Run with the old labels, the same pipeline reproduces `results/x0/` (all numbers).
`CHECKSUMS.txt` lists every CSV. `x0_rerun_log.txt` compares against the shadow's regenerated outputs, not the committed ones.

| Quantity | As printed (`results/x0`) | Corrected |
|---|---|---|
| Abstention OK / OU / PK / PU | 0.43 / 0.48 / 0.00 / 0.13 | 0.61 / 0.71 / 0.003 / 0.18 |
| kappa between the two judges | 0.62 [0.57, 0.67] | 0.91 [0.88, 0.94] |
| Margin | +0.30 [+0.20, +0.40] | +0.43 [+0.31, +0.54] |
| Margin, relations resampled | [-0.09, +0.43] | [-0.15, +0.62] |
| Unknown-fact gap, raw | +0.34 [+0.25, +0.43] | +0.53 [+0.42, +0.63] |
| Unknown-fact gap, label-corrected | +0.18 [+0.04, +0.31] | +0.35 [+0.19, +0.50] |
| same, relations then questions resampled | [-0.01, +0.37] | [+0.06, +0.65] |
| same, left-out relations with an interval including 0 | 1 of 16 | 0 of 16 |
| Margin, three label corrections | +0.14 [-0.01, +0.29] | +0.24 [+0.07, +0.41] |
| Margin by scorer, smallest to largest (one-judge corrected excluded) | +0.26 to +0.44 | +0.42 to +0.44 |
| Margin at 10 of 10 | +0.28 [+0.17, +0.38] | +0.38 [+0.26, +0.50] |
| Margin outside country, capital, sport (7 facts) | +0.00 | -0.05 |

Unchanged: everything regex- or label-only (A7 pool, A8 retests, within-relation regex table, R1, R3, prereg table), and "One judge, corrected" (+0.13).
Still true after the correction: every scorer's relation-resampled margin interval includes zero; 80 of 87 obscure-known facts are in three relations; no human validation (0 of 120). A kappa of 0.91 is agreement between two LLM judges.
No longer true: "zero" abstention on popular known facts (one of 300 replies, 0.003); the earlier note that judge choice moves the margin (it no longer does); the label-corrected gap being fragile.
