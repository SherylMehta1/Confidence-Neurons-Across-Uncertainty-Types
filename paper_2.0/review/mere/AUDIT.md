# Audit of the MERe version (2026-10-31)

Written for: the authors, to decide whether to replace the file on OpenReview. Two blind agents audited `mere/main.pdf` (submitted version): a numbers auditor and a cold-read referee (`cold_read_findings.md`). Key claims were then rechecked against the source files.

## Verdict on the submitted version

No number in it was altered from its source: cell rates recomputed from the raw judge files match (113/261, 143/300, 40/300, 0/300). The problems are in what the text claims from those numbers.

## Errors in the submitted version

| # | Submitted text | What the files show | Source |
|---|---|---|---|
| 1 | "no model-side measure shows the obscure names to be unfamiliar" (abstract, Section 6, conclusion) | **Wrong.** Recognition correlates with log page views (rho +0.52); among unknown facts it is 0.11 obscure vs 0.56 popular. The paper quoted only the known-fact cells (0.83 vs 0.75). Recognition still does not explain the gap (5% shrink) | results/round3/e6_summary.txt |
| 2 | "tracks popularity more than knowledge" (title), +0.30 | **The preference sits inside three relations.** Outside them +0.00 [-0.18, +0.31] (7 obscure known, 96 popular unknown); inside +0.46 [+0.38, +0.54] with only 4 popular unknown. The two defining cells hold different kinds of question. New post hoc check | analysis/mere_checks.txt |
| 3 | "holds in every variant by point estimate" | True only of Table 2 rows; the forced-answer readout in the same section is -0.12 | results/factorial_4models/effects.csv |
| 4 | "Qwen, Gemma and Mistral rarely say I don't know"; preference "appears in the two families where we can measure it" | Omits a measure the paper uses for Llama's +0.13: the corrected judge gives obscure-unknown rates 0.76, 0.42, 0.74 and leans toward knowledge (-0.21, -0.11, -0.17). Also omits Gemma's plain-prompt three-judge estimate, +0.01 [-0.03, +0.06] | results/hedge_judge/summary.txt; round3/e5 |
| 5 | +0.34, the "firmest result", not marked post hoc | Comes from a post hoc script, so Section 2's rule mislabels it | analysis/round4_checks.py header |
| 6 | Abstract: "lens readouts point to a generic abstention signal" | Section 6 does not say this; lenses do not locate the effect | paper's own Section 6 |
| 7 | "at most 0.03 ... contain the gold answer"; "refusals, not hidden answers" | 0.03 holds only with name echoes excluded (0.29 of obscure-known abstaining replies with them), and the main text did not say so | analysis/echo_check.txt |
| 8 | "it never does" (popular known) | True of 300 sampled replies; 0.01 on 150 other popular known questions | results/round3/e4_summary.txt |
| 9 | "All label corrections" | Three corrections; the paraphrase relabel and name-cue control are not included | Table 2, note b |
| 10 | Steering "adds nothing" with no model named | Llama only; Gemma's residual on its defensible scorer is +0.05 [+0.03, +0.07] | results/round3/e4_summary.txt |

## New in the audit, in the paper's favor

Within capital, country and sport, known facts still show a popularity effect: 0.46 vs 0.00, +0.46 [+0.38, +0.54] (80 vs 18 questions; post hoc). So the popularity effect holds at both knowledge levels within comparable relations. Only "more than knowledge" is fragile.

## What the corrected copy does

`mere/main_v2.pdf`: leads with the popularity effect at both knowledge levels, states the comparison with knowledge as unsettled, and corrects items 1-10. It also makes the cold read's framing changes (one term per quantity, "answers correctly when forced", entropy as a rival for the unknown-fact gap, a conditional steering reading). New title: "In Llama, Saying 'I Don't Know' Depends on Entity Popularity, Even Among Facts It Does Not Know". The body ends on page 8; all checks pass. No result file was changed.

## Left undone in both versions

- Table 2 prints +0.35 for a value the source gives as +0.355 (the auditor's recomputation, +0.3552, rounds to +0.36).
- No human labels for the judges; no within-relation estimate of the preference. The design cannot give one: 4 popular unknown questions sit in the three relations.
- Blank n cells in Table 2 (values exist: name-cue 342, both-agree 231, regrade 344, alias drop 339, guessable 335, joint 278, namesake 384; these are the auditor's figures, not rechecked).
- The auditor could not trace T=0.7, the 24-token limit, sample seed 0, or plan labels on the first five measure rows from the files it was allowed to read.
- Cut from the appendix for the page limit: regraded-label details, strict-decline rates, namesake medians, surprisal coefficients, onset table, cross-model agreement. These remain in the UniReps version.

## The UniReps paper has the same problems

`paper/main.tex` (submitted 11 Oct) carries items 1, 2, 6 and 7, and its title has the same issue as item 2. Its Section 3.1 says "no model-side measure shows the obscure names to be unfamiliar". It has not been changed. Whether to tell UniReps or correct it at camera-ready is an author decision.
