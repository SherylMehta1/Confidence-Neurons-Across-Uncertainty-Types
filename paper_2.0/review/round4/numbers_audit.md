# Numbers audit: paper/main.tex against the raw result files (round 4)

Auditor scope: every quantitative statement, comparison, count, interval, hash prefix and scope word in main.tex. Macros were expanded through numbers.tex and tab_onsets.tex. Evidence came only from the result files listed in the brief. I did not open author notes, CLAIMS, LAB_LOG, PROJECT_*, WHAT_CHANGED, FINDINGS.md, README.md, review/ or revision/.

Repo root for evidence paths below: `C:/Users/starb/Desktop/Sher/Confidence-Neurons-Across-Uncertainty-Types` ("repo/"). Paper analysis files are `C:/Users/starb/Desktop/Sher/paper/analysis/` ("an/").

Independent recomputations done for this audit:
- Two-judge and three-judge cell rates and contrasts for all four models, recomputed from `repo/results/hedge_measure_v2/items_*.jsonl` and `judge_*_*.jsonl`. They match summary.txt and e5_summary.txt to three decimals (Llama two-judge 0.477/0.433/0.133/0.000, pop-know +0.300; majority 0.747/0.644/0.207/0.000, +0.437). The identity "preference = OK minus PU" holds algebraically and numerically.
- SHA-256 of all 16 plan files. Every printed prefix matches.
- Pool: `repo/data/familiarity_v3/all_measured.jsonl` has 2,675 rows. Two rows have a null subject ("What is the capital of ?", "Who is the author of ?"), so 2,673 are scored, matching the analyses. 16 relations; 10 forced samples and 20 free replies per item.
- Steering EVAL split: 150 questions × 3 replies = 450 replies per condition (`round3/e4/replies_llama.jsonl`).
- Figure provenance JSONs: plotted values equal the summary values.
- Orphan macros: 525 macros are defined in numbers.tex, 347 are used and 178 are unused. No macro is undefined and no macro is duplicated. main.pdf (17:18) is newer than main.tex (17:17) and numbers.tex (16:09).

Status key: **exact**: same value at the printed precision with no rounding question. **rounding-ok**: printed value is a correct rounding of the file value. **mismatch**: value, count, configuration or scope differs from the evidence. **no-evidence-found**: no listed file holds the value.

## Item table

### Abstract
| id | claim | evidence (file : field) | value in file | status |
|---|---|---|---|---|
| A1 | "says IDK to 43% of questions it answers correctly…" (\tcObsKnownAbsPct) | an/lockin_checks.txt : cell obscure HighlyKnown abstention | 0.433 | rounding-ok |
| A1b | "43% **of questions** … correctly in at least 9 of 10 forced samples" | same; data rule in all_measured.jsonl | 0.433 is the mean share of **replies** per question, not a share of questions; "known" also requires a correct greedy answer | mismatch (minor wording) |
| A2 | "on popular entities the rate is 0%" | lockin_checks : popular HighlyKnown abstention | 0.000 [0.000, 0.000] | exact |
| A3 | "answer entropy is similar in the two cases" | lockin_checks : entropy | 0.951 [0.828, 1.080] vs 0.858 [0.801, 0.915] | exact |
| A4 | gap holds with relabel / regrade / name-no-answer | round3/e1_summary T1; e7_summary T7a, T7c | +0.322 [+0.157,+0.490]; +0.234 [+0.119,+0.348]; +0.429 [+0.312,+0.542] | exact |
| A5 | "Gemma-2-9B rarely abstains under a plain prompt" | hedge_measure_v2/summary gemma; e5_summary gemma cells | max two-judge cell 0.047; max majority cell 0.107 | exact |
| A6 | permissive: Gemma tracks popularity more than knowledge | round3/e9_summary gemma | diff +0.297 [+0.190,+0.400], FOLLOWS POPULARITY | exact |
| A7 | "answer entropy follows knowledge" | factorial_4models/effects.csv entropy | know > pop in all 4, CIs of the difference exclude 0 | exact |

### Introduction
| id | claim | evidence | value | status |
|---|---|---|---|---|
| I1 | Llama preference +0.30 [+0.20,+0.40] | hedge_measure_v2/summary llama primary pop-know | +0.300 [+0.204,+0.398] | rounding-ok |
| I2 | "survives four label and measurement checks" | e1 T1, e7 T7a, e5 llama, e7 T7c | four estimates, all CIs > 0 | exact |
| I3 | Gemma permissive +0.30 [+0.19,+0.40] | e9_summary gemma | +0.297 [+0.190,+0.400] | rounding-ok |
| I4 | "Under the plain prompt it rarely abstains" | as A5 | ≤0.107 | exact |
| I5 | "Only Llama and, under a permissive prompt, Gemma let us ask" | κ in hm summary / e5 / e9 | Llama 0.62 (two judges); Gemma 0.64 with rates ≤0.11 (plain), 0.67 (permissive) | exact |

### Section 2 (Setup)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| S1 | "16 relations of PopQA" | all_measured.jsonl : distinct prop | 16 | exact |
| S2a | "ten samples" | all_measured.jsonl : len(samples) | 10 | exact |
| S2b | forced samples at T=0.7 | none in result files | — | no-evidence-found |
| S3 | known = greedy + ≥9 samples correct; "unknown if no sample is" | all_measured.jsonl : frac_correct by slick | HighlyKnown ≥0.9; Unknown 0.0; one "Maybe" item has frac_correct 0.0 (correct greedy), so Unknown also requires a wrong greedy | mismatch (minor definition) |
| S4 | 2,303 / 2,392 / 2,282 / 2,378 questions | factorial_4models/summary "HighlyKnown/Unknown"; cells.csv n sums | 2303 / 2392 / 2282 / 2378 | exact |
| S5 | "(cell counts in app:robust)" for all four models | main.tex App A | App A gives only Llama's cell counts | mismatch (minor pointer) |
| S6 | up to 100 per cell, 387 total | items_llama.jsonl | 100/87/100/100 = 387 | exact |
| S7 | three replies, T=0.7, up to 96 tokens | gens_llama.jsonl (3 samples); hedge_measure_v2_criterion.md (T, 96) | 3; T=0.7; 96 | exact |
| S8 | κ=0.62 [0.57, 0.67] | hm summary; lockin_checks (2) | 0.619 [0.568, 0.668] | rounding-ok |
| S9 | pre-specified bar 0.60 | hedge_measure_v2_criterion.md l.7 | κ ≥ 0.60 | exact |
| S10 | other families below bar | hm summary | 0.30 / 0.47 / 0.39 | exact |
| S11 | "Intervals are 95% percentile bootstraps over questions within cells" (global) | unireps_checks/summary (pair CI is a regression-intercept CI); LPM CIs bootstrap over questions, not within cells | not true of every interval | mismatch (minor; interval method) |
| S12 | preference = OK − PU in the balanced 2×2 | recomputation | 0.433 − 0.133 = 0.300 | exact |

### Section 3.1 (Llama)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| R1 | obscure known: IDK in 0.43 of replies | lockin_checks | 0.433 | rounding-ok |
| R2 | popular known: "never does" | lockin_checks; recompute (0/300 replies) | 0.000 | exact |
| R3 | entropy 0.95 vs 0.86 nats | factorial cells.csv entropy | 0.951 / 0.858 | rounding-ok |
| R4 | pop effect +0.39 [+0.33,+0.45] | hm summary primary pop | +0.388 [+0.328,+0.449] | rounding-ok |
| R5 | know effect +0.09 [+0.03,+0.15] | hm summary primary know | +0.089 [+0.031,+0.147] | rounding-ok |
| R6 | preference +0.30 [+0.20,+0.40] | hm summary | +0.300 [+0.204,+0.398] | rounding-ok |
| R7 | "Its sign holds under **five** other measures of abstention" | Table 2 measures block | Table 2 lists **six** non-primary rows (CLARIFY, 3 judges, 70B decline, regex 96-tok, regex 24-tok, corrected judge). It is five only if the two regex rows count as one measure | mismatch (minor count) |
| R8 | range "+0.13 to +0.44" | hedge_judge summary; factorial effects; e5 | min +0.132; max +0.442 (+0.437) | rounding-ok |
| R9 | entropy rises 2.14 (knowledge) and 0.51 (popularity) | effects.csv llama entropy | 2.139 / 0.513 | rounding-ok |
| R10 | relabel κ=0.79 | an/round3_label_checks (2) | 0.788 | rounding-ok |
| R11 | "Of the 87 obscure known facts, 27 stay known" | e1_summary T3 obscure crosstab | 27+49+11=87; HK→HK 27 | exact |
| R12 | these abstain at 0.46 | e1 T1 cell unc/High | 0.457 (n=27) | rounding-ok |
| R13 | relabel preference +0.32 [+0.16,+0.49] | e1 T1 diff | +0.322 [+0.157,+0.490] | rounding-ok |
| R14 | "entropy still follows knowledge" | e1 T2 | know − pop +1.402 [+1.020,+1.730] | exact |
| R15 | 9 of 100 popular unknown regraded known, 31 between | e7 T7a control | 9 / 60 / 31 | exact |
| R16 | regraded preference +0.23 [+0.12,+0.35] | e7 T7a | +0.234 [+0.119,+0.348] | rounding-ok |
| R17 | third judge κ=0.74, preference +0.44 | e5_summary llama | 0.741; +0.437 | rounding-ok |
| R18 | ≤0.03 of IDK replies per cell contain gold answer | round3_label_checks (3) | 0.00 / 0.00 / 0.03 / nan | exact |
| R19 | strict declines +0.43 [+0.31,+0.54] | e7 T7c | +0.429 [+0.312,+0.542] | rounding-ok |
| R20 | recognition 0.83 vs 0.75 | e6 K3 cell means | 0.831 / 0.750 | rounding-ok |
| R21 | shrinks popularity coefficient by 5% | e6 (a) | "shrinks by 5%" (0.355 → 0.337) | exact |
| R22 | namesake medians 11–13 vs 1 | an/ambiguity_check.txt | 11 / 13 / 1 | exact |
| R23 | unique-name obscure known: 0.43 | ambiguity_check single-homonym | 0.434 (n=73) | rounding-ok |
| R24 | spelling explains part of the pair gap but barely moves the pool coefficient | unireps_checks/summary | +0.758 → +0.434; −1.871 → −1.791 | rounding-ok |
| R25 | surprisal uncorrelated with page views | e2 S1 | −0.001 | exact |
| R26 | recognition higher for obscure known | e6 K3 | 0.831 > 0.750 | exact |
| R27 | hedge log-odds "equally" −0.12 [−0.72,+0.45] | effects.csv llama hedge_logodds | −0.122 [−0.722,+0.449] | rounding-ok |
| R28 | free "I" log-odds +8.4 vs +3.1 | e1 T4 logodds_I_free | +8.365 / +3.111 | rounding-ok |
| R29 | free entropy +0.16 (knowledge) vs +0.05 (popularity) | e1 T4 entropy_free | +0.160 / +0.053 | rounding-ok |

### Section 3.2 (Other families)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| F1 | entropy follows knowledge in all four families | effects.csv entropy | all 4 | exact |
| F2 | Qwen, Gemma, Mistral rarely say IDK | hm summary primary unc/Unkn | 0.050 / 0.047 / 0.097 | rounding-ok |
| F3 | judges disagree on them | hm κ | 0.30 / 0.47 / 0.39 | exact |
| F4 | Gemma three-judge κ=0.64 | e5 | 0.643 | rounding-ok |
| F5 | "abstains in at most 0.11 of replies in any cell" | e5 gemma cells; recompute | max 0.107 | rounding-ok |
| F6 | "four-family test **on the forced-answer hedge readout** found both conditions in Gemma and one in Qwen and Mistral" | factorial_4models/summary.txt "governing hedging readout" | counts right (REPLICATES/PARTIAL/PARTIAL). But the governing readout was **hedge_rate** (regex on free replies) for Qwen, Mistral and the Llama reference; hedge_logodds governed Gemma only | mismatch (configuration) |
| F7 | "that readout shows no preference in Llama either" | factorial summary LLAMA | true for hedge_logodds (−0.122 [−0.722,+0.449]). On the test's governing readout for Llama, hedge_rate gives +0.442 [+0.345,+0.534] → REPLICATES | mismatch (configuration / selective) |
| F8 | permissive κ 0.71, 0.67 | e9 | 0.709 / 0.670 | rounding-ok |
| F9 | Llama permissive +0.47 [+0.35,+0.59] | e9 | +0.470 [+0.349,+0.587] | rounding-ok |
| F10 | Gemma permissive +0.30 [+0.19,+0.40] | e9 | +0.297 [+0.190,+0.400] | rounding-ok |
| F11 | Gemma 0.41 obscure known, 0.00 popular known | e9 cells | 0.407 / 0.0 | rounding-ok |
| F12 | "Cross-model agreement **on that readout** … met a post hoc rule for one of three pairs" | an/cross_model_agreement.txt | count right (Mistral only). But "that readout" refers back to permissive-prompt abstention, while the analysis uses plain-pool forced-answer hedge log-odds and entropy | mismatch (configuration / referent) |

### Section 3.3 (Mechanistic)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| M1 | three directions fitted on half; dose on a second set; scored on a third | e4 summary; replies split select/eval 150/150 | as stated | exact |
| M2 | Llama abstention 0.01 → 0.10 | e4 EVAL none/fam judges | 0.007 / 0.100 | rounding-ok |
| M3 | +0.09 [+0.06,+0.12] over random | e4 D1 | +0.092 [+0.062,+0.125] | rounding-ok (upper bound is a half value) |
| M4 | short of pre-specified +0.10 bar | e4_steering_v3.md D1; e4 "NOT MET" | bar 0.10 | exact |
| M5 | Gemma 0.00 → 0.04 at 4× | e4 gemma | 0.000 / 0.044; k*=4 | rounding-ok |
| M6 | cosine 0.95 | e4/geometry_llama.json | 0.948 | rounding-ok |
| M7 | "abstaining questions are mostly obscure" | e4 gives only 650 vs 603 counts, no arm breakdown | — | no-evidence-found (plausible from cells.csv hedge_rate) |
| M8 | removing it leaves +0.01 [−0.00,+0.02]; Gemma +0.02 [+0.01,+0.03] | e4 D2 | +0.010 [−0.001,+0.024]; +0.017 [+0.005,+0.032] | rounding-ok |
| M9 | within-obscure cosine 0.91; +0.01 [+0.00,+0.03] (three judges) | e8 llama | 0.913; +0.014 [+0.000,+0.029] | rounding-ok |
| M10 | fam − know +0.06 [+0.02,+0.09] | e4 D3 llama | +0.056 [+0.024,+0.089] | rounding-ok |
| M11 | Gemma entropy know − fam +0.66 [+0.58,+0.74] | e4 D4 gemma | +0.659 [+0.577,+0.741] | rounding-ok |
| M12 | popularity raises hedge term in all four base models | e3_window_effects hedge_term pop | all CIs > 0 (Gemma-2-2B +0.098 [+0.031,+0.168]) | exact |
| M13 | "Knowledge raises it **about as much**" (all four) | e3 hedge_term | Gemma-2-2B: know +0.369 vs pop +0.098, diff −0.271 [−0.377,−0.151] (≈3.8×). True for the other three | mismatch (scope; the caption's "about as large or larger" is accurate) |
| M14 | instruct partners lean further toward knowledge | e10 | instruct diff < base diff for all 3 pairs, both terms | exact |

### Limitations
| id | claim | evidence | value | status |
|---|---|---|---|---|
| L1 | "cells of at most 100 questions" | cells.csv; lockin pool | true for abstention cells only; entropy, regex and lens cells reach 1,179 | mismatch (minor scope) |
| L2 | models 2–9B | model list | 2B–9B | exact |
| L3 | facts known ≥9/10 or never | all_measured.jsonl | consistent | exact |

### Table 1 (tab:cells)
| id | cell | evidence | value | status |
|---|---|---|---|---|
| T1-1 | OK abstention 0.43 [0.35,0.51], n 87 | lockin_checks (1b) | 0.433 [0.352,0.513] n=87 | rounding-ok |
| T1-2 | OU 0.48 [0.41,0.55], n 100 | lockin | 0.477 [0.410,0.547] | rounding-ok |
| T1-3 | PK 0.00 [0.00,0.00], n 100 | lockin | 0.000 [0,0] | exact |
| T1-4 | PU 0.13 [0.08,0.19], n 100 | lockin | 0.133 [0.083,0.190] | rounding-ok |
| T1-5 | OK entropy 0.95 [0.83,1.08] | lockin | 0.951 [0.828,1.080] | rounding-ok |
| T1-6 | OU entropy 3.51 [3.46,3.56] | lockin | 3.510 [3.457,3.562] | rounding-ok |
| T1-7 | PK entropy 0.86 [0.80,0.92] | lockin | 0.858 [0.801,0.915] | rounding-ok |
| T1-8 | PU entropy 2.58 [2.47,2.69] | lockin | 2.577 [2.469,2.685] | rounding-ok |
| T1-9 | caption: entropy on full pool | lockin (n 87/1179/676/361) | as stated | exact |

### Table 2 (tab:robust)
| id | row | evidence | value | status |
|---|---|---|---|---|
| T2-1 | primary, n 387, +0.30 [+0.20,+0.40] | hm summary | +0.300 [+0.204,+0.398] | rounding-ok |
| T2-2 | DONT_KNOW or CLARIFY +0.31 [+0.21,+0.41] | hm secondary | +0.311 [+0.209,+0.414] | rounding-ok |
| T2-3 | three judges +0.44 [+0.32,+0.55] | e5 | +0.437 [+0.322,+0.550] | rounding-ok |
| T2-4 | 70B declines, n 387, +0.43 [+0.31,+0.54] | e7 T7c (n 100/87/100/100) | +0.429 [+0.312,+0.542] | rounding-ok |
| T2-5 | regex 96-token +0.43 [+0.31,+0.55] | hm regex | +0.429 [+0.306,+0.547] | rounding-ok |
| T2-6 | regex 20 × 24-token, n 2,303, +0.44 [+0.34,+0.53] | effects.csv hedge_rate; free list len 20 | +0.442 [+0.345,+0.534] | rounding-ok |
| T2-7 | corrected judge, n 2,303, +0.13 [+0.03,+0.24] | hedge_judge/summary llama | +0.132 [+0.027,+0.239] (CI method not stated; corrections from 1,515 judged generations) | rounding-ok (interval method not recoverable) |
| T2-8 | relabeled, n 253, +0.32 [+0.16,+0.49] | e1 T1 (n 110+27+74+42=253) | +0.322 [+0.157,+0.490] | rounding-ok |
| T2-9 | both agree +0.31 [+0.15,+0.48] | round3_label_checks (2) | +0.314 [+0.147,+0.480] | rounding-ok |
| T2-10 | regraded +0.23 [+0.12,+0.35] | e7 T7a | +0.234 [+0.119,+0.348] | rounding-ok |
| T2-11 | alias dropped +0.19 [+0.07,+0.31] | round3_label_checks worst case | +0.189 [+0.068,+0.308] | rounding-ok |
| T2-12 | guessable removed +0.30 [+0.19,+0.41] | an/guessability.csv judge_noguess | +0.2997 [+0.192,+0.411] | rounding-ok |
| T2-13 | FE + knowledge only, n 387, +0.35 | an/round2_checks (2) n=387; also e6/ambiguity n=384 | +0.355 | rounding-ok (half value printed down) |
| T2-14 | plus entity recognition, **n 387**, +0.34 [+0.27,+0.41] | e6 (a) "two judges **n=384** with" | +0.337 [+0.266,+0.411], n=384 | mismatch (count) |
| T2-15 | plus namesake +0.34 [+0.27,+0.41] | ambiguity_check LPM 2 (n=384) | +0.340 [+0.267,+0.412] | rounding-ok |
| T2-cap | corrected by one judge, Qwen2.5-32B-Instruct | hedge_judge/summary header | as stated | exact |

### Appendix A text
| id | claim | evidence | value | status |
|---|---|---|---|---|
| AA1 | pool 87/85/1179 obscure; 676/285/361 popular | lockin (3); all_measured.jsonl minus 2 null-subject rows | as stated | exact |
| AA2 | stratified 100 per cell or whole cell, seed 0 | items_llama.jsonl; criterion | counts as stated | exact |
| AA3 | "10,000 within-cell bootstraps (seed 0); LPM 2,000 bootstraps" | only e4/e8 state 10,000; hm summary, lockin, e1, e5, e7, e9, round2 and e6 state no count or seed | — | no-evidence-found |
| AA4 | regex recall 0.99 | hm summary | 0.99 | exact |
| AA5 | agreement with majority 0.98 / 0.84 / 1.00 | lockin (2) | 0.983 / 0.842 / 0.996 | rounding-ok |
| AA6 | 253; 62 / 169 / 15 / 7; agreement 0.91; κ 0.79 | round3_label_checks (2) | as stated; 0.913; 0.788 | rounding-ok |
| AA7 | most other changes move popular known facts to the dropped band | e1 T3 popular HK→Maybe | 312 of 676 | exact |
| AA8 | consensus cells 99 / 27 / 70 / 35 | round3_label_checks | as stated | exact |
| AA9 | pop +0.38 [+0.29,+0.47]; know +0.06 [−0.03,+0.15] | e1 T1 | +0.383 [+0.294,+0.472]; +0.061 [−0.030,+0.150] | rounding-ok |
| AA10 | entropy know − pop +1.40 [+1.02,+1.73] | e1 T2 | +1.402 [+1.020,+1.730] | rounding-ok |
| AA11 | 70B regraded every unknown fact | e7 "1540, fully graded 1540" (=1179+361) | as stated | exact |
| AA12 | 60 / 9 / 31; 86 / 2 / 12 | e7 T7a | as stated | exact |
| AA13 | alias drop: 0.48 of them; +0.19 [+0.07,+0.31] | round3_label_checks | con 0.48; +0.189 [+0.068,+0.308] | rounding-ok |
| AA14 | declines 0.66 / 0.00; hedged ≤0.03 per cell | e7 T7c | 0.659 / 0.0; max 0.025 | rounding-ok |
| AA15 | greedy + 5 samples; ρ=+0.52; coef −0.03 [−0.06,+0.00]; shrink 5% / 3% | e6 | 6 completions; +0.516; −0.027 [−0.057,+0.004]; 5% / 3% | rounding-ok |
| AA16 | popularity gap among unknown facts +0.34 [+0.26,+0.43] | guessability.csv judge_all popU | +0.343 [+0.257,+0.430] (from all items, not the guessable-filtered set the paragraph discusses) | rounding-ok |
| AA17 | regex non-guessable n 2,106; +0.48 [+0.36,+0.59] | guessability.csv regex_noguess | +0.480 [+0.361,+0.592] | rounding-ok |
| AA18 | medians 1 / 5 / 11 / 13; 73 at 0.43 | ambiguity_check | as stated; 0.434 | rounding-ok |
| AA19 | obscure known: 0.53 of free replies contain a gold alias; 0.72 of non-abstaining | an/round2_checks (1) gives 0.53 / 0.72. The newer **an/echo_check.txt** (17:25, after numbers.tex at 16:09) shows these include subject-name echoes; excluding echoes: **0.25 / 0.44** | 0.53 / 0.72 (incl. echoes) | mismatch (stale; alias matching inconsistent with round3_label_checks, which excludes echoes) |
| AA20 | "for 24 of 87 questions, one free reply **gives the correct answer** and another says IDK" | round2_checks: 24 of 87; echo_check.txt (echoes excluded): **9 of 87** | 24 vs 9 | mismatch (stale; overstated) |
| AA21 | LPM pop +0.29 [+0.23,+0.36]; know −0.05 [−0.14,+0.03]; entropy +0.14 | round2_checks (2) | +0.294 [+0.227,+0.362]; −0.054 [−0.140,+0.030]; +0.137 | rounding-ok |
| AA22 | 55 pairs; +0.76 → +0.43 [+0.18,+0.69] | unireps_checks summary; check1_pairs.csv | 55; +0.758; +0.434 [+0.182,+0.686] (CI is a regression-intercept interval, method not stated in paper) | rounding-ok (interval method not recoverable) |
| AA23 | 2,303 questions; −1.87 → −1.79 logits per SD | unireps M1/M2 logpop_z | −1.871 / −1.791, n 2303 | exact |
| AA24 | pairs compare only known facts | unireps "pop_given_known" | as stated | exact |

### Appendix B (Table 3 and text)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| AB1 | two-judge κ 0.62 / 0.30 / 0.47 / 0.39 | hm summary | as stated | exact |
| AB2 | Fleiss κ 0.74 / 0.41 / 0.64 / 0.54 | e5 | 0.741 / 0.409 / 0.643 / 0.535 | rounding-ok |
| AB3 | regex recall 0.99 / 0.58 / 0.48 / 0.73 | hm summary | as stated | exact |
| AB4 | both DK, OU 0.48 / 0.05 / 0.05 / 0.10 | hm summary | 0.477 / 0.050 / 0.047 / 0.097 | rounding-ok |
| AB5 | DK or CLARIFY, OU 0.65 / 0.32 / 0.22 / 0.25 | hm summary | 0.647 / 0.317 / 0.220 / 0.253 | rounding-ok |
| AB6 | three judges, OU 0.75 / 0.10 / 0.11 / 0.19 | e5_summary.csv obs_unk | 0.747 / 0.097 / 0.107 / 0.19 | rounding-ok |
| AB7 | regex, OU 0.85 / 0.18 / 0.07 / 0.21 | hm summary | 0.847 / 0.180 / 0.073 / 0.210 | rounding-ok |
| AB8 | three-judge preference +0.44 / – / +0.01 / – | e5 | +0.437 / +0.013 | rounding-ok |
| AB9 | hedge log-odds popularity 2.93 / 1.14 / 1.79 / 2.48 | effects.csv | 2.933 / 1.138 / 1.789 / 2.480 | rounding-ok |
| AB10 | knowledge 3.05 / 3.63 / −0.32 / 2.08 | effects.csv | 3.055 / 3.632 / −0.318 / 2.077 | rounding-ok |
| AB11 | difference −0.12 / −2.49 / +2.11 / +0.40 | effects.csv | −0.122 / −2.494 / +2.107 / +0.403 | rounding-ok |
| AB12 | four-family row: reference / one holds / both hold / one holds; caption says the conditions are tested "on this readout" (hedge log-odds) | factorial summary VERDICTs and "governing hedging readout" | verdicts right; governing readout hedge_rate for Qwen and Mistral (and the Llama reference, which REPLICATES on it) | mismatch (configuration; same issue as F6) |
| AB13 | CIs: three judges [+0.32,+0.55], [−0.03,+0.06]; log-odds [−0.72,+0.45], [−3.41,−1.64], [+1.79,+2.42], [−0.55,+1.34] | e5; effects.csv | [0.322,0.550], [−0.030,0.057]; [−0.722,0.449], [−3.413,−1.644], [1.787,2.424], [−0.545,1.344] | rounding-ok |
| AB14 | best pair Qwen-32B and Llama-70B, κ=0.96 on Llama | e5 pairwise | 0.955 (best pair in every family) | exact |
| AB15 | permissive cells: Llama 0.84 / 0.69 / 0.22 / 0.00; Gemma 0.54 / 0.41 / 0.11 / 0.00 | e9 | 0.84 / 0.693 / 0.223 / 0.0; 0.537 / 0.407 / 0.11 / 0.0 | rounding-ok |
| AB16 | Gemma licensing threshold 0.25 | e9_permissive_prompt.md; e9 summary | ≥0.25 (0.537) | exact |
| AB17 | existing two-judge question sets | e9 n (387; 400) | as stated | exact |

### Appendix C (Steering)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| AC1 | layer round(0.55(L−1)) | geometry json (17 of 32; 23 of 42) | 17.05→17; 22.55→23 | exact |
| AC2 | norms 4.8 / 3.1 / 7.0; 62.4 / 66.5 / 105.3 | geometry json | 4.838 / 3.149 / 7.034; 62.42 / 66.47 / 105.27 | rounding-ok |
| AC3 | dose grid {0.5, 1, 2, 4, 8} | e4 SELECT table | as stated | exact |
| AC4 | k* on 150 SELECT, 0.10 rule; Llama 1×, Gemma 4× | e4; kstar json | 1; 4 (fam beats random by >0.10 first at those doses) | exact |
| AC5 | Gemma regex 0.00 at 1×, 0.02 at 2× | e4 SELECT fam | 0.002 / 0.020 | rounding-ok |
| AC6 | 150 EVAL; 21 overlap | lockin (3) | 150; 21 | exact |
| AC7 | Llama row 0.01 (0.03), 0.008, 0.04 (0.10), 0.09 (0.25), 0.10 (0.23), 0.02 (0.02) | e4 EVAL | 0.007 (0.031), mean 0.0082, 0.044 (0.102), 0.089 (0.253), 0.100 (0.233), 0.018 (0.024) | rounding-ok |
| AC8 | Gemma row 0.00 (0.00), 0.003, 0.04 (0.03), 0.20 (0.74), 0.04 (0.27), 0.02 (0.07) | e4 EVAL | 0.000 (0.002), 0.0026, 0.042 (0.033), 0.196 (0.736), 0.044 (0.271), 0.020 (0.073) | rounding-ok |
| AC9 | 150 questions, 3 replies each | replies_llama.jsonl | 450 per eval condition | exact |
| AC10 | primary D1 / D2 / D3, both models | e4 | +0.092 [+0.062,+0.125] / +0.042 [+0.022,+0.063]; +0.010 [−0.001,+0.024] / +0.017 [+0.005,+0.032]; +0.056 [+0.024,+0.089] / +0.002 [−0.027,+0.031] | rounding-ok |
| AC11 | secondary D1 +0.17 [+0.13,+0.22], +0.11 [+0.08,+0.15]; D2 +0.02 [+0.00,+0.04], +0.05 [+0.03,+0.07] | e4 secondary | +0.173 [+0.128,+0.221], +0.114 [+0.078,+0.152]; +0.017 [+0.002,+0.036], +0.049 [+0.028,+0.072] | rounding-ok |
| AC12 | cos 0.95 / 0.63 | e4 | 0.948 / 0.631 | rounding-ok |
| AC13 | no condition produced degenerate text | e4 degenerate | 0.000 everywhere | exact |
| AC14 | within-level 587 vs 55, 36 vs 520; cos 0.91 / 0.47 | e8 | as stated; 0.913 / 0.472 | rounding-ok |
| AC15 | within-level rates Llama 0.02 / 0.02 / 0.19 / 0.22 / 0.03; Gemma 0.00 / 0.01 / 0.11 / 0.41 / 0.03 | e8 primary | 0.018 / 0.0174 / 0.187 / 0.222 / 0.031; 0.000 / 0.006 / 0.111 / 0.407 / 0.031 | rounding-ok |
| AC16 | D1w +0.17 [+0.12,+0.22], +0.10 [+0.07,+0.14]; D2w +0.01 [+0.00,+0.03], +0.03 [+0.01,+0.04] | e8 | +0.169 [+0.125,+0.217], +0.105 [+0.069,+0.144]; +0.014 [+0.000,+0.029], +0.025 [+0.008,+0.044] | rounding-ok |
| AC17 | Llama: carried by abstention | e8 "CARRIED BY ABSTENTION" | as stated | exact |
| AC18 | Fig. steering caption: "three-judge rates in app:steer" | main.tex App C | App C gives three-judge **differences** for E4 (secondary scorer) but no three-judge **rates**. The only three-judge rates printed are for E8, a different direction set (E4 rates exist in e4_summary "(secondary) majority-rule rates") | mismatch (caption vs content, minor) |
| AC19 | Fig. steering plotted values | fig_steering.provenance.json vs e4 | identical | exact |

### Appendix D (Lens)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| AD1 | window [1/3, 2/3]; 2,000 within-cell bootstraps | e3 / e10 headers | as stated | exact |
| AD2 | Table lwRows (16 estimates + CIs) | e3_window_effects.csv | all match (e.g., Llama +0.671 [+0.593,+0.750]; OLMo hedge +0.612 [+0.529,+0.696]) | rounding-ok |
| AD3 | Fig. lens caption: (a) popularity > knowledge in three models; (b) knowledge about as large or larger except OLMo, slightly smaller with overlapping CIs | e3 diffs | logodds diff CIs exclude 0 for Llama, Gemma-2-2B, OLMo; OLMo hedge diff +0.081 [−0.013,+0.179] | exact |
| AD4 | 5 hedge tokens / 6 answer tokens (token lists) | none in result files | — | no-evidence-found |
| AD5 | Table itRows (12 estimates + CIs) | e10_summary.csv | all match (e.g., Gemma-2-9B instruct hedge −1.281 [−1.555,−0.998]) | rounding-ok |
| AD6 | criterion held for no pair | e10 "NO CHANGE SHOWN" ×6 | as stated | exact |
| AD7 | onset table (8 rows: layers, F, K, D [CI], outcome, mid popularity) | jlens_cascade/outcomes.csv | all match (e.g., OLMo 32 / 19 / 14 / −5 [−5,−5] / contradicts / +0.293) | exact |
| AD8 | 1 supports, 1 contradicts, 6 undecided of eight | jlens summary | SUPPORTS 1, CONTRADICTS 1, UNDECIDED 6 | exact |
| AD9 | onset caption: intervals for D "often collapse to a point" | outcomes.csv | 2 of 8 have zero width | mismatch (minor wording) |

### Appendix E (Cross-model)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| AE1 | knowledge-differs n 313 / 60 / 108 | cross_model_agreement.csv | as stated | exact |
| AE2 | hedge agreement +0.11 [−0.00,+0.22], +0.38 [+0.13,+0.60], +0.25 [+0.06,+0.42] | csv | +0.108 [−0.001,+0.215], +0.384 [+0.133,+0.603], +0.247 [+0.058,+0.425] | rounding-ok |
| AE3 | entropy −0.17 [−0.28,−0.07], +0.03 [−0.24,+0.28], −0.07 [−0.27,+0.12] | csv | −0.174 [−0.278,−0.069], +0.033 [−0.235,+0.278], −0.075 [−0.267,+0.124] | rounding-ok |
| AE4 | gap +0.28 [+0.14,+0.42], +0.35 [−0.03,+0.74], +0.32 [+0.06,+0.57] | cross_model_agreement.txt | as stated | exact |
| AE5 | rule not met / not met / met | txt verdicts | NOT SUPPORTED / NOT SUPPORTED / PREDICTION HOLDS | exact |
| AE6 | same-knowledge n 1,784 / 1,977 / 2,000; hedge 0.50 / 0.38 / 0.64; entropy 0.46 / 0.54 / 0.69 | csv | 0.503 / 0.376 / 0.640; 0.458 / 0.541 / 0.687 | rounding-ok |

### Appendix F (Surprisal)
| id | claim | evidence | value | status |
|---|---|---|---|---|
| AF1 | Spearman −0.001 | e2 S1 | −0.001 | exact |
| AF2 | two-judge +0.013 [−0.026,+0.059] | e2 S2(a) | same | exact |
| AF3 | regex +0.07 [+0.05,+0.10] | e2 S2(b) | +0.071 [+0.046,+0.096] | rounding-ok |
| AF4 | with arm +0.01 [−0.01,+0.03] | e2 S3(b) | +0.012 [−0.008,+0.033] | rounding-ok |
| AF5 | point-biserial +0.02; partial −0.06 | an/surprisal_check.txt | +0.018; −0.059 | rounding-ok |

### Appendix G (Hashes; computed sha256 of file bytes)
| id | plan | printed | computed | status |
|---|---|---|---|---|
| AG1 | qwen_factorial_criterion.md | eba35dec | eba35dec | exact |
| AG2 | gemma_mistral_factorial_criterion.md | b53d9a27 | b53d9a27 | exact |
| AG3 | hedge_judge_criterion.md | ae0bb807 | ae0bb807 | exact |
| AG4 | hedge_measure_v2_criterion.md | 81acfc9f | 81acfc9f | exact |
| AG5 | jlens_cascade_criterion.md | c8e04372 | c8e04372 | exact |
| AG6 | e1_independent_labels.md | af50f53f | af50f53f | exact |
| AG7 | e2_name_surprisal.md | 2b69e06c | 2b69e06c | exact |
| AG8 | e3_lens_split.md | b2b916ce | b2b916ce | exact |
| AG9 | e4_steering_v3.md | 18c74883 | 18c74883 | exact |
| AG10 | e4_amendment_third_judge.md | 7bed3826 | 7bed3826 | exact |
| AG11 | e5_third_judge.md | f35aa4d9 | f35aa4d9 | exact |
| AG12 | e6_entity_knowledge.md | 937b46e3 | 937b46e3 | exact |
| AG13 | e7_label_and_decline_audit.md | 1498be91 | 1498be91 | exact |
| AG14 | e8_steering_within_level.md | 05474954 | 05474954 | exact |
| AG15 | e9_permissive_prompt.md | f6459163 | f6459163 | exact |
| AG16 | e10_instruct_lens.md | 6af179ce | 6af179ce | exact |
| AG17 | footnote: amendment's internal date is one day after its recorded hash time | amendment line 1 "2026-10-10 ~22:10 UTC"; HASHES_round3.txt 2026-10-09T22:02:16Z | ≈1 day | exact |

### Figures
| id | claim | evidence | value | status |
|---|---|---|---|---|
| FIG1ab | Fig. 1 (a, b): "values and n in Table 1" | fig_dissociation.provenance.json | means identical; CIs come from a separate bootstrap draw (OU upper 0.543 vs 0.547 in the table; PU lower 0.087 vs 0.083) | rounding-ok (negligible) |
| FIG1c | Fig. 1c: entropy effects in four instruct models | provenance vs effects.csv | identical | exact |
| FIG3 | Fig. lens plotted values | fig_lens_base.provenance.json vs e3_window_effects.csv | identical | exact |

## Summary

**Counts (209 items):** exact 88, rounding-ok 101, mismatch 16, no-evidence-found 4.

No numeric value in the paper was altered from its source file, and every hash prefix matches. No mismatch reverses the headline Llama result or the Gemma permissive-prompt result. Every macro value traces to a file. The mismatches are about how results are described (configuration, scope, counts) and about two stale free-reply numbers.

### Critical mismatches (each changes a stated sub-conclusion)
1. **Four-family test misdescribed (F6, F7, AB12; Sec 3.2 and Table 3 caption).** `factorial_4models/summary.txt` names the "governing hedging readout" per model. It is `hedge_rate` (regex on free replies) for Qwen, Mistral and the Llama reference, and `hedge_logodds` for Gemma only. On its governing readout, Llama's reference shows a preference (+0.442 [+0.345,+0.534], REPLICATES). Correction: say the test used regex free-reply rates for Qwen and Mistral and forced-answer hedge log-odds for Gemma. Then limit "that readout shows no preference in Llama either" to the log-odds readout (Llama −0.12 [−0.72,+0.45]), which is the readout that governed Gemma's verdict.
2. **Free-reply "correct answer" counts are stale (AA19, AA20; App A "Free replies").** The newer `paper/analysis/echo_check.txt` (17:25, after numbers.tex was generated at 16:09) shows that the round2_checks alias matches include echoes of the subject name. With echoes excluded, only 0.25 (not 0.53) of obscure-known free replies, and 0.44 (not 0.72) of non-abstaining ones, contain the answer. And only **9 of 87** (not 24) questions have both a correct reply and an abstention. Correction: use the echo-excluded values from echo_check.txt, as round3_label_checks already does. At minimum, drop "gives the correct answer" for the 24.

### Minor mismatches
- **F12 (Sec 3.2):** "Cross-model agreement on that readout" points back to permissive-prompt abstention. The analysis actually uses plain-prompt forced-answer hedge log-odds and entropy (cross_model_agreement.csv). The count of one of three pairs is right.
- **M13 (Sec 3.3):** "Knowledge raises it about as much" is wrong for Gemma-2-2B: knowledge +0.37 vs popularity +0.10, diff −0.27 [−0.38,−0.15], from e3_window_effects.csv. Use the caption's wording "about as much or more".
- **T2-14 (Table 2):** the "plus entity recognition" row prints n = 387. e6_summary says n = 384. The namesake row is also n = 384.
- **R7 (Sec 3.1):** "five other measures", but Table 2 has six non-primary measure rows. It is five only if the two regex rows count as one measure.
- **A1b (Abstract):** "43% of questions" is a mean share of replies. The "9 of 10 samples" definition also leaves out the correct-greedy requirement.
- **S3 (Sec 2):** "unknown if no sample is [correct]". The data also require a wrong greedy answer: one item with 0/10 correct samples and a correct greedy answer is in the dropped band.
- **S5 (Sec 2):** "cell counts in App A" covers Llama only. Qwen, Gemma and Mistral counts are in factorial_4models/cells.csv but are not printed.
- **S11 (Sec 2):** "Intervals are 95% percentile bootstraps over questions within cells" is not true of every interval. The matched-pair CI [+0.18,+0.69] is a regression-intercept interval, and the LPM CIs bootstrap over questions rather than within cells.
- **AC18 (Fig. steering caption):** "three-judge rates in App C". Only the three-judge differences for E4 are printed. The E4 majority rates exist in e4_summary but are not shown.
- **L1 (Limitations):** "cells of at most 100 questions" holds only for the abstention sample. The entropy, regex and lens cells reach 1,179.
- **AD9 (onset caption):** intervals "often collapse to a point", but only 2 of 8 do.

### No evidence found
- S2b: T=0.7 for the forced samples.
- AA3: "10,000 within-cell bootstraps (seed 0); LPM 2,000 bootstraps". Only e4 and e8 state 10,000. No file states a seed or the LPM bootstrap count.
- M7: "abstaining questions are mostly obscure". No arm breakdown of the 650 abstain-set questions exists, though it is plausible from cells.csv.
- AD4: the hedge and answer token lists.

### Other checks requested
- **Altered values:** none found.
- **Best run vs mean:** none found. Random-direction rates are means of five directions, recomputed to match: 0.0082, 0.0026, 0.0174, 0.006.
- **Arithmetic:** all differences checked (pop − know, OK − PU, D1/D2/D3, D = K − F, cross-model Δ, shrink percentages 5.07% and 3.2%). All are correct.
- **Rounding of half values:** printing is inconsistent but every case is within rounding: 0.355 → 0.35; 0.125 → 0.12; 0.025 → 0.03 (twice); 0.005 → 0.01; 0.915 → 0.92; 2.685 → 2.69.
- **Intervals whose method or n cannot be recovered:** the corrected-judge CI [+0.03,+0.24] (hedge_judge/summary gives no method; corrections come from 1,515 judged generations applied to n = 2,303); the matched-pair surface-form CI [+0.18,+0.69].
- **Stale values:** echo_check.txt is the only evidence file newer than numbers.tex (see critical 2). main.pdf is newer than main.tex and numbers.tex.
- **Orphans:**
  - 178 macros in numbers.tex are defined but unused. Several cite sources outside the evidence set (direction_reuse_v2, direction_nohedge, circuit_familiarity_v3, singlebos), and `\ciRecovery` cites numbers.tex itself. None is printed.
  - The evidence includes an unreported fifth family, OLMo-2-7B-Instruct (hedge_measure_v2/*olmo2_7b*, an/explore_olmo.txt: two-judge κ 0.24, Fleiss 0.45, unmeasured, diff −0.02). The paper's "four families" framing does not mention it. This is consistent with "whether other families share the preference is open", but it is an omission.
