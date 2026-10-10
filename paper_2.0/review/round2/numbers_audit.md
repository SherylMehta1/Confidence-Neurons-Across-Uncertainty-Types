# Numbers audit, round 2: main.tex against evidence files

Auditor: independent paper-to-evidence check (no prior knowledge of the project; human-written notes not opened).
Date: 2026-10-10.
Paper: `Sher/paper/tex/main.tex`, macros in `Sher/paper/numbers.tex`.
R = `Sher/Confidence-Neurons-Across-Uncertainty-Types`.

## Method

1. Every number, interval, count, comparison and scope word in the abstract, text, captions and tables was extracted, and each macro was resolved to its value in `numbers.tex`.
2. Each value was traced to the file and field named in the macro comment, then checked against that file. Where possible it was also recomputed independently:
   - Two-judge cell means: recomputed from `item_rates.csv` for all four models.
   - Guessability: re-ran `guessability.py` from `all_measured.jsonl` and `item_rates.csv` into scratch. The output matched `guessability.csv` to every printed digit.
   - Window means of F and K: recomputed from `jlens_cascade/curves.csv`.
   - Steering rates: recomputed by dose from `steer.csv`. The k* choice for Gemma was verified.
   - Pair gaps: recomputed from `check1_pairs.csv`.
   - SHA-256 prefixes: computed for every `R/*_criterion.md`.
   - Macro usage: checked programmatically. All 156 macros used in main.tex are defined. 96 defined macros are unused.
3. Design statements were checked against the criterion files and `build_familiarity_twins.py`. The cross-model rule and verdicts were checked against `paper/PREREG/cross_model_agreement.md` and `cross_model_agreement.py`.

Status key:
- **exact**: identical to the stored value, or an integer count.
- **rounding-ok**: correct rounding of a longer stored value.
- **mismatch**: text and evidence disagree, or a selection, scope or configuration problem.
- **no-evidence-found**: the claim could not be traced in the evidence.
- **n/a**: a citation claim, outside this audit.

## Summary counts

| Status | Printed values |
|---|---|
| exact | 144 |
| rounding-ok | 117 |
| mismatch (table rows) | 19 rows, from 16 distinct issues: 2 critical (C1, C2) and 14 minor (m1–m14) |
| no-evidence-found | 0 outright; G5 (\ciRecovery 0.83 [0.78, 0.87]) traces only to the older macro file, whose upstream CSV is outside the evidence set |
| n/a (citations) | 2 |
| **Total** | **282 printed values in 211 table rows** |

The counts are per printed value: "(×n)" in a row counts n, and D5 counts as 8. A mismatch row counts once. Four rows (A9, I5, F3c, L8) show issue C1. Some m-issues concern wording or design descriptions, not printed numbers. Their rows are A-, S-, St-, Li-, Co-, B-, C-, E- and H-rows marked "mismatch".

No altered values were found. No stale macro values were found: every macro used reproduces from its named source at the printed precision. No count errors were found. All 7 SHA-256 prefixes in Appendix G match the current criterion files. The cross-model verdicts (not met / not met / met) are correct under the plan's rule. The paper's own statement of that rule is wrong (see C2).

---

## Critical mismatches (each would change a stated conclusion)

### C1. Base-model "knowledge effect" is reported for 2 of 4 base models; OLMo's is opposite in sign

**Where the claim appears:**
- Abstract: "base models of three families already carry a popularity effect, alongside a knowledge effect, in their decoded hedge readout".
- Intro (iii): "responds to popularity, as well as to knowledge".
- Sec 3.3: "The readout also responds to knowledge (Fig. 3b): in the window the unknown-minus-known effect is +0.36 in Llama-3.1-8B and +0.37 in Gemma-2-9B … Pretraining alone yields a readout sensitive to popularity, not blind to knowledge."
- Fig. 3 caption: "responds to popularity (a) and to knowledge (b)".

**Evidence:** `results/jlens_cascade/curves.csv` gives the mean of K over rel_depth in [1/3, 2/3]. The same values are in `fig_lens_base.provenance.json` "b/…" mid_mean and in the macros `\jlGemmaTwoBaseMidK` and `\jlOlmoBaseMidK`, which are defined but unused.

| Base model | Family | Window K | Window F | Reported in text? |
|---|---|---|---|---|
| Llama-3.1-8B | Llama | +0.362 | +0.671 | yes |
| Gemma-2-9B | Gemma | +0.375 | +0.387 | yes |
| Gemma-2-2B | Gemma | **+0.110** | +0.401 | no |
| OLMo-3-7B | OLMo | **−0.303** | +0.293 | no |

**Further details:**
- In OLMo, K's per-layer 95% CI lies entirely below zero at 5 of the 10 window layers. It is above zero at only 3 of the 10.
- OLMo is one of the "three families" in the abstract. Its decoded hedge readout has a negative knowledge effect in the window: unknown facts decode as less hedge-like.
- No interval exists anywhere for the window mean of K. The criterion (`jlens_cascade_criterion.md`, hash c8e04372) pre-specifies only the window mean of F for base models. The K window comparison is therefore post hoc, but it is not labelled so (see m6).

**Suggested correction:** report all four values (`\jlLlamaBaseMidK` +0.36, `\jlGemmaNineBaseMidK` +0.37, `\jlGemmaTwoBaseMidK` +0.11, `\jlOlmoBaseMidK` −0.30). Then:
- Change the abstract to e.g. "…already carry a popularity effect in their decoded hedge readout; in Llama and Gemma-2 it is accompanied by a positive knowledge effect, while in OLMo the knowledge effect in the same window is negative".
- Qualify the Fig. 3 caption the same way.
- Label the K-window comparison post hoc.
- If an interval is wanted, it must be computed, because none exists in the evidence.

### C2. The cross-model rule as written in Appendix F does not reproduce the reported verdicts

**Where the claim appears:**
- App. F: "The plan, including the rule (hedge agreement and the hedge-minus-entropy gap both above zero)…"
- Conclusion: "our pre-set rule held for one".

**Plan and code:**
- `PREREG/cross_model_agreement.md`: "The prediction holds for a pair if, on knowledge-differs questions, ρ(hedge_logodds) > 0 **with CI excluding 0** and Δ > 0 **with CI excluding 0**."
- `cross_model_agreement.py` implements it as `holds = h["lo"] > 0 and lo > 0`.

**Evidence (`cross_model_agreement.txt`):**

| Pair | ρ hedge | Δ | Plan verdict | Verdict under the rule as written in the paper (point > 0) |
|---|---|---|---|---|
| Llama–Qwen | +0.11 [−0.00, +0.22] | +0.28 [+0.14, +0.42] | not met | met |
| Llama–Gemma | +0.38 [+0.13, +0.60] | +0.35 [−0.03, +0.74] | not met | met |
| Llama–Mistral | +0.25 [+0.06, +0.42] | +0.32 [+0.06, +0.57] | met | met |

The table's verdict row (`\cmQwenRule` etc.) and the conclusion's "held for one" match the plan. A reader who applies the rule as the paper states it gets 3 of 3, not 1 of 3.

**Suggested correction:** "(hedge agreement and the hedge-minus-entropy gap both above zero, with 95% intervals excluding zero)".

---

## Minor mismatches

- **m1. Configuration of the never-abstained control is under-described** (App. C, Sec 3.2, Fig. 2 caption).
  - The paper says it "refits the obscure-minus-popular direction on working-half questions with no abstaining sample". That implies the same knowledge-averaged construction as the main direction.
  - `nohedge_direction_criterion.md` defines the primary direction (d_nohedge_U) as mean(obscure, **Unknown**, hedge 0) − mean(popular, **Unknown**, hedge 0). Only unknown facts are used. The knowledge-balanced version (d_nohedge_both) is secondary and gives 0.087; it is not reported.
  - "No abstaining sample" means hedge rate 0 under the regex over the 20 stored 24-token generations.
  - Fix: add "using unknown facts only".
- **m2. Fig. 3 caption: "OLMo's popularity effect reverses in late layers, outside the tested window."**
  - The late reversal is correct: F's CI is below 0 at L24–L30.
  - But OLMo's F CI also lies below zero inside the window, at L14 (rel 0.45, −0.06 [−0.11, −0.02]) and L17 (rel 0.55, −0.36 [−0.55, −0.18]).
  - Gemma-2-2B also reverses late (L23–24, rel 0.92–0.96), which the caption does not mention.
  - The window means are unaffected.
  - Fix: "…OLMo's popularity effect dips below zero at two window layers and reverses in late layers, as does Gemma-2-2B's at the last layers."
- **m3. Interval method is not uniform, although Sec 2 says "with 95% item-bootstrap intervals".**
  - (a) Table 1 "One LLM judge, corrected" +0.13 [+0.03, +0.24]: `hedge_judge_criterion.md` states the CI comes from 5,000 bootstrap resamples of judged **generations** within strata, not items.
  - (b) App. A "+0.43 [+0.18, +0.69] at equal surface form" is the intercept of a pair-level regression (`unireps_checks/summary.txt`). The interval method is not stated in the evidence, so it cannot be recovered.
  - (c) Cross-model ρ intervals reuse ranks computed on the full subset (stated in the script docstring as an approximation). This is not mentioned in the paper.
  - Fix: name the method for (a) and (b).
- **m4. App. B caption scope.** "Top: judge agreement and abstention in the obscure, unknown cell". The κ row and the regex-recall row are model-level quantities, not obscure-unknown-cell quantities.
- **m5. App. B text: "Intervals for the bottom rows".** Only the intervals for the popularity − knowledge row are given. They match: `\loLlamaDiffCI`, etc. The pop and know intervals exist as unused macros (`\lo*PopCI`, `\lo*KnowCI`). Fix: "bottom row", or add them.
- **m6. Labelling of confirmatory and post hoc analyses is incomplete.** Sec 2 says "Confirmatory analyses followed criterion notes …; the rest are labelled post hoc." The following are neither listed in App. G nor labelled post hoc:
  - (a) the single-judge correction (Table 1 row 2), which has its own criterion note, `hedge_judge_criterion.md` (SHA-256 prefix ae0bb807), not listed in App. G;
  - (b) the window means of K in base models (Sec 3.3; see C1);
  - (c) the lens timing in App. E (layers 16/28/12, `singlebos/circuit_familiarity/jlens_summary.txt`);
  - (d) the neuron finding and the circuit recovery 0.83 (`l5_behavioral_criterion.md` covers only the flip-rate bar);
  - (e) Steering v2 inherits ℓ*, the item set, the detector, the intervention positions and the 0.10 bar from `direction_reuse_criterion.md` (prefix 3281f804). App. G lists only the v2 hash (da0ad000).

  Section naming is consistent elsewhere: App. F is headed "(exploratory)" and its text and App. G say "post hoc", which matches the plan's "exploratory … labelled post hoc in the paper".
- **m7. App. A: "on the regex measure over the full pool (n = 2,106)".**
  - 2,106 is the pool after removing guessable questions (`guessability.csv` regex_noguess).
  - The full pool is 2,303 (regex_all, diff +0.44).
  - Fix: "over the factorial pool without guessable questions".
- **m8. Guessability definition omits a condition.** `guessability.py` requires a gold alias of **≥ 3 characters** to appear in the name. The paper says only "a gold alias appears in the subject's name".
- **m9. Factorial verdicts are not reported.**
  - `gemma_mistral_factorial_criterion.md` says "every model's verdict reported". `qwen_factorial_criterion.md` says PARTIAL means "the paper says … partial in Qwen".
  - `factorial_4models/summary.txt` gives Qwen PARTIAL, Gemma REPLICATES (on hedge_logodds) and Mistral PARTIAL.
  - The paper gives the numbers (App. B) but never the verdicts. `hedge_measure_v2_criterion.md` (κ < 0.60) also asks that "the regex verdicts are reported with their known recall"; the recall is reported, the verdicts are not.
  - The numbers in App. B are consistent with the verdicts ("Only Gemma's forced-answer readout follows popularity more than knowledge").
- **m10. Cross-model plan deviations.**
  - (a) The plan names `hedge_rate` (free-generation regex) as a secondary readout to be "Reported". The paper omits it. On knowledge-differs questions it agrees for all three pairs with CIs excluding 0: Qwen +0.50 [+0.35, +0.62], Gemma +0.37 [+0.19, +0.54], Mistral +0.26 [+0.01, +0.48] (unused macros `\cm*DiffRegex`).
  - (b) The plan's "Caveat to state" is not stated in App. F. The caveat reads: Llama's forced-answer hedge readout responds to knowledge as much as to popularity, so it is a weaker abstention proxy.
  - (c) The conclusion calls the forced-answer hedge log-odds "abstention readouts". The Setup reserves "abstention" for the free prompt.
- **m11. "In every family the knowledge effect is at least four times the popularity effect" (Sec 3.1).**
  - True for the point estimates: Llama 2.139/0.513 = 4.17, Gemma 11.6, Mistral 4.79, and Qwen's popularity effect is negative.
  - Llama's margin is thin. It is not established by the intervals: 4 × 0.606 (pop upper) > 2.045 (know lower).
  - Suggest "about four times or more", or state that it holds for the point estimates.
- **m12. Limitations: "other norms … were not tested".**
  - For Gemma, the k* calibration ran the obscure-minus-popular direction and the random directions at 1× and 2× (abstention 0.000 and 0.007; `direction_reuse_v2/gemma9b/steer.csv`).
  - Only the comparison directions were run at k* alone.
  - Fix: "the comparison directions were tested only at the chosen norm".
- **m13. Onset rule paraphrases (App. D).**
  - The criterion's onset also requires "a positive maximum".
  - D's bootstrap recomputes onsets "with rule (a) only".
  - Both are omitted. No value changes.
- **m14. Layer rule (App. C).** "The layer at 0.55 of depth" is a paraphrase of ℓ* = round(0.55 × (L − 1)) (`direction_reuse_criterion.md`). It gives 17 of 32 and 23 of 42, as printed. Rounding-ok, but the exact rule is not recoverable from the text.

---

## Full item table

Abbreviations:
- HM = `results/hedge_measure_v2/summary.txt`
- FA = `results/factorial_4models/` (eff = effects.csv, cells = cells.csv, sum = summary.txt)
- HJ = `results/hedge_judge/summary.txt`
- UC = `results/unireps_checks/summary.txt`
- DR-L / DR-G = `results/direction_reuse_v2/{llama,gemma9b}/summary.txt`
- NH = `results/direction_nohedge/llama/summary.txt`
- JO / JC = `results/jlens_cascade/{outcomes,curves}.csv`
- GS = `paper/analysis/guessability.csv`
- CM = `paper/analysis/cross_model_agreement.{csv,txt}`

### Abstract

| id | claim | evidence (file: field) | value in file | status |
|---|---|---|---|---|
| A1 | popularity effect +0.39 | HM llama primary pop | +0.388 | rounding-ok |
| A2 | knowledge +0.09 | HM llama primary know | +0.089 | rounding-ok |
| A3 | "far more" | HM pop−know | +0.300 [+0.204, +0.398] | exact (CI excludes 0) |
| A4 | "also after removing answers that can be guessed" | GS judge_noguess pop / know / diff | +0.373 / +0.074 / +0.300 [+0.192, +0.411] | exact |
| A5 | entropy follows knowledge "as it does in three other families" | FA eff entropy, qwen/gemma/mistral | know 1.230/1.079/1.413 vs pop −0.019/0.093/0.295 | exact |
| A6 | steering makes "Llama and Gemma-2" say IDK about known facts | DR-L, DR-G own_fam | 0.287, 0.220 on popular and HighlyKnown items | exact |
| A7 | never-abstained version "adds only +0.04" | NH d_nohedge_U − random | +0.041 | rounding-ok |
| A8 | base models of "three families" carry a popularity effect | JO midF for llama3.1-8b, gemma-2-9b, gemma-2-2b, olmo | all > 0, CIs exclude 0 (Llama, Gemma, OLMo = 3 families) | exact |
| A9 | "alongside a knowledge effect" | JC window K, 4 base models | +0.36, +0.37, +0.11, **−0.30** | **mismatch (C1)** |
| A10 | "Where abstention could be measured" (Llama only) | HM κ | 0.62 vs 0.30/0.47/0.39 | exact |

### Introduction

| id | claim | evidence | value | status |
|---|---|---|---|---|
| I1 | popularity +0.39, knowledge +0.09 | HM | +0.388, +0.089 | rounding-ok (×2) |
| I2 | "answer uncertainty follows knowledge in all four families tested" | FA eff entropy | know > pop in all 4; diff CIs all < 0 | exact |
| I3 | steering: obscure-minus-popular does "far more than an unknown-minus-known direction" | DR-L DD hedge +0.200 [+0.140, +0.267]; DR-G +0.220 [+0.160, +0.287] | — | exact |
| I4 | "mostly through components shared with abstention" | NH: d_nohedge_U effect +0.047 vs d_fam +0.267 | 18% | exact |
| I5 | (iii) base readout responds to popularity "as well as to knowledge" | JC window K | see C1 | **mismatch (C1)** |

### Setup (Sec 2)

| id | claim | evidence | value | status |
|---|---|---|---|---|
| S1 | 16 relations of PopQA | data/familiarity_v3/gate_report.json n_relations; UC "relations 16" | 16 | exact |
| S2 | bottom and top deciles | build_familiarity_twins.py --lo/--hi-quantile | 0.10 / 0.90 | exact |
| S3 | greedy + ten samples, T = 0.7 | builder --n-samples 10, temperature 0.7; gate_report n_samples 10 | 10, 0.7 | exact |
| S4 | known = greedy correct and ≥ 90% of samples; unknown = none correct | builder slick_class | ≥ 0.9; frac 0 and greedy wrong | exact |
| S5 | 2,303 questions (Llama) | FA cells, sum of llama n | 1179+87+361+676 = 2303 | exact |
| S6 | 2,392 (Qwen) | FA cells | 1209+54+741+388 = 2392 | exact |
| S7 | 2,282 (Gemma) | FA cells | 1129+122+410+621 = 2282 | exact |
| S8 | 2,378 (Mistral) | FA cells | 1178+71+417+712 = 2378 | exact |
| S9 | forced-answer log-odds pop +2.93 | FA eff llama hedge_logodds pop | 2.9329 | rounding-ok |
| S10 | know +3.05 | same, know | 3.0547 | rounding-ok |
| S11 | "about equally" | same, diff | −0.122 [−0.722, +0.449] | exact (CI spans 0) |
| S12 | up to 100 per cell | hedge_measure_v2_criterion.md | 100 per cell, all if fewer | exact |
| S13 | 387 in total | HM "387 items"; item_rates llama rows | 387 (100+87+100+100) | exact |
| S14 | three replies, T = 0.7, up to 96 tokens | criterion | 3, 0.7, 96 | exact |
| S15 | judges Qwen2.5-32B-Instruct, Gemma-2-27B-it | criterion | same | exact |
| S16 | three labels; abstains when both say DONT_KNOW | criterion | same | exact |
| S17 | κ = 0.62 | HM llama kappa | 0.62 | exact |
| S18 | "just above" a pre-set bar of 0.60 | criterion κ ≥ 0.60 | 0.60 | exact |
| S19 | κ Qwen 0.30 | HM | 0.30 | exact |
| S20 | κ Gemma 0.47 | HM | 0.47 | exact |
| S21 | κ Mistral 0.39 | HM | 0.39 | exact |
| S22 | definitions of the pop and know effects | criterion / guessability.py stat() | match | exact |
| S23 | "95% item-bootstrap intervals" (all effects) | various | not uniform | **mismatch (m3)** |
| S24 | "the rest are labelled post hoc" | App. G vs analyses present | several unlabelled | **mismatch (m6)** |

### Sec 3.1 and Fig. 1

| id | claim | evidence | value | status |
|---|---|---|---|---|
| F1a | caption (a) n = 387, 3 replies each | fig_dissociation.provenance a/* n | 87+100+100+100 = 387 | exact |
| F1b | caption (b) n = 2,303 | provenance b/* n | 1179+87+676+361 | exact |
| F1c | caption (c) four families | provenance c/* | 4 models; values = FA eff entropy | exact |
| F1d | "Means with 95% item-bootstrap intervals" | fig_dissociation.py header | within-cell percentile bootstrap, 2,000 | exact |
| R1 | obscure-unknown 0.48 | HM primary unc/Unkn; recomputed from item_rates | 0.477 | rounding-ok |
| R2 | obscure-known 0.43 | same | 0.433 | rounding-ok |
| R3 | popular-unknown 0.13 | same | 0.133 | rounding-ok |
| R4 | popular-known 0.00 | same | 0.000 | exact |
| R5 | pop +0.39 | HM | 0.388 | rounding-ok |
| R6 | [+0.33, +0.45] | HM | [+0.328, +0.449] | rounding-ok |
| R7 | know +0.09 | HM | 0.089 | rounding-ok |
| R8 | [+0.03, +0.15] | HM | [+0.031, +0.147] | rounding-ok |
| R9 | pop − know +0.30 | HM; recomputed (0.3885 − 0.0885) | 0.300 | rounding-ok |
| R10 | [+0.20, +0.40] | HM | [+0.204, +0.398] | rounding-ok |
| R11 | knowledge "about a quarter" of popularity | 0.089 / 0.388 | 0.23 | rounding-ok |
| R12 | "keeps its sign under four other measurements" | Table 1 rows 1, 2, 4, 5 | all > 0, CIs exclude 0 | exact |
| R13 | entropy know +2.14 | FA eff llama entropy know | 2.1390 | rounding-ok |
| R14 | entropy pop +0.51 | same, pop | 0.5128 | rounding-ok |
| R15 | "in every family … at least four times" | FA eff entropy ratios | 4.17 / (pop < 0) / 11.6 / 4.79 | rounding-ok (point estimates only; m11) |
| R16 | 30 of 87 obscure known guessable | GS obs_known_guessable / total; re-run | 30 / 87 | exact (×2) |
| R17 | without guessable: +0.30 | GS judge_noguess diff | 0.2997 | rounding-ok |
| R18 | [+0.19, +0.41] | GS | [0.1919, 0.4108] | rounding-ok |
| R19 | n = 335 | GS judge_noguess n_total | 335 | exact |
| R20 | obscure known still 0.44 | GS judge_noguess mean_obs_known | 0.4386 (n = 57) | rounding-ok |
| R21 | pool popularity coefficient −1.87 | UC hedge GLM M1 logpop_z | −1.871 | rounding-ok |
| R22 | → −1.79 with surface controls | UC M2 logpop_z | −1.791 | rounding-ok |
| R23 | the controls are "word frequency, token count and rarity" | UC M2 terms zipf_mean, zipf_min, n_tok, tok_logid | match | exact |
| R24 | "Two post hoc checks" | App. G | labelled post hoc | exact |

### Sec 3.2 and Fig. 2

| id | claim | evidence | value | status |
|---|---|---|---|---|
| F2a | 150 held-out popular, known questions | DR-L / DR-G n; steer.csv 150 per condition | 150 | exact |
| F2b | regex detector | direction_reuse_criterion.md HEDGE_RE; fig_steering.py uses `hedged` | regex | exact |
| F2c | dots = individual random directions; hatched = never-abstained, Llama only | fig_steering.py lines 46–55 | match | exact |
| F2d | error bars: 95% bootstrap over questions | fig_steering.py | 2,000 resamples over 150 items | exact |
| St1 | fitted on half of each model's questions, last position of free prompt, differences of cell means | v1 / v2 criteria | random half (seed 0); v2 position fix | exact |
| St2 | five random directions | DR-L random0–4 | 5 | exact |
| St3 | Llama 1× natural norm | DR-L k* | 1 | exact |
| St4 | Gemma-2-9B-it 4× | DR-G k*; recomputed own − random at k = 1/2/4: 0.000/0.007/0.220 | 4 | exact |
| St5 | Llama 0.02 → 0.29 | DR-L none / own_fam | 0.020 / 0.287 | rounding-ok (×2) |
| St6 | unknown-minus-known 0.09 | DR-L inst_know | 0.087 | rounding-ok |
| St7 | random 0.03 | DR-L random mean | 0.0253 | rounding-ok |
| St8 | Gemma 0.00 → 0.22 | DR-G none / own_fam | 0.000 / 0.220 | exact (×2) |
| St9 | "no control moved it" | DR-G inst_know, base_fam, random0–4 | all 0.000 | exact |
| St10 | Gemma: unknown-minus-known raised entropy +0.68 more | DR-G DD entropy | +0.681 | rounding-ok |
| St11 | [+0.59, +0.77] | DR-G | [+0.593, +0.770] | rounding-ok |
| St12 | Llama "both raised it similarly", −0.03 | DR-L DD entropy; E none 0.824, know 1.225, fam 1.256 | −0.031 | rounding-ok |
| St13 | [−0.11, +0.05] | DR-L | [−0.110, +0.047] | rounding-ok |
| St14 | never-abstained version +0.04 over random | NH | +0.041 | rounding-ok |
| St15 | [+0.01, +0.07] | NH | [+0.013, +0.073] | rounding-ok |
| St16 | "most of the effect runs through components shared with abstaining" | NH verdict HEDGING INTENT NOT EXCLUDED | consistent | exact |
| St17 | description of the never-abstained fit | nohedge criterion: d_nohedge_U is Unknown-only | — | **mismatch (m1)** |

### Sec 3.3 and Fig. 3

| id | claim | evidence | value | status |
|---|---|---|---|---|
| F3a | four base models | fig_lens_base provenance | 4 | exact |
| F3b | bands 95% item-bootstrap; shaded window fixed before runs | jlens criterion [1/3, 2/3] | match | exact |
| F3c | "(b) responds … to knowledge" | provenance b/* mid_mean | +0.36, +0.37, +0.11, −0.30 | **mismatch (C1)** |
| F3d | "OLMo's popularity effect reverses in late layers, outside the tested window" | JC olmo F by layer | also negative at L14 and L17 inside the window; Gemma-2-2B reverses late too | **mismatch (m2)** |
| L1 | prompt "Question: … Answer: The answer is" | jlens criterion base prompt | same | exact |
| L2 | window fixed before the run | criterion [1/3, 2/3] | same | exact |
| L3 | obscure-minus-popular positive in all four, all CIs exclude 0 | JO midF base rows; summary "PREDATES … yes" | +0.67 [+0.59, +0.75], +0.39 [+0.34, +0.43], +0.40 [+0.33, +0.47], +0.29 [+0.19, +0.38] | exact |
| L4 | knowledge +0.36 Llama-3.1-8B | JC window K mean | 0.3620 | rounding-ok |
| L5 | knowledge +0.37 Gemma-2-9B | JC | 0.3746 | rounding-ok |
| L6 | popularity +0.67 | JO midF | 0.6709 | rounding-ok |
| L7 | popularity +0.39 | JO midF | 0.3865 | rounding-ok |
| L8 | selection of L4–L5 (2 of 4 base models) | unused macros jlGemmaTwoBaseMidK, jlOlmoBaseMidK | +0.11, −0.30 | **mismatch (C1)** |
| L9 | eight models | JO rows; summary "8 of 8 planned" | 8 | exact |
| L10 | 1 supporting | JO / summary | 1 | exact |
| L11 | 1 contradicting | same | 1 | exact |
| L12 | 6 undecided | same | 6 | exact |

### Related work, limitations, conclusion

| id | claim | evidence | value | status |
|---|---|---|---|---|
| RW1 | "a commit-or-abstain circuit across ten models" | citation | — | n/a |
| Li1 | abstention measured reliably in one model | HM κ | 1 of 4 ≥ 0.60 | exact |
| Li2 | steering norm chosen on the evaluation questions | v2 criterion: k* on the same 150 items | — | exact |
| Li3 | "other norms … were not tested" | DR-G steer.csv own_fam and random at k = 1, 2 | partly tested | **mismatch (m12)** |
| Li4 | all models 2–9B parameters | model list (8B, 7B, 9B, 2B, 7B, 7B) | — | exact |
| Co1 | "reliably for two of three pairs" | CM knowledge-differs hedge_logodds CIs | Qwen [−0.001, +0.215]; Gemma [+0.13, +0.60]; Mistral [+0.06, +0.42] → 2 of 3 | exact |
| Co2 | "pre-set rule held for one" | CM txt verdicts | 1 (Mistral) | exact |
| Co3 | "abstention readouts" (forced-answer log-odds) | plan's terminology; Setup reserves "abstention" for the free prompt | — | **mismatch (m10c)** |

### Appendix A: Table 1 and text

| id | claim | evidence | value | status |
|---|---|---|---|---|
| T0 | five measurements | table rows | 5 | exact |
| T1a | regex: 20 per question, 24 tokens | UC "free generations per item: 20"; builder n_free 20, 24 tokens | 20, 24 | exact |
| T1b | 2,303 questions | FA cells | 2303 | exact |
| T1c | +0.44 | FA eff llama hedge_rate diff | 0.4423 | rounding-ok |
| T1d | [+0.34, +0.53] | same | [0.3445, 0.5337] | rounding-ok |
| T2a | one judge (Qwen2.5-32B-Instruct), corrected for recall and precision on a labelled subset | HJ header; hedge_judge criterion | match | exact |
| T2b | +0.13 | HJ llama corrected pop − know | +0.132 | rounding-ok |
| T2c | [+0.03, +0.24] | HJ | [+0.027, +0.239] | rounding-ok (method: generation bootstrap, m3) |
| T3a | 3 per question, 96 tokens, 387 questions | HM | match | exact |
| T3b | +0.30 [+0.20, +0.40] | HM primary | +0.300 [+0.204, +0.398] | rounding-ok (×2) |
| T4 | +0.31 [+0.21, +0.41] | HM secondary | +0.311 [+0.209, +0.414] | rounding-ok (×2) |
| T5 | +0.43 [+0.31, +0.55] | HM regex on new replies | +0.429 [+0.306, +0.547] | rounding-ok (×2) |
| AA1 | single judge gives the smallest difference | T1–T5 | 0.13 is the minimum | exact |
| AA2 | two-judge measurement made primary before it ran | hedge_measure_v2_criterion.md | stated | exact |
| AA3 | regex recall on 96-token replies 0.99 | HM llama | 0.99 | exact |
| AA4 | guessable definition (alias in name, or modal answer ≥ 20%) | guessability.py | also requires alias ≥ 3 chars | **mismatch (m8)** |
| AA5 | popularity gap among unknown facts +0.34 | GS judge_all popU | 0.3433 | rounding-ok |
| AA6 | [+0.26, +0.43] | GS | [0.2567, 0.4300] | rounding-ok |
| AA7 | without guessable: pop +0.37 | GS judge_noguess pop | 0.3733 | rounding-ok |
| AA8 | know +0.07 | GS | 0.0736 | rounding-ok |
| AA9 | regex "over the full pool" n = 2,106 | GS regex_noguess n_total (full = 2303) | 2106 | exact value; **wording mismatch (m7)** |
| AA10 | regex diff +0.48 | GS regex_noguess diff | 0.4799 | rounding-ok |
| AA11 | [+0.36, +0.59] | GS | [0.3612, 0.5916] | rounding-ok |
| AA12 | 55 matched pairs of known facts | UC pop_given_known; check1_pairs.csv | 55 | exact |
| AA13 | gap +0.76 | check1_pairs mean d_hedge (recomputed) | 0.7582 | rounding-ok |
| AA14 | falls to +0.43 | UC equal surface form | +0.434 | rounding-ok |
| AA15 | [+0.18, +0.69] | UC | [+0.182, +0.686] | rounding-ok (method unrecoverable, m3) |
| AA16 | excluding name-inferable: +0.79 | UC; recomputed (inferable_u == 0) | 0.7875 | rounding-ok |
| AA17 | over 36 pairs | UC n = 36; recomputed | 36 | exact |
| AA18 | regression covers 2,303 questions, relation fixed effects | UC "items: 2303"; "relation fixed effects" | match | exact |
| AA19 | "(post hoc)" labels | App. G | consistent | exact |

### Appendix B: Table 2 and text

| id | claim | evidence | value | status |
|---|---|---|---|---|
| B1 | bar 0.60; κ 0.62 / 0.30 / 0.47 / 0.39 | HM; criterion | same | exact (×5) |
| B2 | both DONT_KNOW, obscure-unknown 0.48 / 0.05 / 0.05 / 0.10 | HM primary unc/Unkn; recomputed | 0.477 / 0.050 / 0.047 / 0.097 | rounding-ok (×4) |
| B3 | DONT_KNOW or CLARIFY 0.65 / 0.32 / 0.22 / 0.25 | HM secondary | 0.647 / 0.317 / 0.220 / 0.253 | rounding-ok (×4) |
| B4 | regex 0.85 / 0.18 / 0.07 / 0.21 | HM regex | 0.847 / 0.180 / 0.073 / 0.210 | rounding-ok (×4) |
| B5 | regex recall 0.99 / 0.58 / 0.48 / 0.73 | HM | same | exact (×4) |
| B6 | log-odds pop +2.93 / +1.14 / +1.79 / +2.48 | FA eff hedge_logodds pop | 2.933 / 1.138 / 1.789 / 2.480 | rounding-ok (×4) |
| B7 | log-odds know +3.05 / +3.63 / −0.32 / +2.08 | FA eff | 3.055 / 3.632 / −0.318 / 2.077 | rounding-ok (×4) |
| B8 | pop − know −0.12 / −2.49 / +2.11 / +0.40 | FA eff | −0.122 / −2.494 / 2.107 / 0.403 | rounding-ok (×4) |
| B9 | caption: top block "in the obscure, unknown cell" | κ and recall are model-level | — | **mismatch (m4)** |
| B10 | "at most 0.10" (Qwen, Gemma, Mistral) | HM primary | max 0.097 | exact |
| B11 | intervals −0.72 to +0.45 / −3.41 to −1.64 / +1.79 to +2.42 / −0.55 to +1.34 | FA eff diff_lo/hi | [−0.722, 0.449] / [−3.413, −1.644] / [1.787, 2.424] / [−0.545, 1.344] | rounding-ok (×4) |
| B12 | "Intervals for the bottom rows" | only the diff row given | — | **mismatch (m5)** |
| B13 | "Only Gemma's forced-answer readout follows popularity more than knowledge" | FA diff CIs: only Gemma excludes 0 on the positive side | consistent | exact |
| B14 | Gemma knowledge effect "slightly negative" | FA −0.318 [−0.498, −0.138] | — | exact |
| B15 | factorial verdicts not stated | FA sum: Qwen PARTIAL, Gemma REPLICATES, Mistral PARTIAL | — | **mismatch (m9)** |

### Appendix C: steering details

| id | claim | evidence | value | status |
|---|---|---|---|---|
| C1 | random half | v1 criterion, seed 0 | — | exact |
| C2 | layer 17 of 32 | DR-L | 17 / 32 | exact (×2) |
| C3 | layer 23 of 42 | DR-G | 23 / 42 | exact (×2) |
| C4 | "the layer at 0.55 of depth, fixed before the runs" | v1 criterion round(0.55 × (L − 1)) | 17.05→17; 22.55→23 | rounding-ok (m14) |
| C5 | added at the last prompt position and every generated position | v1 criterion | same | exact |
| C6 | k ∈ {1, 2, 4, 8} | v2 criterion | same | exact |
| C7 | beat the mean of five random directions by 0.10 | v2 criterion "own − random ≥ 0.10" | same | exact |
| C8 | on the 150 evaluation questions | DR n | 150 | exact |
| C9 | all directions scaled to that norm | v1/v2 criteria | same | exact |
| C10 | never-abstained control description | nohedge criterion | Unknown-only | **mismatch (m1)** |
| C11 | run for Llama only | nohedge criterion | same | exact |
| C12 | Gemma dissociation at four times the natural norm | DR-G DD → yes at k* = 4 | — | exact |

### Appendix D: lens onsets and Table 3

| id | claim | evidence | value | status |
|---|---|---|---|---|
| D1 | onset rule | jlens criterion | omits "positive maximum" | rounding-ok (m13) |
| D2 | supports / contradicts definitions | criterion | match | exact |
| D3 | base models labelled on their own prompt, same rule | criterion | match | exact |
| D4 | pairs reported side by side | criterion; table order | match | exact |
| D5 | 8 rows × (layers, F onset, K onset, D, outcome) = 40 values | JO n_layers, onset_F, onset_K, D, outcome | all match | exact (×40) |
| D6 | 8 mid-popularity values (+1.58, +0.67, +0.21, +0.71, +0.39, +0.40, +0.40, +0.29) | JO midF | 1.580, 0.671, 0.213, 0.711, 0.387, 0.403, 0.401, 0.293 | rounding-ok (×8) |

D5 is counted as 8 in the summary counts (one per table row), not 40. All 40 values were checked.

### Appendix F: cross-model table and text

| id | claim | evidence | value | status |
|---|---|---|---|---|
| E1 | Spearman, same vs different knowledge, 95% item bootstrap | plan; script (10,000 bootstraps) | match | exact |
| E2 | the rule: "both above zero" | plan: both with CI excluding 0 | — | **mismatch (C2)** |
| E3 | plan written before the run; point estimates already seen → post hoc | plan status line | match | exact |
| E4 | n knowledge differs 313 / 60 / 108 | CM n | same | exact (×3) |
| E5 | hedge ρ +0.11 [−0.00, +0.22] | CM | 0.108 [−0.0009, 0.215] | rounding-ok (×2) |
| E6 | +0.38 [+0.13, +0.60] | CM | 0.384 [0.133, 0.603] | rounding-ok (×2) |
| E7 | +0.25 [+0.06, +0.42] | CM | 0.247 [0.058, 0.425] | rounding-ok (×2) |
| E8 | entropy ρ −0.17 [−0.28, −0.07] | CM | −0.174 [−0.278, −0.069] | rounding-ok (×2) |
| E9 | +0.03 [−0.24, +0.28] | CM | 0.033 [−0.235, 0.278] | rounding-ok (×2) |
| E10 | −0.07 [−0.27, +0.12] | CM | −0.075 [−0.267, 0.124] | rounding-ok (×2) |
| E11 | gap +0.28 [+0.14, +0.42] | CM txt | same | exact (×2) |
| E12 | gap +0.35 [−0.03, +0.74] | CM txt | same | exact (×2) |
| E13 | gap +0.32 [+0.06, +0.57] | CM txt | same | exact (×2) |
| E14 | rule: not met / not met / met | CM txt NOT SUPPORTED / NOT SUPPORTED / PREDICTION HOLDS; plan rule | match | exact (×3) |
| E15 | n same knowledge 1,784 / 1,977 / 2,000 | CM | same | exact (×3) |
| E16 | same-knowledge hedge ρ +0.50 / +0.38 / +0.64 | CM | 0.503 / 0.376 / 0.640 | rounding-ok (×3) |
| E17 | same-knowledge entropy ρ +0.46 / +0.54 / +0.69 | CM | 0.458 / 0.541 / 0.687 | rounding-ok (×3) |
| E18 | plan's secondary readout (hedge_rate) and caveat | plan: "Reported"; "Caveat to state" | omitted | **mismatch (m10)** |

### Appendix E (app:circuit): timing, neurons and the circuit

| id | claim | evidence | value | status |
|---|---|---|---|---|
| G1 | abstention readout reaches half its final size at layer 16 | singlebos jlens_summary "log-odds L16" | 16 | exact |
| G2 | entropy gap at layer 28 | same | 28 | exact |
| G3 | 12 layers later | 28 − 16 | 12 | exact |
| G4 | 20 heads + 100 MLP neurons | circuit_familiarity_v3/behavioral_summary.txt | 20 + 100 | exact (×2) |
| G5 | recovers 0.83 [0.78, 0.87] | R/paper/numbers.tex \circuitHedgeCUVThreeFull(CI) (from fig6_familiarity_v3_full_cells.csv, n = 141) | 0.83 [0.78, 0.87] | exact against the designated source; underlying CSV not in the evidence set (no-evidence-found beyond the macro) |
| G6 | removing its state removes abstention in 0.98 | behavioral_summary control_to_uncertain circuit flip | 0.981 | rounding-ok |
| G7 | injecting produces abstention in 0.40 | uncertain_to_control flip | 0.398 | rounding-ok |
| G8 | "fell short of the behavioural bar fixed in advance" | l5_behavioral_criterion.md bar flip ≥ 0.50 in both directions | 0.398 < 0.50 | exact |
| G9 | "ten models" (cited work) | citation | — | n/a |

### Appendix G: timing

| id | claim | evidence (sha256 of R/<file>, first 8 hex) | value | status |
|---|---|---|---|---|
| H1 | four-family design eba35dec | qwen_factorial_criterion.md | eba35dec | exact |
| H2 | b53d9a27 | gemma_mistral_factorial_criterion.md | b53d9a27 | exact |
| H3 | two-judge measurement 81acfc9f | hedge_measure_v2_criterion.md | 81acfc9f | exact |
| H4 | steering da0ad000 | direction_reuse_v2_criterion.md | da0ad000 | exact |
| H5 | never-abstained control 797f52d4 | nohedge_direction_criterion.md | 797f52d4 | exact |
| H6 | lens study c8e04372 | jlens_cascade_criterion.md | c8e04372 | exact |
| H7 | circuit behaviour 1c99afd4 | l5_behavioral_criterion.md | 1c99afd4 | exact |
| H8 | post hoc list (guessability, spelling, cross-model) | — | incomplete; see m6 (hedge_judge_criterion.md = ae0bb807 and direction_reuse_criterion.md = 3281f804 not listed) | **mismatch (m6)** |

Note: the prefixes show that the current files match the recorded hashes. Whether each note predates its run cannot be verified from the files.

---

## Checklist of the requested checks

- **Altered values:** none. Every macro used reproduces its source at the printed precision.
- **Selected values:** C1 (2 of 4 base-model knowledge effects reported; the omitted ones include a sign reversal). m10a (the cross-model secondary readout, which the plan said to report, is omitted). m5 (only one row's intervals given).
- **Configuration mismatch:** m1 (never-abstained direction uses unknown facts only). m8 (guessability alias-length condition). m14 and m13 (paraphrased rules). No mismatch was found in models, layers, doses, token budgets, sample counts, temperatures or judges.
- **Count mismatch:** none. 387, 2,303/2,392/2,282/2,378, 335, 2,106, 55, 36, 150 and the 313/60/108 and 1,784/1,977/2,000 cross-model counts all verified. The n = 87 cell and the 30/87 guessable count were recomputed.
- **Arithmetic recomputation:**
  - Llama two-judge effects from cell means: 0.3885 / 0.0885 / 0.300.
  - Entropy ratios: 4.17 / 11.6 / 4.79.
  - Knowledge-to-popularity ratio for abstention: 0.23.
  - Steering random mean: 0.0253.
  - Gemma k* rule at k = 1/2/4: 0.000 / 0.007 / 0.220.
  - Lens window means: F matches outcomes.csv; K as tabled in C1.
  - 28 − 16 = 12.
  - ℓ* = round(0.55 × 31) = 17 and round(0.55 × 41) = 23.
  - Factorial n sums.
  - All correct.
- **Caption vs content:**
  - All three provenance files match the captions on n, panels and interval method.
  - Exceptions: Fig. 3 caption (C1, m2) and the Table 2 caption (m4).
- **Scope words:**
  - "all four", "every family", "all intervals excluding zero", "at most 0.10", "only Gemma", "Llama only", "no control moved it" and "eight models" all verified.
  - "Alongside a knowledge effect" for three families fails (C1).
  - "Full pool" is mis-scoped (m7).
  - "Other norms were not tested" overstates (m12).
- **Interval method recoverable:** yes for HM, FA, GS, CM, steering, NH and the lens. Not stated, or not item-level, for the hedge-judge correction and the surface-form intercept (m3).
- **Stale values:** none found. The only value taken from an older file (\ciRecovery) matches that file. Its upstream CSV was not in scope.
- **Exploratory and post hoc labels:**
  - The cross-model analysis is consistent: "(exploratory)" heading, "post hoc" in text and App. G, matching the plan.
  - Guessability and spelling are consistently labelled post hoc.
  - Several other non-confirmatory analyses carry no label (m6).
- **Cross-model rule and verdicts vs plan:**
  - The verdicts and the "two of three / held for one" statements match the plan and the code.
  - The paper's paraphrase of the rule drops "CI excluding 0" (C2).
  - The plan's secondary readout and caveat were not reported (m10).
