# Numbers audit: main.tex against the evidence (round 3)

Auditor: independent paper-to-evidence check. I did not judge the science.
Scope: every number, interval, count, comparison word and scope word in `paper/main.tex` (abstract, text, tables, captions, appendices), with each macro followed to `paper/numbers.tex` and then to the evidence file it names.
Not opened: FINDINGS.md, OVERVIEW, LAB_LOG, CLAIMS, PROJECT_CONTEXT, RESPONSE_LETTER, `paper/review/*` (except to write this file), `paper/revision/*`, and any notes written by people.

R = `C:/Users/starb/Desktop/Sher/Confidence-Neurons-Across-Uncertainty-Types`. P = `C:/Users/starb/Desktop/Sher/paper`.

## How I checked

- **Summary files.** Each macro was compared with its source file to the printed precision.
- **Independent recomputation from raw files** (scripts in the session scratchpad; no repo file was changed):
  - hedge_measure_v2: cell sizes, 3 replies per item, two-judge κ, cell means and effects for all four families, from `items_*`, `gens_*` and `judge_*` jsonl.
  - E5: majority-of-three cell means and effects, from the same judge files.
  - E1: T1 cell sizes, means and effects; T4 effects (`e1_llama.jsonl` + `item_rates.csv`).
  - E3: window point estimates for all four base models from `e3/<model>/lens.csv` and `labels.jsonl`, including which layers fall in the window.
  - E4: judged abstention rate per condition and model; entropy means; SELECT/EVAL disjointness; k* rule recomputed from the SELECT dose table.
  - E6: entity_known cell means, Spearman ρ, and the n of two-judge items with a covariate.
  - The κ between the original and paraphrase labels, from the E1 T3 table.
  - All reproduce the summary files exactly.
- **Bootstrap intervals** were taken from the summary files, not re-run. The method and number of resamples were read from the code.
- **Hashes.** SHA-256 was computed for every criterion note.
- **Hash timing.** `PREREG/HASHES_round3.txt` timestamps were compared with the modification times of the E1–E6 result files.
- **Figures.** The `plotted` values in the three provenance files were compared with the evidence. Their input modification times were compared with the current files; all are unchanged since plotting.

Status key:
- **exact**: identical at the printed precision.
- **rounding-ok**: the file value rounds correctly to the printed value.
- **mismatch**: the value, count, comparison word or scope word is not supported as written.
- **no-evidence-found**: nothing in the permitted evidence confirms or refutes it.

## Summary counts

| Status | Items |
|---|---|
| exact | 161 |
| rounding-ok | 181 |
| mismatch | 24 |
| no-evidence-found | 4 |
| **Total** | **370** |

How items were counted:
- Each table cell value counts as one item (the 8 onset rows count as 8 items, the 16 window-effect values as 16). Their intervals are counted with the values.
- The 24 mismatch rows come from 20 distinct issues: 2 critical (C1, C2), 3 moderate (M-a, M-b, M-c) and 15 minor. C1 appears in 4 rows and M-b in 3.
- The four no-evidence-found items:
  - "no human validated the judges";
  - hash timing for the 7 older criterion notes;
  - the random-direction dots in Fig. 2;
  - the claim about entropy neurons.

Main findings:
- **No numeric value is wrong.** Every number printed in the paper matches its source at the printed precision. In particular:
  - all 13 criterion hashes printed in the paper match their files;
  - all three figures were drawn from the current evidence files.
- **The mismatches are all in wording.** Each is a comparison word, scope word, count word, verdict word, selective report or caption/content issue. Two of them could change a stated conclusion (C1, C2).

## Critical mismatches (would change a stated conclusion)

### C1. The four-family preregistered verdicts are never reported, and Gemma's is REPLICATES

- **Where.** Sec. 3.2: "The popularity preference is thus Llama's, not general." Abstract: "Gemma-2-9B … shows no popularity preference." Limitations: "Gemma lacks it." App. H lists the four-family design (eba35dec, b53d9a27) as confirmatory.
- **Evidence.** `results/factorial_4models/summary.txt`:
  - VERDICT Qwen PARTIAL, Gemma **REPLICATES**, Mistral PARTIAL.
  - Gemma's pool hedge rate is 0.036, below the 0.05 floor, so its governing readout is first-token hedge log-odds, fixed in advance. On that readout popularity − knowledge = +2.11 [+1.79, +2.42]; condition 1 is yes and condition 2 is yes.
- **What the criterion notes require.**
  - `gemma_mistral_factorial_criterion.md` requires "every model's verdict reported".
  - `hedge_judge_criterion.md` says "Locked factorial verdicts are not changed by this audit".
  - `hedge_measure_v2_criterion.md` supersedes only "the regex", and only where κ ≥ 0.60. Gemma's κ is 0.47, and its verdict was not regex-based.
  - So no note supersedes Gemma's REPLICATES.
- **What the paper does.** It shows the number (App. B, "Hedge log-odds: popularity − knowledge", Gemma +2.11) but never names the verdict. The three-judge result that Gemma shows no preference (E5, NEITHER, +0.01 [−0.03, +0.06]) is correct for judged free-reply abstention.
- **Why it matters.** "Not general" is stated without saying that the confirmatory four-family test returned REPLICATES for Gemma on its preregistered hedging readout.
- **Suggested correction.**
  - Sec. 3.2, after the E5 sentence, add: "Under the four-family criterion fixed before those runs, the forced-answer hedge readout gave PARTIAL for Qwen and Mistral and REPLICATES for Gemma (popularity − knowledge +2.11 [+1.79, +2.42], App. B). Gemma's judged abstention in free replies, however, shows no popularity preference."
  - Scope the closing sentence: "The popularity preference in judged abstention is thus Llama's, not general."

### C2. "Not seen in base models" holds only for the hedge term; the full readout shows popularity > knowledge in 3 of 4 base models

- **Where.** Sec. 3.3, Base models: "Llama-Instruct's preference for popularity over knowledge is not seen in base models."
- **Evidence.** `results/round3/e3_summary.txt` (E3, criterion b2b916ce), popularity − knowledge in the mid-depth window:

| Model | Full readout (H − A) | Hedge term (H) |
|---|---|---|
| Llama-3.1-8B | +0.31 [+0.16, +0.46] | −0.01 [−0.17, +0.15] |
| Gemma-2-2B | +0.29 [+0.16, +0.42] | −0.27 [−0.38, −0.15] |
| OLMo-3-7B | +0.60 [+0.41, +0.76] | +0.08 [−0.01, +0.18] |
| Gemma-2-9B | +0.01 [−0.06, +0.08] | −0.02 [−0.15, +0.11] |

  So on the full readout, three of four base models do favour popularity, with intervals excluding zero. On the hedge term, none does.
- **Criterion.** E3 L2 makes the full readout "required reporting". The paper reports it (App. D table; Fig. 3a) but draws the conclusion from the hedge term alone, without saying so.
- **Why it matters.** The sentence carries the claim that the preference arises with instruction tuning.
- **Suggested correction.** "In the hedge-token term, no base model shows Llama-Instruct's preference for popularity over knowledge. In the full readout, three of four do, because the answer-token term also differs between obscure and popular entities (App. D)."

## Other mismatches (moderate)

### M-a. "Only entropy separates the directions"

- **Where.** Sec. 3.3, Steering.
- **Evidence.** `e4_summary.txt`:
  - Llama D3 (obscure−popular minus unknown−known, judged abstention) = +0.056 [+0.024, +0.089], CI excluding 0.
  - Secondary majority scorer: Llama +0.118 [+0.080, +0.158]; Gemma −0.058 [−0.116, −0.000].
  - So abstention also separates the directions in Llama.
- **Criterion.** E4 D3 asks the paper to "say that the ranking is expected from the cell abstention rates". The paper does not.
- **Suggested correction.** "Entropy separates the directions only in Gemma: … In Llama, the obscure-minus-popular direction exceeds unknown-minus-known in abstention (+0.06 [+0.02, +0.09]), as expected from the cell abstention rates."

### M-b. "about as large" / "about as much" / "as much to knowledge" for base-model hedge terms

- **Where.** Abstract; Fig. 3 title; Sec. 3.3.
- **Evidence.** E3 hedge term, popularity versus knowledge:
  - Llama: 0.55 vs 0.56.
  - Gemma-2-9B: 0.28 vs 0.30.
  - OLMo: 0.61 vs 0.53.
  - **Gemma-2-2B: 0.10 vs 0.37**, with difference −0.27 [−0.38, −0.15]. Here knowledge is about 3.8 times popularity.
- **Suggested correction.** "as large or larger" or "at least as large" (abstract: "but at least as much to knowledge").

### M-c. Name surprisal "neither stands in for popularity nor predicts abstention"

- **Where.** App. F.
- **Evidence.** `e2_summary.txt`:
  - S2(a), two judges: +0.013 [−0.026, +0.059]. This one is reported.
  - S2(b), regex on n = 2,303: **+0.071 [+0.046, +0.096]**, CI excluding 0.
  - S3(b), with the popularity arm added: +0.012 [−0.008, +0.033].
  - The criterion's verdict, NOT SUPPORTED (it needs both a and b), is correctly applied. But "nor predicts abstention" is contradicted by outcome (b), which the paper does not report.
- **Suggested correction.** "… its coefficient on two-judge abstention is +0.013 [−0.026, +0.059] per SD. On the regex measure it is +0.07 [+0.05, +0.10], which falls to +0.01 [−0.01, +0.03] once popularity is in the model. So it does not stand in for popularity."

## Minor mismatches

1. **"keeps its sign under four other measurements"** (Sec. 3.1). Table 1 has **five** other measurement rows besides the primary: regex at 24 tokens, corrected judge, +CLARIFY, regex at 96 tokens, and three-judge majority. It also has three relabelling or covariate rows. The range +0.13 to +0.44 is correct over all of them. Fix: "five other measurements".
2. **"answers correctly every time it is made to answer"** (Intro). The rule is greedy correct plus at least 90% of the 10 samples correct (`build_familiarity_twins.py` l.95), so 9 of 10 qualifies. Fix: "nearly every time" or "at least nine times in ten".
3. **Lens prediction "failed"** (Sec. 3.3). The jlens criterion's cross-model wording for 1 SUPPORTS / 1 CONTRADICTS / 6 UNDECIDED is "mixed, reported model by model" (`summary.txt`: "CROSS-MODEL CLAIM: mixed"). Fix: "was not supported (mixed: …)".
4. **"adds little abstention, and only through its abstention component"** (Abstract). The verdict words match E4 (D2 below the 0.10 bar, so "carried by the abstention component"). But the residual is not zero:
   - Gemma D2 = +0.017 [+0.005, +0.032];
   - majority scorer: Llama +0.017 [+0.002, +0.036], Gemma +0.049 [+0.028, +0.072].
   - Consider "mainly through".
5. **One-prompt readouts placed under "Independent labels"** (App. A, end of that paragraph). The E1 T4 free-prompt "I" log-odds and entropy use the **original** labels (`e1_summary.txt`: "T4 … (original labels)"). They follow "Under the new labels …" with no change of label stated. Fix: add "(original labels)".
6. **Entity-knowledge row and paragraph: n not disclosed.** There are 384 of 387 two-judge items and 2,293 of 2,303 regex items, because 11 pool items lack a Wikidata description (2,662 of 2,673 graded). Table 1's "same" in the Replies column implies the full 387.
7. **Table 1 row "entity knowledge as covariate"** is an LPM popularity coefficient, not "popularity effect minus knowledge effect" as the caption says. The row label says so, but the value is not comparable with the other rows.
8. **Table 1 relabelled and guessable rows: "same" replies, different question counts.** These use 253 of 387 (paraphrase Maybe items dropped) and 335 of 387 questions. The main text gives obscure-known n = 27 for the first; neither n is in the table.
9. **"correctly says what the entity is for 0.83 of obscure known facts"** (Sec. 3.1). 0.83 is the mean share of 6 completions graded CORRECT per fact, not a share of facts. Fix: "Llama's descriptions of the entity are graded correct in 0.83 of completions for obscure known facts (popular: 0.75)".
10. **App. B caption, "Bottom: effects on first-token hedge-versus-answer log-odds".** The bottom block's first row, "Three judges: popularity − knowledge", is judged abstention, not log-odds.
11. **Circuit recovery 0.83 [0.78, 0.87]** (App. G). This is the control→uncertain direction only. The uncertain→control full set gives 1.03 [0.99, 1.07] (`R/paper/figures/fig6_familiarity_v3_full_cells.csv`). The value is correct, but which direction it is goes unstated.
12. **E4 secondary scorer D3 not reported.** The amendment (7bed3826) says the majority rule is reported for D1–D3. The paper gives majority D1 and D2 but not D3 (Llama +0.12 [+0.08, +0.16]; Gemma −0.06 [−0.12, −0.00]).
13. **App. H omits `hedge_judge_criterion.md`** (sha256 prefix ae0bb807). It governs Table 1's "One LLM judge, corrected" row, which is thus neither listed as confirmatory nor labelled post hoc.
14. **E4 amendment self-dating.**
    - The note says "written 2026-10-10 ~22:10 UTC".
    - `HASHES_round3.txt` records its hash at 2026-10-09T22:02:16Z.
    - The E4 judge files were last modified 2026-10-09T22:47–22:53Z.
    - If the note's own date were right, it would postdate the judging. The recorded hash time precedes the judge outputs' completion, so the paper's claim is supported by the hash log, but the note's internal date is wrong. It cannot be edited without changing its hash; better to mention the discrepancy if asked.
15. **Lens onset rule** (App. D). The paper's SUPPORTS/CONTRADICTS rule omits the criterion's second clause ("or F has an onset and K has none", and vice versa). It doesn't affect any of the 8 outcomes; all have both onsets defined.

## Interval methods (recoverable; not stated in the paper where marked)

- **Recoverable and consistent with "95% item-bootstrap":**
  - factorial effects, hedge_measure_v2, E1, E5, guessability: 10,000 within-cell item bootstraps;
  - E3 and lens D: 2,000 within-cell bootstraps (stated in App. D);
  - E4: 10,000 item bootstraps.
- **Stated nowhere in the paper:**
  - LPMs in E2, E6 and round2_checks: 2,000 item bootstraps.
  - Corrected-judge CI: 5,000 bootstraps of judged generations within strata (`hedge_judge_criterion.md`).
  - Equal-surface-form gap +0.43 [+0.18, +0.69]: an OLS intercept with an HC3 robust CI (`unireps_confound_checks.py` l.188), not a bootstrap.
  - Name-spelling GLM coefficients: SEs clustered by relation.
  - Cross-model ρ intervals: ranks are computed once and reused in the resamples (documented approximation).
- **Suggestion.** One sentence in Setup ("Other intervals: …") would cover these.

## Confirmatory and post hoc labelling

| Analysis | Paper label | Criterion note (sha256 prefix computed) | Verdict in file | Paper wording | OK? |
|---|---|---|---|---|---|
| Two-judge measurement | confirmatory | hedge_measure_v2 (81acfc9f ✓) | Llama FAMILIARITY > KNOWLEDGE; others UNMEASURED | "follows popularity more than knowledge"; others "do not" agree | yes |
| Four-family factorial | confirmatory | qwen (eba35dec ✓), gemma_mistral (b53d9a27 ✓) | Qwen PARTIAL, Gemma REPLICATES, Mistral PARTIAL | verdicts not reported | **no (C1)** |
| Independent labels E1 | confirmatory | af50f53f ✓ | T1 SURVIVES; T2 NOT CIRCULAR | "holds"; "entropy still follows knowledge" | yes |
| Name surprisal E2 | confirmatory | 2b69e06c ✓ | S2 NOT SUPPORTED | "neither … nor predicts abstention" | partly (M-c) |
| Lens split E3 | confirmatory | b2b916ce ✓ | L1 HEDGE-SIDE (4/4) | "sits in the hedge-token term" | yes |
| Steering E4 + amendment | confirmatory | 18c74883 ✓, 7bed3826 ✓ | D1 NOT MET; D2 CARRIED BY THE ABSTENTION COMPONENT (both models) | "short of the 0.10 bar"; "moves a generic abstention signal" | yes |
| Third judge E5 | confirmatory | f35aa4d9 ✓ | Llama FOLLOWS POPULARITY; Gemma NEITHER; Qwen, Mistral UNMEASURED | matches | yes |
| Entity knowledge E6 | confirmatory | 937b46e3 ✓ | K1 SURVIVES ENTITY KNOWLEDGE | "holds … controlling for …" | yes |
| Lens cascade | confirmatory | c8e04372 ✓ | SUPPORTS 1 / CONTRADICTS 1 / UNDECIDED 6; cross-model "mixed"; predates instruction tuning: yes | "failed" | wording (minor 3) |
| Circuit behaviour | confirmatory | 1c99afd4 ✓ | removal 0.98, injection 0.40 < 0.50 → not surviving (EQUIVOCAL) | "short of a behavioural bar" | yes |
| Earlier steering | superseded | da0ad000 ✓, 797f52d4 ✓ | NOT REUSED | "superseded" | yes |
| Corrected single judge | unlabelled | hedge_judge (ae0bb807, not printed) | Llama condition 1 yes | in Table 1 | label missing (minor 13) |
| Guessability, spelling, free replies, entropy control | post hoc | none | n/a | labelled post hoc | yes |
| Cross-model agreement | post hoc ("exploratory") | cross_model_agreement.md (ca29f0d4, not printed; it says post hoc itself) | Qwen, Gemma NOT SUPPORTED; Mistral PREDICTION HOLDS | not met / not met / met | yes |

**Hash timing.**
- E1–E5 hashes were recorded at 2026-10-09T21:28Z. The first result files followed: E2 at 21:50Z, E1 and E5 at 22:01Z, E3 at 02:59–03:13Z on 10-10, E4 judging finished at 22:47Z.
- The E6 hash was recorded at 2026-10-10T03:23:38Z, before the Wikidata, generation and judge files (03:32, 03:38, 04:07Z).
- For the seven older notes, no hash-time record is among the permitted evidence. Git has history only for `l5_behavioral_criterion.md` (commit 59f5ec5, 2026-10-02). These are marked no-evidence-found, not mismatches.

## Full item table

Abbreviations:
- HM = `results/hedge_measure_v2/summary.txt`; FA = `results/factorial_4models/{effects,cells}.csv` and `summary.txt`.
- HJ = `results/hedge_judge/summary.txt`; UC = `results/unireps_checks/summary.txt`.
- E1–E6 = `results/round3/e*_summary.*`; JL = `results/jlens_cascade/outcomes.csv` and `summary.txt`.
- GS = `paper/analysis/guessability.csv`; CM = `paper/analysis/cross_model_agreement.{csv,txt}`; RC = `paper/analysis/round2_checks.txt`.
- "recomp" = also recomputed from raw files.

### Abstract

| id | claim | evidence (field) | file value | status |
|---|---|---|---|---|
| A1 | obscure known says IDK in 0.43 | HM llama primary unc/High (recomp) | 0.433 | rounding-ok |
| A2 | against 0.00 for popular | HM primary con/High | 0.000 | exact |
| A3 | holds with relabelled knowledge | E1 T1 verdict | SURVIVES | exact |
| A4 | holds with a third judge | E5 llama verdict | FOLLOWS POPULARITY | exact |
| A5 | holds without guessable answers | GS judge_noguess diff | +0.30 [+0.19, +0.41] | exact |
| A6 | holds controlling for entity knowledge | E6 K1 | SURVIVES ENTITY KNOWLEDGE | exact |
| A7 | entropy follows knowledge in four families | FA condition 2, all four | yes ×4 | exact |
| A8 | … and under the independent labels | E1 T2 | NOT CIRCULAR | exact |
| A9 | one prompt: "I" lean follows popularity, entropy follows knowledge | E1 T4 | I: +8.37 vs +3.11; ent: +0.05 vs +0.16, diff CI excl 0 | exact |
| A10 | steering "adds little … only through its abstention component" | E4 D1/D2 | NOT MET; CARRIED; Gemma D2 +0.017 [+0.005, +0.032] | mismatch (minor 4) |
| A11 | base hedge readout responds to popularity "but as much to knowledge" | E3 hedge_term | G2B 0.10 vs 0.37 | mismatch (M-b) |
| A12 | Gemma the only other family measurable | E5 Fleiss κ ≥ 0.60 only Llama, Gemma | 0.741, 0.643 | exact |
| A13 | Gemma shows no popularity preference | E5 NEITHER; FA Gemma REPLICATES | conflict undisclosed | mismatch (C1) |

### Introduction and Setup

| id | claim | evidence | file value | status |
|---|---|---|---|---|
| S1 | answers correctly "every time" | build_familiarity_twins.py l.95 | greedy + ≥ 0.9 of samples | mismatch (minor 2) |
| S2 | 16 relations | UC "relations 16" | 16 | exact |
| S3 | bottom/top deciles | qwen_factorial_criterion (quantile 0.1/0.9) | 0.1/0.9 | exact |
| S4 | greedy + 10 samples, T = 0.7 | criterion; builder | 10, 0.7 | exact |
| S5 | known ≥ 90% + greedy; unknown if none correct | builder l.95–98 | same | exact |
| S6 | Llama 2,303 | FA cells n sum | 1179+87+361+676 = 2303 | exact |
| S7 | Qwen 2,392 | FA cells | 2392 | exact |
| S8 | Gemma 2,282 | FA cells | 2282 | exact |
| S9 | Mistral 2,378 | FA cells | 2378 | exact |
| S10 | up to 100 per cell | items_llama.jsonl (recomp) | 100/87/100/100 | exact |
| S11 | 387 in total | HM | 387 | exact |
| S12 | three replies, T = 0.7, up to 96 tokens | criterion; gens 3 per item (recomp) | 3, 0.7, 96 | exact |
| S13 | judges Qwen2.5-32B, Gemma-2-27B | criterion; judge files | same | exact |
| S14 | Llama κ = 0.62 | HM (recomp 0.619) | 0.62 | exact |
| S15 | pre-set bar 0.60 | hedge_measure_v2_criterion | 0.60 | exact |
| S16 | other families below bar | HM κ | 0.30/0.47/0.39 | exact |
| S17 | "no human validated the judges" | n/a | n/a | no-evidence-found |
| S18 | effect definitions; 95% item bootstrap | criterion; code | 10,000 within-cell | exact |
| S19 | criterion notes hashed before each run | HASHES_round3 vs result mtimes | E1–E6 verified; 7 older unverifiable | no-evidence-found (older) |

### Figure 1 caption and provenance

| id | claim | evidence | value | status |
|---|---|---|---|---|
| F1 | n = 387 questions, 3 replies | provenance a/* n: 87+100+100+100 | 387 | exact |
| F2 | obscure known n = 87 | provenance a/uncertain/HighlyKnown n | 87 | exact |
| F3 | (b) n = 2,303 | provenance b/* n sum | 2303 | exact |
| F4 | (c) four families | provenance c/* = FA effects | match to 16 digits | exact |
| F5 | panel a values | provenance vs HM | 0.433/0.477/0.133/0.000 | exact |
| F6 | 95% item-bootstrap intervals | provenance ci fields | present | exact |
| F7 | inputs current | provenance mtimes vs files | identical | exact |

### Section 3.1

| id | claim | evidence | value | status |
|---|---|---|---|---|
| M1 | 0.43 vs 0.00 | HM | 0.433 / 0.000 | rounding-ok |
| M2 | entropy "nearly equal" in those two cells | FA cells entropy | 0.951 vs 0.858 (CIs overlap) | exact |
| M3 | pop +0.39 | HM | +0.388 | rounding-ok |
| M4 | CI [+0.33, +0.45] | HM | [+0.328, +0.449] | rounding-ok |
| M5 | know +0.09 | HM | +0.089 | rounding-ok |
| M6 | CI [+0.03, +0.15] | HM | [+0.031, +0.147] | rounding-ok |
| M7 | diff +0.30 | HM | +0.300 | exact |
| M8 | CI [+0.20, +0.40] | HM | [+0.204, +0.398] | rounding-ok |
| M9 | "four other measurements" | Table 1 | five measurement rows | mismatch (minor 1) |
| M10 | range lower +0.13 | HJ corrected pop−know | +0.132 | rounding-ok |
| M11 | range upper +0.44 | FA llama hedge_rate diff | +0.442 | rounding-ok |
| M12 | "keeps its sign" | all Table 1 rows > 0 with CI > 0 | yes | exact |
| M13 | entropy +2.14 nats (knowledge) | FA | +2.139 | rounding-ok |
| M14 | +0.51 (popularity) | FA | +0.513 | rounding-ok |
| M15 | knowledge ≥ 4× popularity in all four | FA | 4.17, n/a (Qwen pop −0.02), 11.6, 4.79 | exact |
| M16 | "Five checks, fixed in advance unless stated" | E1/E5/E6 prereg; guessable post hoc stated; one-prompt = E1 T4 (prereg, descriptive) | 5 | exact |
| M17 | E1 pop +0.38 | E1 T1 (recomp) | +0.383 | rounding-ok |
| M18 | E1 know +0.06 | E1 | +0.061 | rounding-ok |
| M19 | E1 diff +0.32 | E1 | +0.322 | rounding-ok |
| M20 | CI [+0.16, +0.49] | E1 | [+0.157, +0.490] | rounding-ok |
| M21 | obscure known 0.46 | E1 cell | 0.457 | rounding-ok |
| M22 | n = 27 | E1 cells n | 27 | exact |
| M23 | know − pop entropy +1.40 | E1 T2 | +1.402 | rounding-ok |
| M24 | CI [+1.02, +1.73] | E1 T2 | [+1.020, +1.730] | rounding-ok |
| M25 | third judge Llama-3.3-70B, 2 of 3 rule | E5 criterion; file names | same | exact |
| M26 | κ = 0.74 | E5 Fleiss | 0.741 | rounding-ok |
| M27 | diff +0.44 | E5 | +0.437 | rounding-ok |
| M28 | CI [+0.32, +0.55] | E5 | [+0.322, +0.550] | rounding-ok |
| M29 | entity known 0.83 obscure known | E6 K3 (recomp) | 0.831 | rounding-ok (wording: minor 9) |
| M30 | popular 0.75 | E6 | 0.750 | exact |
| M31 | ρ = 0.52 | E6 K3 (recomp) | +0.516 | rounding-ok |
| M32 | shrinks by only 5% | E6 (a) | 5% (0.355 → 0.337) | exact |
| M33 | +0.34 | E6 (a) obscure | +0.337 | rounding-ok |
| M34 | CI [+0.27, +0.41] | E6 | [+0.266, +0.411] | rounding-ok |
| M35 | guessable diff +0.30 | GS judge_noguess | +0.2997 | rounding-ok |
| M36 | CI [+0.19, +0.41] | GS | [+0.192, +0.411] | rounding-ok |
| M37 | "I" log-odds pop +8.4 | E1 T4 | +8.365 | rounding-ok |
| M38 | know +3.1 | E1 T4 | +3.111 | rounding-ok |
| M39 | entropy know +0.16 | E1 T4 | +0.160 | exact |
| M40 | entropy pop +0.05 | E1 T4 | +0.053 | rounding-ok |
| M41 | surprisal "did not even correlate" | E2 S1 | −0.001 | exact |

### Section 3.2

| id | claim | evidence | value | status |
|---|---|---|---|---|
| FA1 | entropy pattern in all four | FA condition 2 | yes ×4 | exact |
| FA2 | others "rarely" say IDK; two judges disagree | HM obscure-unknown 0.05/0.05/0.10; κ < 0.60 | as stated | exact |
| FA3 | Gemma κ = 0.64 | E5 | 0.643 | rounding-ok |
| FA4 | Gemma diff +0.01 | E5 | +0.013 | rounding-ok |
| FA5 | CI [−0.03, +0.06] | E5 | [−0.030, +0.057] | rounding-ok |
| FA6 | at most 0.11 even obscure unknown | E5 cells (max = obs-unk) | 0.107 | rounding-ok |
| FA7 | Qwen κ = 0.41 | E5 | 0.409 | rounding-ok |
| FA8 | Mistral κ = 0.54 | E5 | 0.535 | rounding-ok |
| FA9 | Qwen, Mistral "remain unmeasured" | E5 | UNMEASURED | exact |
| FA10 | "popularity preference is Llama's, not general" | FA verdict Gemma REPLICATES (undisclosed) | conflict | mismatch (C1) |

### Figure 2 caption and provenance

| id | claim | evidence | value | status |
|---|---|---|---|---|
| FS1 | 150 popular known questions | e4/log.txt; replies (recomp) | EVAL 150 × 3 | exact |
| FS2 | held out from fitting and dose choice | log FIT 1336, outside-FIT 341; SELECT ∩ EVAL = 0 (recomp) | disjoint | exact |
| FS3 | panel values | provenance a/* = e4_summary judges | match | exact |
| FS4 | entropy panel | provenance b/* = e4 entropy (recomp) | match | exact |
| FS5 | "dots: individual random directions" | provenance stores only random mean | not stored | no-evidence-found |
| FS6 | 95% bootstrap over questions | provenance ci | present | exact |

### Section 3.3, Steering

| id | claim | evidence | value | status |
|---|---|---|---|---|
| ST1 | Llama none 0.01 | E4 EVAL none judges (recomp) | 0.007 | rounding-ok |
| ST2 | fam 0.10 | E4 | 0.100 | exact |
| ST3 | +0.09 | D1 | +0.092 | rounding-ok |
| ST4 | CI [+0.06, +0.12] | D1 | [+0.062, +0.125] | rounding-ok |
| ST5 | short of the 0.10 bar | D1 verdict | NOT MET | exact |
| ST6 | majority +0.17 | D1 maj | +0.173 | rounding-ok |
| ST7 | Gemma 0.00 → 0.04 | E4 | 0.000 → 0.044 | rounding-ok |
| ST8 | at 4× natural norm | kstar_gemma9b.json | 4 | exact |
| ST9 | cosine 0.95, "nearly parallel" | geometry_llama.json | 0.948 | rounding-ok |
| ST10 | perp adds +0.01 | D2 | +0.010 | exact |
| ST11 | CI [−0.00, +0.02] | D2 | [−0.001, +0.024] | rounding-ok |
| ST12 | +0.02 with three judges | D2 maj | +0.017 | rounding-ok |
| ST13 | "moves a generic abstention signal" | D2 verdict | CARRIED BY THE ABSTENTION COMPONENT | exact |
| ST14 | "Only entropy separates the directions" | D3 Llama | +0.056 [+0.024, +0.089] | mismatch (M-a) |
| ST15 | "and only in Gemma" (entropy) | D4 Llama | +0.003 [−0.087, +0.092] | exact |
| ST16 | +0.66 nats | D4 Gemma | +0.659 | rounding-ok |
| ST17 | CI [+0.58, +0.74] | D4 | [+0.577, +0.741] | rounding-ok |
| ST18 | fit on half / dose on second set / score third | log | FIT/SELECT/EVAL | exact |

### Figure 3 and Base models

| id | claim | evidence | value | status |
|---|---|---|---|---|
| FL1 | four base models, three families | E3 | Llama, Gemma ×2, OLMo | exact |
| FL2 | pre-set mid-depth window | e3 criterion; code (recomp layer sets) | ℓ/(L−1) ∈ [1/3, 2/3] | exact |
| FL3 | title "about as much to knowledge" | E3 hedge_term G2B | 0.10 vs 0.37 | mismatch (M-b) |
| FL4 | provenance values | fig_lens_base.provenance = e3_window_effects | match | exact |
| FL5 | magnitudes not comparable across models | n/a (design statement) | n/a | exact |
| B1 | pop effect positive in all four (full readout) | E3 logodds | 0.67/0.39/0.40/0.29, all CI > 0 | exact |
| B2 | sits in hedge term, all four CIs exclude zero | E3 L1 | 4 of 4, HEDGE-SIDE | exact |
| B3 | knowledge effect on that term "about as large" | E3 | G2B 0.37 vs 0.10 | mismatch (M-b) |
| B4 | OLMo full-readout knowledge negative | E3 | −0.303 [−0.396, −0.203] | exact |
| B5 | signal exists before instruction tuning | JL "PREDATES … yes" | yes | exact |
| B6 | preference "not seen in base models" | E3 full readout diffs | 3 of 4 pop > know | mismatch (C2) |
| B7 | prediction "failed" | JL cross-model | "mixed" | mismatch (minor 3) |
| B8 | 1 supporting | JL | 1 | exact |
| B9 | 1 contradicting | JL | 1 | exact |
| B10 | 6 undecided of eight | JL | 6 of 8 | exact |

### Related work, Limitations, Conclusion

| id | claim | evidence | value | status |
|---|---|---|---|---|
| L1 | ours uninstructed and free-form | free prompt = chat template alone | yes | exact |
| L2 | popularity preference "shown for one model; Gemma lacks it" | E5; FA (C1) | see C1 | mismatch (C1, same root) |
| L3 | 2–9B parameters | models studied | 2B–9B | exact |
| L4 | "only the first holds in every family" | FA condition 2 ×4; abstention not | yes | exact |

### Appendix A, Table 1 and text

| id | claim | evidence | value | status |
|---|---|---|---|---|
| T1 | regex: 20 per question, 24 tokens | UC "free generations per item 20"; criterion "originally 24" | 20, 24 | exact |
| T2 | +0.44 | FA llama hedge_rate diff | +0.442 | rounding-ok |
| T3 | [+0.34, +0.53] | FA | [+0.345, +0.534] | rounding-ok |
| T4 | corrected +0.13 | HJ | +0.132 | rounding-ok |
| T5 | [+0.03, +0.24] | HJ | [+0.027, +0.239] | rounding-ok |
| T6 | primary +0.30 [+0.20, +0.40] | HM | +0.300 [+0.204, +0.398] | rounding-ok |
| T7 | +CLARIFY +0.31 | HM secondary | +0.311 | rounding-ok |
| T8 | [+0.21, +0.41] | HM | [+0.209, +0.414] | rounding-ok |
| T9 | regex 96 tokens +0.43 | HM regex | +0.429 | rounding-ok |
| T10 | [+0.31, +0.55] | HM | [+0.306, +0.547] | rounding-ok |
| T11 | three judges +0.44 [+0.32, +0.55] | E5 | +0.437 [+0.322, +0.550] | rounding-ok |
| T12 | relabelled +0.32 [+0.16, +0.49] | E1 | as M19–M20 | rounding-ok |
| T13 | relabelled row "same" replies | E1 T1 n = 253 of 387 | n not given | mismatch (minor 8) |
| T14 | guessable removed +0.30 [+0.19, +0.41] (post hoc) | GS; n = 335 | as M35–M36 | rounding-ok |
| T15 | entity covariate +0.34 [+0.27, +0.41] | E6 | +0.337 [+0.266, +0.411] | rounding-ok |
| T16 | caption "pop effect minus know effect" vs covariate row; n = 384 | E6 | coefficient, n = 384 | mismatch (minor 6, 7) |
| T17 | caption: one judge Qwen2.5-32B; subset labelled by that judge | HJ "1515 judged generations" | yes | exact |
| T18 | corrected judge gives the smallest difference | Table 1 | 0.13 = min | exact |
| T19 | 96-token two-judge measure made primary before it ran | hedge_measure_v2 criterion | yes | exact |
| T20 | regex recall on 96-token replies 0.99 | HM | 0.99 | exact |
| IL1 | one paraphrase per relation, prefix "Answer:", fresh samples, 90%/0% | E1 criterion | same | exact |
| IL2 | agreement "moderate" | E1 T3 table → Cohen κ ≈ 0.52 (recomp) | 0.52 | exact |
| IL3 | paraphrase moves many popular known facts to the intermediate band | E1 T3 popular HK → Maybe | 312 of 676 | exact |
| IL4 | pop +0.38 [+0.29, +0.47] | E1 | +0.383 [+0.294, +0.472] | rounding-ok |
| IL5 | know +0.06 [−0.03, +0.15] | E1 | +0.061 [−0.030, +0.150] | rounding-ok |
| IL6 | entropy know − pop +1.40 [+1.02, +1.73] | E1 T2 | same | rounding-ok |
| IL7 | "I" pop +8.4, know +3.1 | E1 T4 | +8.365, +3.111 | rounding-ok |
| IL8 | diff +5.3 [+4.4, +6.0] | E1 T4 | +5.254 [+4.424, +6.045] | rounding-ok |
| IL9 | entropy pop +0.05, know +0.16 | E1 T4 | +0.053, +0.160 | rounding-ok |
| IL10 | diff −0.11 [−0.18, −0.03] | E1 T4 | −0.106 [−0.182, −0.029] | rounding-ok |
| IL11 | T4 placed under "new labels" | E1 T4 uses original labels | config | mismatch (minor 5) |
| EK1 | "{subject} is" after "What is {subject}?", greedy + 5 samples | E6 criterion; e6_gen 6 completions | same | exact |
| EK2 | 70B judge vs English Wikidata description | E6 criterion; e6_wikidata | same | exact |
| EK3 | 0.83 / 0.75 | E6 K3 | 0.831 / 0.750 | rounding-ok |
| EK4 | coef −0.03 | E6 (a) entity_known_z | −0.027 | rounding-ok |
| EK5 | CI [−0.06, +0.00] | E6 | [−0.057, +0.004] | rounding-ok |
| EK6 | LPM with relation fixed effects | E6 criterion | yes | exact |
| EK7 | 5% two-judge / 3% regex shrink | E6 (a)/(b) | 5% / 3% | exact |
| EK8 | sample sizes not stated (384, 2,293) | E6 | n/a | mismatch (minor 6) |
| G1 | guessable definition (≥ 3 chars; modal answer ≥ 20%) | guessability.py l.3–26 | same | exact |
| G2 | among unknown facts popularity gap +0.34 | GS judge_all popU | +0.343 | rounding-ok |
| G3 | [+0.26, +0.43] | GS | [+0.257, +0.430] | rounding-ok |
| G4 | 2,106 non-guessable | GS regex_noguess n_total | 2106 | exact |
| G5 | +0.48 [+0.36, +0.59] | GS | +0.480 [+0.361, +0.592] | rounding-ok |
| FR1 | 0.53 of free replies contain gold alias | RC obscure HK | 0.53 | exact |
| FR2 | 0.72 of non-abstaining | RC | 0.72 | exact |
| FR3 | 24 of 87 questions | RC | 24 of 87 | exact |
| FR4 | pop +0.29 [+0.23, +0.36] | RC two judges with entropy | +0.294 [+0.227, +0.362] | rounding-ok |
| FR5 | know −0.05 [−0.14, +0.03] | RC | −0.054 [−0.140, +0.030] | rounding-ok |
| FR6 | +0.14 per SD entropy | RC entropy_z | +0.137 | rounding-ok |
| NS1 | over 2,303 questions, relation FE | UC items 2303; GLM relation FE | 2303 | exact |
| NS2 | −1.87 → −1.79 logits/SD | UC M1/M2 logpop_z | −1.871 / −1.791 | rounding-ok |
| NS3 | word frequency, token count, token rarity | UC M2 terms | zipf_mean/min, n_tok, tok_logid | exact |
| NS4 | 55 matched pairs of known facts | UC; check1_pairs.csv | 55 | exact |
| NS5 | gap +0.76 | UC raw hedge gap | +0.758 | rounding-ok |
| NS6 | → +0.43 [+0.18, +0.69] | UC equal surface form (OLS HC3) | +0.434 [+0.182, +0.686] | rounding-ok |

### Appendix B (families table and text)

| id | claim | evidence | value | status |
|---|---|---|---|---|
| FT1–4 | two-judge κ: Llama, Qwen, Gemma, Mistral | HM (recomp) | 0.62, 0.30, 0.47, 0.39 | exact ×4 |
| FT5–8 | Fleiss κ | E5 | 0.741, 0.409, 0.643, 0.535 → 0.74, 0.41, 0.64, 0.54 | rounding-ok ×4 |
| FT9–12 | regex recall | HM | 0.99, 0.58, 0.48, 0.73 | exact ×4 |
| FT13–16 | both DK, obscure-unknown | HM (recomp) | 0.477, 0.050, 0.047, 0.097 → 0.48, 0.05, 0.05, 0.10 | rounding-ok ×4 |
| FT17–20 | DK or CLARIFY | HM | 0.647, 0.317, 0.220, 0.253 → 0.65, 0.32, 0.22, 0.25 | rounding-ok ×4 |
| FT21–24 | three-judge majority | E5 (recomp) | 0.747, 0.097, 0.107, 0.19 → 0.75, 0.10, 0.11, 0.19 | rounding-ok ×4 |
| FT25–28 | regex obscure-unknown (96 tokens) | HM | 0.847, 0.180, 0.073, 0.210 → 0.85, 0.18, 0.07, 0.21 | rounding-ok ×4 |
| FT29–30 | three judges pop − know: Llama, Gemma | E5 | +0.437, +0.013 → +0.44, +0.01 | rounding-ok ×2 |
| FT31–34 | hedge log-odds popularity | FA | +2.933, +1.138, +1.789, +2.480 | rounding-ok ×4 |
| FT35–38 | hedge log-odds knowledge | FA | +3.055, +3.632, −0.318, +2.077 | rounding-ok ×4 |
| FT39–42 | hedge log-odds pop − know | FA | −0.122, −2.494, +2.107, +0.403 | rounding-ok ×4 |
| FT43–48 | CIs in text (three judges ×2; log-odds ×4) | E5; FA | [+0.32, +0.55], [−0.03, +0.06]; [−0.72, +0.45], [−3.41, −1.64], [+1.79, +2.42], [−0.55, +1.34] | rounding-ok ×6 |
| FT49 | Qwen-32B and Llama-70B agree best across families; κ = 0.96 on Llama | E5 pairwise (best in all four) | 0.955 | rounding-ok |
| FT50 | Llama forced-answer hedge readout follows pop and know about equally | FA | 2.93 vs 3.05, diff CI spans 0 | exact |
| FT51 | caption "Bottom: … log-odds" | first bottom row is judged abstention | content | mismatch (minor 10) |
| FT52 | caption "Cell rows: obscure, unknown cell" | as above | yes | exact |

### Appendix C (steering)

| id | claim | evidence | value | status |
|---|---|---|---|---|
| SD1 | ℓ* = round(0.55(L−1)) | geometry json: Llama 17 of 32; Gemma 23 of 42 | 17, 23 | exact |
| SD2 | random half for fitting | log FIT 1336 of 2673 | half | exact |
| SD3 | abst = rate ≥ 0.5 minus rate 0 | criterion; geometry 650 vs 603 | same | exact |
| SD4 | perp rescaled to original norm | criterion | same | exact |
| SD5 | k ∈ {0.5, 1, 2, 4, 8} × fam norm | E4 dose table | same | exact |
| SD6 | last prompt + every generated position | criterion | same | exact |
| SD7 | k* smallest beating random mean by 0.10 (regex), on 150 | E4 SELECT table (recomp): Llama k = 1 (+0.30), Gemma k = 4 (+0.22) | 1, 4 | exact |
| SD8 | scored on 150 further, both judges | log; judges (recomp) | yes | exact |
| SD9–14 | Table, Llama judges: none, random, know, abst, fam, perp | E4 (recomp) | 0.007, 0.008, 0.044, 0.089, 0.100, 0.018 | rounding-ok ×6 |
| SD15–19 | Table, Llama regex: none, know, abst, fam, perp | E4 | 0.031, 0.102, 0.253, 0.233, 0.024 | rounding-ok ×5 |
| SD20–25 | Table, Gemma judges | E4 (recomp) | 0.000, 0.003, 0.042, 0.196, 0.044, 0.020 | rounding-ok ×6 |
| SD26–30 | Table, Gemma regex | E4 | 0.002, 0.033, 0.736, 0.271, 0.073 | rounding-ok ×5 |
| SD31 | Table caption 150 questions, 3 replies | recomp 450 replies per condition | yes | exact |
| SD32 | D1 Llama +0.09 [+0.06, +0.12] | E4 | as ST3–ST4 | rounding-ok |
| SD33 | D1 Gemma +0.04 [+0.02, +0.06] | E4 | +0.042 [+0.022, +0.063] | rounding-ok |
| SD34 | majority D1 +0.17 [+0.13, +0.22], +0.11 [+0.08, +0.15] | E4 | +0.173 [+0.128, +0.221], +0.114 [+0.078, +0.152] | rounding-ok |
| SD35 | majority D2 +0.02 [+0.00, +0.04], +0.05 [+0.03, +0.07] | E4 | +0.017 [+0.002, +0.036], +0.049 [+0.028, +0.072] | rounding-ok |
| SD36 | D2 +0.01 [−0.00, +0.02], +0.02 [+0.01, +0.03] | E4 | +0.010 [−0.001, +0.024], +0.017 [+0.005, +0.032] | rounding-ok |
| SD37 | D3 +0.06 [+0.02, +0.09], +0.00 [−0.03, +0.03] | E4 | +0.056 [+0.024, +0.089], +0.002 [−0.027, +0.031] | rounding-ok |
| SD38 | majority D3 not reported | amendment requires D1–D3 | omitted | mismatch (minor 12) |
| SD39 | regex counts far more abstentions than judges for steered replies | E4 | e.g. Gemma abst 0.74 vs 0.20 | exact |
| SD40 | earlier experiment Llama 0.02 → 0.29 | direction_reuse_v2/llama | 0.020 → 0.287 | rounding-ok |
| SD41 | earlier dose chosen on evaluation questions | direction_reuse_v2.py (same `held` set for doses and k*) | yes | exact |
| SD42 | cosine 0.95 / 0.63 | geometry; E4 | 0.948 / 0.631 | rounding-ok |
| SD43 | no condition degenerate | E4 EVAL degenerate | 0.000 all | exact |

### Appendix D (lens)

| id | claim | evidence | value | status |
|---|---|---|---|---|
| LE1 | hedge term = logsumexp of hedge tokens; full = minus answer term | E3 criterion; lens.csv columns | same | exact |
| LE2 | window ℓ/(L−1) ∈ [1/3, 2/3] | code; recomp layers (Llama 11–20 of 0–30) | same | exact |
| LE3 | 2,000 within-cell bootstraps | e3 summary header | 2,000 | exact |
| LE4 | base models labelled on own prompt, same rule | jlens criterion | same | exact |
| LE5–20 | window-effects table, 16 point values | E3 (recomp) | all match | rounding-ok ×16 |
| LE21 | window-effects table, 16 intervals | E3 csv | all match | rounding-ok (counted with values) |
| LE22 | onset definition | jlens criterion | matches; omits the "F onset, no K onset" clause | mismatch (minor 15) |
| LE23–30 | onset table rows (layers, F, K, D, outcome, mid F) | JL outcomes.csv | all 8 rows match | exact ×8 |

### Appendix E (cross-model)

| id | claim | evidence | value | status |
|---|---|---|---|---|
| CM1–3 | knowledge-differs n | CM | 313, 60, 108 | exact ×3 |
| CM4–6 | hedge ρ | CM | +0.108 [−0.001, +0.215]; +0.384 [+0.133, +0.603]; +0.247 [+0.058, +0.425] | rounding-ok ×3 |
| CM7–9 | entropy ρ | CM | −0.174 [−0.278, −0.069]; +0.033 [−0.235, +0.278]; −0.075 [−0.267, +0.124] | rounding-ok ×3 |
| CM10–12 | gap Δ | CM txt | +0.28 [+0.14, +0.42]; +0.35 [−0.03, +0.74]; +0.32 [+0.06, +0.57] | exact ×3 |
| CM13–15 | rule | CM txt | NOT SUPPORTED, NOT SUPPORTED, PREDICTION HOLDS → not met / not met / met | exact ×3 |
| CM16–18 | same-knowledge n | CM | 1784, 1977, 2000 | exact ×3 |
| CM19–21 | same hedge ρ | CM | 0.503, 0.376, 0.640 | rounding-ok ×3 |
| CM22–24 | same entropy ρ | CM | 0.458, 0.541, 0.687 | rounding-ok ×3 |
| CM25 | rule text and post hoc status | cross_model_agreement.md | matches | exact |

### Appendix F (surprisal)

| id | claim | evidence | value | status |
|---|---|---|---|---|
| SU1 | base Llama-3.1-8B, "Here is a name:" | E2 criterion | same | exact |
| SU2 | Spearman −0.001 | E2 S1 | −0.001 | exact |
| SU3 | +0.013 [−0.026, +0.059] | E2 S2(a) | same | exact |
| SU4 | "neither stands in … nor predicts abstention" | E2 S2(b) +0.071 [+0.046, +0.096] | selective | mismatch (M-c) |

### Appendix G (timing and circuit)

| id | claim | evidence | value | status |
|---|---|---|---|---|
| CI1 | hedge half at layer 16 | singlebos jlens_summary | L16 | exact |
| CI2 | entropy half at layer 28 | same | L28 | exact |
| CI3 | 12 layers later | 28 − 16 | 12 | exact |
| CI4 | 20 heads + 100 neurons | behavioral_summary; fig6 csv | same | exact |
| CI5 | recovery 0.83 | R/paper/numbers.tex → fig6 csv (control_to_uncertain full set) | 0.827 | rounding-ok |
| CI6 | [0.78, 0.87] | fig6 csv | [0.784, 0.871] | rounding-ok |
| CI7 | direction of the 0.83 unstated (other direction 1.03) | fig6 csv | selection | mismatch (minor 11) |
| CI8 | removal 0.98 | circuit_familiarity_v3 behavioral_summary | 0.981 | rounding-ok |
| CI9 | injection 0.40 | same | 0.398 | rounding-ok |
| CI10 | short of the bar fixed in advance | l5 criterion bar 0.50 | 0.398 < 0.50 | exact |
| CI11 | entropy-correlated neurons behaved as word-frequency/temperature neurons | n/a | not in evidence list | no-evidence-found |

### Appendix H (hashes)

| id | claim | file | computed sha256[:8] | status |
|---|---|---|---|---|
| H1 | eba35dec | qwen_factorial_criterion.md | eba35dec | exact |
| H2 | b53d9a27 | gemma_mistral_factorial_criterion.md | b53d9a27 | exact |
| H3 | 81acfc9f | hedge_measure_v2_criterion.md | 81acfc9f | exact |
| H4 | c8e04372 | jlens_cascade_criterion.md | c8e04372 | exact |
| H5 | 1c99afd4 | l5_behavioral_criterion.md | 1c99afd4 | exact |
| H6 | af50f53f | PREREG/e1_independent_labels.md | af50f53f | exact |
| H7 | 2b69e06c | PREREG/e2_name_surprisal.md | 2b69e06c | exact |
| H8 | b2b916ce | PREREG/e3_lens_split.md | b2b916ce | exact |
| H9 | 18c74883 | PREREG/e4_steering_v3.md | 18c74883 | exact |
| H10 | 7bed3826 | PREREG/e4_amendment_third_judge.md | 7bed3826 | exact |
| H11 | f35aa4d9 | PREREG/e5_third_judge.md | f35aa4d9 | exact |
| H12 | 937b46e3 | PREREG/e6_entity_knowledge.md | 937b46e3 | exact |
| H13 | da0ad000 | direction_reuse_v2_criterion.md | da0ad000 | exact |
| H14 | 797f52d4 | nohedge_direction_criterion.md | 797f52d4 | exact |
| H15 | post hoc list complete | hedge_judge_criterion.md (ae0bb807) not listed | omission | mismatch (minor 13) |
| H16 | E4 amendment written before judging | note says 2026-10-10 ~22:10 UTC; hash log 2026-10-09T22:02Z; judges done 22:47Z | internal date wrong | mismatch (minor 14) |
| H17 | four-family design confirmatory, verdicts reported | FA verdicts not in paper | omission | mismatch (C1) |
