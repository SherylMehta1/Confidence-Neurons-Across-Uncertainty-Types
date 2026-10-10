# Project explainer: when does a language model say "I don't know"?

*As of 7 Oct 2026. Written for a newcomer. The 5-7 Oct results are not yet committed to git, so the paths below point to untracked files in the repo `Confidence-Neurons-Across-Uncertainty-Types`.*

## 1. The question in one paragraph

When a chatbot answers "I don't know", the common assumption is that it is reporting uncertainty: its internal probabilities over possible answers are spread out (high entropy), and that spread becomes words. We tested this on factual questions from PopQA. We varied two things separately: **how famous the question's subject is**, and **whether the model actually knows the answer** (checked by sampling its answers).

In Llama-3.1-8B-Instruct, whether the model says "I don't know" depends mostly on how famous the name is. Its answer uncertainty depends mostly on whether it knows the fact. Three other model families show the same pattern for uncertainty but rarely say "I don't know" at all. The signal that a name is unfamiliar is already decodable in base models, before instruction tuning.

This matters for two reasons. Confidence methods that read the output distribution may be measuring a different variable from the one that drives the "I don't know". And the pattern matches a 30-year-old finding about people: our sense that we can answer follows how familiar the question's terms feel, not whether the answer is in memory.

## 2. Glossary

| Term | Plain meaning | Where in the data |
|---|---|---|
| hedging | the model's reply says it does not know or cannot verify | `hedge_rate` in `data/*/all_measured.jsonl`; judges in `results/hedge_measure_v2/` |
| popularity / familiarity | Wikipedia pageviews of the subject (PopQA `s_pop`); obscure = bottom 10%, popular = top 10% within a relation | `s_pop`, `arm` (uncertain = obscure, control = popular) |
| known / unknown | the model gives the right answer in at least 9 of 10 samples, greedy included (HighlyKnown), or in 0 of 10 (Unknown) | `slick` |
| answer uncertainty (entropy) | how spread out the model's next-word probabilities are right after "The answer is" | `entropy` |
| hedge log-odds | first-token preference for hedge words (" not", " I", " unknown", …) over generic answer words | `data/*/hedge_logodds.jsonl` |
| factorial | the 2 × 2 of obscure/popular × known/unknown | `data/factorial_v3*` |
| criterion file | a pass/fail rule written and hashed before a run | `*_criterion.md` at the repo root |
| Jacobian lens | a prefitted decoder that shows which output words a layer's activations push toward | `results/jlens_cascade/` |
| steering | adding a direction to the model's activations and watching its behaviour change | `results/direction_reuse_v2/` |
| circuit | 20 attention heads + 100 MLP neurons found by activation patching (older work) | `results/circuit_*` |

## 3. The story arc

| Phase | Dates | Question asked then | Answer | What changed it |
|---|---|---|---|---|
| Confidence neurons | 6 Aug - mid Aug | Do entropy-regulating neurons exist and recur across uncertainty types? | Entropy-correlated neurons are word-frequency and "temperature" neurons; none is specific to uncertainty | Neuron-level nulls |
| Circuit and "decision vs spread" | mid Aug - 1 Sep | Is hedging a decision carried by a circuit, separate from entropy? | A 20-head + 100-neuron set carries 78-107% of the hedge readout; random sets carry 2-4% | Submitted to InterpScience, 1 Sep |
| Rejection and audit | 29 Sep - 4 Oct | Does the circuit change behaviour, not just a first-token score? | Removing the circuit's state abolishes hedging (0.98), but injecting it works only 0.39 of the time: EQUIVOCAL under a locked bar | Rejection (jargon, "circuit is post hoc", overclaiming); L5 behavioural test (e1ef8dc) |
| Familiarity vs knowledge | 19 Sep (data); 5-7 Oct (UniReps) | Does "I don't know" follow familiarity or knowledge? Across families? Before instruction tuning? | Llama: familiarity (+0.30). Entropy follows knowledge in 4 families. Familiarity signal present in base models. | Lit review for UniReps; detector and judge audits forced re-measurement |

## 4. Every claim

| Claim (plain words) | Status | Evidence | Locked in advance? | Caveat |
|---|---|---|---|---|
| Llama's "I don't know" follows name familiarity more than knowledge | **supported** | `results/hedge_measure_v2/summary.txt`, Llama: +0.30 [+0.20, +0.40] | yes (`hedge_measure_v2_criterion.md`) | judge κ 0.62, just over the 0.60 bar; no human validation yet |
| Answer uncertainty follows knowledge, not familiarity, in Llama, Qwen, Gemma, Mistral | **established** | `results/factorial_4models/summary.txt`, condition 2 in 4 of 4 | yes (qwen/gemma_mistral criteria) | prefilled first token only |
| The familiarity effect is not about how the name is spelled | supported | `results/unireps_checks/` check 1 (Llama +0.76 → +0.43 with spelling controls on the headline contrast; pool coefficient unchanged) | no (post hoc control) | spelling explains part of one contrast |
| Steering the obscure-vs-popular direction makes Llama and Gemma say "I don't know" about facts they know; the knowledge direction does much less | supported | `results/direction_reuse_v2/` (Llama 0.02 → 0.29; Gemma 0 → 0.22) | yes (v2 criterion) | mostly carried by hedge-associated components: a no-hedge familiarity direction gives only +0.04 (`results/direction_nohedge/`) |
| In Gemma-2-9B, familiarity steering moves hedging and knowledge steering moves entropy (double dissociation) | supported (one model) | `results/direction_reuse_v2/gemma9b/summary.txt` | yes | needed 4× dose |
| A familiarity signal is decodable in base models (Llama, Gemma, OLMo) | **supported** | `results/jlens_cascade/` (criterion claim 2; survives spelling controls) | yes (`jlens_cascade_criterion.md`) | observational, separately fitted lenses |
| Familiarity is decoded before knowledge in depth | **retired** | `results/jlens_cascade/` "mixed" (1 / 1 / 6) | yes | readout-dependent |
| Instruction tuning reuses the base model's familiarity direction | **retired** | `results/direction_reuse*/` | yes | base direction fitted on a different prompt format |
| The pattern holds for "language models" in general | **retired** | Qwen and Mistral PARTIAL; hedging unmeasured for 3 of 4 | yes | they rarely say "I don't know" |
| Entropy-correlated "confidence neurons" are frequency/temperature machinery | established | RESULTS.md claims 1-2 | partly | older work |
| A 20-head + 100-neuron set carries the hedge readout | established for first-token log-odds; behaviour only for removal | RESULTS.md claim 5; `results/circuit_familiarity_v3/behavioral*` | criterion fixed in code | the first-token score transfers even when generation does not change |
| The lens timing: hedge half-formed by L16, entropy by L28 | established | `results/singlebos/FINDINGS.md` | no | Llama familiarity only |

## 5. Pre-registered versus post hoc

| Criterion | Where, when | Strength of lock | Outcome |
|---|---|---|---|
| Gate thresholds for twins | UNCERTAINTY_DEFINITION.md, before gating (Aug) | stated in prose, before running | — |
| Hedger-vs-confabulator falsification test | preregistration_commit.md, c034557 (1 Oct) | immutable commit | passed on a knife edge (0.703 vs 0.7) |
| L5 behavioural validation | l5_behavioral_criterion.md, 59f5ec5 (2 Oct) | immutable commit | EQUIVOCAL |
| Qwen factorial | qwen_factorial_criterion.md, sha eba35dec, 6 Oct 07:26 UTC | hash recorded before the run; untracked | PARTIAL |
| Gemma / Mistral factorials and cross-model claims | gemma_mistral_factorial_criterion.md, b53d9a27, 6 Oct 09:45 | hash before the run; untracked | Gemma REPLICATES (on log-odds), Mistral PARTIAL |
| Lens cascade | jlens_cascade_criterion.md, c8e04372, 6 Oct 10:05 | hash before the run; untracked | mixed; base-model claim supported |
| Direction reuse v1 / v2 | 3281f804 / da0ad000 | hash before the run | v1 NOT TESTABLE ×5; v2 NOT REUSED ×2 |
| Judge audit; two-judge measurement | ae0bb807; 81acfc9f | hash before the run | regex inadequate for 3 models; Llama FAMILIARITY > KNOWLEDGE |
| No-hedge steering control | 797f52d4 | hash before the run | HEDGING INTENT NOT EXCLUDED |

**Post hoc** (label as such in the paper): the name-spelling and guessable-answer checks; the hedge-only lens readout; the detector-gap sensitivity; the "injection fails when the target is known" pattern; instruct models' familiarity signal being 3-4× their base models'.

**Caveat on the locks:** the 5-7 Oct criteria are hashed in session logs and on disk but **not yet committed**. Commit them first so git shows their timing.

## 6. Knife edges

- **Llama two-judge κ = 0.62** (bar 0.60): the headline's measurement validity clears by 0.02.
- **Hedger-vs-confabulator test 0.703** vs a 0.7 bar.
- **L5 injection 0.398 [0.301, 0.495]** vs a 0.50 floor. With the hedge-detector gap fixed it is 0.481 [0.380, 0.574], whose CI includes 0.50.
- **Qwen's pool-level hedge rate 0.051** vs the 0.05 floor that decided which readout governed.
- **Gemma's REPLICATES flips to PARTIAL** under the corrected detector (pool rate 0.036 → 0.059).
- **Gemma-2-9B base onset order:** SUPPORTS by one layer.

## 7. Decision-status map

| Decision | Status | What happened | Recorded |
|---|---|---|---|
| Venue: UniReps extended abstract | decided | user, 5 Oct | PROJECT_CONTEXT.md |
| Title scoped to Llama | decided by criterion | Mistral PARTIAL | factorial_4models/FINDINGS.md |
| Human validation of judges | open | kit prepared 7 Oct | results/hedge_measure_v2/HUMAN_VALIDATION_README.md |
| Commit 5-7 Oct work | open, asked repeatedly | — | — |
| Repo private during review | open | public repo shows results | litreview synthesis §7 |
| Wire numbers.tex into the paper | moved without a record (old paper); the new paper must use a numbers file | 0 of 239 macros used in the old main.tex | commit 6e235d9 |
| "LM-Polygraph baselines" promised in paper/LITREVIEW.md §8 for a main-track version | dropped by silence | never run | paper/LITREVIEW.md |
| Reasoning uncertainty (reviewer 14N6) | dropped; scope it out in the limitations | — | — |

## 8. What is running now

Nothing. All jobs are finished and copied to the repo (untracked). molab boxes four to eight are idle.

## 9. What is missing

- Plan documents cited by results: l5_behavioral_validation.md, factorial_circuit_level_test.md, uu_falsification_test.md, regenerate_final_numbers.md.
- The submitted InterpScience source (outside this workspace, `shshk/`).
- Human validation of the judges (not done).
- A numbers file for the new paper (to be created in `evidence`).

## 10. Where to look

| Topic | File |
|---|---|
| All 5-7 Oct results in one page | `results/OVERVIEW_2026-10-07.md` |
| Familiarity × knowledge (4 models) | `results/factorial_4models/FINDINGS.md` |
| Proper hedging measurement | `results/hedge_measure_v2/FINDINGS.md` |
| Lens across 8 models | `results/jlens_cascade/FINDINGS.md` |
| Steering | `results/direction_reuse_v2/FINDINGS.md`, `results/direction_nohedge/FINDINGS.md` |
| Confounds | `results/unireps_checks/FINDINGS.md` |
| Lens BOS bug | `results/jlens_bos_audit/FINDINGS.md`, `results/singlebos/FINDINGS.md` |
| Lit review | `../litreview/00_SYNTHESIS_UniReps.md` |
| Prior reviews | `paper/reviews_interpscience.md` |

## Worked example (copied from `data/factorial_v3_pop_given_known`, pair 0000, Llama-3.1-8B-Instruct)

| | Obscure, known | Popular, known |
|---|---|---|
| Question | "What is the capital of Belorechensky District?" | "What is the capital of People's Republic of China?" |
| Pageviews (`s_pop`) | 188 | 724,517 |
| Greedy answer after "The answer is" | "Belorechensk." (10/10 samples correct: known) | "Beijing." (10/10 correct: known) |
| First-token entropy | 1.206 | 0.051 |
| Hedge rate, 20 free generations (no prefill) | **1.0** | **0.0** |
| Free generation #1 | "I do not have information about the capital of Belorechensky District." | "The capital of the People's Republic of China is Beijing." |
| First-token hedge log-odds | +2.64 | −3.15 |

Llama knows both answers when made to answer, but says "I don't know" only for the unfamiliar name.

**Odd thing visible in this very first row:** "Belorechensk" can be read off the name "Belorechensky District". This is the *name-inferable answer* confound. It affects 35% of the obscure-known items in this contrast, and the checks in `results/unireps_checks/` show the hedging gap persists (+0.79) when those items are removed.
