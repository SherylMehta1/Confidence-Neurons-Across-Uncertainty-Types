# Claims ledger

<!-- One row per claim. Never delete a row; change its status and log why in LAB_LOG.md.
     Status: established | supported | equivocal | retired | exploratory (references/hypothesis.md).
     Paths are relative to the repo C:/Users/starb/Desktop/Sher/Confidence-Neurons-Across-Uncertainty-Types.
     Type: existence | systematic | hedged | narrow. "Falsifier" = the result that would make us retract or narrow it. -->

| id | claim (plain words, direction, scope) | status | type | evidence file(s) | number source | locked in advance? | falsifier | caveat | where in paper |
|---|---|---|---|---|---|---|---|---|---|
| C1 | In Llama-3.1-8B-Instruct, whether the model says "I don't know" depends more on how familiar the entity is than on whether it knows the fact: popularity effect +0.39 vs knowledge effect +0.09, difference **+0.30 [+0.20, +0.40]** | **supported** | narrow (one model, PopQA entity questions) | results/hedge_measure_v2/summary.txt, item_rates.csv | summary.txt "llama … pop-know +0.300 [+0.204,+0.398]" | yes: hedge_measure_v2_criterion.md (81acfc9f, 7 Oct; hash recorded, not yet committed) | Human labels disagree with the both-judges rule (κ < 0.6), or the difference CI includes 0 | Judge κ 0.62 vs a 0.60 bar; no human validation yet. Same sign under 4 instruments (+0.13 to +0.44). Knowledge also has a small effect (+0.09, CI > 0). Hedging is measured in free generation; entropy at the prefill. | Abstract, Result 1, Fig. 1 |
| C2 | Answer uncertainty (first-token entropy after "The answer is") follows knowledge, not familiarity, in all four families: knowledge effect ≥ 4× popularity (Llama 2.14 vs 0.51; Mistral 1.41 vs 0.30; Gemma 1.08 vs 0.09; Qwen 1.23 vs −0.02) | **established** | systematic (4 families, 7-9B instruct) | results/factorial_4models/summary.txt, effects.csv | effects.csv rows readout=entropy | yes: qwen_factorial_criterion.md, gemma_mistral_factorial_criterion.md (condition 2) | Popularity effect ≥ knowledge effect in a family | First token only; no detector involved; independently audited (7 Oct) | Abstract, Result 1, Fig. 1 |
| C3 | The familiarity effect on Llama's hedging is not explained by how the name is spelled or by answers guessable from the name | exploratory (post hoc) | narrow | results/unireps_checks/FINDINGS.md, summary.txt | pool logit −1.87 → −1.79; headline contrast +0.76 → +0.43; non-inferable +0.79 | no | Surface form absorbs the popularity coefficient | Spelling explains ~40% of the 55-pair contrast | Result 1 (one sentence) |
| C4 | Steering along the obscure-vs-popular direction makes the instruct model say "I don't know" about popular facts it knows, much more than the knowledge direction at equal norm: Llama 0.02 → 0.29 (knowledge direction 0.09; Δ +0.20 [+0.14, +0.27]); Gemma-2-9B-it 0 → 0.22 (Δ +0.22 [+0.16, +0.29]) | **supported** | narrow (2 models) | results/direction_reuse_v2/*/summary.txt | summary.txt S and DD lines | yes: direction_reuse_v2_criterion.md (da0ad000) | Effect ≤ random directions | **Qualified:** a familiarity direction fitted only on never-hedged items gives +0.04 [+0.01, +0.07] (results/direction_nohedge/), so the effect is carried mostly by hedge-associated components; Gemma needed 4× dose. Word as controllability, not use (concurrent che2026steering: steering ≠ the model using the feature) | Result 2, Fig. 2 |
| C5 | In Gemma-2-9B-it the intervention double dissociation holds: familiarity direction → hedging (+0.22), knowledge direction → entropy (+0.68 [+0.59, +0.77]) | supported | existence (one model) | results/direction_reuse_v2/gemma9b/summary.txt | DD line | yes (v2 criterion) | — | One model; in Llama only the hedging half holds | Result 2 |
| C6 | A familiarity signal is present in the lens-decoded "can't answer" readout of base models (Llama-3.1-8B, Gemma-2-9B, Gemma-2-2B, OLMo-3-7B), mid-depth, all CIs > 0 | **supported** | systematic (3 families, 4 base models) | results/jlens_cascade/summary.txt, outcomes.csv; FINDINGS.md exploratory section | outcomes.csv midF, midF_lo | yes: jlens_cascade_criterion.md claim 2 (c8e04372) | A base model's mid-depth familiarity CI includes 0 | Observational; separately fitted lenses; survives a second readout and spelling controls (post hoc) | Result 3, Fig. 3 |
| C7 | The hedge readout forms about 12 layers before the entropy gap (Llama familiarity: half-formed at L16 vs L28) | supported | narrow | results/singlebos/circuit_familiarity/jlens_summary.txt | "first layer reaching 50%" line | no (from earlier work) | — | Reproduced with a single BOS; Llama familiarity only | Result 3 or appendix |
| C8 | Neurons selected by correlation with entropy are word-frequency and temperature neurons; none is specific to uncertainty on the arm where the model hedges | established | systematic (2 models) | RESULTS.md claims 1-2; results/frequency_causal_*, results/gated_specificity_summary.txt | RESULTS.md | partly | — | Earlier work; one sentence in the extended abstract | Intro / appendix |
| C9 | A 20-head + 100-neuron set carries the hedge readout (first-token log-odds recovery 0.83 [0.78, 0.87] vs random 0.02-0.04); in generation, removing its state abolishes hedging (0.98) but injecting it works only 0.40 [0.30, 0.50] (all positions; 0.398 [0.301, 0.495]) | equivocal (behaviourally) | narrow | results/circuit_familiarity_v3/faithfulness_limit141*, behavioral*; numbers.tex `\circuitHedgeCUVThreeFull` | numbers.tex | criterion fixed in code (circuit); L5 behavioural bar locked (59f5ec5) | — | The first-token score transfers even when generation does not change | Appendix |
| C10 | Instruct models' familiarity signal is 3-4× their base models' (Llama −0.89 vs −0.28; Gemma −0.39 vs −0.14) | exploratory (post hoc) | hedged | results/jlens_cascade/FINDINGS.md exploratory section | table | no | — | Different lenses per model | Discussion (one clause) |
| C11 | Llama's forced-answer hedge readout agrees with each other family's on questions where only one model knows the fact, more than their entropies agree | exploratory (post hoc; plan PREREG/cross_model_agreement.md written before this run, estimates seen before) | narrow | paper/analysis/cross_model_agreement.csv/.txt | macros cm* | plan fixed before run, but post hoc | pre-set rule (hedge rho > 0 and Delta > 0, CIs excl. 0) not met | Rule met for 1 of 3 pairs (Mistral); Delta reliable for Qwen and Mistral; Gemma n = 60 | Conclusion (one clause), App. E |

## Open items

- Human validation of the two judges: **not done** by [user] decision (2026-10-09). C1 stays supported; the limitations say so. The kit remains in `results/hedge_measure_v2/` for later.
- Hedging (C1) and entropy (C2) are measured on different prompts (free generation vs forced-answer prefill). Decided 2026-10-09 [user]: state and justify this in Methods (PROJECT_CONTEXT.md, Methods notes for write); no new experiment.

## Retired claims

| id | date retired | bar | observed | how the paper reports it |
|---|---|---|---|---|
| R1 "decision vs spread" as headline | 2026-10-02 | L5 behavioural: flip ≥ 0.50 and ≥ 0.20 over null, both directions | injection 0.398 [0.301, 0.495] → EQUIVOCAL | Appendix sentence; motivates the current framing |
| R2 familiarity decoded before knowledge in depth | 2026-10-06 | ≥ 5 of 8 SUPPORTS, ≤ 1 CONTRADICTS | 1 / 1 / 6 → mixed; order depends on the readout | "A prediction from human memory research that failed" (one sentence) |
| R3 instruction tuning reuses the base familiarity direction | 2026-10-07 | v1: ≥ 3 of 5 REUSED; v2: both REUSED | v1 0 of 5 (NOT TESTABLE); v2 NOT REUSED ×2 | Limitations or appendix |
| R4 the hedging dissociation holds for "language models" | 2026-10-06 / 07 | both Gemma and Mistral REPLICATE; judges κ ≥ 0.60 | Mistral PARTIAL; hedging unmeasured for Qwen, Gemma, Mistral (κ 0.30-0.47) | Scope sentence; title Llama-scoped |
| R5 "Qwen, Gemma, Mistral are confabulators" | 2026-10-07 | judge audit: confabulator label needs a corrected rate < 0.25 | not licensed (regex recall 0.10-0.14) | Replace with "rarely say they do not know" |

## Per-claim evidence audit (mode evidence, 2026-10-09)

**C1 (Llama hedging: familiarity > knowledge)**
- **Discriminates between:** hedging driven by cue familiarity vs by retrieval failure. Under the rival, the knowledge effect would be ≥ the popularity effect; observed +0.09 vs +0.39.
- **Strongest boring explanation:** the detector, the judges or truncation create the gap. Ruled out as far as it can be: the same sign under 4 instruments (regex 24 tok \faLlamaRegexDiff, judge 24 tok \hjLlamaDiff, two judges 96 tok \hmLlamaDiff, regex 96 tok \hmLlamaRegexDiff), and regex recall 0.99 on full replies.
- **Second boring explanation:** name spelling or guessable answers. Post hoc checks: the pool coefficient barely moves (\scPoolPop → \scPoolPopSurface).
- **Reliability risks:** κ 0.62 vs a 0.60 bar; 387 items × 3 samples; one model.
- **Decisive follow-up:** human labels of 120 replies (kit ready).

**C2 (entropy follows knowledge, 4 families)**
- **Discriminates:** answer uncertainty driven by knowledge vs by familiarity.
- **Boring explanation:** the prefill makes unknown items produce long-tail first tokens regardless of the model's knowledge. Unlikely: the knowledge labels come from the same prompt, and the effect holds within popularity strata.
- **Reliability:** 4 families, 2,282-2,392 items each; independent audit passed.
- **Decisive follow-up:** none needed for the extended abstract.

**C6 (familiarity signal in base models)**
- **Discriminates:** familiarity installed by instruction tuning vs present after pretraining.
- **Boring explanation:** lens artefacts or name spelling. The signal survives a second readout and spelling controls (post hoc).
- **Reliability:** separately fitted lenses per model, so magnitudes are not comparable across models; observational.
- **Decisive follow-up:** a causal base-model steering test (not run).

## Drift list (numbers in write-ups vs macros)
- C9 injection: 0.39 in this file → 0.40 (macro \ciInjectFlip). Fixed above.
- No-hedge cosine: results/direction_nohedge/FINDINGS.md says 0.61; the macro 
hCos is 0.60 (raw 0.605). The paper uses the macro.
- Old paper familiarity recovery 0.80 / 1.07 → current 0.83 [0.78, 0.87] (\ciRecovery). The old draft is not reused.

## Changes in revise round 1 (2026-10-10; review/round1, revision/round1)

- **C1:** unchanged status (supported). Added a post hoc guessability control: difference \gsDiff \gsDiffCI without guessable questions. Disclosed that at the forced-answer position Llama's hedge readout follows popularity and knowledge about equally (\loLlamaDiff \loLlamaDiffCI). Title now scoped to Llama ([user] 1b).
- **C2:** status unchanged (established as a measured pattern). Wording narrowed: knowledge labels and entropy share the forced-answer distribution, so the pattern is expected by construction. The paper presents entropy as the reference readout ([user] 2a).
- **C3:** still exploratory. "Rules out spelling" replaced by the coefficient change; the matched-pair attenuation is reported in App. A.
- **C4:** status unchanged. Worded as "can produce abstention", with a pointer to Ferrando/Brzezinka as prior steering. The never-abstained control is marked Llama-only. The dose rule is stated (chosen on the evaluation questions; conceded in Limitations).
- **C5:** "double dissociation" replaced by "entropy moved the other way" in the main text; App. C calls it a dissociation of steering effects at 4x norm.
- **C6:** wording narrowed. The base-model hedge readout responds to popularity and to knowledge (mid-window K reported). It no longer says "the familiarity signal predates instruction tuning" without qualification.
- **R2 (failed cascade):** the per-model onsets table is now in App. D; "emerge together" was removed (numbers audit N2).
- **C11:** added (exploratory).

## Changes after round-3 experiments (2026-10-10; PREREG e1-e5)

- **C1** (Llama abstention: popularity > knowledge): supported, strengthened.
  - Survives independent labels (E1 T1, \eoDiff \eoDiffCI).
  - Three-judge majority κ 0.74 (E5, \tjLlamaDiff).
  - Guessability control holds; one-prompt check: first-token "I" log-odds follow popularity.
- **C2** (entropy follows knowledge, 4 families): established. No longer circular: holds with paraphrase labels (E1 T2).
- **C4** (steering): **downgraded to equivocal/negative.**
  - E4 D1 not met with held-out dose and judges (Llama +0.09, Gemma +0.04).
  - D2: the abstention-orthogonal part ≈ 0 under both scorers. Reported as a negative result.
- **C5** (Gemma double dissociation): the entropy half survives (D4 +0.66); the abstention half is not met.
- **C6** (base models): narrowed. The popularity effect sits in the hedge term in 4/4 (E3 L1), but the knowledge effect on that term is about as large; Llama-Instruct's popularity preference is not shown in base models.
- **New C12:** Gemma-2-9B abstention is measurable with three judges (κ 0.64) and shows no popularity preference (E5); Qwen and Mistral remain unmeasured.
- **New negative:** E2 name surprisal does not track page views (ρ −0.001) and does not predict abstention.
