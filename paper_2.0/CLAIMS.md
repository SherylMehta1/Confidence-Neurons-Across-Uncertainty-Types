# Claims ledger

<!-- One row per claim. Never delete a row; change its status and log why in LAB_LOG.md.
     Status: established | supported | equivocal | retired | exploratory (references/hypothesis.md).
     Paths are relative to the data repo (Confidence-Neurons-Across-Uncertainty-Types) unless they start with paper/.
     Type: existence | systematic | hedged | narrow. "Falsifier" = the result that would make us retract or narrow it. -->

**Last updated 2026-10-10**, after review round 3 (chair: ACCEPT with camera-ready conditions) and the pre-registered experiments E1-E10. Rows C1-C11 existed before round 3; their status and caveats were revised, never deleted. Rows C12-C17 are new. "Changed" says what moved this round.

| id | claim (plain words, direction, scope) | status | changed in round 3 | evidence | locked in advance? | caveat | where in paper |
|---|---|---|---|---|---|---|---|
| C1 | In Llama-3.1-8B-Instruct, saying "I don't know" tracks entity popularity more than knowledge: obscure known 0.43 [0.35, 0.51] vs popular known 0.00 at similar entropy (0.95 vs 0.86 nats); popularity +0.39 vs knowledge +0.09, difference **+0.30 [+0.20, +0.40]** | **supported** | Strengthened: holds under independent labels (E1, +0.32), consensus labels (+0.31), semantic regrading of unknown facts (E7a, +0.23), a third judge (E5, +0.44), strict declines naming no candidate (E7c, +0.43), an entity-recognition covariate (E6, 5% shrink) and a namesake covariate (post hoc, +0.34). Disclosed: the difference equals obscure-known minus popular-unknown abstention | results/hedge_measure_v2/summary.txt; results/round3/e1, e5, e6, e7 summaries; paper/analysis/round3_label_checks.txt, ambiguity_check.txt | yes (hedge_measure_v2_criterion.md 81acfc9f; E1 af50f53f, E5 f35aa4d9, E6 937b46e3, E7 1498be91) | No human validation of the judges (two-judge kappa 0.62 [0.57, 0.67]). 40 of 100 popular-unknown facts were not unknown by meaning; the result survives relabelling | Abstract, 3.1, Fig. 1, App. A |
| C2 | Answer uncertainty (first-token entropy under "The answer is") follows knowledge more than popularity in all four families (Llama 2.14 vs 0.51 nats) | **established** in Llama; supported across families | Narrowed wording: "as the labelling implies" across families; in Llama it holds under independent labels (E1: knowledge minus popularity +1.40). The ">= 4x" ratio was dropped (no interval; undefined for Qwen) | results/factorial_4models/effects.csv; results/round3/e1_summary.txt | yes (qwen_factorial eba35dec, gemma_mistral_factorial b53d9a27, E1) | Partly by construction outside Llama, because labels and entropy share a distribution | Abstract, 3.1, 3.2, Fig. 1c |
| C3 | The popularity effect is not explained by name spelling or by answers guessable from the name | exploratory (post hoc) | Both name-form results now in 3.1; the matched-pair loss (+0.76 to +0.43) is reconciled with the small regression shift (pairs compare known facts only) | results/unireps_checks/; paper/analysis/guessability.csv | no | Name form explains about 40% of the matched-pair gap | 3.1, App. A |
| C4 | Steering along the obscure-minus-popular direction makes Llama say "I don't know" about popular facts it knows, far more than the knowledge direction (old: 0.02 to 0.29) | **retired** (see R6) | The held-out-dose rerun (E4) gives 0.01 to 0.10, +0.09 over random, short of the +0.10 bar; the earlier number was inflated by choosing the dose on the evaluation questions and scoring with a regex | results/round3/e4_summary.txt | E4 18c74883 | Replaced by C14 | 3.3, App. C |
| C5 | In Gemma-2-9B-it, steering doubly dissociates: popularity direction raises abstention, knowledge direction raises entropy | equivocal | Entropy half holds (unknown-minus-known raises entropy +0.66 [+0.58, +0.74] nats more); the abstention half is scored by two judges who disagree on Gemma (majority-of-3: +0.11) | results/round3/e4_summary.txt | E4 | Gemma's two-judge kappa is below the 0.60 bar | 3.3, App. C |
| C6 | Base models' lens-decoded hedge readout responds to popularity (mid-depth, four base models, three families) | supported, narrowed | The popularity effect sits in the hedge-token term in 4/4 (E3), but knowledge moves that term at least as much in each, so it is not a preference for popularity; on the full readout popularity exceeds knowledge in 3/4 because the answer-token term differs | results/round3/e3_summary.txt, e3_window_effects.csv | jlens_cascade c8e04372; E3 b2b916ce | Separately fitted lenses; forced-answer position only | 3.3, App. D, Fig. 3 (appendix) |
| C7 | The hedge readout forms about 12 layers before the entropy gap (Llama familiarity: L16 vs L28) | supported | Unchanged; moved to App. G | results/singlebos/circuit_familiarity/jlens_summary.txt | no | Two readouts of one contrast, Llama only | App. G |
| C8 | Entropy-correlated neurons are word-frequency and temperature neurons; none is specific to uncertainty | established | Unchanged | RESULTS.md claims 1-2 | partly | Earlier work | App. G |
| C9 | A 20-head + 100-neuron set carries the first-token hedge readout (0.83 [0.78, 0.87] popular-to-obscure; 1.03 [0.99, 1.07] obscure-to-popular); removing it abolishes abstention (0.98) but injecting it works only 0.40 | equivocal (behaviourally) | Both patch directions now named; criterion hash corrected to the committed note (3e1932a9) | results/circuit_familiarity_v3/ | l5_behavioral 3e1932a9 | The first-token score transfers even when generation does not change | App. G |
| C10 | Instruct models' familiarity signal is 3-4x their base models' | **retired** (see R7) | E10 compared base and instruct on the same readout: instruct models lean further towards knowledge, and the pre-set rule for "instruction tuning increases the popularity preference" was met for no pair | results/round3/e10_summary.txt | E10 6af179ce | Lenses differ between the two models of a pair | App. D |
| C11 | Llama's forced-answer hedge readout agrees with other families' beyond shared knowledge labels | exploratory (post hoc) | Now summarised in 3.2: rule met for 1 of 3 pairs (Mistral) | paper/analysis/cross_model_agreement.txt | plan written before the run, estimates seen | Gemma n = 60 | 3.2, App. E |
| C12 | When the prompt permits "I don't know", Gemma-2-9B-it's abstention also tracks popularity more than knowledge: +0.30 [+0.19, +0.40], Fleiss kappa 0.67, obscure known 0.41 vs popular known 0.00 | **supported** | New (E9). Replaces "Gemma's preference cannot be estimated" for the permissive prompt only; under the plain prompt that statement stands | results/round3/e9_summary.txt | E9 f6459163 | One prompt; judges, not humans | Abstract, 3.2, App. B |
| C13 | Under the same permissive prompt, Llama's preference strengthens: +0.47 [+0.35, +0.59], kappa 0.71 | supported | New (E9) | results/round3/e9_summary.txt | E9 | As C12 | 3.2, App. B |
| C14 | Steering along obscure-minus-popular adds abstention only through its overlap with an abstention direction, even when that direction is fitted on obscure questions alone (Llama: cosine 0.91; +0.01 [+0.00, +0.03] left) | supported (negative) | New (E8; verdict CARRIED BY ABSTENTION). Answers the chair's objection that the earlier projection was built in by construction | results/round3/e8_summary.txt | E8 05474954 | First E8 run had swapped doses and is discarded (LAB_LOG) | 3.3, App. C |
| C15 | Under the plain prompt, Qwen, Gemma and Mistral rarely say "I don't know" (at most 0.11 for Gemma in any cell) and the judges disagree, so their preference cannot be estimated | supported | New wording (chair item 1); replaces "Gemma shows no popularity preference" | results/hedge_measure_v2/summary.txt; results/round3/e5_summary.txt | E5 | The pre-registered four-family test on the forced-answer hedge readout returned Qwen PARTIAL, Gemma REPLICATES, Mistral PARTIAL, but that readout shows no preference in Llama either | 3.2, App. B |
| C16 | Name surprisal under the base model does not stand in for popularity (Spearman with log page views -0.001) | supported (negative) | Both pre-registered outcomes now reported; the regex coefficient's drop when popularity is added is unresolved | results/round3/e2_summary.txt; paper/analysis/surprisal_check.txt | E2 2b69e06c | Short common-word names | 3.1, App. F |
| C17 | Neither model-side measure shows the obscure names to be unfamiliar to Llama (surprisal null; recognition 0.83 obscure known vs 0.75 popular known), so the result has the shape the misfire idea predicts but does not establish its mechanism | supported | New framing (chair item 5): the paper no longer presents itself as a test the misfire idea passed | results/round3/e2, e6 summaries | E2, E6 | Recognition is graded by an LLM | Intro, 3.1 |

## Open items

- Human validation of the judges: **not done** by [user] decision (2026-10-09). Limitations say so.
- Title scope: still "In Llama". C12 shows Gemma follows popularity under the permissive prompt; whether to widen the title is an author decision.
- Exploratory OLMo-2-7B-Instruct (not in the paper): judges disagree (Fleiss kappa 0.45), so unmeasured under the plain prompt; the permissive-prompt run is the natural next step (paper/analysis/explore_olmo.txt).

## Retired claims

| id | date retired | bar | observed | how the paper reports it |
|---|---|---|---|---|
| R1 "decision vs spread" as headline | 2026-10-02 | L5 behavioural: flip >= 0.50 and >= 0.20 over null, both directions | injection 0.398 [0.301, 0.495] -> EQUIVOCAL | App. G sentence |
| R2 familiarity decoded before knowledge in depth | 2026-10-06 | >= 5 of 8 SUPPORTS, <= 1 CONTRADICTS | 1 / 1 / 6 -> not supported (mixed) | App. D |
| R3 instruction tuning reuses the base familiarity direction | 2026-10-07 | v1: >= 3 of 5 REUSED; v2: both REUSED | v1 0 of 5; v2 NOT REUSED x2 | not in the extended abstract |
| R4 the abstention dissociation holds for "language models" | 2026-10-06 / 07 | Gemma and Mistral REPLICATE; judges kappa >= 0.60 | unmeasured outside Llama under the plain prompt | title Llama-scoped; C12 adds Gemma under the permissive prompt |
| R5 "Qwen, Gemma, Mistral are confabulators" | 2026-10-07 | corrected rate < 0.25 | not licensed | "rarely say they do not know" |
| R6 strong steering claim (C4) | 2026-10-10 | E4: D1 >= +0.10 over random, CI > 0, dose chosen on held-out questions | +0.09 [+0.06, +0.12] -> NOT MET | 3.3: "short of the pre-set bar"; earlier figure flagged as overstated in App. C |
| R7 instruct familiarity 3-4x base (C10) | 2026-10-10 | E10: instruct difference > base, non-overlapping CIs | met for no pair; instruct leans towards knowledge | App. D |
| R8 "Gemma shows no popularity preference" (round-2 abstract) | 2026-10-10 | chair round 3, K1: a near-floor null cannot show absence | withdrawn | replaced by C15 (plain prompt) and C12 (permissive prompt) |
| R9 "the instruct preference is not seen in base models" (round-2 3.3) | 2026-10-10 | chair round 3, K2: full readout shows popularity > knowledge in 3 of 4 base models | withdrawn | 3.3 now describes both panels (C6) |
| R10 "only through its abstention component" (round-2 abstract) | 2026-10-10 | chair round 3, K4: the projection was built in by construction | withdrawn, then re-tested | C14 (E8) supports the reading with a within-level direction |

## Per-claim evidence audit (mode evidence, 2026-10-09; historical, written before round 3, kept for the record)

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
