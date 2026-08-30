# Uncertainty: definition, types, and dataset standard

This document fixes what "uncertainty" means in this project, which manipulations count as
uncertainty types, and what every dataset must satisfy before any neuron- or circuit-level
analysis uses it. It is written to be pre-registered: the thresholds below are set before the
data are gated, and keep rates are reported, not tuned.

## 1. Uncertainty is defined relative to the model, at three levels

| Level | Definition | Measurement | Not to be confused with |
|---|---|---|---|
| Distributional | the model's next-answer-token distribution is spread | first-token entropy after the assistant prefill (fp32 softmax); top-1 mass; top-2 gap | "the question is hard"; a confident refusal template has *low* entropy |
| Behavioral | the model acts uncertain | hedging / abstention rate in free generation (regex + judge); verbalized 0–100 confidence | paraphrase changes; format effects |
| Representational | the model internally registers "I do not know this" | a causal variable: a subspace whose interchange between twins flips the behavioral readout | a probe that merely correlates with correctness |

The word *confidence* is reserved for the behavioral level. No component earns it by moving
entropy alone.

Ground truth for "the model should be uncertain" follows the uncertainty-estimation literature
(Kadavath et al. 2022 P(IK); Gekhman et al. 2024 SliCK; Yang et al. 2025 CASAL): it is defined by
**sampled correctness under the final prompt**, never by annotation. Per item: greedy answer plus
N = 10 samples at T = 0.7, graded against gold aliases (exact match) with an LLM judge for ties.

| Class (SliCK) | Rule | Role here |
|---|---|---|
| HighlyKnown | greedy correct and >= 9/10 samples correct | control arm |
| MaybeKnown / WeaklyKnown | in between | excluded |
| Unknown | 0/10 correct | uncertain arm *candidate* |

Within the Unknown class, two sub-populations are recorded via `n_distinct` (number of distinct
sampled answers) and the greedy answer: *never heard of it* (dispersed guesses) and *cannot resolve
the reference* (e.g. "the author of *Heaven*" — the model says several works share the title). Both
are model-relative uncertainty; analyses may split them.

Three populations are kept separate (Orgad et al. 2025; HACK 2025; "Too consistent to detect"
2025): consistently correct (control), **dispersed / hedging** (the uncertain arm), and
**consistently wrong** (low entropy, no hedging) — the last is reported as a separate negative
control, because both entropy and probes are known to fail on it.

## 2. Types are manipulations, not labels

A type is admissible only if it has a twin rule: the uncertain prompt and its control differ in
exactly one thing, with the same answer-token format.

| Type | Manipulation (uncertain vs control) | Source | Answer format |
|---|---|---|---|
| A. Fabricated entity | non-existent entity vs real entity, same template | UnknownBench NEC (existing) | entity / short phrase |
| B. Obscure real entity (familiarity) | Unknown-class subject vs HighlyKnown subject, same relation template; popularity as the a-priori proxy | PopQA; EntityQuestions | entity |
| B'. Known subject, unknown attribute | same subject, relation the model cannot answer vs one it can | PopQA / EntityQuestions | entity |
| C. Contested knowledge | two disagreeing sources vs one source, same question; gated on the model's own memory answer | ConflictQA | entity |
| D. Context-dependent | bare question vs question + resolving clause (date / place) | SituatedQA | entity / year |
| E. Missing premise | problem with one number deleted vs original | UMWP | number (separate prefill regime) |
| F. Aleatoric (control) | many equally valid answers vs unique answer | hand-built | word / number |
| G. Non-existent vs real (cross-check of A) | fabricated vs real entity in identical templates | HalluLens NonExistentRefusal | entity |

Excluded by this rule: annotator-defined ambiguity (AmbigQA as shipped), single-passage
contradiction (CounterFact "Redefine", NQ-Swap), future / controversial / philosophical
known-unknowns (KUQ, SelfAware: hedging without distributional uncertainty), false-premise sets
(FalseQA: confident correction), multiple-choice sets (letter tokens), long-form sets.

## 3. Dataset standard (every type, every model)

1. **Model-relative gate, per twin pair, per model.** Keep a pair iff
   (a) control is HighlyKnown; (b) uncertain is Unknown; (c) first-token entropy(uncertain) −
   entropy(control) >= 0.5 nats; (d) hedging rate(uncertain) − hedging rate(control) >= 0.3 over
   3 free-generation samples. Report keep rates per type and per model, and how many
   correctness-defined pairs fail (c)/(d).
2. **Twin rule**: one change; question token-length difference <= 3 (else dropped; for context
   types the passages differ by design, so the rule applies to the question and the total length
   is recorded); identical chat template and prefill; the answer token type matched within pair.
3. **Size and splits**: >= 80 surviving pairs per type; 70 / 30 working / held-out assigned once,
   by seed; held-out never used for selection.
4. **Provenance**: source dataset and revision, source ids, popularity / relation fields, the
   sampled answers and grades, the gate values. Every file carries a provenance sidecar.
5. **Format controls reported**: prompt length, question word, answer type, per arm.
6. **Dissociation control present**: types F and C show that distributional and behavioral uncertainty
   can come apart (spread over valid answers / over sources, with no hedging); they are gated without the
   hedge rule and are never used to claim a neuron or circuit is "for uncertainty".
7. **UE benchmark tie-in**: the surviving pairs are run through LM-Polygraph (MSP, token entropy,
   semantic entropy, SAR, EigenScore, P(True), verbalized) and AUROC for correctness is reported
   alongside our two gate signals, on identical items.

## 4. What the standard buys

Neuron tests (entropy-adjusted interaction, dose-response, behavior) and circuit tests (patching,
EAP-IG, interchange interventions) consume the same twin files. The cross-type questions — does
one variable or one circuit carry several types? — are only meaningful when every type passed
the same gate in the same model. Qwen2.5-7B-Instruct gets its own gated sets; "uncertain" is a
property of a (model, prompt) pair, not of a prompt.

Note on arms without a gold answer on the uncertain side (contested, aleatoric): `uncertain_not_unknown` is
vacuously 0 there, so the gate in practice is control HighlyKnown + entropy gap + question length (+ hedging
gap where required). ConflictQA evidence passages contain a PopQA gold alias in 4992 / 7947 rows (62.8%); the
candidate generator keeps only those.

## 5. Gate outcomes so far (keep rates are reported, not tuned)

| Type | Model | Candidates | Kept | Entropy (U vs C, nats) | Hedge (U vs C) | Main failure |
|---|---|---|---|---|---|---|
| B familiarity (PopQA) | Llama-3.1-8B-Instruct | 1151 | 195 (136/59) | 3.69 vs 0.93 | 0.93 vs 0.01 | len 125, hedge 102, entropy 48 |
| B familiarity (PopQA) | Qwen2.5-7B-Instruct | 1151 | 24 | 2.19 vs 0.55 | 0.56 vs 0.00 | entropy 237, hedge 199: Qwen guesses confidently when it does not know |
| F aleatoric (dissociation control) | Llama | 120 | 41 | 1.52 vs 0.55 | 0.00 vs 0.00 | len 51 (no hedge gate) |
| F aleatoric (dissociation control) | Qwen | 120 | 39 | 1.07 vs 0.07 | 0.00 vs 0.00 | len 60 |
| D situated (SituatedQA) | Llama | 600 | 3 | 3.21 vs 1.68 over the 3 kept pairs; over all 600 candidates the gap is −0.20 nats (the dated question is *not* higher-entropy than the bare one) | - | control not HighlyKnown 519/600: the model does not know the 2021-vintage dated facts, so a resolving clause cannot make it confident. Type D is dropped for this model pair. |
| D situated (SituatedQA) | Qwen | 600 | 0 | - | - | control not HighlyKnown 554/600: same failure as Llama; type D dropped. |
| C contested (ConflictQA) | Llama | 600 | 240 (168/72), gated without the hedge rule; 234 after dropping 6 pairs whose ChatGPT parametric answer already matched the gold (originals in `pre_memory_guard/`) | 1.83 vs 0.48 | 0.01 vs 0.00 | first pass 0/600 was a pipeline bug (whole-prompt length rule; half of ConflictQA's evidence passages lack the answer). With the hedge rule only 6 pairs survive: the distribution spreads over the two sources (8/10 distinct sampled answers) but free generation commits to one source without hedging. Type C is therefore a second **dissociation type** (like F), not a behavioral-uncertainty arm. 302 controls remain not-HighlyKnown (the model often answers 'not stated in the source' even when it is). |
| C contested (ConflictQA) | Qwen | 600 | 184 (128/56), gated without the hedge rule; 178 after the same post-hoc filter | 1.30 vs 0.04 | 0.00 vs 0.00 | 343 HighlyKnown controls; 159 fail the entropy gap. Same dissociation as Llama: spread over sources, no hedging. |

## 6. Grader audit (2026-08-24)

A Qwen2.5-32B-Instruct judge re-graded a stratified sample (grader verdict x side x hedged) of the alias grader's
decisions (`scripts/judge_audit.py`, `results/judge_audit_*`): Llama familiarity 96.3% agreement (kappa 0.91,
n = 269), Llama contested 95.4% (kappa 0.70, n = 65), aleatoric 100%; Qwen familiarity 94.6% (kappa 0.85, n = 241),
Qwen contested 100%. Disagreement is lower on hedged answers (1.5%) than on non-hedged ones (6.1%), so grader
error does not track the behavioral readout. Errors were permissive whole-answer substring matches on elaborated
answers (7/81 grader positives) and paraphrases (3/188 grader negatives). The grader now matches aliases against the
answer head (text before the first sentence break / clause marker) with whole-word containment; on the audited
items this removes 2 of the 6 false positives on Llama familiarity and changes nothing elsewhere. Gated sets in
`data/` were built with the old rule; the difference is below 1% of graded answers.
