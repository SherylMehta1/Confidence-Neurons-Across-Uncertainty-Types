# Confidence Neurons Across Uncertainty Types

## Abstract

Stolfo et al. (2024) found neurons in GPT-2 that regulate output entropy without promoting any
specific token. We test whether the same kind of neuron exists in Llama-3.1-8B-Instruct, and whether
the *same* neurons regulate entropy regardless of what's causing the uncertainty — ambiguity, missing
parametric knowledge, or contradictory context. We build matched uncertain/control prompt sets for
all three, detect candidate neurons via split-half-validated activation-entropy correlation, and
causally test them with mean-ablation under FDR-corrected statistics.

## Hypotheses

- **H1 — Shared mechanism**: common neurons causally regulate entropy across all three uncertainty types.
- **H2 — Specialized mechanisms**: each type has its own distinct, non-overlapping neuron population.
- **H3 — Partial overlap**: some neurons shared across types, others category-specific.
- **H4 — No robust mechanism detected**: neither shared nor category-specific causal effects survive rigorous testing.

## Headline Result (bf16 rerun, current)

Full pipeline re-run unquantized (bf16 weights, fp32 entropy) with matched control prompts, a
stratified split-half detection over uncertain + control prompts, held-out evaluation, and two
ablation references (`scripts/run_all.sh`; analysis in `results/rerun_analysis/SUMMARY.md`,
`results/rerun_analysis_keyset_pooled/`). Numbers below are from the committed result files.

| Question | Result |
|---|---|
| Do the stimuli induce uncertainty? (`results/induction_check.csv`) | Only lack-of-knowledge: entropy 2.85 vs 1.24 on controls (*d* = 2.0). Ambiguity 2.09 vs 2.34 (n.s.); contradictory context 0.92 vs 1.25 — **wrong direction**. The three-way comparison is not executable with these stimuli. |
| Detection (`results/candidate_neurons_bf16.json`) | 27 split-half-stable candidates (\|r\| 0.50–0.71, 23 of 27 in L28–31); 0 overlap with the original 15, 4 with the 17 below. All 55 unique neurons across the three sets were carried through every test. |
| Uncertain-vs-control interaction, sign-flip / label permutation, BH-FDR α = 0.01 (`results/ablation_bf16_*`) | **One neuron survives: L31_N11541 on lack-of-knowledge** — uncertain −0.029 nats vs control +0.001, interaction *p* ≈ 1e-4, paired *d*z −0.66, held-out *p* ≤ 2e-3, activation slope *p* = 2e-10; robust to the in-distribution `pooled_controls` reference (*p* = 4e-5, *d*z −0.57). L31_N6772 marginal; L20_N5595 was a reference artifact. No control neuron is significant on the interaction or held-out tests. |
| L31_N2477 (Phase-2 survivor) | A general entropy regulator: monotonic dose-response (ρ = 0.86), 94–109% RMSNorm-mediated, but mean-ablation shifts confident controls as much (interaction *p* 0.2–0.7). |
| Stolfo weight criteria (`results/stolfo_*_summary.txt`) | 0 of 55 pass under the norm-matched null (one, L29_N6625, passes the looser random-null rule). High null-space fraction, low weight norm throughout. |
| Mechanism of L31_N11541 (`results/frozen_norm_bf16_keyneurons.csv`) | 39% norm-mediated, no frequency effect — a direct-logit neuron, not a temperature or frequency mechanism. |
| Token-frequency neurons (`results/token_frequency_neurons*`, `results/frequency_causal_v2_*`) | 9 of 55 candidates are frequency neurons by weights (≥ 99th pct; ≈ 0.6 expected). Causally, against a temperature-matched baseline, each neuron's pull toward frequent tokens is stable across all three prompt sets (*r* = 0.52–0.62) and 6× that of random neurons; weight scores predict magnitude (*r* = 0.34) but the sign in only 5 of 9. |
| Behavior (`results/behavioral_bf16*.csv`) | Model hedges on 77% of unknown-entity prompts vs 0.8% of controls; clamping L31_N11541 (mean, ±2σ) changes the hedge rate by ≤ 2 points. Entropy regulation and abstention are dissociable. |

**Reading:** activation–entropy correlation in this model selects token-frequency neurons and a general
temperature neuron, plus one direct-logit neuron that is uncertainty-specific but does not reach behavior.
No single neuron should be called a "confidence neuron", and no cross-category claim is supported.

## Original v3 analysis (NF4, superseded — kept for the record)


17 candidate neurons detected, split-half-stable (min-half \|r\| 0.63–0.78, layers 20–31). Causal
ablation testing against those candidates:

| Test | Result |
|---|---|
| Per (neuron × category) significance, FDR-corrected (F1) | 0 / 51 survive (pooled n=120); 0 / 51 on held-out only (n=36) |
| Candidate vs. matched control, two-sided studentized permutation on signed means, FDR-corrected (F5) | 1 / 51 survives (L31_N3330, lack-of-knowledge; exploratory — controls were ablated to the *uncertain*-prompt mean and have a different baseline entropy, so this is a conservative, confounded comparison) |
| Paired effect size, Cohen's *d*z = mean(shift)/SD(shift) | −0.21 to +0.22 (pooled); −0.31 to +0.26 (held-out). The previously reported ≈ 0.001–0.007 divided the mean shift by the SD of *baseline entropy* (a between-prompt spread), not the SD of the paired shifts, and understated the effect by ~30×. |
| Mixed model per neuron, prompt random intercept (ICC 0.34–0.41, all fits converged), FDR-corrected (F4) | 0 / 51 survive (smallest *p* = 0.037) |
| Equivalence (TOST, SESOI *d*z = 0.2 / 0.1), FDR-corrected (F2) | 0 / 51 equivalent to zero at either bound — the data can neither confirm nor rule out small effects |

Induction quality (uncertain vs. control mean top-1 probability):

| Category | Uncertain | Control | Gap |
|---|---|---|---|
| Lack of knowledge | 0.285 | 0.753 | 0.468 |
| Ambiguity | 0.391 | 0.533 | 0.142 |
| Contradictory context | 0.850 | 0.804 | −0.046 |

Cross-category correlation of per-neuron effect sizes:

| Category pair | *r* | 95% bootstrap CI | permutation *p* |
|---|---|---|---|
| Ambiguity — Contradictory context | 0.457 | [−0.02, 0.76] | 0.068 |
| Ambiguity — Lack of knowledge | −0.450 | [−0.85, 0.10] | 0.072 |
| Contradictory context — Lack of knowledge | −0.146 | [−0.62, 0.35] | 0.579 |

None significant after BH correction within the 3-pair family (F3, α = 0.01); every CI spans zero
(n = 17 neurons), and the negative ambiguity–lack-of-knowledge correlation trends apart rather than
together. On held-out prompts only, all three |*r*| ≤ 0.33 (*p* ≥ 0.19). A prompt-level permutation
test of the neuron × category interaction (`results/test_h1_interaction.py`) is also null:
F(32, 5712) = 1.13, *p* = 0.29, partial η² = 0.006. Read together, these three tables point to **H4**: detection finds real, stable
correlations (table 1's premise), but neither individual causal effects nor cross-category agreement
survive (tables 1 and 3) — and contradictory context never even establishes a clean uncertain-vs-control
contrast to test against (table 2).

## Experiments

- **Detection**: correlate MLP activations (layers 20–31) with entropy on a pooled 252-prompt
  baseline, split-half validated (neuron must be top-60 by \|r\| in *both* independently-computed
  halves) — 17 of ~172,000 scanned survive.
- **Causal test**: mean-ablate each candidate, measure entropy shift on the full prompt set and on
  matched controls.
- **Statistics**: sign-flip permutation + bootstrap CI per neuron/category, candidate-vs-control
  permutation test, cross-category Pearson correlation. Benjamini–Hochberg FDR at α = 0.01 is
  applied **separately within each of five families** (never pooled across them): F1 per-cell
  shifts (51 tests; pooled and held-out runs are separate families), F2 TOST equivalence (51 per
  SESOI), F3 cross-category correlations (3), F4 mixed-model per-neuron coefficients (51;
  `entropy_shift ~ 0 + C(neuron_id)` with a prompt random intercept), F5 candidate-vs-control (51).
- **Power / equivalence**: with n = 120 prompts per cell the minimum detectable paired effect at 80%
  power is *d*z ≈ 0.26 (α = 0.05) or ≈ 0.43 (α = 0.01/51); in entropy units, median MDE ≈ 0.005
  nats (α = 0.05) / ≈ 0.009 nats (α = 0.01/51), vs. observed |mean shift| ≤ 0.004 nats. On the 36
  held-out prompts the MDE is *d*z ≈ 0.48 / 0.84. TOST at SESOI *d*z = 0.2 and 0.1 declares 0 / 51
  cells equivalent to zero, so the nulls above are underpowered non-detections, not evidence of
  absence: effects up to *d*z ≈ 0.2 remain compatible with the data (`results/power_mde.csv`).
- **Induction check**: uncertain-set mean top-1 prob. vs. control — 0.285 vs. 0.753
  (lack-of-knowledge), 0.391 vs. 0.533 (ambiguity), 0.850 vs. 0.804 (contradictory context, no real
  separation).

## Data

| Category | Uncertain prompts | Matched controls | n |
|---|---|---|---|
| Ambiguity | AmbigQA questions, ≥2 distinct annotated answers | AmbigQA `singleAnswer` questions | 120 + 120 |
| Lack of knowledge | UnknownBench cloze questions, fabricated entities (9 verified templates) | Same templates, real entities | 120 + 120 |
| Contradictory context | `"Redefine: {fact}. {continuation}"`, false object, subject pre-verified as known | Same construction, true object, matched | 120 + 120 |

Each set: 84 working / 36 held-out. Model: `meta-llama/Llama-3.1-8B-Instruct` (or the ungated mirror via
`CN_MODEL_ID=unsloth/Meta-Llama-3.1-8B-Instruct`). **Precision:** the original v3 analysis above used the NF4 4-bit
model; the current headline results are from the bf16 rerun, and every result file carries a `.provenance.json`
sidecar recording precision, model id, library versions, and data hashes.

## Repo Structure

```
shared/                       model_utils (load_model, tokenize_prompt), prompt_format, detection,
                               ablation, logit_lens, baselines, provenance, schema_utils,
                               run_ablation_pipeline (old_detection / run_phase34 are deprecated shims)
scripts/                      one entry point per stage, bf16 by default, precision recorded in outputs:
                               detect.py, run_ablation.py, frozen_norm.py, dose_response.py,
                               mechanism_check.py, induction_check.py, behavioral_test.py,
                               frequency_causal.py, analyze_rerun.py; run_all.sh = full bf16 rerun
data/<category>/              prompts.jsonl, controls.jsonl (+ review / knows-fact logs, provenance)
person_A_ambiguity/           preprocess_ambiguity.py           <- the single live data builder for A
person_B_lack_of_knowledge/   nec_templates.py, screen_templates.py,
                               rebuild_lack_of_knowledge_whitelisted.py   <- live builders for B
person_C_contradictory_context/ preprocess_contradictory_context.py, LIMITATIONS.md  <- live builder for C
  each also holds results/results_v3.csv, control_results_v3.csv
analysis/stolfo_criteria.py   weight-based candidate identification (independent of activations)
analysis/token_frequency_neurons.py   unigram-frequency (Stolfo "token frequency") neuron check
candidate_neurons.json        validated candidates + detection provenance
full_correlation_distribution.json
results/                      stats_significance.py, mixed_model_stats.py, merge_and_analyze.py,
                               test_h1_interaction.py + output CSVs
archive/                      superseded material, never run at HEAD: pipelines/ (old data builders,
                               see archive/PIPELINES_README.md), notebooks/ (the marimo run record,
                               see archive/notebooks/README.md), old results (archive/README.md)
PHASE3_GUIDE.md               the run order: data -> induction check -> detect -> ablation -> stats
RESULTS_SCHEMA.md  DATA_SOURCES.md  SETUP.md  REFERENCES.md
```

Entry points: `python person_A_ambiguity/preprocess_ambiguity.py`,
`python person_B_lack_of_knowledge/screen_templates.py` then `rebuild_lack_of_knowledge_whitelisted.py`,
`python person_C_contradictory_context/preprocess_contradictory_context.py`, then the `scripts/`
pipeline in the order given in `PHASE3_GUIDE.md`.

## Next Steps

**Before submission:**
- Second model (Llama-3.1-8B base; Qwen2.5-7B-Instruct): `CN_MODEL_ID=... bash scripts/run_all.sh` plus
  `scripts/behavioral_test.py`, `analysis/token_frequency_neurons.py`, `scripts/frequency_causal.py`.
- Seed sweep of the split-half partition (`scripts/detect.py --seed N`) to report candidate-set stability.
- Multi-neuron behavioral test (all frequency neurons together; L31_N11541 + L31_N6772) — the single-neuron
  behavioral null is the weakest-powered result.
- Redesign the ambiguity and contradictory-context stimuli until they pass `scripts/induction_check.py`,
  or scope the paper to lack-of-knowledge.

**Done (this rerun):** bf16 re-run of all candidate sets with matched controls and held-out evaluation;
Stolfo criteria on all sets with a norm-matched null; frozen-RMSNorm decomposition; dose-response;
behavioral test; token-frequency neurons (weights and temperature-matched causal test on all categories).

## References

Full list in `REFERENCES.md`. Core: Stolfo, Belinkov & Sachan (2024); Gurnee et al. (2024); Min et
al. (2020, AmbigQA); Liu et al. (2024, UnknownBench); arXiv:2509.10663.

