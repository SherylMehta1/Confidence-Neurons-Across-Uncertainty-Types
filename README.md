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

## Headline Result

17 candidate neurons detected, split-half-stable (min-half \|r\| 0.63–0.78, layers 20–31). Causal
ablation testing against those candidates:

| Test | Result |
|---|---|
| Per (neuron × category) significance, FDR-corrected | 0 / 51 survive |
| Candidate vs. matched control, FDR-corrected | 0 / 51 survive |
| Standardized effect sizes | ≈ 0.001–0.007 (below "small") |

Induction quality (uncertain vs. control mean top-1 probability):

| Category | Uncertain | Control | Gap |
|---|---|---|---|
| Lack of knowledge | 0.285 | 0.753 | 0.468 |
| Ambiguity | 0.391 | 0.533 | 0.142 |
| Contradictory context | 0.850 | 0.804 | −0.046 |

Cross-category correlation of per-neuron effect sizes:

| Category pair | *r* | *p* |
|---|---|---|
| Ambiguity — Contradictory context | 0.457 | 0.062 |
| Ambiguity — Lack of knowledge | −0.450 | 0.070 |
| Contradictory context — Lack of knowledge | −0.146 | 0.577 |

None significant, and the negative ambiguity–lack-of-knowledge correlation trends apart rather than
together. Read together, these three tables point to **H4**: detection finds real, stable
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
  permutation test, cross-category Pearson correlation, all jointly FDR-corrected.
- **Induction check**: uncertain-set mean top-1 prob. vs. control — 0.285 vs. 0.753
  (lack-of-knowledge), 0.391 vs. 0.533 (ambiguity), 0.850 vs. 0.804 (contradictory context, no real
  separation).

## Data

| Category | Uncertain prompts | Matched controls | n |
|---|---|---|---|
| Ambiguity | AmbigQA questions, ≥2 distinct annotated answers | AmbigQA `singleAnswer` questions | 120 + 120 |
| Lack of knowledge | UnknownBench cloze questions, fabricated entities (9 verified templates) | Same templates, real entities | 120 + 120 |
| Contradictory context | `"Redefine: {fact}. {continuation}"`, false object, subject pre-verified as known | Same construction, true object, matched | 120 + 120 |

Each set: 84 working / 36 held-out. Model: `meta-llama/Llama-3.1-8B-Instruct`, bf16, fp32 entropy.

## Repo Structure

```
shared/                      model_utils, prompt_format, detection, ablation, logit_lens,
                              schema_utils, run_phase34 (per-category causal-test runner)
data/<category>/              prompts.jsonl, controls.jsonl
analysis/stolfo_criteria.py   weight-based candidate identification (independent of activations)
candidate_neurons.json        17 validated candidates + detection provenance
full_correlation_distribution.json
person_A_ambiguity/  person_B_lack_of_knowledge/  person_C_contradictory_context/
  per-category preprocessing + results/results_v3.csv, control_results_v3.csv
results/                      mixed_model_stats.py, stats_significance.py, merge_and_analyze.py
                               + output CSVs (effect_sizes, candidate_vs_control, etc.)
RESULTS_SCHEMA.md  DATA_SOURCES.md  SETUP.md  REFERENCES.md
```

## Next Steps

**Ready to run:**
- Weight-based (Stolfo) check — `analysis/stolfo_criteria.py`, scores candidates vs. a random-neuron
  null on output-weight norm, unembedding null-space fraction, LogitVar. No forward passes needed.
- Reverse-nominated causal test — run `run_category()` on whatever Stolfo-criteria surfaces, to
  localize whether the null is a detection-method problem or a real absence of effect.

**Bigger lifts:**
- Joint ablation of the full candidate set together — checks for a distributed effect single-neuron
  ablation would miss.
- Behavioral link — does ablation change hedging/abstention in generated text, not just entropy?
- Base vs. instruct comparison, connecting to arXiv:2504.02904.

## References

Full list in `REFERENCES.md`. Core: Stolfo, Belinkov & Sachan (2024); Gurnee et al. (2024); Min et
al. (2020, AmbigQA); Liu et al. (2024, UnknownBench); arXiv:2509.10663.

