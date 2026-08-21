# archive/

Files moved out of the live tree because they were computed on a **superseded
candidate set or a superseded precision/baseline** and would otherwise be
mistaken for current results. Nothing here is used by any live script
(`results/*.py` read only `person_*/results/results_v3.csv` and
`control_results_v3.csv`).

Current candidate set: the 17 neurons in `candidate_neurons.json`
(split-half validated, 252-prompt pooled baseline). Every per-neuron file
below was verified with pandas to contain **0 of the 17** current candidates
(the only exception is `stolfo_criteria.csv`, whose 1000-neuron random null
draw happens to include L30_N10312; its `candidate` group is the old 15).

The superseded "old 15" set (`L23_N12156, L24_N7891, L26_N11322, L26_N2788,
L29_N10092, L29_N10191, L29_N11308, L29_N8568, L29_N9228, L30_N13513,
L30_N3533, L30_N5509, L30_N6621, L30_N7102, L31_N2477`) came from the
earlier, non-split-half detection run.

## Per-person result files (`archive/person_*/results/`)

| File | Computed on | Why archived |
|---|---|---|
| `results.csv` (1800 rows = 15 x 120) | old 15 neurons, NF4 4-bit model, mean-ablation, working/held_out split | superseded by `results_v3.csv` (17 neurons) |
| `results_bf16_unquantized.csv` (A/B: 4 neurons x 120; C: 1 neuron x 120) | old-15 subset re-run with the **bf16 unquantized** model | Phase-2 bf16 held-out replication evidence -- see below |
| `significance_summary.csv`, `spread_summary.csv`, `working_vs_heldout.csv` (15 rows) | per-person summaries of `results.csv` (old 15, NF4) | old set; per-person ad-hoc tests (Wilcoxon / t) replaced by `results/stats_significance.py` |
| `control_summary.csv`, `control_significance_summary.csv` (5 rows) | 5 random **control neurons** (a different axis from the matched control *prompts* now in `control_results_v3.csv`) | old era; not comparable to the current design |
| `frozen_norm_L31_N2477.csv` (84 working prompts) | old-15 neuron L31_N2477 ablated with the final RMSNorm scale frozen | see corruption note below |

### Phase-2 bf16 held-out replication (`results_bf16_unquantized.csv`)

This is the only bf16 (unquantized) ablation evidence in the repo. It covers
4 old-15 neurons for ambiguity and lack-of-knowledge and 1 for contradictory
context, each on all 120 prompts (84 working / 36 held_out). It is kept
because the v3 numbers (17 neurons) were all produced with the NF4 4-bit
model and a bf16 re-run of the 17 is still required; this file shows what a
bf16 run of the pipeline looked like and its mean |shift| (~0.015) is of the
same order as the NF4 values (~0.010-0.013). It must not be merged with
`results_v3.csv` -- different neurons, different precision.

### `frozen_norm_L31_N2477.csv` -- corruption note

* **person_C's copy is corrupted**: the `frozen_norm_ablated_entropy`
  column was produced with bf16-quantized ablated entropies (the original
  entropy is fp32, the ablated one is not), so `shift_under_frozen_norm` is
  dominated by quantization rounding and is not interpretable.
* **person_A's and person_B's copies are numerically valid**, but the
  frozen-norm intervention clamped **all positions** (not only the final
  token), so they answer a different question than the per-position
  ablation in `results*.csv`. Treat them as exploratory only.

## `archive/results/`

| File | What it was | Why archived |
|---|---|---|
| `mechanism_check_shared.json` | logit-lens top tokens / direct-effect scores for the **old 15** neurons | old set; re-run `shared/logit_lens.py` on the 17 |
| `old_mechanism_check_shared.json` | an even earlier version of the same check (old 15) | superseded twice |
| `test_l29_category_difference.py` | ad-hoc Welch t-tests for two old-15 neurons (L29_N8568, L26_N2788) on `results.csv` | targets neurons no longer in the candidate set; uncorrected, post-hoc |
| `test_merge.py` | Pearson r + PCA + average-rank on `merged_summary.csv` | superseded by `results/stats_significance.py` (permutation p, bootstrap CI, BH within the 3-pair family) |

## `archive/analysis/`

| File | What it was | Why archived |
|---|---|---|
| `overlap_analysis.py`, `overlap_result.txt` | hard-coded "significant" neuron sets per category from the old 15 (raw-threshold, no FDR) and their pairwise overlaps | the input sets were never FDR-corrected and no longer exist in the candidate list |

## `archive/stolfo_criteria.csv`

Output of the weight-based Stolfo criteria (output-weight norm, unembedding
null-space fraction, LogitVar) for the **old 15** candidates vs a
1000-neuron random null. It must be re-run on the 17 current candidates
before any weight-based claim is made.
