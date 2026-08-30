# Results CSV Schema (v4)

Ablation results are written by `shared/run_ablation_pipeline.py` (`run_category`) /
`scripts/run_ablation.py` as `<out_dir>/results_<category>.csv`, one row per neuron x prompt,
with uncertain prompts AND matched controls in the same file (`is_control` tells them apart).
Column order is the contract (`shared.ablation.RESULT_COLUMNS`); the merge scripts depend on it.

| Column | Type | Description |
|---|---|---|
| `neuron_id` | string | e.g. `"L20_N1083"` (layer 20, neuron 1083) |
| `layer` | int | Layer index |
| `neuron_idx` | int | Neuron index within layer |
| `category` | string | `"ambiguity"` / `"lack_of_knowledge"` / `"contradictory_context"` |
| `prompt_id` | string | Matches `prompt_id` in the data file (`prompts.jsonl` or `controls.jsonl`) |
| `orig_entropy` | float | fp32 next-token entropy (nats) before ablation |
| `ablated_entropy` | float | Entropy after mean-ablation (last position clamped to `mean_val`) |
| `entropy_shift` | float | `ablated_entropy - orig_entropy` |
| `orig_top1_prob` | float | Top-1 token probability before ablation |
| `ablated_top1_prob` | float | Top-1 token probability after ablation |
| `direct_effect_score` | float | Logit lens: max over vocab of the absolute direct logit of the gamma-folded output weight (fp32, weights-only, constant per neuron) |
| `split` | string | `"working"` or `"held_out"` -- for control prompts too (never `"control"`) |
| `is_control` | bool | `True` for matched-control prompts (from `controls.jsonl`), `False` for uncertain prompts |
| `orig_activation` | float | The neuron's last-token activation on the unablated pass of this prompt |
| `mean_val` | float | The value the neuron was clamped to (identical for all rows of a neuron within a category run) |
| `mean_source` | string | `"general_baseline"` (shared 60-prompt baseline, `shared/baselines.py`), `"category_working"` (category's own working-split uncertain prompts), or `"pooled_controls"` (working-split uncertain + control prompts) |
| `precision` | string | `"bf16"` (unquantized) or `"nf4"` (4-bit); `"fp32"`/`"fp16"` if loaded that way |

Example row:
```
L20_N1083,20,1083,ambiguity,amb_0001,2.31,1.98,-0.33,0.41,0.52,0.02,working,False,1.734,0.212,general_baseline,bf16
```

## Compatibility with older files

Pre-v4 files (`person_*/results/results.csv`, `control_results*.csv`, `results/full_precision/*`)
have only the first 12 columns and encode controls as `split == "control"`. They lack
`is_control`, `orig_activation`, `mean_val`, `mean_source`, `precision`; when merging, treat missing
columns as unknown (do not back-fill `precision` -- the old runs were nf4 unless documented
otherwise) and map `split == "control"` to `is_control = True`.

## Sibling artifacts

Every artifact-writing function writes `<artifact stem>.provenance.json` next to its output
(`shared/provenance.py`): model_id, precision, quant config, dtype, transformers/torch versions,
candidate file sha256, data file sha256s, baseline prompt sha256 + count, seed, layer_range,
top_k_per_half, top_k_final, min_abs_corr, git HEAD sha, timestamp, plus function-specific fields.
`run_category` also writes `<out_dir>/ablation_means.json` keyed by category
(`{category: {mean_source, baseline_prompt_sha256, n_baseline_prompts, precision, means: {neuron_id: mean_val}}}`).

Other scripts' CSVs (`scripts/frozen_norm.py`, `scripts/dose_response.py`,
`scripts/induction_check.py`) carry their own documented columns (see each script's docstring);
they share the `neuron_id/layer/neuron_idx/category/prompt_id/split/is_control` identifiers.

Do not rename columns or change types without telling the team.
