# Archived marimo notebooks (historical run record -- do not run)

`A_marimo_nb.py`, `B_marimo_nb.py`, `C_marimo_nb.py` (and their `_cleaned` variants) are the
Kaggle/RunPod marimo sessions in which Persons A, B and C actually produced the committed
artifacts. They are kept verbatim as the provenance record for those artifacts. They are NOT
runnable at HEAD and must not be used to regenerate anything:

- `from shared.detection import load_candidate_neurons` raises ImportError (the function no longer exists).
- Cells that index `candidate_neurons.json` raise KeyError on the current candidate JSON layout.
- Notebook A's data-building cell (`exec(open(".../preprocess_ambiguity_v2.py"))` and the cell that
  rewrites `REVIEW_LOG_PATH` / `data/ambiguity/prompts.jsonl`) would OVERWRITE the committed ambiguity data.
- All three contain `git add` / `git commit` / `git push` cells (e.g. A lines ~1600-1635) that push
  from inside the notebook.
- They reference file names that no longer exist (`preprocess_*_v2.py`, `preprocessing_ambiguity_2/`,
  `shared/rescue_untruncated_candidates.py`).

## Which cells produced which artifacts, and at which precision

| Artifact (now under `archive/` or `results/`) | Notebook cell(s) | Model precision |
|---|---|---|
| Initial candidate detection (`candidate_neurons.json`, `full_correlation_distribution.json`) | B: detection + `rescue_untruncated_candidates` cells (~1500-1640) | **NF4** (`load_model(quantize=True)`) |
| v3 ablations: `person_*/results/results_v3.csv`, `control_results_v3.csv` | A/B/C: `run_category` cells (B ~1740) | **NF4** |
| Mechanism check: `results/mechanism_check_shared.json` | C ~281 / ~963 | **NF4** |
| C frozen-norm: `person_C_.../results/frozen_norm_L31_N2477.csv` | C ~621-708 (`load_model_fnc(quantize=False)` was used for A/B only; C's first run was NF4) | **NF4** |
| Phase-2 bf16 reruns on the OLD 15 candidates: `results_bf16_unquantized.csv` | A ~848, B ~386, C ~376 (`quantize=False`) | bf16 |
| Held-out replication: `working_vs_heldout.csv` | A ~540/~992, B ~508, C ~498 | bf16 |
| A/B frozen-norm: `frozen_norm_L31_N2477.csv` (A, B) | A ~1137, B ~798 | bf16 |
| Stolfo criteria: `stolfo_criteria.csv` | B ~630-734 | bf16 (weights only) |
| Data builds: `data/*/prompts.jsonl`, `controls.jsonl` | A ~805/~1568 (ambiguity), B ~1067/~1122 (screen + rebuild), C ~1240/~1453 | tokenizer only, except C's knows-fact filter (NF4) and B's template screen (bf16) |

In short: detection, the v3 ablations, the mechanism check and C's frozen-norm run were produced
under NF4; the Phase-2 bf16 reruns / held-out / A-B frozen-norm / Stolfo runs were bf16 but on the
old 15-candidate list. Every NF4-produced number has to be reproduced in bf16 before it is reported.

## Replacements

Use the scripts under `scripts/` (one stage each, bf16 by default, precision recorded in every output):
`detect.py`, `run_ablation.py`, `frozen_norm.py`, `dose_response.py`, `mechanism_check.py`,
`induction_check.py`. Data regeneration lives in the per-person `preprocess_*.py` entry points
(see `PHASE3_GUIDE.md`).
