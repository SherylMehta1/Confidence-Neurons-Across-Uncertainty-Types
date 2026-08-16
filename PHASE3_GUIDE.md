# Phase 3 Implementation Guide

How to apply these files to your repo, in order. Every file here is written to slot into your existing structure (`shared/`, `person_X_.../`, `results/`) — copy them in at the matching paths and they import your existing, unmodified `shared/model_utils.py`, `shared/ablation.py`, `shared/logit_lens.py`, `shared/schema_utils.py` directly. Nothing here changes Phase 1's fixes to those files.

I wrote and syntax-checked all of this, but I don't have GPU access or your gated Llama-3.1 weights, so none of it has been run against the real model. Treat every script the way your own `SETUP.md` treats `00_hello_world.ipynb`: run it, look at the printed sanity output, confirm it looks right, *then* trust it.

## Why this order

Data and detection changes feed everything downstream, so they have to land before any new Phase 3/4 numbers mean anything. Detection specifically has to happen *after* the position fix, because the split-half baseline prompts should be built from the corrected (prefilled) format — running detection on old-format prompts and then measuring effects on new-format prompts would make the two halves of the pipeline inconsistent with each other.

## Step 0 — add the new shared files

Copy in as new files (don't touch the existing Phase 1/2 files, these are additions):
```
shared/prompt_format.py
shared/detection_v2.py
shared/run_phase34.py
results/mixed_model_stats.py
```
Add to `requirements.txt`: `statsmodels>=0.14.0`

## Step 1 — fix induction position + build controls (all three, can run in parallel)

**Person B** (Kaggle/RunPod, tokenizer is enough, model optional but recommended for the verification step at the end):
```bash
python person_B_lack_of_knowledge/preprocess_lack_of_knowledge_v2.py
```
Produces `data/lack_of_knowledge/prompts.jsonl` (position-fixed) and `data/lack_of_knowledge/controls.jsonl` (matched answerable twins). With model loaded, also run:
```python
from shared.prompt_format import verify_induction_quality
verify_induction_quality(model, tokenizer, records[:25])       # expect LOW mean top1
verify_induction_quality(model, tokenizer, control_records[:25])  # expect HIGHER top1
```
If the two distributions look similar, don't proceed — the factoid filter needs another pass first.

**Person A** — Stage 1/2 (automated filters) can run standalone; Stage 3 is still a manual review step, unavoidable, but now produces a *committed* artifact:
```bash
python person_A_ambiguity/preprocess_ambiguity_v2.py   # runs Stage 1-2, prints candidate count
# -- do the manual review by hand against the criteria already documented
#    in preprocess_ambiguity.py's Stage 3 docstring --
# write each decision as one line to approval_tracker.jsonl:
#   {"raw_prompt": "...", "decision": "approved"}
#   {"raw_prompt": "...", "decision": "rejected"}
git add person_A_ambiguity/approval_tracker.jsonl   # commit this. this is the fix for
                                                     # "ambiguity data contradicts its own
                                                     # review" -- without this file committed,
                                                     # the 120-item set isn't reproducible.
```
Then, in Python (or uncomment the bottom of the script):
```python
from person_A_ambiguity.preprocess_ambiguity_v2 import finalize_from_approval_log_v2, build_ambiguity_controls
finalize_from_approval_log_v2()   # writes data/ambiguity/prompts.jsonl, position-fixed
build_ambiguity_controls()        # writes data/ambiguity/controls.jsonl, no review needed
```
This is also where you settle the open question from our last exchange — confirm this run, not the pre-audit script, is what produced the data before treating any ambiguity Phase 2/3 result as real.

**Person C** (needs model+tokenizer loaded, same convention as your existing `build_dataset.py`):
```python
exec(open("person_C_contradictory_context/preprocess_contradictory_context_v2.py").read())
records, control_records = build_and_save(model, tokenizer, verbose_knows_fact=True)
```
Pilot with `verbose_knows_fact=True` on a small run first (temporarily lower `n_target` to ~20) and read the printed `top1=... vs true_object first token=...` lines — confirm the match logic is doing what you expect before trusting it at scale, same as the project's convention for every other new filter. `model_knows_fact()` now requires the model's actual top-1 token to match `target_true`'s first token, not just a confidence threshold (see the corrected file — the first draft I sent only checked confidence and never compared token identity, which didn't actually implement "top token = true object"). `knows_fact_min_prob` is available as an optional secondary confidence floor on top of the match, off by default.

Produces `data/contradictory_context/prompts.jsonl` (now model-knows-filtered) and `data/contradictory_context/controls.jsonl` (true-object, subject/relation-matched to the working set 1:1 — `cc_ctrl_0007` and `cc_0007` are always the same subject).

Also worth doing here, once model+tokenizer are loaded anyway: sanity-check whether contradictory-context actually has the same turn-boundary issue as A/B. It wasn't flagged as broken before, and the entropy values in the current `results.csv` don't show the classic peaked/near-zero signature — but it hasn't been directly checked either:
```python
from shared.prompt_format import verify_induction_quality
verify_induction_quality(model, tokenizer, records[:25])
```
If top1 comes back mostly < 0.95, leave C's prompt structure as-is. If not, apply the same `build_completion_prompt` fix used for A/B (Redefine prompts already end mid-sentence, so this is a small change if it turns out to be needed).

**Once all three are done**, commit `data/*/prompts.jsonl`, `data/*/controls.jsonl`, and `person_A_ambiguity/approval_tracker.jsonl` to GitHub. This closes out the "clean ambiguity data," "fix lack-of-knowledge position," "restore model-knows filter," and "matched control prompts" rows of the plan in one pass.

## Step 2 — redo detection (shared, do together, on the NEW data)

Build a baseline pool from the *fixed* working-split prompts across all three categories (recommend ≥300 total — check `candidate_neurons.json`'s existing `provenance.n_baseline_prompts` first; if the old one was much smaller, that's worth noting in the writeup as part of why detection needed redoing):

```python
import json
baseline_prompts = []
for cat_file in ["data/ambiguity/prompts.jsonl", "data/lack_of_knowledge/prompts.jsonl",
                  "data/contradictory_context/prompts.jsonl"]:
    with open(cat_file) as f:
        baseline_prompts += [json.loads(l)["chat_formatted_prompt"] for l in f
                              if json.loads(l)["split"] == "working"]

from shared.detection_v2 import detect_candidate_neurons_split_half, save_candidate_neurons_v2
candidates, full_dist = detect_candidate_neurons_split_half(
    model, tokenizer, baseline_prompts, layer_range=range(20, 32),  # match whatever range Phase 2 used
)
save_candidate_neurons_v2(candidates, full_dist, baseline_prompts, seed=42)
```
This **overwrites `candidate_neurons.json`** — it's meant to. The Phase 2 candidate set was frozen from an unvalidated top-15; this is its replacement. Commit the new `candidate_neurons.json` and the new `full_correlation_distribution.json` together. If the split-half-stable set comes back smaller than 15 (very possible — that's the point of the check), don't pad it back up; a smaller, validated list is the correct, more honest outcome.

## Step 3 — re-run Phase 3 (mechanism check) + Phase 4 (ablation) on the new candidates + new data + controls

`shared/logit_lens.py` and `shared/ablation.py` are unchanged — only the *inputs* changed (new candidates, new prompts, new controls). One call per person:

```python
from shared.run_phase34 import run_category

run_category(
    model, tokenizer, category="lack_of_knowledge",
    prompts_path="data/lack_of_knowledge/prompts.jsonl",
    controls_path="data/lack_of_knowledge/controls.jsonl",
    candidates_path="candidate_neurons.json",
    out_results_path="person_B_lack_of_knowledge/results/results_v3.csv",
    out_control_results_path="person_B_lack_of_knowledge/results/control_results_v3.csv",
)
# repeat for ambiguity (person_A) and contradictory_context (person_C),
# swapping the category/prompts_path/controls_path/out_*_path arguments
```
Commit all six output CSVs. These are the files `results/mixed_model_stats.py` reads.

## Step 4 — upgraded statistics (shared, last)

```bash
python results/mixed_model_stats.py
```
Writes `results/mixed_model_results.csv` (mixed-model sanity check), `results/effect_sizes.csv`, and `results/candidate_vs_control.csv` (the actual candidate-vs-matched-control significance test, FDR-corrected). Also re-run your existing `results/stats_significance.py` against the new `results_v3.csv` files (update its `RESULT_FILES` dict to point at `*_v3.csv`) and `results/merge_and_analyze.py` for the cross-category correlation — **this time actually regenerate and recommit `merged_summary.csv` / `cross_category_correlation.csv`**, since the audit found the currently-committed versions are stale relative to even the Phase 2 data, let alone Phase 3's.

## "Done when" checklist (from your own plan, made concrete)

- [ ] `data/*/controls.jsonl` exists for all three categories, committed
- [ ] `person_A_ambiguity/approval_tracker.jsonl` exists, committed
- [ ] `verify_induction_quality` shows candidate prompts clearly lower top1 than their matched controls, for A and B at minimum
- [ ] `candidate_neurons.json` regenerated via split-half validation, committed alongside `full_correlation_distribution.json`
- [ ] `results_v3.csv` + `control_results_v3.csv` exist for all three people, committed
- [ ] `results/candidate_vs_control.csv` shows which (neuron, category) pairs have a real effect *beyond* what the same neuron does on low-uncertainty matched prompts
- [ ] `results/merged_summary.csv` and `results/cross_category_correlation.csv` regenerated from `results_v3.csv`, not the stale Phase-1/2 versions
- [ ] Every script in this list is committed (this guide's whole point is making sure nothing here becomes another "results-generation notebook that was never pushed")
