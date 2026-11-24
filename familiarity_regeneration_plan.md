# Post-Dedup Regeneration Plan — Familiarity Circuit Pipeline
### ACL Rolling Review, October 2026 cycle — "Confidence Neurons Across Uncertainty Types"

**Trigger:** L0's dedup fix changed the canonical familiarity twin dataset (14,267 → 13,068 raw PopQA rows deduped; candidate pool 1,151 → 1,124; pairs 181 → 207; working/held-out 126/55 → 144/63).
**Status as of writing:** only L0 (dataset regeneration) has been done. Nothing downstream has been rerun yet.
**This doc covers:** why a rerun is required at all, exactly which files are now stale, the order to regenerate them in, and how to preserve the old results for comparison rather than silently losing them.

---

## 1. Why this is required — not optional, not cosmetic

Every circuit-discovery script in this repo — head verification, neuron attribution, the rank-1 direction estimate, faithfulness/recovery measurement — was run against `data/familiarity/{prompts,controls}.jsonl` **as it existed before the dedup fix**. That's a specific, fixed set of 181 pairs (126 working / 55 held-out). Confirmed directly from `results/circuit_familiarity/direction_patch.provenance.json`: `"n_pairs": 59` for the held-out split used in circuit analysis — consistent with the old, pre-dedup numbers, not the new ones.

Replacing `data/familiarity/` with the deduped version doesn't just add or remove a few data points behind an already-fixed analysis — it changes:
- **Which items are available to discover a circuit from in the first place.** Head verification and neuron attribution are themselves fit *on* the twin pairs. A different (even if mostly overlapping) set of pairs can surface a different circuit — not necessarily the same heads/neurons with slightly different numbers, but potentially different heads/neurons entirely.
- **Which pairs the held-out evaluation set is measuring recovery against.** Even if the discovered circuit were identical, `faithfulness.csv` and `direction_patch.csv`'s recovery numbers are averages over the held-out pairs — a different, larger held-out set (63 vs. 55) will not have the same mean/variance as before.

So this isn't "rerun a script to get updated decimal points" — it's "the object being measured (the circuit) and the sample it's measured against both changed." Everything built on top needs to be treated as stale until regenerated.

---

## 2. What's stale — the full list, by pipeline stage

Two driver scripts define the actual dependency chain for the familiarity arm: `scripts/run_circuit.sh` (circuit discovery) and `scripts/run_familiarity_arm.sh` (neuron/ablation tests on top of the gated arm). Everything they produce for `familiarity` is stale:

| Stage | Script | Output(s) | Feeds into |
|---|---|---|---|
| 1. Position map | `circuit_position_map.py` | `results/circuit_familiarity/position_map.csv.gz` | the Jacobian-lens/relay figure (Figure 2 in the earlier screenshot), and informs which layer range (`13-19`) the direction-patching scripts use |
| 2. Head discovery | `circuit_heads.py` | `head_attribution.csv`, `head_verification.csv` | defines the "component set" half of the circuit |
| 3. Neuron discovery | `circuit_mlp.py` | `neuron_attribution.csv`, `neuron_verification.csv`, `mlp_patching.csv` | the other half of the component set |
| 4. Faithfulness | `circuit_faithfulness.py` | `faithfulness.csv`, `faithfulness_r10.csv` | **L1** (component-set cell) and **L2** (necessity/null testing) |
| 5. Rank-1 direction | `circuit_direction_patch.py` | `direction_patch.csv` | **L1** (rank-1-direction cell) |
| 6. Direction null | `circuit_direction_patch_u2c_null.py` | `direction_patch_u2c_null.csv` | null-comparison baseline for the direction |
| 7. Hedge-matched null | `circuit_direction_patch_u2c_hedgematched.py` | `direction_patch_u2c_hedgematched.csv` | the stricter null L1's `fig6` reads (previously not yet generated for familiarity at all — see the earlier L1 doc) |
| 8. Pruning | `circuit_prune.py` | `prune_curve.csv`, `prune_marginals.csv` | minimality/necessity claims |
| 9. Jacobian lens | the `jlens_*` scripts | `jlens_*.csv` / `.json` | the lens-based timing analysis (demoted to appendix per the ACL format change, but still needs to be correct) |
| 10. Twin patching | `scripts/twin_patching.py` (via `run_familiarity_arm.sh`) | `results/twin_patching_familiarity.csv.gz` | Figure 2 (entity-to-readout relay), which is captioned "195 pairs" / "60 pairs" — both now wrong under the new split |
| 11. Ablation | `run_ablation.py` ×3 (via `run_familiarity_arm.sh`) | `results/ablation_bf16_familiarity_new/_dir/_key` | neuron/direction/keyset ablation results |
| 12. Direction bridge | `direction_bridge.py` (via `run_familiarity_arm.sh`) | `results/bridge_familiarity/` | steering/bridge analysis |
| 13. Induction check | `induction_check.py --categories familiarity` (via `run_familiarity_arm.sh`) | `results/induction_check_familiarity.csv` | induction-head confound check |

**Everything in this table is stale.** None of it should be treated as current once `data/familiarity/` is replaced.

---

## 3. Preserve, don't overwrite — back up before regenerating

Every script above is protected by `guard_output()`, which refuses to run if its target file already exists **unless `--overwrite` is passed**. Regenerating in place therefore requires `--overwrite` everywhere — which means the pre-dedup results will be destroyed the moment each script reruns, unless they're moved first. Following the same principle used for the hedge-matched null earlier (keep both, never silently lose one), back up the entire stale set before regenerating anything:

```bash
mv results/circuit_familiarity            results/circuit_familiarity_pre_dedup
mv results/twin_patching_familiarity.csv.gz results/twin_patching_familiarity_pre_dedup.csv.gz
mv results/ablation_bf16_familiarity_new    results/ablation_bf16_familiarity_new_pre_dedup
mv results/ablation_bf16_familiarity_dir    results/ablation_bf16_familiarity_dir_pre_dedup
mv results/ablation_bf16_familiarity_key    results/ablation_bf16_familiarity_key_pre_dedup
mv results/bridge_familiarity                results/bridge_familiarity_pre_dedup
mv results/induction_check_familiarity.csv   results/induction_check_familiarity_pre_dedup.csv
```

This preserves the ability to answer "did deduping the data change which heads/neurons get identified as the circuit, or just the recovery numbers?" — itself worth a line in the paper's limitations/methods section either way, not just a silent rerun.

---

## 4. The order to regenerate in, and why that order

The order below follows the actual dependency chain — each stage reads the previous stage's output, so running out of order either fails outright or silently uses stale inputs.

### Step 1 — Rerun the core circuit-discovery chain

```bash
CN_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct ARMS=familiarity bash scripts/run_circuit.sh
```

This runs, in the fixed internal order `run_circuit.sh` already encodes: `circuit_position_map.py` → `circuit_heads.py` → `circuit_mlp.py` → `circuit_faithfulness.py`. Each stage's `.done` marker means a rerun resumes cleanly if interrupted — but since these are now targeting a changed dataset, **the old `.done` markers in `runs/` must be cleared first** (or the script will think these stages already ran and skip them, silently leaving the stale outputs in place). Before running, check the original run's logs for any non-default flags (`WINDOW`, `N_HEADS`, `N_NEURONS`, `LIMIT`, `MAP_LIMIT`, `VERIFY`) so the rerun is comparable to the original, not just using bare script defaults.

**Why this goes first:** everything else — the rank-1 direction, faithfulness numbers, twin patching, ablations — either directly reads the position map / verified heads / verified neurons this stage produces, or needs to be run on the same twin split this stage established as current.

### Step 2 — Rerun the rank-1 direction and its nulls

```bash
python scripts/circuit_direction_patch.py --category familiarity --layers 13-19 --limit 60 --overwrite
python scripts/circuit_direction_patch_u2c_null.py --category familiarity --layers 13-19 --limit 60 --n-random 10 --seed 42 --overwrite
python scripts/circuit_direction_patch_u2c_hedgematched.py --category familiarity --layers 13-19 --limit 60 --n-random 10 --seed 42 --overwrite
```

**Why this goes second, not first:** the `13-19` layer range was chosen based on the *old* position map (where the entity-to-readout "relay" happens, per the earlier layer-decay figure). Step 1 regenerates `position_map.csv.gz` on the new pairs — check that output before locking in `13-19` again. If the relay layers shifted at all under the new dataset, this range should shift with it, not be copied blindly from the old run.

**Why this goes before Step 3:** L1's `fig6` and L2's null-testing both consume these files; nothing after this point should be treated as final until this step is done.

### Step 3 — Rerun the remaining familiarity-arm stages

```bash
bash scripts/run_familiarity_arm.sh
```

This covers: induction check, three ablation variants (new/direction/keyset), twin patching, and the direction bridge/steering analysis. This depends on Step 1's candidate/verified sets (`results/candidate_neurons_bf16.json`, `results/candidates_direction_familiarity.json`, `results/candidates_keyset.json` — confirm these candidate-set JSONs themselves don't also need regenerating from the new circuit before this stage runs; if they were built from the old circuit's discovered heads/neurons, they're stale too and are a prerequisite here, not just an input).

**Why this goes last:** twin patching and ablation are downstream diagnostics/robustness checks on the already-discovered circuit — they should reflect whatever circuit Step 1 actually identifies on the new data, not run in parallel with or ahead of it.

### Step 4 — Only now, L1 and L2's actual figure/analysis work

Once Steps 1–3 are complete and their outputs verified, L1 (retarget `fig6` to familiarity, add the entropy panel, replace `ci95()` with a pair-level bootstrap) and L2 (null/necessity testing) can proceed against data that will still be current when the paper is finalized.

---

## 5. What can be done now, without waiting on GPU time

The **code** changes L1 needs (parameterizing `fig6` by category, adding the `entropy_rec` panel, swapping in `boot_ci()` from `stats_upgrades.py`) don't depend on the actual data values — only on the file schema, which is unchanged by this regeneration. This code can be written and tested now against the old `circuit_familiarity_pre_dedup` files (or the still-untouched `circuit_conflict` files, same schema) purely to confirm it runs correctly — then pointed at the freshly regenerated files once Steps 1–3 finish. Nothing it currently outputs should be treated as a real result until then.

---

## 6. Verification checklist before treating any of this as final

- [ ] `runs/*.done` markers cleared for every familiarity-arm stage before rerunning (otherwise stages silently skip and stale outputs persist)
- [ ] Old results backed up per §3, not overwritten
- [ ] Layer range for direction-patching scripts re-confirmed against the *new* position map, not copied from the old run
- [ ] Candidate-set JSONs (`candidates_direction_familiarity.json`, `candidates_keyset.json`, etc.) checked for whether they themselves need regenerating from the new circuit before Step 3
- [ ] New pair counts (63 held-out, not 55/59) reflected in every figure caption that currently states a pair count (e.g. Figure 2's "60 pairs" / "195 pairs" captions)
- [ ] A short note prepared for the manuscript's methods/limitations section on whether the discovered circuit changed materially after deduping (heads/neurons same vs. different), using the `_pre_dedup` backups for the comparison
- [ ] Only after all of the above: L1 and L2's actual analysis/figure work proceeds on the new files
