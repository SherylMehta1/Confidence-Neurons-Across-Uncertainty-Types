# L1 — The Dissociation Table (Figure 1)
### ACL Rolling Review, October 2026 cycle — "Confidence Neurons Across Uncertainty Types"
### STATUS: UNBLOCKED — circuit-discovery pipeline has been regenerated post-dedup; this can now proceed for real

**Window:** originally Days 1–4; now starting after the regeneration (circuit survived, per the rerun results) and the bridge-direction fix (option 3b) both landed
**GPU cost:** ~0 hours for assembling the figure from existing data; a small amount if the hedge-matched null for familiarity still needs generating (see §0 and Step 1)
**Blocks:** nothing downstream directly, but it produces the paper's dissociation figure and is the first line of defense against the most obvious reviewer objection

---

## 0. Read this first — what changed since the last version of this doc, and one thing to confirm

Two things happened since this doc was last written that directly affect L1:

1. **The circuit-discovery pipeline was regenerated** against the post-dedup familiarity dataset (207 pairs). Result: the circuit held — 26/30 verified heads identical, faithfulness and rank-1-direction recovery both *strengthened*, minimality tightened from 20 to 10 components. This is genuinely good news for L1: `faithfulness.csv` and `direction_patch.csv` are current again, not stale.
2. **The entropy readout showed a sign flip worth carrying into L1's figure explicitly**: faithfulness entropy (control→uncertain) went from −0.132 (old) to +0.144 (new) — both small, near zero, but opposite sign. This is exactly the kind of thin/unstable cell the dissociation figure should show plainly with a real CI, not smooth over.

**One thing not yet confirmed — check before starting Steps 2–6:** the naming ambiguity surfaced during the bridge-direction fix (`familiarity` vs. `familiarity_v2` are still two separate, un-merged directories) applies here too. Confirm which dataset the regenerated `results/circuit_familiarity/` actually corresponds to — was `run_circuit.sh` run with `--category familiarity` (reading the old, un-fixed `data/familiarity/`) or `--category familiarity_v2` (reading the actual post-dedup data)? The reported "207 pairs" / "60 held-out" numbers match `familiarity_v2` exactly, which is reassuring, but confirm the actual `--category` flag used (check the run's provenance file) before treating `results/circuit_familiarity/` as trustworthy — if it was generated under a `familiarity_v2` category name, the real output directory would be `results/circuit_familiarity_v2/`, not `results/circuit_familiarity/`, and every path below needs the same adjustment the bridge fix already made.

---

## 1. Why this lane exists

The paper claims there's a mechanism specifically responsible for *hedging/doubt* that is separable from a mechanism that just tracks *predictive uncertainty in general*. The obvious reviewer objection:

> "You haven't shown two separate things — you've shown one signal, read out two different ways."

L1 heads this off with a 2×2: **representation** (a distributed component set vs. a single compressed rank-1 direction) crossed with **readout** (hedge log-odds vs. entropy). If all four cells told the same story, that would support "one signal, two readouts." If they diverge, that's evidence against it.

---

## 2. What's in the data now (post-regeneration)

Two parallel scripts, designed to be compared against each other, both already rerun on the post-dedup dataset:

| Representation | Script | Output file | Docstring says |
|---|---|---|---|
| Component set (verified heads + neurons) | `scripts/circuit_faithfulness.py` | `.../faithfulness.csv` | patches the verified component set jointly, reports recovered hedge/entropy gap, both directions, vs. random size-matched null sets |
| Rank-1 direction | `scripts/circuit_direction_patch.py` | `.../direction_patch.csv` | "**Decider experiment 1 — rank-1 direction patching vs the component circuit**... If rank-1 recovery ≈ the component set's recovery, the mechanism is a low-rank directional channel that the components implement; if it falls short, the component set carries more than the direction." |

Both share an identical schema (`pair, set, direction, logodds_rec, entropy_rec`) and both directions (`control_to_uncertain`, `uncertain_to_control`) are populated for the real row — confirmed previously on the pre-dedup version, and the regeneration report confirms the same shape holds post-dedup (both faithfulness and direction-patch numbers reported for both directions).

**Current known values, post-regeneration** (component set / faithfulness side; rank-1 numbers per the regeneration report):

| | control→uncertain | uncertain→control |
|---|---|---|
| **Component set, hedge (`logodds_rec`)** | +0.910 | (confirm — not explicitly listed in the rerun summary) |
| **Component set, entropy (`entropy_rec`)** | **+0.144 (sign-flipped from −0.132 pre-dedup)** | (confirm) |
| **Rank-1 direction, hedge (`logodds_rec`)** | +0.546 | +0.963 |
| **Rank-1 direction, entropy (`entropy_rec`)** | (confirm — not reported in the rerun summary) | (confirm) |

The rerun summary you were given didn't report every cell explicitly (it focused on the headline "circuit survived" comparison) — pull the actual regenerated CSVs directly for the cells marked "confirm" before building the figure, rather than assuming they follow the same direction of change as the ones that were reported.

---

## 3. Where the real remaining gap is

Same as before, mostly still true — the gap is in the figure-assembly code, not the underlying data, though item 3 below now needs a fresh status check:

1. **`fig6` in `make_paper_figures.py` is still wired to `circuit_conflict`**, not `circuit_familiarity` — this hasn't been touched yet.
2. **`fig6` still only plots `logodds_rec`** — no `entropy_rec` panel. Given the entropy sign-flip finding in §0, this panel is now more clearly worth building, not less — it's exactly the cell that needs honest visualization.
3. **Hedge-matched null status for familiarity — unconfirmed post-regeneration.** Before this L1 doc was last updated, `direction_patch_u2c_hedgematched.csv` didn't exist for familiarity. It's unclear whether the circuit-discovery regeneration (§0) regenerated this file too, or only the core chain (position map → heads → mlp → faithfulness → direction patch → null). Check `results/circuit_familiarity/` (or `_v2`, per §0) for this file before assuming it's there.
4. **The CI helper (`ci95()`) is still a normal-approximation SEM, not a bootstrap.** Unchanged from before — still needs replacing with `boot_ci()`.

---

## 4. Process, step by step

**Step 1 — Confirm the hedge-matched null situation for familiarity, post-regeneration.**
Check whether `direction_patch_u2c_hedgematched.csv` already exists for the current familiarity results directory (per §0's naming check). If not:
```
python scripts/circuit_direction_patch_u2c_hedgematched.py --category <familiarity or familiarity_v2, per §0> --layers 13-19 --limit 60 --n-random 10 --seed 42
```
The layer range `13-19` does not need re-derivation this time — the regeneration's own position-map rerun already confirmed the relay peak lands at L17 within this same window, independently re-derived on the new pairs rather than copied from the old run. `13-19` is safe to reuse as-is.

**Overwrite safety:** do not pass `--overwrite` — same reasoning as before, category-scoped paths mean nothing collides, and omitting it is a free extra check.

**Step 2 — Retarget `fig6` to familiarity, without breaking conflict.**
Parameterize by category rather than hardcoding `circuit_conflict`.

**Overwrite safety:** `save(fig, name)` has no existence check at all — give the familiarity version a distinct name (`fig6_familiarity`, not bare `fig6`) so it can never silently overwrite the conflict figure.

**Step 3 — Add the missing entropy panel.**
Duplicate the `logodds_rec` bar-plotting logic for `entropy_rec`. Given §0's finding, make sure this panel actually renders the control→uncertain entropy cell clearly enough to show it's near zero and sign-unstable — this is a legitimate, reportable result (a genuinely weak/inconsistent cell in one specific direction), not something to visually downplay.

**Step 4 — Replace the CI computation with a real per-pair bootstrap.**
Use `boot_ci()` from `scripts/stats_upgrades.py` for all four cells. **Batch this with the other outstanding bootstrap work**: the bridge-direction fix (option 3b) also left a pair-level bootstrap CI on `cos(familiarity, hedging) = +0.67` unfinished, using the exact same function. Do both in one pass rather than writing/testing the bootstrap machinery twice.

**Step 5 — Assemble the final dissociation figure.**
Representation × readout × direction, each cell with a real bootstrap CI. Caption should state plainly what dissociation looks like (cells differing) vs. "one signal, two readouts" (statistically indistinguishable cells) — and should explicitly call out the entropy control→uncertain cell as thin/near-zero given §0's finding, rather than letting a reader assume all four cells are equally solid.

**Step 6 — Sanity-check against the manuscript's existing narrative.**
Confirm the resulting pattern — now including the *strengthened* post-regeneration numbers — doesn't quietly contradict whatever the draft currently claims about component-set vs. rank-1 primacy. Given the circuit measurably strengthened after the dedup fix, this is also a good moment to double check whether any manuscript text drafted against the old (weaker) pre-dedup numbers needs updating alongside the figure.

---

## 5. Data-quality flag, still open

Both files show `logodds_rec`/`entropy_rec` values sitting exactly at clip boundaries (`2.0` / `-1.0`) fairly often — check `recovery()` in `circuit_common.py`. Confirm this clipping behavior on the *regenerated* data too (it may have changed shape now that the circuit strengthened) before bootstrapping — the bootstrap CIs will understate true variance if a meaningful fraction of pairs are sitting at these bounds.

---

## 6. Timeline (revised — this is now the "do it" phase, not "wait")

| Step | Work | Exit check |
|---|---|---|
| 1 | Confirm which category directory is real (§0); check/generate hedge-matched null for familiarity | Clear on which `results/circuit_familiarity*/` directory is current; hedge-matched null file present |
| 2–4 | Retarget `fig6`; add entropy panel; replace `ci95()` with `boot_ci()` (batched with the bridge-cosine CI) | Familiarity dissociation figure renders, both readouts, real bootstrap CIs, distinct output filename |
| 5–6 | Finalize figure; sanity-check against manuscript text, including any stale pre-dedup numbers | Figure and any dependent manuscript text are both current |

---

## 7. Revision history

- **v1:** assumed no L1 code existed; speculated about a missing rank-1 artifact. Wrong — direct repo inspection found both the rank-1 data and a first-draft comparison figure (`fig6`), just scoped to the wrong category.
- **v2:** corrected Step 1's layer range (familiarity's actual `13-19`, not conflict's `11-21`); added overwrite-safety guidance for both CSV generation (already safe) and figure saving (not safe by default).
- **v3 (this version):** updated for the post-regeneration state — circuit confirmed to have survived and strengthened; entropy sign-flip finding folded into Step 3/5's guidance; layer range confirmed re-derived (not just reused) at L17; flagged the unresolved `familiarity`/`familiarity_v2` naming ambiguity as something to check before trusting `results/circuit_familiarity/`'s identity; bootstrap step merged with the outstanding bridge-cosine CI work from option 3b.
