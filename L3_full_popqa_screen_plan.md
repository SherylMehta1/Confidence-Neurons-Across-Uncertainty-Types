# L3 — The Full PopQA Screen
### Explanation, required fixes, and timeline

Verified this session against a fresh, full (non-shallow) clone of `l0-fixes` (commit `6441cd8`) — every number below comes from actually reading `scripts/build_familiarity_twins.py`, `UNCERTAINTY_DEFINITION.md`, `README.md`, and the real, committed `data/familiarity_v2/gate_report.json` / `all_measured.jsonl`, not from the plan text alone.

---

## 0. What "L3" actually is, and one number I can't re-verify here

L3 doesn't exist as a named script or lane inside the repo — it's the master-plan's label for **widening the familiarity candidate pool far beyond what `build_familiarity_twins.py` currently draws**, because three downstream pieces need a bigger, differently-shaped pool than the current one:

- **L2e (saturation to N=500)** needs ~500 held-out pairs; there are 63 today.
- **L4a (frequency×knowledge factorial)** needs items that are *popular-but-unknown* and *obscure-but-known* — cells the current sampling never draws (see §1).
- **L4b (U/U falsification, full version)** needs a much larger pool of Unknown items to build a well-powered Hedger-vs-Confabulator split.

One correction to carry forward: earlier in this engagement a **~14,267-item full PopQA test-set size** was cited. I could not re-verify that number in this session — this sandbox's network is allowlisted to package registries and GitHub only, and a direct request to the Hugging Face datasets API returned `403 Forbidden` when I tried just now. So that figure should be treated as **unconfirmed** until checked directly. The good news: getting the real number costs nothing and needs no GPU — it's just the dataset-load-and-dedupe step that already exists at the top of `build_familiarity_twins.py` (lines 150-165), which prints `n_raw` and the deduped count. Running that alone, before deciding any sampling budget, is Step 0 below.

---

## 1. Why the current sampling can't support L4a, specifically

Confirmed by reading the candidate-selection loop in `build_familiarity_twins.py` (lines 172-187):

```python
for prop in props:
    items = [r for r in rows if r["prop"] == prop and r.get("s_pop") is not None]
    ...
    lo = [r for r in items if r["s_pop"] <= np.quantile(pops, args.lo_quantile)]   # bottom 10%
    hi = [r for r in items if r["s_pop"] >= np.quantile(pops, args.hi_quantile)]   # top 10%
    ...
    for r in lo[: args.n_per_relation]: cands.append(dict(r, arm="uncertain"))
    for r in hi[: args.n_per_relation]: cands.append(dict(r, arm="control"))
```

Every candidate is drawn from the **extreme 10% tails of popularity, per relation**, and its `arm` label (`"uncertain"` vs `"control"`) is assigned from which tail it came from — *before* any scoring happens. Knownness (the SliCK class) is only discovered afterward, by generation.

This is exactly right for the familiarity/L0/L1/L2a-d work, where popularity is used as an *a priori proxy* for knownness and the goal is a clean twin-pair dataset. But it structurally cannot answer L4a's actual question — **does the circuit track popularity or knownness, when the two disagree?** — because the sampling never looks at the middle 80% of the popularity distribution, so it can never surface:

- **popular-but-unknown** items (a well-known entity, but a fact about it the model happens not to know), or
- **obscure-but-known** items (a low-frequency entity the model surprisingly does know).

These off-diagonal cells are the entire point of the factorial. Confirmed via the real, committed `data/familiarity_v2/gate_report.json`: current scored pool is **1,646 items over 16 relations** (`all_measured.jsonl`), all of them by construction sitting in one of the two popularity tails — 904 Unknown, 499 HighlyKnown, 243 Maybe, but every single one of them was drawn *because* it was already at a popularity extreme. There is no way to build a 4-cell factorial out of data that was never sampled from the middle of the distribution or cross-tabulated against real popularity values in the first place.

## 2. What's genuinely new code vs. what's a bigger run of existing code

| Need | New code required? | Why |
|---|---|---|
| L2e — more held-out pairs (target ~500) | **No.** Reuse `build_familiarity_twins.py` as-is with a larger candidate budget (see §3.2). The existing gate + pairing + 70/30 split already scales to any pool size. | The keep-rate is already known (207/1646 ≈ 12.6%), so the needed candidate-pool size to reach ~500 pairs is a straightforward extrapolation, not new logic. |
| L4a — frequency×knowledge factorial | **Yes, on two fronts.** (a) The candidate-sampling loop needs a new mode that draws across the *full* popularity range per relation, not just the tails. (b) A new script is needed to bucket the resulting scored pool into the 4 quadrants and build matched pairs within/across them — the existing pairing loop (lines 239-261) only ever pairs "Unknown-from-lo-tail" against "HighlyKnown-from-hi-tail," which by construction discards the off-diagonal cells. | Confirmed via grep: no factorial/quadrant-construction script exists anywhere in the repo. `frequency_causal.py` (mentioned in the README) is an unrelated, older neuron-level test (Stolfo et al.-style frequency-neuron dose-response) — it does not touch twin-pair construction and doesn't reduce this work at all. |
| L4b — full U/U falsification | **No new script**, but needs the bigger pool. `build_uu_hedge_confab.py` already reads `all_measured.jsonl` and applies the pre-registered `hedge_rate >= 0.8` / `<= 0.1` thresholds; it just needs a much larger set of scored Unknown items to draw from to get adequate power per relation, not just pooled across all of them. | Confirmed: the script's filtering logic (`arm=="uncertain" and slick=="Unknown"`) is already general-purpose; it was only data-starved before. |

## 3. What to actually build and run

### 3.1 Step 0 — size the job before committing to it (no GPU, minutes)

Run just the dataset-load-and-dedupe block that already exists at the top of `build_familiarity_twins.py` (or a 10-line standalone script that imports and calls the same logic) to get, per relation:
- total deduped PopQA test-set rows,
- the real popularity distribution (not just the two tails already sampled),
- how many rows fall in, say, each popularity decile.

This replaces the unverified ~14,267 figure with a real number, and tells you immediately whether "the full test set" is even the right target, or whether a large stratified sample gets you everything L4a/L2e/L4b need at a fraction of the compute.

### 3.2 New candidate-sampling mode

Add a `--stratify` mode (or a new `--full-screen` flag) to `build_familiarity_twins.py`'s candidate loop: instead of only `lo`/`hi` at the 10th/90th percentile, sample across **all deciles** per relation, with a per-decile cap (e.g., `--n-per-decile`, analogous to today's `--n-per-relation`). Every drawn item is scored exactly as today (greedy + 10 samples + entropy + 20 free-gens) and written to `all_measured.jsonl` — the checkpoint/resume machinery (`item_key`, `load_checkpoint`) already works unchanged on a larger and differently-shaped candidate list, since it keys purely on `(prop, subj)`.

Two sizing decisions to make once Step 0's real numbers are in hand:
- **Per-decile cap.** Needs to be large enough that, after the ~12.6% keep-rate-equivalent filtering (adjusted, since off-diagonal cells will likely have a *different* keep rate than the tails — this itself is worth reporting, not assuming), each of the 4 factorial cells has enough surviving items. `UNCERTAINTY_DEFINITION.md`'s own dataset standard (§3.3) sets a bar of "≥80 surviving pairs per type" for the existing single-axis types — a reasonable starting target per factorial cell, though this is a call worth making explicitly rather than inheriting silently.
- **Whether to cut `n_samples`/`n_free` for the screening pass.** The current per-item cost (greedy + 10 sampled-correctness generations + 20 free-generation hedge samples ≈ 31 generations/item) is real money at 4-10x the current 1,646-item pool. A cheaper first-pass triage (e.g., `n_samples=5`, `n_free=8` just to bucket by popularity×approximate-knownness) followed by a full-budget re-score only of items that land in an interesting quadrant is a legitimate way to cut cost — but changes the SliCK classification's reliability, so it should be a deliberate, stated choice, not a quiet default change.

### 3.3 New script: factorial construction

`scripts/build_frequency_knowledge_factorial.py` (name illustrative): reads the widened `all_measured.jsonl`, buckets items per relation into {high-popularity, low-popularity} × {HighlyKnown, Unknown} using the real popularity distribution from Step 0 (not the old fixed 10/90 cutoffs, which were calibrated for a tails-only design), and builds matched twin pairs within and across cells using the same length/entropy/hedge gate already used in `build_familiarity_twins.py` — reusing `slick_class()`, `is_correct()`, `normalize()` etc. directly, the same way `gate_twins.py` already imports them. Output: four twin-pair files (or one file with a `cell` column), through the same `data/<category>/{prompts,controls}.jsonl` schema so the existing circuit pipeline (`circuit_position_map.py` → `circuit_heads.py` → `circuit_mlp.py` → `circuit_faithfulness.py` → direction patching) runs on it unmodified.

### 3.4 L2e and L4b — no new code, just point them at the wider pool

Once the widened `all_measured.jsonl` exists:
- Rerun `build_familiarity_twins.py`'s existing pairing/gate logic (or a thin wrapper that skips re-scoring, since everything is already checkpointed) to get the larger working/held-out split; regenerate the circuit pipeline on it exactly as was done after the L0 dedup fix, following the same "preserve, don't overwrite" pattern (`_pre_l3` suffix on the old `results/circuit_familiarity_v2/` directory before rerunning with `--overwrite`).
- Rerun `build_uu_hedge_confab.py` unchanged against the bigger `all_measured.jsonl` to get a better-powered Hedger/Confabulator split.

---

## 4. Timeline

This is the long pole in the schedule (per the master plan's honest read: the 21 Sept factorial-balance gate is already unreachable), so the sequencing below treats compute time as the real constraint, not engineering time.

| Day | Task |
|---|---|
| **Day 1** | Step 0 (dataset sizing, no GPU) — get the real deduped test-set size, per-relation counts, and decile boundaries. Decide the per-decile cap and whether to use a cheaper screening budget, based on real numbers rather than the unverified 14,267 estimate. |
| **Day 1-2** | Write the `--stratify` candidate-sampling mode in `build_familiarity_twins.py`. Smoke-test on one or two relations with a small cap before committing to the full run. |
| **Day 2 onward** | Kick off the widened screening run in the background — it's checkpointed, so it can run unattended across multiple sessions without risk, and other unblocked work (L4 pre-registration commit, L6 manuscript edits, anything else still open) can proceed in parallel while it runs. |
| **Once screening finishes** | Write `build_frequency_knowledge_factorial.py`; build the 4-cell pairs; regenerate the circuit pipeline on both the L2e-widened familiarity pool and the new factorial dataset, preserving old results under a `_pre_l3` suffix per the standing rule. |
| **Same window** | Rerun `build_uu_hedge_confab.py` against the widened pool for L4b. |
| **After both regenerate cleanly** | L4a and L4b's real, pre-registered falsification runs (step 7 of the master ordered list) become possible — this is the point at which the deferred L4 pre-registration commit (already decided to happen "before beginning L4," not now) needs to actually land. |

No hard day-count estimate is given for the screening run itself — that number falls straight out of Step 0's real per-relation counts and per-item generation cost, which aren't guessable from here. Recommend getting Step 0's numbers first and sizing the rest of this timeline against them rather than committing to a duration now.

---

## 5. Open decisions to make before starting, not after

1. **Per-decile candidate cap and total screening budget** — depends on Step 0's real numbers.
2. **Full-budget scoring vs. cheap-triage-then-rescore** — a real cost/reliability tradeoff, not a default to inherit silently.
3. **Minimum N per factorial cell** to consider L4a adequately powered — `UNCERTAINTY_DEFINITION.md`'s existing "≥80 pairs per type" bar is the natural anchor, but the off-diagonal cells may simply not exist in enough quantity naturally (a popular-but-unknown item for this specific model may be rare) — worth deciding up front what happens if a cell comes back underpowered even after the wider screen, rather than treating it as pass/fail against a hard number without any Plan B.
