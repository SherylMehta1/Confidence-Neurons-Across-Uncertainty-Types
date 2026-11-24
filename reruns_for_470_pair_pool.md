# Reruns Needed to Make `familiarity_v3` (470 pairs) the Complete Dataset

Scope: only the experiments that currently exist on the 207-pair pool (`familiarity_v2`) but not on the 470-pair pool (`familiarity_v3`), plus the one existing gap even within `familiarity_v3` itself (the full-scale figure). Four items. Nothing else.

All commands assume they're run from the repo root on `l0-fixes`, with the model (`unsloth/Meta-Llama-3.1-8B-Instruct`, bf16) loadable exactly as the existing `familiarity_v3` runs already do — same environment that produced `results/circuit_familiarity_v3/faithfulness.csv` etc.

---

## 1. Leave-one-token-out (LOTO)

**What it does:** checks whether the circuit's headline hedge-recovery number depends on any single token in the 5-token hedge set (`" not"`, `" I"`, `" unknown"`, `" unclear"`, `" no"`). It's a two-step design: an expensive step that runs the model once and caches every log-probability the readout could ever need, and a free step that recomputes the recovery number under each "drop one token" variant purely by re-combining those cached numbers — no further model calls.

**Why rerun on the 470-pair pool:** it currently only exists for the 207-pair pool. If the paper's headline circuit numbers move to the 470-pair pool, this check needs to exist on the same pool, or the paper is explaining one dataset's circuit using a check run on a different one.

**A real gotcha to know before running this:** the caching script's own self-check compares its output against the *already-committed* `faithfulness.csv` and `direction_patch.csv` in the target directory — that's how it verifies it reran the identical circuit correctly. In `results/circuit_familiarity_v3/`, those two files reflect the **60-held-out-pair** slice, not the full 141-pair one (the full slice lives in separately-named `faithfulness_limit141.csv` / `direction_patch_limit141.csv`, which the self-check does *not* know to look at). So:

- **Run it at `--limit 60`** (matching the existing reference files) and the self-check works with zero code changes. This gives you LOTO on the same 60-pair slice that's already backing the committed `fig6_familiarity_v3` figure — internally consistent, no extra work.
- Running it at `--limit 141` instead would need `loto_recompute.py`'s `REFERENCE` dict pointed at the `_limit141` files first (a one-line edit), otherwise the script's own sanity check will silently compare against the wrong reference and could report a false mismatch.

**Process:**

```bash
# Step 1 — GPU, caches every log-prob the readout could need (one model load, one forward pass per pair per condition)
python scripts/circuit_loto_cache.py \
    --category familiarity_v3 \
    --out-dir results/circuit_familiarity_v3 \
    --limit 60 \
    --n-heads 20 --n-neurons 100 --n-random-sets 30 \
    --layers 13-19 --n-random-dirs 10 --seed 42

# Step 2 — CPU only, free, recomputes recovery under each token-dropped variant
python scripts/loto_recompute.py --dir results/circuit_familiarity_v3
```

(`--n-heads`/`--n-neurons`/`--layers`/`--n-random-sets`/`--n-random-dirs`/`--seed` above are copied from the provenance already recorded for `familiarity_v3`'s own `faithfulness.csv` — matching them is what makes the self-check pass.)

**Output:** `results/circuit_familiarity_v3/loto.csv` and `loto_summary.txt` — same shape as the existing `familiarity_v2` version, so it slots into the numbers-tracking script the same way once it exists.

---

## 2. Causal timing

**What it does:** a layer-by-layer sweep of residual-stream patching, cumulative from layer 0 up to each candidate layer, to find where the hedge decision and the entropy gap each become causally "locked in." This is what produced the "hedge resolves by L8/L12, entropy not until L13/L18" finding on the 207-pair pool.

**Why rerun on the 470-pair pool:** same reason as LOTO — this doesn't exist for `familiarity_v3` at all yet, and it's the mechanistic explanation *of* the circuit numbers that would become canonical.

**Cost note:** this is the more expensive of the two reruns — it's a genuine full-layer sweep (32 layers × 2 directions × every held-out pair), not a cache-once design. The script's own docstring estimates roughly 4 minutes for 60 pairs; scaling to the full 141-pair held-out set puts this at somewhere around 8–10 minutes, not counting model load time. Budget for it accordingly — it's real, non-trivial GPU time, unlike LOTO's caching step.

**Process — one command covers both directions and both readouts (hedge, entropy) in a single run:**

```bash
python scripts/circuit_causal_timing.py \
    --category familiarity_v3 \
    --out-dir results/circuit_familiarity_v3 \
    --mode up_to \
    --positions suffix \
    --limit 141
```

- `--mode up_to` is deliberate: this is the informative sweep (recovery rises from the null floor toward 1 as the transition layer is crossed). The other mode, `--mode from`, is degenerate by construction (flat at ~1.0 for every layer) and is only kept in the script to show the degeneracy, not to be cited — don't run it.
- `--positions suffix` is deliberate too: it's what isolates "the layer at which the decision reaches the readout position," as opposed to `--positions all`, which mostly just measures the entity-token swap and saturates early for a different reason.
- `--limit 141` uses the full held-out pool, matching the "canonical = 470-pair draw" choice. (If you decide instead to keep this check on the same 60-pair slice as the committed figure, for full internal consistency with item 1 above, use `--limit 60` here too — either choice is defensible, but pick the same N as LOTO's, not a third number.)

**Output:** `results/circuit_familiarity_v3/causal_timing_up_to_suffix.csv` and `causal_timing_up_to_suffix_summary.csv`/`.txt` — same naming convention as the existing `familiarity_v2` files.

---

## 3. The dissociation figure at the full 141-pair scale

**What it is:** the actual plotted figure (bar chart + `_cells.csv`) comparing the rank-1 direction against the full component set, at both readouts and both directions. This one is subtly different from items 1–2: the *numbers* behind it already exist for the full 141-pair pool (`faithfulness_limit141.csv`, `direction_patch_limit141.csv`), but the figure-generation script has never been pointed at them — it's only ever been run against the plain `faithfulness.csv`/`direction_patch.csv` files, which are the smaller 60-pair slice. So there is currently no committed figure at the full scale, even though the raw numbers for one already exist.

**The gotcha:** `fig_dissociation.py` hardcodes the three filenames it reads (`faithfulness.csv`, `direction_patch_u2c_null.csv`, `direction_patch_u2c_hedgematched.csv`) inside whatever `--circuit-dir` you give it — there's no flag to tell it "read the `_limit141` versions instead." Two ways to handle this, in order of preference:

- **Small script change (cleanest):** add a `--suffix` argument to `fig_dissociation.py` that gets inserted before `.csv` in each of the three filenames it reads (e.g. `faithfulness{suffix}.csv`), defaulting to empty string so nothing about the existing `familiarity_v2`/`familiarity_v3` (60-pair) figures changes.
- **No-code workaround:** create a scratch directory containing copies (or symlinks) of `faithfulness_limit141.csv` → `faithfulness.csv`, and the equivalent for the two direction-patch files, then point `--circuit-dir` at that scratch directory. Works today with zero script edits, at the cost of a slightly awkward intermediate folder.

**Process (once either fix above is in place):**

```bash
python scripts/fig_dissociation.py \
    --circuit-dir results/circuit_familiarity_v3 \
    --name fig6_familiarity_v3_full \
    --out-dir paper/figures
```

**Output:** `paper/figures/fig6_familiarity_v3_full.pdf` and `fig6_familiarity_v3_full_cells.csv` — the latter is what would let the numbers-tracking script pull CIs for the full-scale numbers from a real committed figure instead of falling back to its own recomputed CI, the same "prefer the figure's own CI" convention it already follows everywhere else.

---

## 4. Bridge/alignment direction, rebuilt from the 470-pair pool (optional — only needed for full consistency, not for the circuit claims themselves)

**What it is:** the two existing bridge runs (`bridge_familiarity_v3`, `bridge_familiarity_v4`) both build their "familiarity direction" from `familiarity_v2` (the 207-pair pool) — that's a hardcoded default in the script, unrelated to which pool is canonical for the main circuit claims. If you want the bridge section to be built from the same canonical draw as everything else, this needs its own rerun; if you're fine with the bridge staying pinned to the 207-pair direction (a defensible, separately-versioned choice), skip this item entirely.

**Process — three steps, all requiring the model (all GPU):**

```bash
# Step 1 — builds the actual direction from the 470-pair pool
python scripts/direction_bridge.py \
    --category familiarity_v3 \
    --hedging-category uu_hedge_confab_v3 \
    --layer-range 20-31 \
    --out-dir results/bridge_familiarity_v5

# Step 2 — bootstrap CI on the cosine, per layer
python scripts/bridge_cosine_bootstrap.py \
    --category familiarity_v3 \
    --hedging-category uu_hedge_confab_v3 \
    --layer-range 20-31 \
    --out-dir results/bridge_familiarity_v5

# Step 3 — permutation null on the cosine, per layer
python scripts/bridge_cosine_permutation.py \
    --category familiarity_v3 \
    --hedging-category uu_hedge_confab_v3 \
    --layer-range 20-31 \
    --out-dir results/bridge_familiarity_v5
```

- The output directory is named `bridge_familiarity_v5` deliberately, not `_v3` or `_v4` — those names are already taken by earlier *bridge-experiment* versions (different U/U pool sizes), unrelated to this dataset switch. Reusing either name would silently overwrite an existing, still-relevant result.
- `--hedging-category uu_hedge_confab_v3` (the 53-pair U/U pool, already built) is used here rather than the original 25-pair one, since there's no reason to pair a newly-widened familiarity pool with the smaller, superseded U/U pool.

---

## Order to run these in

LOTO and causal timing don't depend on each other or on the figure/bridge items — they can run in any order, or in parallel if GPU capacity allows. The figure regeneration (item 3) has no dependency on items 1, 2, or 4 either, since it only reads circuit-faithfulness numbers that already exist. Item 4 (bridge) is fully independent of the other three. There's no forced sequencing here — the only real constraint is deciding, once, which `--limit` (60 or 141) to use consistently across items 1 and 2, so the two mechanistic checks end up describing the same slice of the pool.
