# L2, Step 1 and Step 2 — Leave-One-Token-Out and Causal Timing
### Confidence Neurons Across Uncertainty Types — detail doc for the two tests that lead the master ordered list

Scope: exactly the two items that come first in the agreed no-branching order — **(1) leave-one-token-out (LOTO)** and **(2) causal timing**. Both are confirmed, in this session, against a freshly re-cloned `l0-fixes` branch (commit `6441cd8`, 11 commits, matches everything verified pre-reset) — nothing here is carried over unverified. Neither test touches L3, L4, or anything still blocked.

---

## 0. Why these two, and why first

They're first because they're the only unstarted work with **zero dependencies**. L2a (nulls) and L2b (necessity/minimality) are already done. L2e (saturation) is blocked on L3. L3 itself is a multi-day data-collection job. LOTO and causal timing need nothing but the code and data already on disk today — `results/circuit_familiarity/` (regenerated post-dedup) and the existing `Patch`/`Readout`/`recovery()` machinery in `circuit_common.py`. They can run this weekend, in parallel with starting L3.

Neither script exists in the repo yet:

```
$ ls scripts/ | grep -i "loto\|timing\|token_out\|jlens"
jlens_positions.py
jlens_steer.py
jlens_trajectory.py
```

Only the Jacobian-lens family exists. LOTO and causal timing are both net-new driver scripts, but both are *thin* — they reuse existing primitives rather than inventing new ones.

---

## 1. Test 1 — Leave-one-token-out (LOTO)

### 1.1 What it is and why it matters

The headline hedge/answer log-odds readout is defined in `direction_bridge.py`:

```python
HEDGE_TOKENS  = [" not", " I", " unknown", " unclear", " no"]
ANSWER_TOKENS = [" a", " the", " that", " in", " Yes", " yes"]
```

and computed by `Readout.logodds()` in `circuit_common.py`:

```python
def logodds(self, logits_last):
    lp = torch.log_softmax(logits_last.float(), -1)
    return torch.logsumexp(lp[..., self.hedge_ids], -1) - torch.logsumexp(lp[..., self.answer_ids], -1)
```

This is every result in the paper that cites "log-odds recovery" — `circuit_faithfulness.py`'s `logodds_rec`, the direction-patching cells, the bridge cosine work. All of it sits on top of one 5-token hedge set and one 6-token answer set, chosen once, by hand.

The concern a reviewer will raise: `" I"` is doing a lot of work in that hedge set. It's the only token that isn't semantically a hedge word on its own — it's a hedge token only in the specific completion pattern "I don't know" / "I'm not sure." If `" I"` is carrying most of the log-odds signal, the whole recovery number could be an artifact of one token's raw frequency rather than a genuine "the model is expressing uncertainty" signal. LOTO is the direct test: recompute every headline recovery number with each hedge token removed one at a time (5 runs) and with the full set (1 run), and confirm the qualitative result — set beats random, direction separates from set — survives all 6 versions, especially the one with `" I"` dropped.

### 1.2 The fix that makes this cheap

The naive approach — rerun `circuit_faithfulness.py` (and the direction-patching scripts) 5 more times, once per dropped token — would cost 5-6× the GPU time of the original run, because the expensive part is the forward passes (baseline, patched, source-capture), not the log-odds arithmetic.

That's unnecessary. `Readout.logodds()` only touches the **final-token logits**, which are already being computed and thrown away as an intermediate. The correct design:

1. **Cache once.** Modify (or wrap) the faithfulness/direction-patching forward passes to save the raw last-token logit vector (`logits_last`, full vocab, one float32 vector per pair per condition — target/source/patched, both directions) instead of immediately reducing it to `logodds()`/`entropy()` and discarding it. This is the same fact flagged in the L2 doc: `circuit_faithfulness.py` currently clips and discards raw values immediately — nothing is persisted for offline recomputation despite the plan's claim otherwise. Confirmed again this session: `recovery()`'s only inputs are `logodds()`/`entropy()` scalars computed inline (`circuit_faithfulness.py` lines ~89-95); nothing upstream of that is written to `faithfulness.csv`.
2. **Recompute offline, 6 times, on CPU, in seconds.** For each of the 6 hedge-token-set variants (full set, minus `" I"`, minus `" not"`, minus `" unknown"`, minus `" unclear"`, minus `" no"`), rerun only `torch.logsumexp(lp[..., hedge_ids], -1) - torch.logsumexp(lp[..., answer_ids], -1)` on the cached logits, then `recovery()` on the resulting scalars. Zero additional forward passes.

This is a genuine one-GPU-pass-then-six-CPU-recomputes design, not a corner cut — it's strictly more information (raw logits) sitting behind the same headline numbers.

### 1.3 What needs building

**New script: `scripts/circuit_faithfulness_cache.py`** (or a `--cache-logits` flag added directly to `circuit_faithfulness.py` — cleaner, since it reuses 100% of the existing pair-loading/patching loop and only adds one `np.save`/`torch.save` call per condition instead of discarding).

Concretely, inside the existing loop in `circuit_faithfulness.py` (the loop that currently does `rows.append(dict(pair=k, set="circuit", ..., logodds_rec=recovery(lo, lo_u, lo_c), ...))`), the raw scalars `lo`, `lo_u`, `lo_c` (and their entropy counterparts `Hh`, `H_u`, `H_c`) are already logits-derived scalars — but they're derived from `logits_last`, the full-vocab vector, which is what actually needs caching. So the real change is one level up: in `patch_all()` and wherever `lo_u`/`lo_c` (baseline logits under each twin) are computed, save `logits_last` (or just the 11 needed vocab indices — hedge ∪ answer token ids — to keep the cache tiny) to a per-pair, per-condition file.

- **Cache format:** one `.npz` or parquet file, keyed by `(pair, set, direction, condition ∈ {target, source, patched})`, storing the logit values at exactly the 11 hedge/answer token ids (not the full ~128k vocab — cuts cache size by ~10,000×). Since `HEDGE_TOKENS ∪ ANSWER_TOKENS` is fixed and known ahead of time, cache exactly those 11 scalars per condition. This makes the cache trivially small (~207 pairs × 2 directions × 3 conditions × 11 floats ≈ 14K numbers) and fast to reload.
- **Recompute script: `scripts/loto_recompute.py`** — loads the cache, and for each of the 6 hedge-set variants, recomputes `logodds()` (only the 5-or-4-token `logsumexp` changes; `ANSWER_TOKENS` stays fixed since the question here is specifically about the hedge set, not the answer set) and `recovery()`, reusing `boot_ci()` (replicated from `stats_upgrades.py`, per this branch's established pattern — that module runs its whole analysis as an import side effect, so every downstream script on `l0-fixes` copies the ~5-line function rather than importing it) for a bootstrapped CI per variant.
- **Output:** `results/circuit_familiarity/loto.csv` with columns `variant, direction, logodds_rec_mean, ci_lo, ci_hi, n_pairs` — one row per (dropped-token variant × direction), 12 rows total (6 variants × 2 directions).
- Same cache-once design applies to the rank-1 direction-patching results (`circuit_direction_patch.py`) if LOTO is meant to cover that cell too — worth deciding explicitly: does LOTO apply only to the circuit-set recovery number (the headline claim), or also to the direction-patching comparison? Recommend covering both, since the dissociation claim (L1) also rests on the hedge-token readout, and a reviewer could ask the same "is this just `" I"`" question there too. That doubles the caching work (same pattern, one more script instrumented) but not the GPU cost — still one pass each.

### 1.4 What "pass" looks like

The existing headline claims must survive qualitatively:
- Circuit-set recovery stays materially above the random-null band in all 6 variants (full set and each single-token-dropped version).
- If dropping `" I"` alone collapses the recovery number toward the null band, that's a real finding, not a bug — it means the result is more `" I"`-dependent than currently claimed, and the paper's Limitations section needs to say so explicitly rather than LOTO being silently swept aside. Report whatever comes back.

---

## 2. Test 2 — Causal timing

### 2.1 What it is, and what it replaces

The plan originally used a **Jacobian-lens** analysis (`jlens_trajectory.py`, already in the repo) to ask "where does the hedge decision become linearly readable?" — decoding the residual stream at every layer through a pretrained lens and checking where the two twins' log-odds diverge. That's a useful *correlational* observation, but it's not causal: linear readability doesn't establish that the information patched from that point on is *sufficient* to move the model's actual output. It's also an external dependency (`neuronpedia/jacobian-lens`) and, per the ACL page-budget decisions already made, being demoted to an appendix corroboration rather than a headline result.

**Causal timing** is the causal replacement, and it's a direct extension of the exact patching machinery already used for the circuit-faithfulness test — no new low-level primitive required. The question: instead of patching a *specific set of components* (heads+neurons) at *all* layers simultaneously, patch the **entire residual stream**, cumulatively, starting from each candidate layer ℓ through the final layer, and measure at which ℓ the recovery crosses the "sufficient" threshold. That identifies *when* (which layer onward) the uncertain-vs-control difference becomes causally transportable — the causal analogue of "the trajectory lens shows separation starting around layer 13."

### 2.2 Why this needs no new low-level code

Confirmed again this session, `circuit_common.py`'s `Patch` class already supports exactly this:

```python
class Patch:
    """... edits: list of (layer, kind, position_pairs, source_act, index) where
      kind 'resid'/'mlp' -> replace output[:, tgt_pos, :] with source_act[:, src_pos, :]
      ..."""
```

`kind="resid"` replaces the **entire residual-stream output** of a layer block at the given target positions with the corresponding source-run activations — precisely the "full residual stream patch" causal timing needs. This is registered as a forward hook on `blk` itself (via `component_module(model, layer, "resid")` → `blk`), not on a sub-module, so a single `Patch` call with `edits = [(l, "resid", pmap, src_resid[l], None) for l in range(start_layer, n_layers)]` patches every layer from `start_layer` to the end in one forward pass. (The `index` field is unused for `"resid"`/`"mlp"` kinds — only `"heads"`/`"neurons"` use it to select a sub-slice — so it can be passed as `None` or omitted per the existing `_apply()` logic, which only branches on `kind`.)

`run_capture(model, enc_s, list(range(n_layers)), "resid")` (already-existing helper) gets the full set of source-run residual-stream activations to patch from, in one pass. `run_patched(model, enc_t, edits)` (already-existing helper) runs the patched forward pass. `Readout.logodds()` / `Readout.entropy()` and `recovery()` are unchanged. This means the entire causal-timing test is a new **driver script that loops over candidate start layers**, not new patching logic.

### 2.3 What needs building

**New script: `scripts/circuit_causal_timing.py`**, structured directly on `circuit_faithfulness.py`'s pattern (same imports: `Readout, encode_pair, load_pairs, position_map, recovery, reverse_map, run_capture, run_patched` from `circuit_common`, `add_model_args, guard_output, load_model_from_args` from `_common`):

- **Args:** `--category` (default `familiarity`), `--limit` (held-out pairs, default 60 to match `circuit_faithfulness.py`'s convention), `--layers` (candidate start layers to sweep — recommend every layer 0 through the model's final layer, or a coarser stride of 2 to cut cost, since this is now a per-layer sweep rather than a single run), `--seed`.
- **Core loop**, per candidate start layer ℓ:
  1. `run_capture(model, enc_source, list(range(l, n_layers)), "resid")` — capture the *source* twin's residual stream at every layer from ℓ onward.
  2. Build `edits = [(layer, "resid", pmap, src_resid[layer], None) for layer in range(l, n_layers)]` using the existing `position_map(al, group)` alignment (same `all_position_pairs()` helper already defined in `circuit_faithfulness.py` — reuse it verbatim, or promote it into `circuit_common.py` so both scripts import it rather than duplicating).
  3. `run_patched(model, enc_target, edits)` → logits → `Readout.logodds()` / `.entropy()` → `recovery()` against the same target/source baseline scalars used elsewhere.
  4. Repeat for both directions (control→uncertain, uncertain→control), matching `circuit_faithfulness.py`'s convention.
- **Bootstrap CI per layer**, via the replicated `boot_ci()` pattern already established on this branch (10k resamples, seed 0, percentile method) — one CI per candidate start layer, giving a recovery-vs-start-layer curve with uncertainty bands, directly comparable to (and replacing) the Jacobian-lens trajectory plot.
- **Output:** `results/circuit_familiarity/causal_timing.csv`, columns `start_layer, direction, logodds_rec_mean, entropy_rec_mean, ci_lo, ci_hi, n_pairs`. One row per (start layer × direction).
- **A companion figure** (new, or a variant folded into `fig_dissociation.py`'s pattern rather than `make_paper_figures.py`'s `save()`, which — per the standing "preserve, don't overwrite" rule — has no overwrite guard): recovery-vs-start-layer, with the existing L13-19 (familiarity) / L11-21 (conflict) component-patching window drawn as a reference band, so the figure directly shows "here's where the causal window starts, corroborating (or not) the range already used for rank-1 direction patching."

### 2.4 A cost note worth flagging explicitly

Unlike LOTO, causal timing genuinely needs one forward pass **per candidate start layer** (patching cumulatively from ℓ onward requires a distinct patched run for each ℓ, since the patched region changes) — this is not reducible to a cache-and-recompute trick, because the *causal intervention itself* differs per layer, not just the readout. If sweeping every layer individually (32 layers for Llama-3.1-8B) is too expensive against 60 held-out pairs × 2 directions, use `--layers` with a stride of 2 or 4 first to locate the approximate transition zone, then refine with single-layer steps only in that neighborhood — same adaptive-resolution logic as a binary search, cutting total passes substantially versus a naive full sweep.

### 2.5 What "pass" looks like

- A clear recovery-vs-layer curve: recovery near the random-null floor for early ℓ, rising to the >0.7 "circuit" threshold at some transition layer, and staying high through to the final layer (patching from later and later layers necessarily includes less of the network's own processing, so recovery should not increase past the transition point — a sanity check on the implementation, not just the science).
- The transition layer roughly aligning with the existing L13-19 window used for rank-1 direction patching is the corroborating result the paper wants (replacing the Jacobian-lens's correlational version of the same claim with a causal one). If it doesn't align, that's a finding to report honestly, not a bug to chase away — same standing principle as the LOTO test and the earlier permutation-null work.

---

## 3. Timeline

Both are independent of each other and of everything else currently blocked (L3, L4). They can run in parallel on separate GPU allocations, or sequentially if only one GPU is available — LOTO is cheaper (1 pass + 6 cheap CPU recomputes) and should go first if resources are shared.

| Day | LOTO | Causal timing |
|---|---|---|
| **Day 1 (today/tomorrow)** | Instrument `circuit_faithfulness.py` with the logit-cache (`--cache-logits` flag or new `circuit_faithfulness_cache.py`); rerun once on `familiarity` to populate the cache (same GPU cost as a normal faithfulness run — this is not an extra run, just an instrumented rerun of one already-planned regeneration, if one is still pending; otherwise one fresh pass). | Write `scripts/circuit_causal_timing.py`; promote `all_position_pairs()` into `circuit_common.py` so both scripts share it. Do a small smoke test (`--layers` = 3-4 hand-picked layers, `--limit 10`) to confirm the patched forward pass runs and produces sane recovery values before committing to a full sweep. |
| **Day 2** | Write `scripts/loto_recompute.py` (offline, CPU-only, uses replicated `boot_ci()`); run all 6 variants × 2 directions; inspect the `" I"`-dropped row specifically. | Run the coarse-stride sweep (every 2-4 layers) across the full held-out set (60 pairs) × 2 directions; identify the approximate transition zone. |
| **Day 3** | Write up the result — a short table (6 variants × 2 directions × recovery + CI) — and decide whether it needs its own doc-update to `claims_and_changes_L0_through_L1.md`-style tracking, or folds directly into the L1/L2 combined write-up. Repeat for the direction-patching cell if scope is extended per §1.3's open question. | Refine with single-layer steps in the identified transition neighborhood; produce the recovery-vs-layer CSV and figure. |
| **Day 4** | — | Compare the causal transition layer against the existing L13-19 direction-patching window; write up the corroboration (or divergence) result. Decide whether this fully replaces `jlens_trajectory.py` in the main text (moving it to appendix, per the existing ACL-page-budget plan) or runs alongside it. |

Both should be done well inside a 3-4 day window, running fully in parallel with L3's (multi-day) data collection starting the same day — neither should be allowed to block or be blocked by L3.

---

## 4. Open decisions to flag before starting

1. **LOTO scope** — headline circuit-set recovery only, or also the rank-1 direction-patching cell (needed if the dissociation claim itself should be shown to survive LOTO too)? Recommend both; doubles instrumentation work, not GPU cost.
2. **Causal timing figure placement** — new standalone figure, or extend `fig_dissociation.py`'s pattern into a new `fig_causal_timing.py`? Either keeps it off `make_paper_figures.py`'s unguarded `save()`, consistent with the standing overwrite-safety rule.
3. **Does causal timing fully replace `jlens_trajectory.py` in the main text**, or run alongside it as belt-and-suspenders (causal in the body, correlational lens corroboration in the appendix, as already planned)? No need to decide before starting the implementation — only before the manuscript section is drafted.

None of these three decisions block writing or running the code above; they only affect how the results get reported once they're in hand.
