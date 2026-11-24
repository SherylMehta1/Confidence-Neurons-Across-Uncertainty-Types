# Fixing `bridge_familiarity`'s Hedging Direction — Option 3b
### Source the hedge-vs-non-hedge contrast from the U/U arm
### ACL Rolling Review, October 2026 cycle — "Confidence Neurons Across Uncertainty Types"

**Decision made:** Option 3b — reuse the plan's U/U arm (hedgers vs. confabulators, drawn from genuinely-unknown items) as the source population for `direction_bridge.py`'s hedging direction, instead of the current `hedge_rate >= 0.5` split within the gated familiarity twin set.
**Reality check before anything else:** the U/U arm does not exist as code or data in this repo yet. Nothing in `scripts/` or `data/` builds it. This is not a flag-flip fix — it requires writing one new script and a small, targeted change to `direction_bridge.py`. The good news, worked out below, is that it does **not** require rerunning the model to re-score anything new.

---

## 1. The mechanism problem, restated precisely

In `scripts/direction_bridge.py`, the hedging direction is currently built here:

```python
if all("hedge_rate" in r for r in work):
    hedged = np.array([float(r["hedge_rate"]) >= 0.5 for r in work])
...
hed[l] = diff_of_means(X[l], unc & hedged, unc & ~hedged)
```

`work` is the **familiarity** category's working-split prompts — the same twin-gated pool used for the familiarity direction, which is pre-selected for `hedge_gap >= 0.3` between each pair. That's why, post-regeneration, only 2 of 144 uncertain-arm working items are genuine non-hedgers: the population this split is drawn from was never meant to contain a balanced hedge/non-hedge contrast, it was built to contrast *known* vs. *unknown*.

**What 3b changes:** the *familiarity* direction stays exactly as it is — computed from the familiarity category's `unc`/`is_ctrl` split, untouched. Only the **hedging** direction's source population changes, to a new category built specifically to have balanced, genuine hedgers and confabulators.

---

## 2. Why this doesn't require rerunning the model

The U/U arm's source population is "genuinely unknown items" — and thanks to **L0's Fix 1 (the retention bug)**, every item the model was ever scored on during the original familiarity screen — including every `Unknown`-classified item that never made it into a twin pair — is already sitting in `data/familiarity/all_measured.jsonl`, with `hedge_rate`, `entropy`, `n_distinct`, `frac_correct`, `s_pop`, and `n_q_tokens` already computed. This is precisely the pool the plan's original U/U design calls for: *"From the screen's Unknown items, pair hedgers (≥0.8 over 20 generations) with confabulators (≤0.1, ≤2 distinct answers), matched on relation, popularity, token counts, and first-token entropy within 0.3 nats."*

Because `all_measured.jsonl` already has all of this, at the current `n_free=20` resolution (post Fix 4), building the U/U arm is a **pure data-wrangling step** — filter, classify, pair, write — not a new model-scoring run. This is exactly the payoff Fix 1's retention writer was built for, showing up in a place well beyond the frequency-twin use case it was originally motivated by.

---

## 3. The new script: `scripts/build_uu_hedge_confab.py`

Modeled directly on `build_familiarity_twins.py`'s existing pairing/gating logic, but with genuinely different selection and matching criteria — this is not a copy-paste, the axis being separated is hedging behavior, not knowledge.

**Source pool:** rows from `data/familiarity/all_measured.jsonl` where `arm == "uncertain"` and `slick_class == "Unknown"` — i.e., items the model was tested on as candidates for the uncertain side of a familiarity twin, and which it genuinely doesn't know the answer to (this is the "genuinely unknown" population the plan's U/U spec calls for).

**Classification, per the plan's pre-registered thresholds:**
- **Hedger**: `hedge_rate >= 0.8` (hedges in at least 16 of 20 free generations)
- **Confabulator**: `hedge_rate <= 0.1` **and** `n_distinct <= 2` (rarely hedges, and commits to a small number of distinct wrong answers rather than casting around)

Note this is very close to, but not identical to, the existing `is_correct`-derived "consistently wrong" definition already used elsewhere in the codebase (`n_distinct <= 1 and frac_correct == 0.0`) — the plan's own spec is slightly more permissive (`≤2` distinct, not `≤1`, and doesn't reference `frac_correct` since these are all already `Unknown`-classed and thus `frac_correct` uncertainty-adjacent). Use the plan's literal spec here rather than silently reusing the stricter existing definition, since the two serve different purposes (one is a negative-control exclusion in a different pairing flow; this is a direct classification for U/U's contrast).

**Matching** (within relation, same pattern as `build_familiarity_twins.py`'s existing pairing loop):
- Same `prop` (relation)
- Popularity (`s_pop`) — closest available match within relation
- Question token length (`n_q_tokens`) — within some `--max-len-diff` (reuse the existing default of 3)
- First-token entropy within 0.3 nats (`abs(hedger["entropy"] - confab["entropy"]) <= 0.3`) — this is a genuinely new gate not present in the existing pairing code, needs to be added explicitly, matching the plan's literal spec

**Output**, in the same repo schema everything else uses, so `load_category()` works unmodified:
```
data/uu_hedge_confab/prompts.jsonl     # hedgers  (is_control=False)
data/uu_hedge_confab/controls.jsonl    # confabulators (is_control=True)
data/uu_hedge_confab/gate_report.json  # counts, matching diagnostics
data/uu_hedge_confab/prompts.provenance.json
```

`is_control=True` for confabulators is a slight repurposing of the "control" field's usual meaning (normally "known"), but it's the correct mechanical fit — `direction_bridge.py`'s `diff_of_means(X[l], unc & hedged, unc & ~hedged)` doesn't care what "control" means semantically, it just needs a boolean split, and `load_category()`'s `(prompts, controls)` split maps cleanly onto `(hedgers, confabulators)`.

---

## 4. The change needed in `direction_bridge.py`

Currently, one `--category` argument drives both the familiarity direction and the hedging direction — that's the coupling causing the problem. Add a second, independent argument:

```python
ap.add_argument("--hedging-category", default=None,
                help="category to source the hedging direction from (default: same as --category)")
```

Then, in `main()`, load a second working set when this is provided:

```python
hedging_cat = args.hedging_category or args.category
if hedging_cat == args.category:
    hed_work = work           # unchanged behavior — same pool for both directions
    hedged = np.array([float(r["hedge_rate"]) >= 0.5 for r in hed_work]) if all("hedge_rate" in r for r in hed_work) else ...
    hed_unc = unc
else:
    h_prompts, h_ctrls = load_category(hedging_cat)
    h_recs = [dict(r, is_control=False) for r in h_prompts] + [dict(r, is_control=True) for r in h_ctrls]
    hed_work = [r for r in h_recs if r["split"] == "working"]
    Xh = last_token_residuals(model, tokenizer, [r["chat_formatted_prompt"] for r in hed_work], layers)
    hedged = np.array([not r["is_control"] for r in hed_work])   # hedgers = "prompts" side = not-control
    hed_unc = np.ones(len(hed_work), dtype=bool)                # every row here is a genuine "uncertain" item by construction
```

and change the direction computation:

```python
hed[l] = diff_of_means(Xh[l] if hedging_cat != args.category else X[l], hed_unc & hedged, hed_unc & ~hedged)
```

This keeps the **default behavior fully backward-compatible** (omit `--hedging-category` and everything works exactly as before, for the `conflict` category or any other arm where the old approach was never broken) — the only thing changing is the specific call used to regenerate `bridge_familiarity`, which now explicitly separates the two source populations:

```
python scripts/direction_bridge.py --category familiarity --hedging-category uu_hedge_confab --layer-range 20-31 --steer --overwrite --out-dir results/bridge_familiarity_v3
```

Note the new output directory name (`bridge_familiarity_v3`, not overwriting `_v2` or the original) — same "preserve, don't overwrite" principle used throughout this project so far.

---

## 5. Order of operations

1. **Confirm the prerequisite is actually in place**: `data/familiarity/all_measured.jsonl` must be the *post-dedup, `n_free=20`* version (i.e., after L0's full regeneration, not the original pre-fix run) — check its row count and a spot-check of `hedge_rate` values (should show fine-grained values, not just quantized thirds) before building anything on top of it.
2. **Write and run `build_uu_hedge_confab.py`** — pure CPU/pandas work, no model calls, should be fast. Inspect `gate_report.json` for the resulting hedger/confabulator counts before proceeding — confirm both groups are now reasonably sized (the whole point of this fix), not just "better than 2, still thin."
3. **Add the `--hedging-category` flag to `direction_bridge.py`**, test it in backward-compatible mode first (omit the flag, confirm output matches a normal familiarity-only run) before testing the new cross-category mode.
4. **Rerun `direction_bridge.py` for familiarity** with `--hedging-category uu_hedge_confab`, writing to a fresh, clearly-named output directory.
5. **Compare against the broken `bridge_familiarity_v2` run**: familiarity-direction numbers (top neurons, familiarity steering curve) should be nearly identical, since that direction's source data didn't change — this is your sanity check that the fix touched only what it was supposed to. The hedging-direction numbers (cosine with familiarity, hedging steering curve, top cos_hedging neurons) are expected to differ, and should now come with a defensible, balanced-group justification behind them.

---

## 6. What this does and doesn't resolve

**Resolves:** the underpowered/degenerate hedging contrast (142 vs. 2) that made the previous cosine and steering numbers untrustworthy. The new contrast is drawn from a population purpose-built to have both classes well represented.

**Does not resolve, and isn't meant to:** whether the *familiarity* direction and this *hedging* direction turn out to be aligned or not is now an honest empirical question again — 3b fixes the measurement, it doesn't presuppose or protect a particular answer. If the new cosine similarity comes back near zero, or even negative, that's a real result to report, not something to second-guess because the old (broken) number was more favorable.

**Relationship to L4's actual falsification test:** this U/U dataset serves double duty — it's what `bridge_familiarity` needs here, and it's also the population the original 31-day plan's L4 falsification criteria are pre-registered against (hedge-circuit recovery ≤0.2 on this population vs. ≥0.7 on the standard arm). Building it now for this fix effectively pulls a piece of L4 forward. Worth flagging explicitly: if L4's actual falsification run later uses a *different* instantiation of the U/U pairing (different seed, different candidate pool state, etc.), the pre-registration commit for L4's thresholds should be checked against whichever exact dataset build ends up used for the falsification test itself — this bridge fix's version doesn't automatically satisfy that pre-registration requirement on its own, since pre-registration is about committing to thresholds *before* seeing that specific run's results.

---

## 7. Verification checklist

- [ ] `data/familiarity/all_measured.jsonl` confirmed to be the post-dedup, `n_free=20` version before building the U/U pool from it
- [ ] `build_uu_hedge_confab.py` output inspected: hedger and confabulator counts both reasonably sized (not just "more than 2")
- [ ] Matching diagnostics checked — relation coverage, popularity/length/entropy balance across the two groups (same spirit as the standardized-mean-difference check used for the familiarity twins)
- [ ] `--hedging-category` backward-compatibility verified (default behavior unchanged when the flag is omitted)
- [ ] Familiarity-direction numbers compared old vs. new run — should be stable, confirming the fix is isolated to the hedging direction only
- [ ] New hedging-direction numbers (cosine, steering) reported honestly, whatever they turn out to be
- [ ] A note added to the manuscript's methods section explaining why the hedging direction for the familiarity arm is sourced from the U/U population rather than the gated twin set — this is a methodological choice worth stating plainly, not something to leave implicit
- [ ] If/when L4's actual U/U falsification test runs, confirm whether it reuses this same dataset build or requires its own pre-registered instantiation
