# The U/U Falsification Test: What It Is and How to Run It

Governed by the pre-registration commit `c034557` (`l0-fixes`). This doc explains the test and gives the exact run order and commands. Nothing here changes anything locked in `c034557` — it's the execution plan for what that commit already specified.

---

## 1. The rundown — what this test actually is

The paper has, up to now, only shown that a circuit controls hedging on prompts where the model is uncertain because it doesn't know a fact (the familiarity arm). The U/U test asks a sharper question: can the same kind of circuit distinguish two models of *behaving* uncertain, within the same population of prompts the model doesn't know?

- **Hedger**: a prompt the model reliably says "I don't know" to (hedge rate ≥ 0.8).
- **Confabulator**: a prompt the model doesn't know either, but instead of hedging, confidently states a wrong answer (hedge rate ≤ 0.1).

Both are "unknown" to the model. Only one of them hedges. If there's a real hedging circuit, it should be possible to find components whose patching recovers the hedge/no-hedge difference between these two — not just between "knows it" and "doesn't know it." That's a materially harder and more specific claim than the familiarity result, which is why it's being held to its own pre-registered, falsifiable bar rather than just reported as a bonus finding.

**The test, as locked in `c034557`:** build the circuit (heads + neurons) from the *working* split of `uu_hedge_confab_v3` (53 pairs), then measure its faithfulness — patched recovery of the hedge signal — on the 23 *held-out* pairs, which must not have informed circuit selection in any way. The circuit counts as confirmed only if recovery exceeds **0.7, plain sample mean, no significance test, in both directions** (removal and injection). This mirrors exactly how the familiarity circuit was validated, applied here to a harder, pre-committed comparison.

**Why this needed a pre-registration commit at all:** the original confabulator *definition* was already adjusted once after an earlier attempt failed to discriminate hedgers from confabulators (the `n_distinct` clause was loosened from 2 to 10 after the fact). `c034557` exists so that this test — the one that actually uses the held-out pairs — can't be adjusted the same way after seeing its result.

---

## 2. One prerequisite gap: the pipeline needs a small, additive flag

The existing circuit scripts (`circuit_heads.py`, `circuit_mlp.py`) already correctly restrict component selection to the `working` split — no changes needed there, confirmed directly in the source.

`circuit_faithfulness.py`, however, is hardcoded to evaluate on the `held_out` split only (with a fallback to "all pairs" if no held-out split exists at all). There is currently **no way to run it against the working split instead** — which is exactly what's needed for the exploratory pass (step 4 below), since the confirmatory pass must be the *first and only* time the held-out 23 are evaluated.

**Proposed fix, additive only (same pattern as the earlier `--suffix`/`--reference-suffix` additions):** add an `--eval-split` argument to `circuit_faithfulness.py`, defaulting to `"held_out"` (byte-identical behavior to today for every existing call), with `"working"` as the only other allowed value for the exploratory pass.

```python
# in main(), alongside the other ap.add_argument(...) calls:
ap.add_argument("--eval-split", default="held_out", choices=["held_out", "working"],
                help="which split to evaluate faithfulness on; 'working' is for "
                     "exploratory estimates only (see uu_falsification_test.md)")
...
# replace the existing load_pairs call:
pairs, how = load_pairs(args.category, args.limit, args.eval_split)
if not pairs:
    pairs, how = load_pairs(args.category, args.limit, None); print("WARNING: no held-out split; using all pairs")
```

This is a one-line behavioral change (an explicit split argument instead of a hardcoded one) and doesn't touch anything else in the script. Every existing familiarity/conflict/aleatoric call that doesn't pass `--eval-split` keeps running exactly as before.

**Important, separate caveat about the exploratory pass this flag enables:** evaluating faithfulness on the *same* working pairs used to select the heads and neurons is in-sample — the same circularity concern the validation already raised for minimality/pruning. The exploratory SD and cross-direction correlation this produces will likely be a little optimistic (too small) relative to true variance, same direction of bias as using a held-out set would correct for. Treat the exploratory numbers as "better grounded than borrowing from the familiarity circuit," not as a clean, unbiased estimate.

---

## 3. Run order

All GPU steps use `unsloth/Meta-Llama-3.1-8B-Instruct`, bf16 — same model/precision as every other circuit build in this project. No dataset work needed: `data/uu_hedge_confab_v3/` already exists (`prompts.jsonl` = confabulator/"uncertain", `controls.jsonl` = hedger/"control", both carrying the `working`/`held_out` split field already, confirmed directly in the files).

### Step A — Position map (working split only)
```bash
python scripts/circuit_position_map.py \
    --category uu_hedge_confab_v3 \
    --split working \
    --limit 53 \
    --out-dir results/circuit_uu_hedge_confab_v3 \
    --overwrite
```
`--split` defaults to `None` (all pairs) in this script specifically — pass `working` explicitly here, or this stage alone would look at the held-out prompts too, before anything else has.

### Step B — Head attribution + verification (working split, enforced internally)
```bash
python scripts/circuit_heads.py \
    --category uu_hedge_confab_v3 \
    --limit 53 --verify 30 \
    --out-dir results/circuit_uu_hedge_confab_v3 \
    --overwrite
```

### Step C — MLP neuron attribution (working split, enforced internally)
```bash
python scripts/circuit_mlp.py \
    --category uu_hedge_confab_v3 \
    --limit 53 --window 12-26 \
    --out-dir results/circuit_uu_hedge_confab_v3 \
    --overwrite
```
`--window 12-26` is the script's own default, matching every prior circuit build (familiarity included) — not something `c034557` pre-registers, just the existing project convention, kept for comparability.

### Step D — Exploratory faithfulness pass (working split — requires the `--eval-split` flag from §2)
```bash
python scripts/circuit_faithfulness.py \
    --category uu_hedge_confab_v3 \
    --circuit-dir results/circuit_uu_hedge_confab_v3 \
    --n-heads 20 --n-neurons 100 \
    --eval-split working \
    --overwrite
```
This is the pass that produces a real, U/U-circuit-specific per-pair SD and cross-direction correlation — in-sample and somewhat optimistic (see §2), but still better grounded than the placeholder borrowed from the familiarity circuit in `c034557`.

### Step E — Rerun the power check with real numbers
```bash
python -c "
import pandas as pd
df = pd.read_csv('results/circuit_uu_hedge_confab_v3/faithfulness.csv')
circ = df[df['set']=='circuit']
piv = circ.pivot(index='pair', columns='direction', values='logodds_rec')
print('sd (control_to_uncertain):', piv['control_to_uncertain'].std())
print('sd (uncertain_to_control):', piv['uncertain_to_control'].std())
print('cross-direction correlation:', piv.corr().iloc[0,1])
"
python scripts/power_analysis_uu.py mde --n 23 --bound 0.7 --sd <real_sd>
python scripts/power_analysis_uu.py power --n 23 --bound 0.7 --sd <real_sd> --joint-rho <real_rho> --effects 0.05 0.1 0.15 0.2
```
**Decision point, per `c034557`'s pre-committed fallback:** if this shows the test is still adequately powered (roughly: 80% power to detect an effect smaller than or comparable to what's actually expected), proceed to Step F. If it comes back meaningfully worse than the provisional estimate in the commit, the committed response is to widen the held-out pool or report the eventual result as exploratory — not to loosen the 0.7 bar.

### Step F — THE confirmatory run (held-out split — the one-shot test)
```bash
python scripts/circuit_faithfulness.py \
    --category uu_hedge_confab_v3 \
    --circuit-dir results/circuit_uu_hedge_confab_v3 \
    --n-heads 20 --n-neurons 100 \
    --limit 23 \
    --overwrite
```
No `--eval-split` flag here — the script's existing default (`held_out`) is exactly correct and exactly what's pre-registered. This is the only time the 23 held-out pairs should be touched by this test.

---

## 4. How to read the result — don't use the script's own "verdict" line as-is

`circuit_faithfulness.py` prints its own `verdict: CIRCUIT (...)` line, but that checks a *different* criterion (control→uncertain recovery alone, plus a sparsity bound) — the convention used for the familiarity circuit, not the one locked for U/U.

**Apply the actual locked rule manually**, reading both direction means straight from `faithfulness.csv` (set == "circuit"):

```python
import pandas as pd
df = pd.read_csv("results/circuit_uu_hedge_confab_v3/faithfulness.csv")
circ = df[df["set"] == "circuit"]
cu = circ[circ["direction"] == "control_to_uncertain"]["logodds_rec"].mean()
uc = circ[circ["direction"] == "uncertain_to_control"]["logodds_rec"].mean()
print(f"control_to_uncertain: {cu:.3f}  (needs > 0.7)")
print(f"uncertain_to_control: {uc:.3f}  (needs > 0.7)")
print("PASSES the locked criterion" if cu > 0.7 and uc > 0.7 else "DOES NOT pass the locked criterion")
```

This is the actual falsification outcome — both numbers above 0.7, bare means, no test statistic, no confidence interval, exactly as `c034557` §C specifies.

---

## 5. What happens with the result

Whatever this returns feeds directly into step 8 of the broader plan ("look at results from [the factorial] and [this test], decide what the paper claims") — it isn't a separate write-up task, it's one of the two inputs that decision needs. If it passes: the paper gets to claim the hedging circuit generalizes beyond "doesn't know vs. knows" to "hedges vs. confabulates while equally not knowing" — a materially stronger result than anything currently in the paper. If it fails, or lands in the power-ambiguous band identified in Step E: that's a reportable negative result in its own right, consistent with this whole project's established pattern of disclosing rather than hiding results that don't hold up (same posture as the `" not"`-token finding and the relation-sorting artifact).
