# Step 9 (L5): Does Patching the Circuit Change What the Model Actually Says?

Scope: the one thing step 9 needs that isn't done yet. Checked directly against the repo (fresh clone, `l0-fixes` @ `3284bc2`, nothing newer) before writing this, because the obvious first guess — "there's already a behavioral-test script, just rerun it" — turned out to be half right and half a trap. Both halves are below.

---

## 1. Why this step exists

Every result in the paper so far — familiarity, U/U, the factorial — is a **logit-probability** claim: patch some heads/neurons from a control twin into an uncertain twin's single forward pass, and check whether the hedge-vs-answer *log-odds at the next token* recovers. That's `circuit_faithfulness.py`'s whole method.

Nothing has yet asked the question a reader will ask immediately: if you patch this circuit and let the model **actually keep talking**, does it say something different — hedge instead of confabulating, or vice versa — or does the probability shift wash out over the next 20–30 generated tokens? That's a real, separate claim, and it's the one `unireps_revision_notes.md` calls C3 and marks as "now required before any decision-vs-spread claim." It's also exactly what step 9/L5 in the 10-step plan means by "behavioral validation."

This is not a falsification test like U/U — it's a validity check on the measurement itself. But it still feeds directly into a headline framing decision (Section 2 below), so it's worth being disciplined about even without a formal pre-registration.

---

## 2. What already exists — and why it doesn't answer the question

Two scripts already do free-generation behavioral testing: `scripts/behavioral_test.py` (single neuron) and `scripts/multi_neuron_behavioral.py` (named neuron sets, plus a verbalized-confidence follow-up question). Both have already been run — `results/multi_neuron_behavioral_summary.csv` and `results/multi_neuron_behavioral_direction_summary.csv` exist with real output.

Checked directly, three things make these runs **not usable as-is** for step 9:

- **They're old.** `git log` shows both scripts last touched at commit `0202d4b` (2026-08-22), and the saved provenance confirms the same `git_head_sha`. That's before `familiarity_v3`, before the circuit pipeline (`circuit_heads.py`/`circuit_mlp.py`/`circuit_faithfulness.py`) existed, before `l0-fixes`, before the independent validation. The neuron sets tested (`freq`, `entropy`, `corrfreq`, `key`, `dirfam`, `dirfam5`) come from an earlier correlation/direction-based candidate-selection approach, not from the heads+neurons the paper now actually calls "the circuit."
- **They only clamp MLP neurons.** `ClampHook`/`MultiClampHook` both hook `mlp.down_proj` only. There's no head-patching anywhere in either script. The circuit that `circuit_faithfulness.py` validates is heads **and** neurons together (`component_set()` reads both `head_verification.csv` and `neuron_attribution.csv`) — a neuron-only behavioral test can't speak to that circuit's behavioral effect, only to a strict subset of it.
- **The existing results are a near-null, worth noting but not conclusive.** Every real neuron set (`freq`, `entropy`, `corrfreq`, `key`, `dirfam`) shows a hedge-rate delta indistinguishable from its random-neuron controls (deltas of ±0.01–0.02 across the board, both real and random). Taken at face value this would already be bad news for decision-vs-spread — but since it's testing the wrong (stale, pre-pipeline) candidate sets with the wrong (neuron-only) intervention, it can't be read as evidence about the actual circuit. It's a reason to take the question seriously, not an answer to it.

**Conclusion: a new script is needed.** This is the opposite situation from step 5 (factorial), where the existing pipeline turned out to need no changes — here the existing pipeline covers a different, superseded object.

---

## 3. What the new script needs to do, and the one design choice it forces

The natural approach is to reuse `circuit_faithfulness.py`'s own patching machinery (`circuit_common.Patch`, `component_set()`, `align()`/`position_map()`) — same heads, same neurons, same source-into-target transplant — but call `model.generate()` instead of reading one forward pass's logits.

That raises a real design question `circuit_faithfulness.py` never had to answer: **patching is defined by aligning positions between two fixed prompts** (prefix/entity/suffix, computed once by `align()`). A single forward pass over the full prompt has well-defined aligned positions. But once generation starts producing new tokens, the uncertain prompt's continuation and the control prompt's continuation diverge token-by-token — there is no principled "source position" to patch a generated position against.

**Resolution:** patch only the prompt positions, during the one forward pass that produces the logits for the first generated token (this is the same forward pass `circuit_faithfulness.py` already scores) — then continue standard greedy decoding, unpatched, for the remaining `--max-new-tokens`. This tests the real claim at stake: does flipping the circuit's state at the decision point change the token actually sampled and the continuation that follows from it, not just the raw log-odds. It's a weaker intervention than clamping every generated position (which `behavioral_test.py` does, but only because clamping to a fixed value is well-defined at any position — transplanting from a twin's specific position is not). This is a one-shot "decision" test, which is exactly in keeping with the paper's own "decision vs. spread" framing — if the circuit only moves the decision point and the rest is just whatever the model would have said anyway, that's itself informative, not a weakness of the test.

Everything else reuses existing pieces directly:
- `component_set()`, `Patch`, `align()`/`position_map()` from the circuit pipeline — unchanged.
- `HEDGE_RE`, `generate_batch()`'s left-padding/batching pattern from `behavioral_test.py` — unchanged.
- The per-pair random-component-set null from `circuit_faithfulness.py` (`--n-random`) — same idea, applied to generation instead of logits.

---

## 4. Scope: which circuit(s), and in what order

**Primary target: `circuit_familiarity_v3`.** This is the paper's headline circuit, the one the decision-vs-spread claim is actually about, and the one with the most held-out pairs (141) to spare. This is the run that resolves the C3 fork.

**Secondary, only if time allows:** `circuit_uu_hedge_confab_v3` (23 held-out — already a thin margin for the faithfulness test; a behavioral rerun on the same 23 pairs adds information but doesn't change the U/U falsification verdict already locked) and the two factorial circuits (`know_given_hi`, `pop_given_unknown` — useful corroboration, not required for any pending decision). Don't let either secondary run block getting the familiarity_v3 answer first.

**Pool:** held-out split only (141 pairs for familiarity_v3) — same discipline as every other confirmatory circuit result in this project, even though this isn't a formal falsification test. There's no working-split "exploratory pass" prerequisite here the way U/U needed one (no new threshold or criterion is being tuned — the circuit, the split, and the comparison metric are all already fixed), so there's no reason to touch working pairs first.

---

## 5. A judgment call worth flagging before running this

Unlike the factorial, this result is going straight into a headline framing decision (decision-vs-spread, stated as the core dissociation claim in the paper). The project's established pattern elsewhere has been: lock the read-off criterion *before* seeing results whenever the result will directly support or sink a specific claim (that's the entire reason the U/U pre-registration commit exists). This step doesn't currently have a locked criterion — "hedge rate changes meaningfully in the circuit condition but not in random-set controls" is directly analogous to U/U's bar, but no threshold or statistical test has been specified.

Worth deciding explicitly: run this as a quick exploratory check first (informal read of the numbers, decide how to report from there), or write a short lock-the-criterion note first (even a lighter one than U/U's, given lower stakes — e.g., "hedge-rate delta in the circuit condition must exceed its own random-set null by some pre-stated margin, both directions"). I'd lean toward the lighter, undocumented version given this is a validity check rather than a falsification test and the paper doesn't currently promise a pre-registered behavioral bar — but this is a call about how much rigor-signaling the paper needs here, not a technical one, so flagging it rather than deciding it.

---

## 6. Script sketch and commands

New script, e.g. `scripts/circuit_behavioral.py` (name picked to sit next to `circuit_faithfulness.py` in the pipeline; adjust to taste):

```bash
python scripts/circuit_behavioral.py \
    --category familiarity_v3 \
    --circuit-dir results/circuit_familiarity_v3 \
    --n-heads 20 --n-neurons 100 \
    --limit 141 \
    --max-new-tokens 24 \
    --n-random 10 \
    --out results/circuit_familiarity_v3/behavioral.csv
```

What it needs to do, concretely:
1. Load heads/neurons via `component_set(circ_dir, n_heads, n_neurons)` — identical to `circuit_faithfulness.py`.
2. Load held-out pairs via `load_pairs(category, limit, "held_out")` — identical.
3. For each pair: tokenize both twins, compute `align()`/`position_map()` exactly as `circuit_faithfulness.py` does.
4. Generate the **clean** continuation for both twins (no patching) via `generate_batch()` (reused from `behavioral_test.py`), batched for throughput.
5. Generate the **patched** continuation: wrap the uncertain prompt's forward pass in `circuit_common.Patch(model, edits)` (same `edits` list `patch_all()` builds) for the prefill step only, then call `model.generate()` — since `Patch` is a context manager registering forward hooks, the natural implementation is to run the first forward step (prefill) inside the `Patch` context to get the first token, remove the hooks, then continue `generate()` for the rest unpatched. (This needs a small custom generation loop rather than a single `model.generate()` call with hooks left on for the whole call — the hooks must come off after the first produced token, which a plain hook-and-call-generate pattern like `behavioral_test.py` uses doesn't support since its hooks are meant to apply at every step.)
6. Reverse direction the same way (patch control-from-uncertain).
7. `--n-random` matched random head/neuron sets, same size, for the null — reuse the random-sampling logic already in `circuit_faithfulness.py`'s main loop.
8. Score: `HEDGE_RE.search()` on each generated continuation (clean, circuit-patched, each random set), per pair, both directions.
9. Summarize per set: hedge rate on the twin that should flip (uncertain→hedges-less-on-patch-from-control direction and vice versa), delta vs. clean, and the circuit's delta vs. the random-set null distribution — same shape as `multi_neuron_behavioral.py`'s summary table, so it's directly comparable to the stale run already on file.

### Reading the result
```python
import pandas as pd
df = pd.read_csv("results/circuit_familiarity_v3/behavioral.csv")
circ = df[df["set"] == "circuit"]
rnd = df[df["set"].str.startswith("random")]
for d in ["control_to_uncertain", "uncertain_to_control"]:
    g = circ[circ["direction"] == d]
    print(d, "hedge delta vs clean:", (g.hedged_patched.mean() - g.hedged_clean.mean()))
print("random null hedge delta:", (rnd.hedged_patched.mean() - rnd.hedged_clean.mean()))
```

---

## 7. What this feeds

Directly resolves the C3 fork in `unireps_forward_plan.md`:
- **If the circuit's patched continuation actually flips hedge/confabulate behavior, clearly above the random-set null:** decision-vs-spread stays the headline, with the existing `" not"`-token-selection caveat (B5) stated alongside it.
- **If it doesn't:** the honest fallback is gbEf's own characterization — a well-built gated twin-pair methodology for isolating uncertainty circuits, with the decision/spread dissociation reported as a finding that didn't survive its own behavioral audit. Also a legitimate extended abstract, just differently centered.

Either way, this is independent of step 8 (writing up the U/U and factorial claims) — see the earlier answer on sequencing. Once this result is in hand, it slots into step 8's synthesis as a third input alongside U/U and the factorial, rather than requiring its own separate claims decision.
