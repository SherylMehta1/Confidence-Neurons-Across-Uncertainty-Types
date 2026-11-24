# Claims & Value Changes — L0 through L1
### "Confidence Neurons Across Uncertainty Types" — ACL October 2026 cycle
### Consolidated record of every number that changed, why, and what it does to the paper's claims

This doc exists because numbers moved at almost every stage of L0 and L1, for different reasons each time (a retention bug, a measurement-resolution fix, a data dedup, a construct-definition problem, a statistical artifact). Each section below is a distinct claim the paper makes or depends on, with the before/after values and — most importantly — what actually changed *scientifically*, not just numerically.

---

## 1. Dataset scale — candidate pool and twin pairs

| Stage | Candidates | Pairs | Working / Held-out | Cause of change |
|---|---|---|---|---|
| Original (pre-L0) | 1,151 | 195 | — | baseline |
| Post Fix 1–4 (`n_free` 3→20, retention, checkpoint, class flags) | 1,151 | 181 | 126 / 55 | stricter, less-noisy `hedge_gap` gating (fewer marginal pairs pass on real measurement vs. sampling noise) |
| Post dedup fix (PopQA-wide, 14,267→13,068 raw rows) | 1,124 | 207 | 144 / 63 | different candidate pool → different quantile cutoffs per relation → re-draw, not a subtraction |

**Effect on claims:** none of these numbers appear as a *scientific* claim on their own — they're dataset-construction facts. But every downstream number in this doc is built on top of whichever pair count was current at the time, which is why the same underlying quantity (e.g. "faithfulness recovery") shows up with three different values across this document. The **final, canonical figure is 207 pairs / 144 working / 63 held-out** (`data/familiarity_v2/`); `data/familiarity/` (the original, 195-pair set) has not been promoted and should not be cited going forward.

---

## 2. Circuit discovery — did deduping change *what* the circuit is

| Metric | Old (195-pair circuit) | New (207-pair circuit) | Read |
|---|---|---|---|
| Verified heads | — | 26/30 identical to old | circuit is the same object |
| Minimality (smallest subset recovering >0.7) | 20 components | 10 components | tighter, stronger minimality claim |
| Twin-patching peak | 0.95 / 1.07 @ L31 | 0.97 / 1.08 @ L31 | unchanged |
| Induction Cohen's d | 3.327 | 3.369 | unchanged |
| Verdict | CIRCUIT | CIRCUIT | holds |

**Effect on claims:** this is the load-bearing check for the entire rest of the paper. If the discovered circuit had materially changed (different heads, different neurons), every number below would be about a *different* mechanism, not a more precisely measured version of the same one. It didn't — **the circuit is confirmed to be the same object**, just measured more precisely and with a tighter minimal subset. This licenses treating everything below as "the same finding, refined," not "a new finding."

---

## 3. Faithfulness (component-set recovery) — headline numbers

| Metric | Old (195-pair) | New (207-pair) | Change |
|---|---|---|---|
| Faithfulness, hedge, control→uncertain | +0.798 | +0.910 | strengthened |
| Faithfulness, entropy, control→uncertain | **−0.132** | **+0.144** | **sign flip** |
| Faithfulness, entropy, injection (uncertain→control) | +0.454 (approx, pre-manuscript-check value +0.45 cited) | +0.54 [0.46, 0.64] | strengthened |
| Paired entropy asymmetry | +0.59 [0.41, 0.77], p<10⁻⁴ | +0.40 [0.27, 0.53], p<10⁻⁴ | weaker but still significant |

**Effect on claims:** the hedge-readout numbers got uniformly *stronger* — good news, no claim needs walking back. The **entropy sign flip is the one number that cannot simply be updated in place** — the manuscript currently states a negative effect; the true (current) effect is positive. This is not a magnitude correction, it changes which direction the effect points, and it's the single highest-priority manuscript fix in this entire document (see §6).

---

## 4. Rank-1 direction recovery

| Metric | Old | New | Change |
|---|---|---|---|
| Rank-1 direction, hedge, control→uncertain (OFF) | +0.464 [0.35, 0.58] | +0.55 [0.47, 0.63] | strengthened, tighter CI |
| Rank-1 direction, hedge, uncertain→control (ON) | +0.812 | +0.96 [0.89, 1.04] | strengthened |
| Rank-1 direction, entropy, control→uncertain (OFF) | +0.363 (pre-dedup raw mean) | +0.40 [0.33, 0.49] | strengthened |

**Effect on claims:** uniformly strengthens the paper's rank-1-direction story — no sign changes, no walk-backs needed here, just updated citations (§6).

---

## 5. The dissociation claim (L1's core result) — did the ordering hold

| | rank-1 direction | full component set |
|---|---|---|
| hedge log-odds, OFF | 0.55 [0.47, 0.63] | **0.91 [0.82, 1.00]** |
| entropy, OFF | **0.40 [0.33, 0.49]** | 0.14 [0.05, 0.24] |

**Effect on claims:** this is the actual payoff of L1 — the component set wins on hedge, the rank-1 direction wins on entropy, with cleanly separated CIs both ways. This is direct evidence *against* "one signal, read out two ways," which was the specific reviewer objection L1 was built to foreclose. The entropy/full-set cell (0.14) is thin — CI clear of zero, but only just — and should be presented as such, not smoothed over. **This claim did not exist in a checkable form before L1**; it's new, not a revision of an old number.

---

## 6. Manuscript staleness — exact replacements needed (`main.tex` 185–197)

| Claim | Currently cites | Correct (v2) value | Priority |
|---|---|---|---|
| Entropy recovery, removal, familiarity | **−0.13** | **+0.14 [0.05, 0.24]** | **sign flipped — fix first** |
| Rank-1 direction, off / on (hedge) | 0.46 [0.35, 0.58] / 0.81 | 0.55 [0.47, 0.63] / 0.96 [0.89, 1.04] | magnitude update |
| Component set (control→uncertain, hedge) | 0.80 | 0.91 [0.82, 1.00] | magnitude update |
| Entropy recovery, injection, familiarity | +0.45 | +0.54 [0.46, 0.64] | magnitude update |
| Paired entropy asymmetry, familiarity | +0.59 [0.41, 0.77], p<10⁻⁴ | +0.40 [0.27, 0.53], p<10⁻⁴ | magnitude update (weaker, still significant) |

**Effect on claims:** the qualitative narrative survives intact and is arguably cleaner post-update ("removing the unknown signal moves the decision with little of the spread" — 0.91 vs 0.14 — and "the direction carries the spread the set doesn't" are both *more* clean-cut now). Only the sign-flip line is a substantive risk if left uncorrected — a reviewer comparing text to a regenerated figure would catch it immediately. **Not yet edited in `main.tex` as of this doc.**

---

## 7. The bridge/hedging-alignment claim — three successive versions of the same number

| Version | Hedge contrast used | cos(familiarity, hedging) | Hedge-steer swing | Status |
|---|---|---|---|---|
| Original (`n_free=3`) | 126 vs 10 | +0.32 | 6.3 | **quantization artifact** — "non-hedgers" were mostly mismeasured borderline cases |
| `_v2` (`n_free=20`, gated twin pool) | 142 vs 2 | **−0.07** | 1.6 | **degenerate** — tool's own code warned "too few hedged/non-hedged prompts" |
| `_v3` (U/U arm, matched) | 25 vs 25 | **+0.67** (L31; range +0.655–0.693 across L20–31) | 11.7 | current, defensible |

**Effect on claims:** the paper's alignment claim roughly **doubled in strength** and, more importantly, changed from resting on a measurement now known to be flawed (twice over — once via quantization, once via degeneracy) to resting on a purpose-built, balanced contrast. The isolation check (familiarity-direction neuron cosines identical to the bit, 0.0 difference across 172,032 neurons between `_v2`/`_v3`) confirms only the hedging side moved. **This is the single largest number change in the whole project**, and it needs the same manuscript-citation check as §6 — not yet confirmed done.

---

## 8. Statistical backing for the +0.67 alignment claim

| Check | Result | What it licenses saying |
|---|---|---|
| Bootstrap (naive 95% CI, per-layer) | [+0.41, +0.67] at L31 — **point estimate sits at the interval's edge, not center** | Do **not** cite as a symmetric interval — see §9 |
| Bootstrap, lower bound only | stays above +0.31 at every layer, +0.41 at L31 | "Alignment is not fragile" — the one robust number from this test |
| Permutation test (10,000 relabelings) | null mean ≈ 0, SD ≈ 0.27; observed value beaten by ≤12/10,000 relabelings at any layer; p ≤ 0.0012 | "Alignment is not producible by an uninformative labeling" — the primary significance claim |

**Effect on claims:** the defensible combined statement is **"cos ≈ 0.66, permutation p ≤ 10⁻³, bootstrap lower bound > 0.3, with the caveat that the hedging direction rests on only 25 matched pairs."** This is stronger and more carefully qualified than anything the earlier (+0.32 or −0.07) numbers could have supported, but it requires stating the small-sample caveat explicitly — omitting it would be the kind of gap a reviewer finds on their own.

---

## 9. What the 95% CI claim specifically becomes

**Naive claim (do not use as written):** "cos = +0.661, 95% CI [+0.41, +0.67]" — implies the true value is centered in that range.

**Why it's wrong:** the point estimate sits at the *upper edge* of its own interval (ci_hi − point ≈ +0.007), the signature of attenuation bias — bootstrap resampling adds independent noise to two separately-estimated, noisily-measured directions (the hedging side especially, at only 25 pairs), and independent noise on two vectors being compared for alignment biases the resampled cosine downward, not because the true value is lower but because resampling (draws with replacement, effectively less distinct data per resample) amplifies estimation noise beyond what the original, undistorted point estimate carries.

**Corrected three-part claim to use instead:**
1. Point estimate: cos = +0.66 at L31 (unaffected by the bias — it's computed once, directly, no resampling)
2. Conservative floor: "alignment remains above +0.31 across all evaluated layers" (the one-sided reading of the interval that's still valid, since the bias only pushes estimates down, making this floor if anything extra-conservative)
3. Significance: permutation p ≤ 10⁻³ (unaffected by the bias — permutation uses all 50 items exactly once, no duplication, no degraded per-draw information)

**Outstanding:** a bias-corrected (BCa) bootstrap, or continued reliance on the floor+permutation combination in place of a symmetric CI. Not yet implemented — the floor+permutation combination is defensible to cite now.

---

## 10. Construct definition — what "confabulator" means, and why it changed

| Definition | Yield | Verdict |
|---|---|---|
| Plan's literal spec: `hedge_rate ≤ 0.1` AND `n_distinct ≤ 2` | 3 items → 1 matched pair | **unworkable** — worse than the problem it was meant to fix |
| Revised: `hedge_rate ≤ 0.1` alone (n_distinct dropped) | 52 items → 36 matched pairs (25 used in the working split for `_v3`) | adopted |

**Why this is a finding, not just a fix:** among low-hedge Unknown items, `n_distinct` averages ~8.5 regardless of hedge status (8.56 hedgers vs. 8.25 non-hedgers — indistinguishable). **This model does not confabulate a single consistent wrong answer when it doesn't hedge — it scatters across many different wrong answers.** That's a reportable behavioral finding about the model, independent of anything it does for the bridge-direction fix.

**Effect on claims:** enables §7's `_v3` result to exist at all (no usable contrast otherwise). Also creates an **open obligation**: the paper's L4 falsification design pre-registers the literal spec's thresholds; since that spec is now known, twice over, to be infeasible, a tagged commit recording the revised definition and this rationale is required **before** L4's actual falsification test runs — not yet done as of this doc.

---

## 11. Outstanding actions, ranked by priority

1. **Fix the entropy sign-flip in `main.tex`** (§6) — highest risk if left as-is.
2. **Add the bridge/alignment claim (§7–9) to the manuscript**, replacing whatever currently cites the old +0.32 figure, with the caveated three-part statement from §9.
3. **Pre-register the revised confabulator definition** (§10) in a tagged commit before L4 runs.
4. **Update every other magnitude in §6's table** — lower risk than #1, but still currently stale.
5. Optional, not blocking: BCa bootstrap for a properly-centered CI (§9); continuous (non-thresholded) robustness check on the alignment claim (§7); decide on `familiarity` vs. `familiarity_v2` naming going forward; fold `fig_dissociation.py` back into `make_paper_figures.py`.
