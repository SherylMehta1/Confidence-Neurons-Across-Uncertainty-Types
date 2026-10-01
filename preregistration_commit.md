# Pre-Registration Commit: U/U (Hedger vs. Confabulator) Falsification Test

Scope note up front: this covers three of the four points discussed — **thresholds (A)**, **the falsification criterion (C)**, and **pool choice (D)**. **Point B (the justification for dropping the `n_distinct` clause) is deliberately not included here**, per decision. One flag worth registering before moving on: the independent validation pass on `l0-fixes` found that at `n_samples=10`, `confab_max_distinct=10` makes the answer-diversity condition vacuous outright (12 of 36 confabulators in the 25-pair pool, 25 of 76 in the 470-pair-era pool, still flag ambiguity in at least half their generations) — so the clause isn't just non-discriminating, it's not really doing anything at the current sample count. That's a fact about the current pipeline, not something this commit needs to resolve; it's noted here only so it isn't silently lost by skipping B.

---

## Purpose

This commit exists to lock the classification thresholds and the pass/fail rule for the U/U (hedger vs. confabulator) comparison **before** the real falsification run touches the held-out pairs. The value of doing this is entirely in the commit's immutable timestamp — a decision written down after seeing results is not a pre-registration, it's a post-hoc rationalization with pre-registration formatting. The concrete reason this matters here specifically: `confab_max_distinct` was already moved from its script default (2) to 10 after the original two-part definition failed to discriminate hedgers from confabulators. That's a disclosed, defensible change, but it is also the exact pattern a pre-registration commit is meant to foreclose going forward. This commit draws the line: everything above it was exploratory, everything below it (the actual held-out test) is confirmatory.

It also closes a line the paper already owes: the existing Future Work text promises a "lockbox confirmation" — converting "survived the split we had" into "predicted in advance." This commit is that lockbox.

---

## A. Classification thresholds — locked

| Parameter | Value | 
|---|---|
| Hedger | `hedge_rate ≥ 0.8` |
| Confabulator | `hedge_rate ≤ 0.1` |
| Matched-pair constraint: length | `max_len_diff = 3` tokens |
| Matched-pair constraint: entropy | `entropy_tol = 0.3` |

**Why these values, not re-derived ones:** these are the thresholds that already produced a stable, twice-independently-replicated split (first at the 25-pair pool, again at the 53-pair pool built from `familiarity_v3`). The current pool's own summary stats show clean separation with no contested middle ground — hedger mean hedge-rate 0.972, confabulator mean 0.022. Re-deriving new thresholds at this stage would mean re-fitting to the same data the falsification test is about to run on, which defeats the purpose of pre-registering at all. Lock what's already validated; this is the lowest-risk, least contestable part of the commit.

---

## C. The falsification criterion — the one real decision

**Recommended and locked: `recovery > 0.7`, as a plain point estimate, applied symmetrically in both directions (removal and injection), decided before the power check is run.**

Breaking down each sub-choice:

**Point estimate vs. CI-lower-bound.** Locked as the plain point estimate (`> 0.7`), not the stricter "CI lower bound > 0.7" form. The only existing precedent for the stricter form in this codebase is `circuit_causal_timing.py`'s crossing-point definition ("first layer with CI lower bound > 0.7"), which is a descriptive device for locating a point along a 32-layer curve — being conservative there costs nothing, since the curve keeps moving. The U/U falsification test is a single, one-shot comparison on 23 held-out pairs, not a curve. Stacking a small N with a stricter statistic risks manufacturing a "failed to confirm" that's actually just underpowered, which is a worse outcome than an honest, clean failure. The plain 0.7 bar is also what the paper's own faithfulness criterion already uses everywhere else, so this keeps the U/U test on the same evidentiary standard as the rest of the paper rather than inventing a harder one just for this test.

**Symmetric vs. one-directional.** Locked as symmetric — the bar must hold in both the removal and injection directions. This matches the paper's own existing convention: every other circuit result in this paper is reported in both directions, with no precedent anywhere for a one-directional claim. Pre-registering only the more favorable direction, on a construct this new, would read as exactly the kind of after-the-fact selection this commit exists to prevent.

**Sequencing relative to the power check.** The power check runs *after* A/C/D are frozen in this commit, and functions as a **go/no-go gate**, not an input that can move the bar. Concretely: once this commit lands, compute power/minimum-detectable-effect for the chosen bar against N=23 held-out pairs, using an effect size in the range already seen elsewhere in this pipeline (roughly 0.3–0.4 recovery gap). If the test comes back adequately powered, run it as specified above. If it comes back underpowered, the correct response — written here in advance, precisely so it can't be chosen after seeing the result — is to say so explicitly and either widen the held-out pool before running the confirmatory test, or report the result as exploratory rather than confirmatory. **What is not an acceptable response:** loosening the 0.7 bar to compensate for low power. That would repeat the `confab_max_distinct` pattern (adjusting a criterion once it's seen to fail) in a new location, which is the one thing this entire commit is designed to make impossible.

**Update: the power-analysis script now exists** (`power_analysis_uu.py`, delivered separately, CPU-only, no repo dependency — add it to `scripts/` as part of this commit). It implements the one-sample bound test above exactly (`--mode bound`): the locked `--rule point` path (the one that matches this section) uses the exact central t-distribution of the sample mean, appropriate given N=23; an alternate `--rule ci` path is kept only for comparison (see the correction below — it uses the noncentral t-distribution and is *not* the locked rule). A two-sample variant (`--mode diff`) is kept in reserve in case the criterion is ever reframed as a direct hedger-vs-confabulator comparison.

**Correction (2026-10-01):** the script's original default computed power as a one-sided significance test at α=0.05, which is algebraically identical to requiring the one-sided CI lower bound to exceed 0.7 — exactly the stricter form this section argues against using. The script now defaults to `--rule point` (a bare point-estimate crossing, no alpha, matching `circuit_faithfulness.py`'s own existing convention of a plain `mean() > 0.7`), with the old behavior kept available as `--rule ci` for comparison only. The table below is the corrected one.

**First run, using a provisional SD** (0.262 — the real per-pair recovery SD from the only circuit-evaluation data that currently exists, `results/circuit_familiarity_v3/faithfulness_limit141.csv`; the U/U circuit itself hasn't been run yet, so this is a placeholder, not a measured value for this specific test):

| Power target | MDE above the 0.7 bound | True recovery needed |
|---|---|---|
| 0.70 | 0.029 | 0.729 |
| 0.80 | 0.047 | 0.747 |
| 0.90 | 0.072 | 0.772 |

**What this means for the go/no-go gate in C:** at N=23 and this SD estimate, the test has 80% power to detect a true confabulator-circuit recovery of **0.747 or higher** — much closer to the 0.7 pass line than the earlier (incorrect) table suggested. The genuinely ambiguous band is roughly 0.7–0.75, not 0.7–0.84: a result in that narrower range should still be treated as **underpowered, not disconfirmed**, but a result anywhere above ~0.75 is one this test is well-equipped to detect, and shouldn't be second-guessed on power grounds.

**The two-direction conjunction:** the table above is for a single direction. The locked rule in C requires the bound in *both* directions (removal and injection) on the same 23 pairs, and because both are measured on the same pairs, they're correlated rather than independent — so the true joint power is somewhat lower than the single-direction number above, not simply the same number applied twice. Using a provisional cross-direction correlation (~0.51, estimated the same way as the SD, from `faithfulness_limit141.csv`), the joint power at a true effect of +0.1 above the bound drops modestly: **0.966 → 0.940** (`power_analysis_uu.py power --joint-rho 0.51`). (Both numbers here use the same bivariate-normal approximation so they're directly comparable — the script's exact `--rule point` central-t number for single-direction power at this effect is 0.960, very close but not identical; that small gap is the normal-vs-t approximation, not part of the joint penalty.) Small at this effect size, and it doesn't change the conclusion — the test is still well powered above roughly 0.75 — but the single-direction table should be read as an upper bound on the real (joint) test's power, not as the final number.

**Action before the real test runs:** rerun `power_analysis_uu.py` with the real per-pair SD and the real cross-direction correlation once the U/U circuit has been evaluated even once (even an exploratory pass, kept separate from the held-out confirmatory run) — both the 0.262 SD and the 0.51 correlation above are borrowed from a different circuit and should not be treated as final.

---

## D. Pool choice — locked

**`uu_hedge_confab_v3`: 53 working pairs / 23 held-out pairs.**

No real alternative exists to weigh this against — it's the only U/U pool built from the 470-pair-era screen (`familiarity_v3`), and it's already the pool every bridge result currently cited is built from (bridge v5's L31 cosine of 0.658 uses exactly this pool). Choosing a different pool here would reintroduce a pool mismatch the project already resolved elsewhere. Lowest-risk, least contestable point in the commit, same as A.

---

## What this commit should contain, concretely

1. The four threshold values under **A**, written as the actual config/constants used by the gating script, not just prose — so the commit is checkable against the code, not just a description of it.
2. The locked falsification rule from **C**: `recovery > 0.7`, both directions, point-estimate form — stated as the literal pass/fail condition that will be applied to the held-out run, plus the pre-committed fallback behavior if the power check comes back underpowered. Include `power_analysis_uu.py` itself in this commit (now written — see below), and its output at whatever SD is current at commit time, so the power check is checkable rather than asserted.
3. The pool identifier from **D** (`uu_hedge_confab_v3`, 53/23 split), named explicitly so there's no ambiguity about which file the held-out test reads from.
4. A timestamp/commit hash that predates the first time the held-out 23 pairs are touched by the actual confirmatory test — this is what makes the whole exercise meaningful. **Disclosure, not assumed away:** the held-out 23 pairs are not untouched. They sit inside `uu_hedge_confab_v3`, the same pool that fed `bridge_familiarity_v5`'s cosine result (referenced approvingly under D above). That's a non-confirmatory touch — the bridge run used the pool to build and evaluate an alignment direction, not to test the 0.7 hedger/confabulator recovery bar — but it means the pairs aren't pristine, and the commit should say so explicitly rather than imply a clean slate. If this is a problem for the confirmatory test's validity, that's a call to make now, before the test runs, not after.
5. An explicit note that point B (the distinct-answer clause) is out of scope for this commit, so a future reader doesn't mistake the omission for an oversight.

---

## What remains open after this commit lands

- Rerunning `power_analysis_uu.py` with the real (not borrowed) per-pair SD, once available, and confirming the test is adequately powered before touching the held-out pairs — this is now a mechanical step, not an open design question.
- The actual confirmatory run against the 23 held-out pairs — this commit locks the rules, it doesn't run the test.
- Point B, by the scope decision above, stays unresolved and undocumented in this commit specifically.