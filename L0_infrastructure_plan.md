# L0 — Infrastructure Lane
### ACL Rolling Review, October 2026 cycle — "Confidence Neurons Across Uncertainty Types"

**Window:** Days 1–3 (Fri 11 Sep – Sun 13 Sep)
**GPU cost:** 0 hours — this is pure engineering, no model runs
**Blocks:** L4 (falsification experiments) and L5 (behavioural validation) cannot start until L0 is done
**Feeds:** L3 (the screen) — L0's fixes are what make the screen's output usable at all

---

## 1. Why this lane exists

Before any of the big, expensive experiments run, three small bugs/gaps in the existing codebase have to be closed. None of them are hard engineering — each is described as "small" — but each one is a silent failure mode: if left unfixed, the screen (L3) will run, cost GPU-hours, and produce a dataset that can't actually support the 2×2 design the rest of the paper depends on. That's why L0 sits on the critical path with a hard 3-day window before L3's pilot screen begins on Day 4.

The underlying logic: **fix the plumbing once, cheaply, before spending 10+ GPU-hours running data through it.**

---

## 2. The three fixes, in detail

### Fix 1 — Retention bug in `build_familiarity_twins.py`

**The problem:**
The script currently only writes out the items it selects as twins (the matched known/unknown pairs) — it never writes an `all_measured.jsonl` file containing *every* item it scored along the way. That means every "MaybeKnown" item (partially known — inconsistent across samples) and every off-diagonal cell of the intended 2×2 design (frequency × knowledge) is silently discarded the moment the script writes its output.

**Why it matters:**
The whole point of the October plan is to move from a hand-picked, deciled dataset to a full, continuous-distribution screen so that frequency-matched and knowledge-matched twins can be built *after the fact*, from the full pool. If the script keeps throwing away everything except its own narrow twin selection, running the full 14,267-item screen produces no more usable data than the old approach did — the extra data disappears at write time.

**The fix — process:**
1. Locate every point in `build_familiarity_twins.py` where an item is scored (knownness class assigned, popularity looked up, entropy computed).
2. Add a persistent writer that appends **every** scored item — regardless of whether it's ultimately selected as a twin — to `all_measured.jsonl`, with at minimum: item ID, relation type, popularity score, knownness class, entropy, and hedge-rate stats if available at that stage.
3. Keep the existing twin-selection logic and its output untouched — this is additive, not a replacement.
4. Verify: run the script on a small existing sample and confirm `all_measured.jsonl` contains more rows than the twin-selection output (i.e., MaybeKnown and off-diagonal items are now present).

**Definition of done:** `all_measured.jsonl` exists, is non-empty on a test run, and contains items that never appear in the twin-selection output.

---

### Fix 2 — Checkpoint / resume

**The problem:**
Each candidate item costs roughly 14 model-generation calls. The full screen is approximately 200,000 calls in total. There is currently no mechanism to resume a partially completed run — a crash, timeout, or disconnect partway through means starting over from zero.

**Why it matters:**
At this scale, an uninterrupted run is not a safe assumption — session disconnects, GPU pre-emption, or a code error partway through are realistic risks. Without checkpointing, a failure on (say) item 150,000 of 200,000 costs the entire run, which would blow the GPU-hour and calendar budget for L3.

**The fix — process:**
1. Decide the checkpoint granularity — most naturally, one checkpoint write per completed item (or small batch of items), since each item is a self-contained unit of work.
2. On each checkpoint, persist: the ID of the last successfully completed item, and the item's full output, to disk (e.g., append to the same `all_measured.jsonl` being built in Fix 1, or a dedicated progress file).
3. On startup, have the script check for an existing checkpoint/progress file; if found, skip all items already recorded and resume from the next uncompleted item.
4. Verify: start a run, kill it partway through deliberately, restart it, and confirm it resumes rather than reprocessing completed items.

**Definition of done:** killing the screen process at an arbitrary point and restarting it continues from where it left off, with no duplicate or lost items.

---

### Fix 3 — Configurable class flags in `gate_twins.py`

**The problem:**
`gate_twins.py` (lines 86–89) currently hardcodes the thresholds that define "HighlyKnown" vs. "Unknown" directly in the code.

**Why it matters:**
Two of the plan's dated gates — the Sept 15 "is the 2×2 viable?" check and the Sept 21 "is the balance acceptable?" check — explicitly anticipate needing to *adjust* matching criteria (e.g., falling back to relation-stratified matching with wider popularity calipers if the pilot screen doesn't yield enough common-but-unknown items). If the classification thresholds are hardcoded, every adjustment requires a code change and re-review, which is slow and error-prone under a tight calendar. This fix is what makes those gate decisions actually executable on the day they're scheduled.

**The fix — process:**
1. Identify the current hardcoded conditional(s) at `gate_twins.py:86–89` that assign HighlyKnown/Unknown class labels.
2. Replace the hardcoded thresholds with two command-line arguments: `--uncertain-class` and `--control-class` (naming/values should match whatever threshold basis is already used — e.g., accuracy-over-N-samples cutoffs).
3. Ensure the rest of the script reads from these flags rather than the old hardcoded constants, with sensible defaults matching current behavior (so existing results remain reproducible if run with no flags).
4. Verify: run once with default flags and confirm output matches the current (pre-fix) behavior exactly; run once with altered flags and confirm the classification boundary shifts as expected.

**Definition of done:** class thresholds can be changed at the command line without editing the script, and default-flag output is identical to current output.

---

### Fix 4 — Raise `--n-free` from 3 to 20

**The problem:**
The behavioural gate (used to classify whether a model "hedges" or "confabulates" on a given item) currently samples only 3 free generations per item. That gives a measurement resolution of 1-in-3 — i.e., hedge rate can only be reported as roughly 0%, 33%, 67%, or 100%.

**Why it matters:**
The document flags this dataset — behavioral hedge-rate measurement — as **the paper's best contribution**, and it directly feeds the U/U arm (L4), which needs hedge rate measured finely enough to apply thresholds like "≥0.8" (hedger) and "≤0.1" (confabulator) meaningfully. A 1-in-3 resolution can't support those thresholds with any precision. It also feeds L5's correlation plot (hedge log-odds vs. sampled hedging rate), which needs a reasonably continuous y-axis to be meaningful.

**The fix — process:**
1. Change the default value of the `--n-free` parameter from 3 to 20 wherever it's set (config file, script default, or call sites).
2. Confirm this doesn't silently break any existing analysis code that assumes exactly 3 samples per item (e.g., array shapes, indexing).
3. Note the GPU-cost implication: this roughly 6.7× increases the generation cost per item for hedge-rate measurement — this is already priced into L3's GPU-hour estimate, but worth confirming against the actual budget once real per-item timing is known from the pilot.

**Definition of done:** hedge rate for a test item is computed from 20 generations, and all downstream code that consumes hedge rate runs without error on the new sample count.

---

## 3. Day-by-day timeline

| Day | Date | GPU work (elsewhere, same days) | L0 work | Exit check |
|---|---|---|---|---|
| 1 | Fri 11 Sep | rank-band + complement, familiarity (L2) | **Build Fix 1 (retention writer)**; write/review the `all_measured.jsonl` schema | `all_measured.jsonl` produced on a test run and contains off-diagonal items |
| 2 | Sat 12 Sep | same, contested + Qwen (L2) | **Build Fix 2 (checkpoint/resume)**; test kill-and-resume | Resume test passes cleanly, no duplicates |
| 3 | Sun 13 Sep | token leave-one-out, cached (L2) | **Build Fix 3 (class flags) + Fix 4 (`--n-free` → 20)**; full integration test of all four fixes together | A full small-scale dry run of the pipeline (retention + resume + flags + 20 samples) completes end-to-end without manual intervention |

**Hard deadline implication:** L3's pilot screen (2,000 items) is scheduled to start on Day 4 (Mon 14 Sep). All four L0 fixes must be integration-tested together by end of Day 3 — there is no slack day built in before the pilot begins.

---

## 4. Why the ordering matters

The three code fixes are sequenced deliberately, not arbitrarily:

1. **Retention first** — without it, nothing downstream is worth building on top of, since every subsequent run (even test runs of Fix 2 and Fix 3) would still be silently losing data.
2. **Checkpoint/resume second** — this is what makes it *safe* to run larger and longer tests of the pipeline (including the Day 3 integration test and, later, the actual 200k-call full screen) without fear of losing a long run to a crash.
3. **Class flags + sample count last** — these are the two changes that affect *what gets measured and how finely*, and it's easiest to validate them once the underlying data pipeline (retention + resume) is already trustworthy.

---

## 5. Downstream dependency map

```
L0 (Fix 1: retention) ─────┐
L0 (Fix 2: checkpoint) ────┼──► L3 (pilot screen, Day 4) ──► L3 (full screen, Days 5–9)
L0 (Fix 3: class flags) ───┤                                        │
L0 (Fix 4: n-free=20) ─────┘                                        ▼
                                                    L4 (factorial + U/U, Days 9–22)
                                                    L5 (behavioural validation, go/no-go Day 20)
```

If any one of the four fixes is incomplete by end of Day 3, the pilot screen on Day 4 either (a) produces unusable data again (Fix 1), (b) risks total loss on failure (Fix 2), (c) can't be adjusted at the Sept 15/21 gates (Fix 3), or (d) can't support the U/U hedge-rate thresholds (Fix 4). There is no partial-credit version of L0 — all four need to land together.
