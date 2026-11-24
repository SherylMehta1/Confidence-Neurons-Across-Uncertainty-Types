# The Numbers-Tracking System, and Where the Project Stands
### "Confidence Neurons Across Uncertainty Types" — ACL October 2026 cycle
### As of Thursday, 25 September 2026

Two things in one doc, because they're related: (1) the general-purpose script that keeps every number in the manuscript traced to a result file, explained end to end with the real code; (2) a status tracker so you can mark off progress against whatever list you're checking this against.

---

## Part A — The `numbers.tex` system

### A.1 What this is, in one sentence

One script, `scripts/make_numbers_tex.py`, that reads every result file the paper cites and writes `paper/numbers.tex` — a flat list of `\newcommand` macros — so `main.tex` cites `\circuitHedgeCUNew` instead of typing `0.91` by hand. It is **general-purpose and permanent**, not a one-off for L0/L1: today it only has L0/L1 blocks because that's what exists to aggregate; you extend it — never replace it — as L2, L3, L4 results land.

### A.2 Why this exists (the concrete failure it prevents)

Three separate bugs already happened in this project from numbers being hand-typed in more than one place:
1. The entropy recovery sign flip (`-0.13` left in the manuscript after the real value moved to `+0.14`, and then that value itself turned out to be one of four sign-flipping estimates).
2. The alignment cosine typo (`0.67` in one section, `0.661` in another — traced to someone rounding the bootstrap CI's upper bound instead of the point estimate).
3. Every one of the "manuscript staleness" line items in `claims_and_changes_L0_through_L1.md` — a number changed in a result file and the prose kept the old one.

A single generated file that every citation points at can't have this specific failure mode: if the underlying result changes and you rerun the script, every citation of that number updates together. It can still be *wrong* if you feed it the wrong file — it's not magic — but it can't go *inconsistent with itself*.

### A.3 How it works, mechanically

```
result files (CSV/JSON, already committed)
        │
        │  read-only, no analysis performed here
        ▼
scripts/make_numbers_tex.py   (you run this by hand)
        │
        │  writes, always overwriting fully
        ▼
paper/numbers.tex             (flat \newcommand list)
        │
        │  \input{numbers.tex} near the top of main.tex
        ▼
main.tex cites \circuitHedgeCUNew, \pairCountNew, etc.
        │
        ▼
   compile (pdflatex / latexmk / Overleaf)
```

Nothing here is automatic or event-driven. There is no file-watcher, no CI hook (unless you choose to add one later). **You are the trigger.** The rule is: any time a result file the script reads gets regenerated, rerun the script before you next compile.

### A.4 The script itself, as it stands today (L0/L1 scope)

```python
"""
scripts/make_numbers_tex.py

Generate paper/numbers.tex — one \\newcommand macro per number cited in main.tex,
computed directly from committed result files. Rerun any time an upstream result
changes. This script performs NO analysis itself — it only aggregates existing
CSVs/JSON that other scripts already produced.

Usage: python scripts/make_numbers_tex.py
Output: paper/numbers.tex (fully overwritten each run)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "paper" / "numbers.tex"

macros = {}


def add(name, value):
    macros[name] = value


def mean_ci(vals, n_boot=10000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return vals.mean(), lo, hi


# =====================================================================
# L0/L1 — dataset scale, circuit faithfulness, rank-1 direction, bridge
# =====================================================================

# ---- dataset scale ----
gate = json.loads((REPO_ROOT / "data/familiarity_v2/gate_report.json").read_text())
add("pairCountNew", gate["n_pairs"])
add("workingCountNew", gate["n_working"])
add("heldOutCountNew", gate["n_held_out"])

gate_old = json.loads((REPO_ROOT / "data/familiarity/gate_report.json").read_text())
add("pairCountOld", gate_old["n_pairs"])

# ---- component-set faithfulness (old + new circuits) ----
def faithfulness_macros(csv_path, suffix):
    df = pd.read_csv(csv_path)
    circ = df[df.set == "circuit"]
    for direction, tag in [("control_to_uncertain", "CU"), ("uncertain_to_control", "UC")]:
        sub = circ[circ.direction == direction]
        for metric, mtag in [("logodds_rec", "Hedge"), ("entropy_rec", "Entropy")]:
            m, lo, hi = mean_ci(sub[metric].to_numpy())
            add(f"circuit{mtag}{tag}{suffix}", f"{m:.2f}")
            add(f"circuit{mtag}{tag}{suffix}CI", f"[{lo:.2f}, {hi:.2f}]")

faithfulness_macros(REPO_ROOT / "results/circuit_familiarity_v2/faithfulness.csv", "New")
faithfulness_macros(REPO_ROOT / "results/circuit_familiarity/faithfulness.csv", "Old")

# ---- rank-1 direction recovery ----
def direction_macros(csv_path, suffix):
    df = pd.read_csv(csv_path)
    d = df[df.set == "direction"]
    for direction, tag in [("control_to_uncertain", "CU"), ("uncertain_to_control", "UC")]:
        sub = d[d.direction == direction]
        for metric, mtag in [("logodds_rec", "Hedge"), ("entropy_rec", "Entropy")]:
            m, lo, hi = mean_ci(sub[metric].to_numpy())
            add(f"direction{mtag}{tag}{suffix}", f"{m:.2f}")
            add(f"direction{mtag}{tag}{suffix}CI", f"[{lo:.2f}, {hi:.2f}]")

direction_macros(REPO_ROOT / "results/circuit_familiarity_v2/direction_patch.csv", "New")

# ---- minimality: smallest k in prune_curve.csv clearing 0.7 recovery ----
prune = pd.read_csv(REPO_ROOT / "results/circuit_familiarity_v2/prune_curve.csv")
cleared = prune[prune.recovery > 0.7]
add("minimalityK", int(cleared.k.min()) if len(cleared) else "n/a")

# ---- bridge/alignment cosine ----
boot = pd.read_csv(REPO_ROOT / "results/bridge_familiarity_v3/cos_bootstrap.csv")
row31 = boot[boot.layer == 31].iloc[0]
add("alignCosLThirtyOne", f"{row31.cos:.2f}")
add("alignCosFloor", f"{boot.ci_lo.min():.2f}")
add("alignCosRangeLo", f"{boot.cos.min():.3f}")
add("alignCosRangeHi", f"{boot.cos.max():.3f}")

perm = pd.read_csv(REPO_ROOT / "results/bridge_familiarity_v3/cos_permutation.csv")
add("alignPermPMax", f"{perm.p_one_sided.max():.4f}")

# =====================================================================
# L2/L3 additions go here as their result files land (see Part B) — do
# not start a new script; add a new block, following the same pattern.
# =====================================================================

# ---- write ----
with open(OUT, "w") as f:
    f.write("% AUTO-GENERATED by scripts/make_numbers_tex.py — do not hand-edit.\n")
    f.write("% Rerun the script after any result regenerates; this file has no memory of its own.\n\n")
    for name, val in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")

print(f"wrote {len(macros)} macros to {OUT}")
```

### A.5 How to use it, step by step

1. **One-time:** add `\input{numbers.tex}` near the top of `main.tex`, right after `\begin{document}`.
2. **One-time:** go through `main.tex` and replace every hand-typed number this script covers with its macro (`0.91` → `\circuitHedgeCUNew`, `207` → `\pairCountNew`, etc.).
3. **Every time a result changes** (you rerun `circuit_faithfulness.py`, regenerate the bridge cosine, anything under `results/` or `data/` this script reads): run
   ```
   python scripts/make_numbers_tex.py
   ```
   then recompile. Two commands, always in that order, every time.
4. **On Overleaf:** if the prose is edited there, push the regenerated `numbers.tex` into the Overleaf project (Git sync, or manual upload) after every regeneration — Overleaf never sees your `results/` folder on its own.

### A.6 How it grows as L2/L3/L4 land — add a block, don't start over

The pattern is always the same three lines: read the file, compute the number, `add()` it. Concretely, once each of these exists:

```python
# ---- L2c/L2d: LOTO and causal timing (once loto.csv / causal_timing.csv exist) ----
loto = pd.read_csv(REPO_ROOT / "results/circuit_familiarity_v2/loto.csv")
full_set = loto[loto.variant == "full"]
add("lotoFullRecovery", f"{full_set.logodds_rec_mean.mean():.3f}")
minus_i = loto[loto.variant == "minus_I"]
add("lotoMinusIRecovery", f"{minus_i.logodds_rec_mean.mean():.3f}")

timing = pd.read_csv(REPO_ROOT / "results/circuit_familiarity_v2/causal_timing.csv")
crosses = timing[(timing.readout == "hedge") & (timing.logodds_rec_mean >= 0.7)]
add("timingHedgeCrossLayer", int(crosses.start_layer.min()))

# ---- L2e: saturation curve (once saturation_curve.csv exists) ----
sat = pd.read_csv(REPO_ROOT / "results/circuit_familiarity_v3/saturation_curve.csv")
add("saturationRecoveryAtMaxN", f"{sat.recovery.iloc[-1]:.3f}")
add("saturationMaxN", int(sat.n.max()))

# ---- L3/L4a: factorial (once the v3 factorial CSV exists) ----
fact = pd.read_csv(REPO_ROOT / "results/factorial_v3.csv")
row = fact.set_index("contrast").loc["know_given_hi"]
add("factorialKnowGivenHiEntropyGap", f"{row.entropy_gap:.2f}")
add("factorialPopGivenKnownHedgeGap", f"{fact.set_index('contrast').loc['pop_given_known'].hedge_gap:.2f}")
```

Paste blocks like these into the same script, in the marked section, and rerun. `numbers.tex` will then contain both the L0/L1 macros and the new ones, because both code blocks live in the one file and both run every time.

### A.7 Two open calls that affect what the script should emit (not yet decided)

- **Entropy removal recovery:** given the sign instability (−0.13, +0.14, +0.10, −0.06 across four estimates), consider having the macro resolve to literal text — `\newcommand{\entropyRemovalFam}{near zero, sign-unstable across replications}` — rather than a number, so it can't be cited as a precise point value just because a macro exists.
- **Which circuit is canonical** (v2 vs v3 vs a re-run with a raised `LIMIT`): once decided, the `faithfulness_macros()` call should point at that result file specifically, or the script should emit a range across the logged draws instead of one "new" value.

---

## Part B — Project status tracker

Reconciling everything actually confirmed in this engagement, so you can mark progress against your own list. Compared to `master_plan_L0_to_L6.md` (19 Sept), this reflects what's landed since: LOTO, causal timing, and the L3 screen are now done — earlier than that doc's own recommended order expected, since you ran them together.

| Item | Status | What was actually confirmed |
|---|---|---|
| **L0 — infrastructure fixes** | ✅ Done | Retention, checkpoint/resume, class flags, `n_free`→20, PopQA dedup. Final: `data/familiarity_v2/`, 207 pairs (144/63). |
| **L1 — dissociation table** | ✅ Done | Component set wins on hedge (0.91), rank-1 direction wins on entropy (0.40 off); CIs cleanly separated. Entropy-side numbers now known to be small and sign-unstable across draws — report cautiously, not as a precise value. |
| **L2c — causal timing** | ✅ Done | Cumulative residual-stream patching (0..ℓ, suffix positions) reused `Patch(kind="resid")` with zero new low-level code. Hedge decision resolves at L8–12 (u→c / c→u), entropy spread resolves later at L13–18 — mechanistically explains *why* the rank-1 direction (extracted from L13–19) wins on entropy but not hedge in L1. |
| **L2d — leave-one-token-out** | ✅ Done | Cache-once/recompute-offline design validated (full-set reproduction to ~1e-7). `" I"` cleared — dropping it barely moves recovery. `" not"` carries real weight — dropping it narrows the L1 dissociation's hedge-side margin (0.36→0.10, CIs now overlapping), though the entropy leg is untouched by construction. |
| **L3 — the full PopQA screen** | ✅ Done | Ran alongside L2c/L2d, as planned. `--n-per-relation 1000` under the existing 10/90 cutoffs (no new sampling code needed — the original §1 concern about off-diagonal cells being absent was wrong; they were already present, just thin). 2,637 candidates, 470 pairs (329/141) vs v2's 207 (144/63) — the 10/90 design's ceiling; 500 held-out is only reachable by widening the cutoffs (declined for now). |
| **L2e — saturation** | 🟡 Partially done | A 2-point spot-check (N=60 vs N=141) shows no degradation and tightening CIs — the right signature, but not yet the full recovery-vs-N curve the original design called for. See the sweep design already laid out (reuse `circuit_faithfulness.py --limit` across N=10..141, fixed discovered circuit) — cheap, queued, not yet run. Ceiling is 141, not 500, under the current design. |
| **L4a — frequency×knowledge factorial** | 🟡 Partially done | **Dataset-level result already lands the paper's key claim**: knowledge moves entropy (+1.3 to +2.2), popularity moves hedging (+0.6 to +0.8) even between two known items — the decision-vs-spread dissociation, replicated stably across v2 and v3 pools. Circuit-level test (patching, to confirm the same dissociation causally) not yet run. |
| **L4b — U/U arm, full falsification** | 🟡 Partially done | Pool more than doubled (36→76 pairs, 25→53 working) via the L3 screen. Confabulator spec (`n_distinct`) now rejected twice, at two sample sizes — should be written into the pre-registration as a two-replication finding. **Pre-registration commit itself still not done** — this is the one explicit hard gate before any L4 falsification test runs. Bridge cosine bootstrap/permutation not yet rerun at 53-vs-53 (cheap, ~5 min, not yet done). |
| **L5 — behavioral validation** | ⬜ Not started | Go/no-go gate, contingent on L4 landing. |
| **L6 — manuscript** | 🟡 In progress | `numbers.tex` system designed (Part A); not yet built as of this doc — still a to-do, not done. Two confirmed corrections queued: entropy sign-flip (write as unstable, not a signed point value) and the 0.67→0.66 alignment-cosine fix (confirmed against raw `cos_bootstrap.csv`). |

### What's immediately actionable, unblocked, right now

1. **The L4 pre-registration commit** — trivial, still overdue, blocks nothing else but is a hard gate before L4a/L4b's real runs.
2. **The L2e saturation sweep** — reuse existing code, no new design, just run the `--limit` loop across N values instead of the two-point check.
3. **Build `scripts/make_numbers_tex.py`** (Part A) — needed before any more manuscript editing happens, so the L0/L1 fixes land in a form that won't drift again.
4. **The 53-vs-53 bridge cosine rerun** — 5 minutes, tightens the L4b caveat that's currently still quoting the older 25-pair number.

### What's still genuinely gated

- **L4a's circuit-level test** and **L4b's real falsification run** both need the pre-registration commit (#1 above) to land first.
- **Which circuit is canonical** (v2 vs v3 vs raised-`LIMIT`) is still an open decision, not a blocker, but affects what `make_numbers_tex.py` should cite once built.
