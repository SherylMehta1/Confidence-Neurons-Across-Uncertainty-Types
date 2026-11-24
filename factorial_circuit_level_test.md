# Finishing L4: Circuit-Level Confirmation of the Frequency×Knowledge Factorial

Scope: just the immediate next step — the one piece of step 5 that isn't done yet. Everything already complete is listed first so it isn't repeated.

---

## Already done — not part of this step

- The dataset-level factorial itself: four contrasts (`pop_given_known`, `pop_given_unknown`, `know_given_hi`, `know_given_lo`), built twice (207-pair era `factorial_v2_*` and 470-pair era `factorial_v3_*`), both eras agreeing that knowledge moves hedging/entropy more than popularity alone.
- The dataset gating (`build_frequency_knowledge_factorial.py`) — no changes needed, no rebuild needed.
- The headline confound the independent validation flagged — now confirmed directly from `gate_report.json`, with exact numbers (see below), not just the earlier approximate "38 of 46" figure.

**What's missing, and only this:** nothing has run the actual circuit — patching heads and neurons and checking recovery — on any of the four factorial contrasts. Everything so far is a dataset-level behavioral gap (hedge rate, entropy), not a mechanistic one.

---

## What the confound actually looks like, contrast by contrast

Checked directly against each contrast's `gate_report.json` (470-pair era):

| Contrast | n_pairs | n_working | n_held_out | capital/country share |
|---|---|---|---|---|
| `pop_given_known` | 55 | 38 | 17 | **46/55 = 84%** |
| `know_given_lo` | 51 | 35 | 16 | **38/51 = 75%** |
| `know_given_hi` | 248 | 173 | 75 | 28/248 = 11% |
| `pop_given_unknown` | 361 | 252 | 109 | 28/361 = 8% |

The two small contrasts are the confounded ones, and the confound isn't marginal — it's the large majority of each pool. Both are also too small to filter and recover: pulling out capital/country questions would leave `pop_given_known` with ~9 pairs and `know_given_lo` with ~13 — not enough for a working/held-out circuit split at all, let alone a trustworthy one.

The two large contrasts are mostly clean (8–11% capital/country) and have real size: 173/75 and 252/109 working/held-out pairs respectively — both comfortably larger than familiarity's original pools, let alone U/U's 53/23.

**Scoping decision for this step:** build the circuit on `know_given_hi` and `pop_given_unknown` only. Defer `pop_given_known` and `know_given_lo` — they would need a materially different, larger re-screen to be usable at all, which is a separate piece of work, not a quick fix folded into this one.

This scoping isn't a full loss for the original question either: `know_given_hi` tests "does knowledge move hedging when popularity is held high" and `pop_given_unknown` tests "does popularity move hedging when knowledge is held at unknown" — between the two, you still get a real read on whether the circuit tracks knowledge or popularity, just without the other two (confounded) cells of the full 2×2.

---

## The pipeline needed: none of it is new

Checked directly: `circuit_position_map.py`, `circuit_heads.py`, `circuit_mlp.py`, and `circuit_faithfulness.py` take `--category` as a plain string with no special-casing anywhere in `_common.py`'s `load_category`/`data_paths`. Since `data/factorial_v3_know_given_hi/` and `data/factorial_v3_pop_given_unknown/` already exist in exactly the same `prompts.jsonl`/`controls.jsonl`/`gate_report.json` shape the pipeline expects (confirmed directly), **this is a matter of running the existing four-stage pipeline on two new category names — no script needs writing.** (This corrects the earlier assumption that a new circuit-level factorial script needed to be built; it doesn't.)

---

## Commands

Same model, same component budget as every other circuit build in this project (`n_heads=20`, `n_neurons=100`, default `--window 12-26`), for direct comparability.

### `know_given_hi` (173 working / 75 held-out)
```bash
python scripts/circuit_position_map.py --category factorial_v3_know_given_hi --split working --limit 173 --out-dir results/circuit_factorial_v3_know_given_hi --overwrite
python scripts/circuit_heads.py        --category factorial_v3_know_given_hi --limit 173 --verify 30 --out-dir results/circuit_factorial_v3_know_given_hi --overwrite
python scripts/circuit_mlp.py          --category factorial_v3_know_given_hi --limit 173 --window 12-26 --out-dir results/circuit_factorial_v3_know_given_hi --overwrite
python scripts/circuit_faithfulness.py --category factorial_v3_know_given_hi --circuit-dir results/circuit_factorial_v3_know_given_hi --n-heads 20 --n-neurons 100 --limit 75 --overwrite
```

### `pop_given_unknown` (252 working / 109 held-out)
```bash
python scripts/circuit_position_map.py --category factorial_v3_pop_given_unknown --split working --limit 252 --out-dir results/circuit_factorial_v3_pop_given_unknown --overwrite
python scripts/circuit_heads.py        --category factorial_v3_pop_given_unknown --limit 252 --verify 30 --out-dir results/circuit_factorial_v3_pop_given_unknown --overwrite
python scripts/circuit_mlp.py          --category factorial_v3_pop_given_unknown --limit 252 --window 12-26 --out-dir results/circuit_factorial_v3_pop_given_unknown --overwrite
python scripts/circuit_faithfulness.py --category factorial_v3_pop_given_unknown --circuit-dir results/circuit_factorial_v3_pop_given_unknown --n-heads 20 --n-neurons 100 --limit 109 --overwrite
```

The final `circuit_faithfulness.py` call in each block defaults to the `held_out` split with no override needed — same as the U/U confirmatory run, this is the one evaluation that should run once per contrast.

No pre-registration commit is being proposed for this step — unlike U/U, step 5 was never specified in the 10-step plan as a falsifiable, high-stakes claim with its own locked bar. This is closer to the familiarity circuit's original validation: run it, report what comes back, including the random-set null, honestly.

---

## How to read the result — one thing learned from the U/U test, applied here

The U/U run just showed the random-set null varies a lot by arm (0.182 for U/U vs. ~0.02 for familiarity) — the 0.7 bar means something different depending on how easy the comparison is by chance. **Report each contrast's own random-set null alongside its recovery number**, not just the recovery itself, and don't assume a passing score means the same thing it meant for familiarity until the null is checked.

For each contrast, read off `faithfulness.csv` (set == "circuit"):
```python
import pandas as pd
df = pd.read_csv("results/circuit_factorial_v3_<contrast>/faithfulness.csv")
circ = df[df["set"] == "circuit"]
rnd = df[df["set"].str.startswith("random")]
for d in ["control_to_uncertain", "uncertain_to_control"]:
    print(d, circ[circ["direction"] == d]["logodds_rec"].mean())
print("random null:", rnd["logodds_rec"].mean(), "+/-", rnd["logodds_rec"].std())
```

---

## What this feeds

Both contrasts' results go straight into step 8 (deciding the paper's claims), alongside the U/U result already in hand. If the circuit tracks knowledge (transfers on `know_given_hi`, doesn't on `pop_given_unknown`) or the reverse, that's the mechanistic answer the dataset-level factorial could never give on its own — and it's the piece that was always missing, not something this step is redoing.
