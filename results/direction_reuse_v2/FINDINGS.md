# Direction reuse, version 2: position-matched, dose-calibrated steering (7 Oct 2026)

- **Governed by:** `direction_reuse_v2_criterion.md` (sha256 `da0ad000`, written after version 1 returned NOT TESTABLE and before any version 2 run).
- **Script:** `scripts/direction_reuse_v2.py`.
- **Per-pair outputs:** `summary.txt`, `steer.csv`, `geometry.csv`, `log.txt`.
- **Run:** molab box seven.

The fix: directions are now fitted at the same position they are added at (the last position of the free-generation prompt), and the steering dose is calibrated on the instruct model's own direction only.

## Results

| | Llama-3.1-8B-Instruct | Gemma-2-9B-it |
|---|---|---|
| ℓ* (of layers) | 17 (32) | 23 (42) |
| Dose k* (× natural norm), chosen by own direction | **1** | **4** (1×: +0.000, 2×: +0.007 over random) |
| cos(d_fam base, inst) vs null 99th pct | 0.21 vs 0.33: no | 0.49 vs 0.52: no |
| Hedge rate: none / **own familiarity direction** / base direction / knowledge direction / random mean | 0.020 / **0.287** / 0.020 / 0.087 / 0.025 | 0.000 / **0.220** / 0.000 / 0.000 / 0.000 |
| First-token entropy: none / own fam / knowledge | 0.82 / 1.26 / 1.23 | 0.25 / 0.82 / **1.50** |
| Δhedge(fam) − Δhedge(know) [95% CI] | **+0.200 [+0.140, +0.267]** | **+0.220 [+0.160, +0.287]** |
| Δentropy(know) − Δentropy(fam) [95% CI] | −0.031 [−0.110, +0.047] | **+0.681 [+0.593, +0.770]** |
| Intervention double dissociation (criterion) | no (hedging half only) | **yes** |
| Degenerate outputs at k* (any condition) | 0% | 0% |
| **Verdict** | **NOT REUSED** | **NOT REUSED** |

**Claims, per the criterion:**
1. **"Instruction tuning reuses the base model's familiarity direction": not licensed.** In both pairs the base model's direction has no effect on the instruct model's hedging, and the two directions are no more aligned than arbitrary item splits.
2. **Intervention-level double dissociation: holds in Gemma-2-9B-it.** Pushing along the familiarity direction makes the model say it does not know (+0.22) without much change in answer entropy. Pushing along the knowledge direction raises entropy (+1.25) without making it hedge.
3. **Llama-3.1-8B-Instruct: the hedging half holds, the entropy half does not.** The familiarity direction makes it hedge (0.02 → 0.29) far more than the knowledge direction does (→ 0.09). But both directions raise entropy about equally.

## Interpretation and caveats

- **This is the paper's first causal evidence at the level of behaviour, not first-token log-odds,** that the familiarity direction drives "I don't know". It replicates in two families.
- **The base direction is fitted on a different prompt format** ("Question: … Answer:" vs the chat template). That alone could lower base/instruct alignment. The negative reuse result is therefore "the base model's direction, estimated on its own natural prompt, does not drive the instruct model's hedging", not "instruction tuning creates a new representation".
  - Together with the lens result (a familiarity signal is decodable in base models), the defensible reading is: base models encode familiarity, but the direction that controls the instruct model's "I don't know" is not simply the base direction carried over.
- **Gemma's hedging is scored with `HEDGE_RE`,** which undercounts Gemma's hedges (judge audit). The +0.22 effect is therefore conservative.
- **Gemma needed 4× the natural norm.** Llama responded at 1×.
- **Only two pairs.** Version 1 (five pairs, NOT TESTABLE because of the position mismatch) is reported separately in `results/direction_reuse/FINDINGS.md`.

## Qualification (7 Oct, later): `results/direction_nohedge/FINDINGS.md`

For Llama, a familiarity direction fitted only on items the model never hedged on raises hedging by just +0.04 [+0.01, +0.07] over random directions. The full direction raises it by +0.27. So the steering effect above is largely carried by components shared with the model's own hedging, and should be described as "the obscure-vs-popular direction drives 'I don't know'", not "familiarity per se drives it". Gemma was not tested this way.
