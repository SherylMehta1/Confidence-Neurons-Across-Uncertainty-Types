# Direction reuse, version 1: base vs instruct familiarity directions (7 Oct 2026)

- **Governed by:** `direction_reuse_criterion.md` (sha256 `3281f804`, written before any run).
- **Script:** `scripts/direction_reuse.py`.
- **Per-pair outputs:** `summary.txt`, `result.json`, `geometry.csv`, `steer.csv`, `log.txt`.
- **Runs:** molab boxes four and five, 7 Oct.

## Verdicts

| Pair | ℓ* | cos(d_fam base, inst) at ℓ* | Permutation null 99th pct | Condition 1 | Hedge rate: none / own d_fam / base d_fam / random mean | Verdict |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 17 / 32 | 0.41 | 0.49 | no | 0.020 / 0.053 / 0.027 / 0.020 | NOT TESTABLE |
| Gemma-2-9B | 23 / 42 | 0.40 | 0.52 | no | 0 / 0 / 0 / 0 | NOT TESTABLE |
| Gemma-2-2B | 14 / 26 | 0.59 | 0.64 | no | 0 / 0 / 0 / 0 | NOT TESTABLE |
| Qwen2.5-7B | 15 / 28 | 0.36 | 0.32 | **yes** | 0 / 0 / 0 / 0 | NOT TESTABLE |
| Mistral-7B-v0.3 | 17 / 32 | 0.46 | 0.57 | no | 0 / 0 / 0 / 0 | NOT TESTABLE |

**Claims, per the criterion:**
- **"Instruction tuning reuses a familiarity direction already present in the base model": not licensed.** No pair is REUSED.
- **Intervention-level double dissociation: not shown in any model.**

## What the result says, and does not

- **Geometry.** Base and instruct familiarity directions are clearly aligned (cosine 0.36-0.59 at ℓ*). But in 4 of 5 pairs they are no more aligned than directions from an arbitrary split of the same items, which is what the permutation null measures. So base and instruct models share their item geometry in general, and this test gives no evidence that the familiarity direction in particular is preserved. Qwen is the exception (0.36 vs a 99th percentile of 0.32).
- **Steering.** At the natural norm, adding any direction at a single layer barely changes free-generation hedging. Llama's own direction raises hedging from 0.020 to 0.053 (0.093 at 2×). This is far weaker than the 20-head + 100-neuron circuit patch, which abolishes hedging.
- **Two design problems limit what NOT TESTABLE means.**
  1. **Position mismatch.** Directions were fitted at the prefill position ("… The answer is") but added at the last position of the free-generation prompt. Version 2 (`direction_reuse_v2_criterion.md`) fixes this for Llama and Gemma-9B, with a calibrated dose.
  2. **Detector.** Steering outcomes were scored with `HEDGE_RE`. The judge audit later showed it misses 86-90% of Qwen, Gemma and Mistral hedges. For those four instruct models, "0 hedging under every condition" is partly a measurement floor. Only Llama's steering numbers are measured with an adequate detector (regex recall 0.80).
- **Entropy, descriptively.** The knowledge direction raises first-token entropy in every model (e.g. Gemma-2B 0.44 → 0.58, Gemma-9B 0.25 → 0.33). In Llama the familiarity direction raises it about as much (0.82 → 1.14), so there is no dissociation.
