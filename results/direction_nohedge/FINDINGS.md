# Is Llama's familiarity steering effect familiarity itself, or "hedging intent"? (7 Oct 2026)

- **Governed by:** `nohedge_direction_criterion.md` (sha256 `797f52d4`, written before the run).
- **Script:** `scripts/direction_nohedge.py`.
- **Outputs:** `llama/summary.txt`, `llama/steer.csv`, `llama/log.txt`. Run on molab box eight.

## Result: HEDGING INTENT NOT EXCLUDED

| Direction (layer 17, natural norm 4.84, 150 held-out popular known prompts) | Hedge rate |
|---|---|
| none | 0.020 |
| d_fam (obscure − popular, knowledge-balanced; replicates v2 exactly) | **0.287** (+0.267) |
| **d_nohedge_U** (obscure − popular among Unknown items Llama **never** hedged on: 41 vs 108 working-half items) | **0.067** (+0.047) |
| d_nohedge_both (adds the 13 vs 316 zero-hedge HighlyKnown items) | 0.087 |
| 5 random directions (mean) | 0.025 |

- d_nohedge_U minus random: **+0.041 [+0.013, +0.073]**. That is positive, but below the 0.10 bar and only 18% of d_fam's effect, so both conditions fail.
- cos(d_nohedge_U, d_fam) = 0.61; cos(d_nohedge_both, d_fam) = 0.85.

## What it means

- **Most of the familiarity direction's causal effect on hedging comes from the part of the obscure-vs-popular difference that co-varies with the model being about to hedge.** A familiarity direction with hedging held at zero has only a small effect (about +4 points).
- So the steering result in `results/direction_reuse_v2/` shows that **the hedge-associated familiarity contrast** drives "I don't know". It does not show that familiarity as such does, at the residual-stream level.
- **Caveat in the other direction:** obscure unknown items that Llama never hedges on are an unusual subset (plausibly entities it confidently confabulates about). The control direction may therefore under-represent the familiarity signal the model normally uses. This test cannot separate those two readings.

## Consequence for the paper

Reword the causal claim to:

> "Steering along the obscure-vs-popular direction makes Llama and Gemma say they do not know; the effect is largely carried by components shared with the model's own hedging, and a familiarity direction fitted only on never-hedged items has a small effect (+0.04)."

The behavioural familiarity-over-knowledge result (two judges, +0.30) is unaffected.
