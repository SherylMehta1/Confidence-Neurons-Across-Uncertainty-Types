# Is the familiarity steering effect just a "hedging intent" direction? Criterion, stated before the run

Written 7 Oct 2026, after direction reuse v2 (Llama: the own familiarity direction raises hedging 0.020 → 0.287 at the natural norm) and before this run.

## The worry

The familiarity direction is the mean residual difference between obscure and popular items (knowledge-balanced). Obscure items hedge more, so the direction may partly encode "about to say I don't know". Steering along it would then induce hedging without familiarity playing any causal role.

## The control

**Model:** Llama-3.1-8B-Instruct only. Its hedge detector is adequate: regex recall 0.80 on 24-token replies and 0.99 on 96-token replies.

Fit the familiarity direction using only items the model **never** hedged on (hedge_rate = 0 over the 20 stored free generations in `data/familiarity_v3/all_measured.jsonl`), on the working half (seed 0, as in v2), at the v2 position (last position of the free-generation prompt) and layer ℓ* = 17:

- **d_nohedge_U** = mean(obscure, Unknown, hedge 0) − mean(popular, Unknown, hedge 0). Knowledge and hedging are both held fixed. **Primary.**
- **d_nohedge_both** = ½ [d_nohedge_U + (mean(obscure, HighlyKnown, hedge 0) − mean(popular, HighlyKnown, hedge 0))]. Secondary; the obscure-known zero-hedge cell is small (18 items in the full pool).

## Steering

Identical to direction reuse v2:
- the same 150 held-out popular-and-HighlyKnown prompts and the same free-generation prompt;
- greedy, 24 tokens, `HEDGE_RE`;
- the vector added at layer 17 at the last prompt position and every generated position;
- dose k = 1 × ‖d_fam^v2(17)‖, the norm at which v2 found its effect;
- all vectors scaled to that norm.

**Conditions:** none; d_fam (v2's full familiarity direction, refitted here, as a replication); d_nohedge_U; d_nohedge_both; 5 random directions.

## Verdict

**FAMILIARITY PER SE** if d_nohedge_U both:
1. raises hedging ≥ 0.10 over the mean of the random directions; and
2. achieves ≥ 50% of d_fam's effect (each minus "none").

**HEDGING INTENT NOT EXCLUDED** otherwise.

Also reported: cos(d_nohedge_U, d_fam) at layer 17, whether the d_fam effect replicates v2 (within ±0.10), and a 95% prompt-bootstrap CI on d_nohedge_U − random.

## Not to be done after seeing results

Changing the dose, the layer, the zero-hedge threshold, the prompt set or the bars; substituting d_nohedge_both for d_nohedge_U as the primary.
