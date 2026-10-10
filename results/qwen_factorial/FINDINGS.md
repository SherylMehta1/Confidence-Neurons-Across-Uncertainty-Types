# Qwen2.5-7B-Instruct popularity x knowledge factorial (6 Oct 2026)

Governed by `qwen_factorial_criterion.md` (sha256 `eba35dec…`, written 07:26 UTC, before any Qwen item was scored; the uploaded copy was hash-checked before the read-off ran). Pool: `data/familiarity_v3_qwen/` (2,637 PopQA candidates, 2,636 of them shared with Llama's 2,673-item pool, which holds 37 extra items from an earlier run; identical builder arguments, bf16, molab RTX PRO 6000). Numbers: `summary.txt`, `effects.csv`, `cells.csv`, `cross_model_items.csv`. Scripts: `scripts/qwen_factorial_analysis.py`, `scripts/add_hedge_logodds.py`.

## Verdict: PARTIAL

| | Hedging: popularity vs knowledge | Entropy: knowledge vs popularity |
|---|---|---|
| **Llama** (reference) | **+0.63 vs +0.19**, difference +0.44 [+0.34, +0.53] | **+2.14 vs +0.51**, difference +1.63 |
| **Qwen** (criterion) | **+0.047 vs +0.043**, difference +0.003 [−0.02, +0.04]: **fails condition 1** | **+1.23 vs −0.02**, difference +1.25 [+1.07, +1.42]: **passes condition 2** |

- **Governing readout.** Free-generation hedge rate, because Qwen's pool-level hedge rate is 0.051, just above the 0.05 floor. It is 0.053 counting only HighlyKnown/Unknown items, so the choice does not depend on how the pool is defined.
- **If the log-odds readout had governed**, the hedging difference would be −2.49 [−3.41, −1.64]. That is **FAILS**, not REPLICATES. No choice of readout gives REPLICATES.
- **What licenses the paper:** the familiarity claim stays **Llama-scoped**. "Two language models" is not licensed for the hedging half.

## What the Qwen data show

1. **The entropy half replicates, more cleanly than in Llama.** Qwen's first-token entropy follows knowledge (+1.23) and ignores popularity entirely (−0.02 [−0.12, +0.08]). In the pool regression the popularity coefficient is −0.03 (p = 0.54). In Llama it was −0.39.
2. **Qwen barely says "I don't know".** Its free-generation hedge rate is 9.3% even on obscure facts it does not know, against Llama's 85%. Qwen makes something up instead. On the rate scale, popularity and knowledge each move its hedging by about 0.045. In the logistic pool model, popularity still predicts Qwen's hedging with knowledge held fixed (−0.92 per SD, p < 1e-13), half Llama's −1.87. So familiarity affects Qwen's hedging, but there is very little hedging to affect.
3. **A small hedge-without-entropy effect in Qwen.** Among facts Qwen does not know (740 pairs), obscure entities get more hedging (+0.075 [+0.06, +0.09]) with no more entropy (−0.12). That is the dissociation pattern, at a tenth of Llama's size. It survives the name-spelling controls (+0.067). Report it as descriptive.
4. **Name spelling matters in Qwen too, but does not explain popularity.** The popularity coefficient barely moves when surface-form measures are added (−0.92 → −0.88). Qwen's `pop_given_known` contrast (37 pairs) shows no hedging gap at all (+0.03), so there is nothing there to explain.

## By-product that matters for the paper (both models)

**The first-token hedge-vs-answer log-odds, the readout behind every circuit number, does not carry the familiarity dissociation, even in Llama.**

| Llama | Popularity effect | Knowledge effect |
|---|---|---|
| Free-generation hedge rate | +0.63 | +0.19 |
| First-token hedge log-odds (prefill "The answer is") | +2.93 | +3.06 |

- In Llama the log-odds readout responds to knowledge as much as to popularity. In Qwen it responds mostly to knowledge (+3.63 vs +1.14).
- The two readouts use different prompts. Free generation has no prefill; the log-odds is read after "The answer is". The prefilled first token mixes "do I recognise this name" with "do I have an answer".
- This fits check 2 in `results/unireps_checks/FINDINGS.md`: the circuit moves the first-token log-odds fully, yet generation can still answer.
- **Consequence:** the paper cannot argue "the circuit carries the familiarity signal" from log-odds recovery alone. The knowledge-matched factorial transfer on `pop_given_unknown` (0.92 / 1.01, knowledge absent on both sides) helps, but it is itself a first-token log-odds recovery, so it does not escape the readout-versus-generation gap. The generation-time removal test is the behavioural evidence.
- Also report hedge-token probability mass beside log-odds: median 0.098 in Llama and 0.020 in Qwen at the prefill.

## Cross-model agreement (descriptive, same items, 2,097 scored by both)

| Readout | All items | Same knowledge in both (n = 1,784) | Knowledge differs (n = 313) |
|---|---|---|---|
| Hedge rate | ρ +0.37 | +0.33 | **+0.50** |
| Entropy | ρ +0.32 | +0.46 | **−0.17** |
| Hedge log-odds | ρ +0.51 | +0.50 | +0.11 |

- On items where the two models disagree about knowing the fact, their **free hedging still agrees** (ρ +0.50). Qwen's hedging there follows Llama's hedging (β +0.20, p = 0.044), not Qwen's own knowledge (p = 0.12). Their **entropies diverge** (ρ −0.17), because each follows its own knowledge.
- That is the pattern the shared-familiarity "why" account predicts (litreview/G_cross_model_convergence.md §0.4).
- **Caveats:** Qwen's hedge rates are near zero for most items, so ρ rests on a heavily tied minority. It is descriptive and not in the criterion. Treat it as a pointer for a follow-up, not a headline.

## Recommendation for the 4-page paper

- Keep the familiarity × knowledge dissociation as a **Llama** result, and say so in the title or abstract.
- Report Qwen in one paragraph:
  - the entropy half replicates, cleanly;
  - the hedging half does not, because Qwen rarely hedges at all;
  - the locked criterion's verdict is PARTIAL.
- Frame Qwen as a natural contrast case: a model that rarely says "I don't know" by our detector where Llama does (pending the judge audit of the detector), with the same knowledge-driven entropy. That links to Qwen's missing on/off asymmetry and its aleatoric leakage, both already in the paper ("the channels are not kept apart").
- Fix the circuit section's framing per the by-product above.


*Corrected 7 Oct 2026 after an independent audit of every number in this file: wording and figures fixed where the audit found them wrong or overstated.*
