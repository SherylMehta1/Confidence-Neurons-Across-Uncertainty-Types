# Does instruction tuning reuse the base model's familiarity direction? Criterion, stated before the runs

Written 6 Oct 2026, after the lens study (`results/jlens_cascade/FINDINGS.md`) and before any run of this experiment. A written bar, not a full pre-registration.

## Question

An instruct model is fine-tuned from its base model, so the two share a residual-stream coordinate system. The lens study found a familiarity signal mid-stack in every base model.

1. Is the instruct model's familiarity direction the **same direction** as the base model's?
2. Does the **base** model's direction, added to the instruct model, **make it say "I don't know"** about facts it knows?
3. As an intervention version of the factorial: does the familiarity direction move hedging more than answer entropy, and the knowledge direction the reverse?

## Pairs (fixed now)

| Pair | Base | Instruct |
|---|---|---|
| Llama | unsloth/Meta-Llama-3.1-8B | unsloth/Meta-Llama-3.1-8B-Instruct |
| Gemma-9B | unsloth/gemma-2-9b | unsloth/gemma-2-9b-it |
| Gemma-2B | unsloth/gemma-2-2b | unsloth/gemma-2-2b-it |
| Qwen | Qwen/Qwen2.5-7B | Qwen/Qwen2.5-7B-Instruct |
| Mistral | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 |

A pair that fails to load or run is reported as not run, not replaced.

## Items, prompts, labels

- **Items:** the 2,673 PopQA candidates of the lens study.
- **Split:** a fixed random half (seed 0) is the *working* set, used to fit directions; the other half is *held out*, used for steering prompts.
- **Prompts and labels:** each model uses its lens-study prompt (base: `Question: … Answer: The answer is`; instruct: chat template plus "The answer is") and its own SliCK labels from the lens study, or the factorial pools where the prompts are byte-identical. Qwen and Mistral base models are labelled in-run with the same procedure.

## Directions (per model, per layer ℓ, last prompt position, working half)

- **Familiarity direction** d_fam(ℓ) = ½ [mean(obscure, Unknown) − mean(popular, Unknown) + mean(obscure, HighlyKnown) − mean(popular, HighlyKnown)].
- **Knowledge direction** d_know(ℓ) = ½ [mean(obscure, Unknown) − mean(obscure, HighlyKnown) + mean(popular, Unknown) − mean(popular, HighlyKnown)].
- **Intervention layer** ℓ* = round(0.55 × (number of layers − 1)), fixed for all models. 0.55 is the median relative onset depth in the lens study.

## Measures

- **G (geometry):** cos(d_fam^base(ℓ*), d_fam^inst(ℓ*)). Null: 200 permutations of the popularity arm within knowledge cells of the working half, with the same permutation applied to both models; the 99th percentile of null cosines is the bar. Reported at every layer, descriptively.
- **S (steering, behaviour):**
  - **Prompts:** the instruct model's free-generation prompt (chat template, no prefill; the factorial's hedge-rate prompt). The first 150 held-out items that the instruct model labels popular and HighlyKnown.
  - **Decoding:** greedy, 24 new tokens. Hedging is scored by `behavioral_test.HEDGE_RE` (primary); the amended detector of `results/unireps_checks` is reported alongside.
  - **Intervention:** a vector added to the residual stream output of layer ℓ* at the last prompt position and every generated position.
  - **Conditions (all vectors scaled to ‖d_fam^inst(ℓ*)‖, the natural obscure − popular difference):** none; instruct's own d_fam; the base model's d_fam; instruct's d_know; 5 random directions. Secondary: own and base d_fam at 2× scale.
- **E (entropy):** first-token entropy at the prefill prompt (chat + "The answer is") on the same 150 items, under the same conditions.

## Per-pair verdict

**REUSED** if all three hold:
1. G exceeds the 99th percentile of its permutation null;
2. hedge rate under the base direction minus the mean hedge rate under the random directions ≥ 0.10;
3. the base direction's effect (minus "none") ≥ 50% of the instruct model's own-direction effect (minus "none").

Other outcomes:
- **NOT TESTABLE:** the instruct model's own direction raises hedging by < 0.10 over random. In that case steering cannot test reuse; report G alone.
- **NOT REUSED:** otherwise.

## What may be claimed

- **"Instruction tuning reuses a familiarity direction already present in the base model":** only if at least 3 of the 5 pairs are REUSED and none fails condition 1.
- **"Intervening on familiarity moves hedging more than entropy, and on knowledge the reverse"** (per model, as an intervention-level double dissociation): only if, with 95% prompt-bootstrap CIs excluding 0,
  - Δhedge(d_fam) > Δhedge(d_know), and
  - ΔH(d_know) > ΔH(d_fam), where Δ is relative to "none" at equal norm.

  Reported per instruct model; claimed for a model only if both hold.

## Not to be done after seeing results

Changing ℓ*, the 0.10 / 50% bars, the scale, the prompt set, the detector, or the pairs; refitting directions on held-out items; dropping random directions.
