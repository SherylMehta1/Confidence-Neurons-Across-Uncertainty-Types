# Gemma-2-9B-it and Mistral-7B-Instruct-v0.3 factorials: read-off criterion, stated before the runs

Written 6 Oct 2026, AFTER the Qwen result (PARTIAL; `results/qwen_factorial/FINDINGS.md`) and BEFORE any Gemma or Mistral item was scored. Same status as `qwen_factorial_criterion.md`: a written bar, not a full pre-registration.

## What is run, per model

Identical to `qwen_factorial_criterion.md`, with only the model and output directory changed:

| Model | `--model-id` | Pool | Factorial prefix |
|---|---|---|---|
| Gemma-2-9B-it | `unsloth/gemma-2-9b-it` (ungated mirror of `google/gemma-2-9b-it`) | `data/familiarity_v3_gemma` | `factorial_v3_gemma` |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | `data/familiarity_v3_mistral` | `factorial_v3_mistral` |

Builder arguments as for Llama and Qwen: 1,000 candidates per relation, quantiles 0.1/0.9, 10 samples, 20 free generations, seed 42, bf16. Then `scripts/add_hedge_logodds.py` and `scripts/build_frequency_knowledge_factorial.py --max-len-diff 3 --seed 42`.

## Per-model verdict

Exactly the rule in `qwen_factorial_criterion.md`, applied to each model separately:
- 2 × 2 cell-mean effects with 10,000 within-cell item bootstraps;
- condition 1: hedging, popularity effect > knowledge effect, CI excluding 0;
- condition 2: entropy, knowledge effect > popularity effect, CI excluding 0;
- the governing hedging readout is the free-generation hedge rate unless the model's pool-level hedge rate is below 0.05, in which case first-token log-odds governs;
- outcomes REPLICATES / PARTIAL / FAILS as defined there.

## What the paper may claim across models (decided now, before Gemma and Mistral)

Llama is the discovery model. Qwen (PARTIAL), Gemma and Mistral are the replication set.
- **"Language models" (general, in the title):** only if BOTH Gemma and Mistral come out REPLICATES. That makes three of four models, with Qwen reported as the exception.
- **"N of 4 models" in the abstract or body:** always allowed, with N counted as the models that come out REPLICATES (Llama counts as 1), and every model's verdict reported.
- **"Llama" (scoped title):** if fewer than two of Gemma/Mistral replicate.

## Not to be done after seeing the result

Same list as for Qwen (no changing the 0.05 floor, swapping readouts, dropping relations, changing gate or sampling settings, or reseeding). In addition: no swapping either model for another family, size or version after its result is seen. A model that fails to load or crashes is reported as not run, not replaced.
