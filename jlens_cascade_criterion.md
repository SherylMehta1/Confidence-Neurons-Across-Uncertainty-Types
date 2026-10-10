# Jacobian-lens "familiarity before knowledge" study across models: criterion, stated before the runs

Written 6 Oct 2026, before any model in this study was run through a lens. Same status as the factorial criteria: a written bar, not a full pre-registration.

## Question

Across models, does the decoded "I don't know" signal separate obscure from popular entities (**familiarity**) at an earlier layer than it separates unknown from known facts (**knowledge**)? The human feeling-of-knowing literature predicts this order: a fast, familiarity-based judgment, revised later by retrieval (Koriat & Levy-Sadot 2001; Vernon & Usher 2003). A second question: is the familiarity signal already present in base models, before instruction tuning?

## Models (fixed now; official lenses from `neuronpedia/jacobian-lens`, all fitted with the same recipe)

| Model | Weights loaded | Lens file | Prompt |
|---|---|---|---|
| Llama-3.1-8B-Instruct | unsloth/Meta-Llama-3.1-8B-Instruct | llama3.1-8b-it/…/Llama-3.1-8B-Instruct_jacobian_lens.pt | chat |
| Llama-3.1-8B | unsloth/Meta-Llama-3.1-8B | llama3.1-8b/…/Llama-3.1-8B_jacobian_lens.pt | base |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | qwen2.5-7b-it/…/Qwen2.5-7B-Instruct_jacobian_lens.pt | chat |
| Gemma-2-9B-it | unsloth/gemma-2-9b-it | gemma-2-9b-it/…/gemma-2-9b-it_jacobian_lens.pt | chat |
| Gemma-2-9B | unsloth/gemma-2-9b | gemma-2-9b/…/gemma-2-9b_jacobian_lens.pt | base |
| Gemma-2-2B-it | unsloth/gemma-2-2b-it | gemma-2-2b-it/…/gemma-2-2b-it_jacobian_lens.pt | chat |
| Gemma-2-2B | unsloth/gemma-2-2b | gemma-2-2b/…/gemma-2-2b_jacobian_lens.pt | base |
| OLMo-3-1025-7B | allenai/Olmo-3-1025-7B | olmo-3-1025-7b/…/Olmo-3-1025-7B_jacobian_lens.pt | base |

The `unsloth/` repos are ungated mirrors of the Meta and Google weights the lenses were fitted on. A model that fails to load or run is reported as not run, not replaced.

## Items, prompts, labels

- **Items:** the same 2,637 PopQA candidates as every factorial (popularity arm = bottom vs top decile per relation).
- **Chat prompt:** the model's chat template plus the "The answer is" prefill (`build_completion_prompt`).
- **Base prompt:** `Question: {q}\nAnswer: The answer is`. The leading BOS text is stripped before `jlens` tokenizes, because `jlens` adds one itself.
- **Check, per model, before any analysis:** the final-layer logits returned by `jlens` must match a plain forward pass on the same tokens with exactly one BOS (max |Δ logit| < 0.5 on 5 prompts). A model that fails this check is not analysed.
- **Knowledge labels:** the same SliCK labelling as the builder (greedy plus 10 samples at T = 0.7, 12 new tokens, the same alias grader), on the same prompt the lens reads. Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct reuse their existing `familiarity_v3*` labels, which were made on the identical chat prompts. All other models are labelled here. "Maybe" items are excluded.

## Measures (per model, at the last prompt position, every layer ℓ)

- Y(ℓ) = lens-decoded hedge-vs-answer log-odds (`circuit_common.Readout`).
- **Familiarity curve** F(ℓ) = mean over knowledge levels of [Y(obscure) − Y(popular)].
- **Knowledge curve** K(ℓ) = mean over popularity levels of [Y(Unknown) − Y(HighlyKnown)].
- **Onset** of a curve = the first layer where (a) the curve reaches 50% of its maximum over layers, with a positive maximum, and (b) its 95% within-cell item-bootstrap CI excludes 0. Reported as relative depth ℓ / (number of layers − 1).
- **D** = K onset − F onset (in layers). 95% CI from 2,000 within-cell item bootstraps, recomputing both onsets in each replicate with rule (a) only.

## Per-model outcome

- **SUPPORTS** (familiarity first): D's CI lies entirely above 0, or F has an onset and K has none.
- **CONTRADICTS:** D's CI lies entirely below 0, or K has an onset and F has none.
- **UNDECIDED:** anything else.

## What may be claimed across models

- **"The familiarity signal is decoded before the knowledge signal across models":** only if at least 5 of the 8 models are SUPPORTS and at most 1 is CONTRADICTS.
- **"The order differs across models":** if 2 or more are CONTRADICTS.
- **Otherwise:** "mixed", reported model by model.
- **"The familiarity signal predates instruction tuning":** only if, in each of the 4 base models, the mean of F(ℓ) over layers with relative depth in [1/3, 2/3] is positive with a 95% bootstrap CI excluding 0. Report each base and instruct pair side by side either way.

## Descriptive only

Lens entropy curves; the rank of " unknown" by layer and cell; final-layer (model) readouts. Gemma-3 sizes may be added after the eight are done, as a descriptive scale series that does not count toward the claims above.

## Not to be done after seeing results

Changing the 50% threshold, the [1/3, 2/3] window or the model list; changing prompt formats; swapping lenses; reusing labels from a different prompt format.
