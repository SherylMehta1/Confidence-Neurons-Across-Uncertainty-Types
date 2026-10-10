# Jacobian-lens study across eight models: familiarity vs knowledge in the decoded "I don't know" signal (6 Oct 2026)

- **Governed by:** `jlens_cascade_criterion.md` (sha256 `c8e04372`, written 10:05 UTC, before any model in this study was run through a lens).
- **Scripts:** `scripts/jlens_cascade.py` (per model), `scripts/jlens_cascade_analysis.py` (read-off and figure).
- **Outputs:** `summary.txt`, `outcomes.csv`, `curves.csv`, `fig_cascade.{pdf,png}`, and per-model `labels.jsonl`, `lens.csv`, `bos_check.txt`, `lens.provenance.json`.

## Setup in one paragraph

Eight models with official prefitted lenses from `neuronpedia/jacobian-lens`, all fitted with the same recipe:
- **instruct:** Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Gemma-2-2B;
- **base:** Llama-3.1-8B, Gemma-2-9B, Gemma-2-2B, OLMo-3-7B.

Each model reads the same 2,673 PopQA questions. Instruct models get their chat template plus "The answer is"; base models get "Question: … Answer: The answer is". Every item is labelled known or unknown by sampling that model on that prompt.

At the last prompt position, every layer is decoded through the model's lens, giving the hedge-vs-answer log-odds. Two curves come from the 2 × 2 of popularity × knowledge:
- **familiarity:** obscure minus popular, averaged over knowledge;
- **knowledge:** unknown minus known, averaged over popularity.

Before any analysis, each model's `jlens` final-layer logits matched a plain single-BOS forward pass, within 0.0625–0.125 logits.

## Results under the locked rule

| Model | Layers | Familiarity onset (rel. depth) | Knowledge onset | D = K − F [95% CI] | Mid-depth familiarity [CI] | Outcome |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 32 | L17 (0.55) | L17 (0.55) | 0 [0, 0] | +1.58 [+1.39, +1.76] | UNDECIDED |
| Llama-3.1-8B (base) | 32 | L16 (0.52) | L16 (0.52) | 0 [0, +1] | +0.67 [+0.59, +0.75] | UNDECIDED |
| Qwen2.5-7B-Instruct | 28 | L20 (0.74) | L20 (0.74) | 0 [−1, +1] | +0.21 [+0.17, +0.26] | UNDECIDED |
| Gemma-2-9B-it | 42 | L24 (0.59) | L24 (0.59) | 0 [−2, +1] | +0.71 [+0.60, +0.82] | UNDECIDED |
| Gemma-2-9B (base) | 42 | L24 (0.59) | L25 (0.61) | +1 [+1, +2] | +0.39 [+0.35, +0.43] | SUPPORTS |
| Gemma-2-2B-it | 26 | L17 (0.68) | L15 (0.60) | −2 [−2, 0] | +0.40 [+0.36, +0.45] | UNDECIDED |
| Gemma-2-2B (base) | 26 | L16 (0.64) | L16 (0.64) | 0 [0, +1] | +0.40 [+0.33, +0.47] | UNDECIDED |
| OLMo-3-1025-7B (base) | 32 | L19 (0.61) | L14 (0.45) | −5 [−5, −5] | +0.29 [+0.19, +0.38] | CONTRADICTS |

**Claims, per the criterion:**
1. **"Familiarity is decoded before knowledge across models": NOT supported.** The criterion's outcome is "mixed" (1 SUPPORTS, 1 CONTRADICTS, 6 UNDECIDED). The cascade order predicted from human feeling-of-knowing research does not appear.
2. **"The familiarity signal predates instruction tuning": SUPPORTED.** All four base models carry a positive mid-depth familiarity signal in the decoded hedge readout, each 95% CI excluding 0. Within the three families that have a base model here (Llama, Gemma, OLMo), the familiarity signal in the decoded readout does not depend on instruction tuning. This is an observational comparison through separately fitted lenses; the direction-reuse experiment tests it causally.

## Descriptive observations (not tested claims)

- **One shared event, not a cascade.** In 5 of 8 models the familiarity and knowledge onsets fall in the same layer, and in 7 of 8 within 2 layers (OLMo: 5 apart). All onsets lie between 0.45 and 0.74 of depth (median about 0.6). So the main regularity across families is a mid-stack event at a similar relative depth where both signals appear, with OLMo the exception.
- **What happens after the onset differs by model.**
  - In base Llama the familiarity signal persists late while the knowledge signal falls to about zero. In base Gemma-2-2B and OLMo both curves are unstable late (OLMo's familiarity curve turns negative from L23; Gemma-2-2B's ends at -0.57).
  - In instruct Llama, Qwen and both Gemma-2-9B models, a knowledge signal stays large late.
  - This matches the factorial result that the final first-token readout mixes both signals. Eight models and no tested prediction: do not claim an instruction-tuning effect on the late layers.
- **Caveat on OLMo's CONTRADICTS.** OLMo's knowledge curve is mostly negative (as low as −2.4); its "onset" is a small early bump (+0.3 at depth 0.45) that clears 50% of a tiny positive maximum. OLMo also has very few known items (15 obscure-known, 85 popular-known). The verdict stands under the rule, but it reflects a degenerate case of the onset definition, not a clear early knowledge signal.
- **Caveat on Gemma-2-9B base's SUPPORTS:** one layer.
- **Small cells.** Obscure-known cells are small in several base models (Gemma-2-2B 33, Gemma-2-9B 50, OLMo 15), so their knowledge curves are noisier.

## Exploratory robustness check: a hedge-only readout (not in the criterion; run after seeing the results)

The criterion's readout, hedge-vs-answer log-odds, has a known weakness. Its denominator is generic answer words (" a", " the", " that"…). When a model is confident in a specific answer, it stops predicting those, so the log-odds can rise for known items for reasons unrelated to hedging: the "generic answer channel" leak the paper already documents for Qwen's aleatoric arm. The lens files also store a hedge-only measure, the rank of " unknown", which involves no answer words. Recomputing the curves on −log(rank + 1):

| Model | Known items | Onsets F / K | D [CI] | Mid-depth F [CI] | Late K (last third) | Outcome on this readout |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 763 | L14 / L16 | +2 [+1, +2] | +2.24 [+2.08, +2.39] | +1.97 | SUPPORTS |
| Llama-3.1-8B | 511 | L14 / L16 | +2 [+2, +3] | +0.48 [+0.40, +0.56] | −0.22 | SUPPORTS |
| Qwen2.5-7B-Instruct | 444 | L21 / L20 | −1 [−1, −1] | +0.01 [−0.04, +0.07] | +0.75 | CONTRADICTS |
| Gemma-2-9B-it | 744 | L24 / L24 | 0 [0, 0] | +1.23 [+1.15, +1.31] | +1.05 | UNDECIDED |
| Gemma-2-9B | 515 | L23 / L28 | +5 [+5, +9] | +0.37 [+0.31, +0.42] | +1.03 | SUPPORTS |
| Gemma-2-2B-it | 487 | L17 / L15 | −2 [−2, −2] | +0.54 [+0.48, +0.60] | +1.61 | CONTRADICTS |
| Gemma-2-2B | 274 | L17 / L16 | −1 [−1, +5] | +0.14 [+0.09, +0.21] | +0.90 | UNDECIDED |
| OLMo-3-1025-7B | 100 | L16 / L19 | +3 [+3, +4] | +0.62 [+0.52, +0.72] | +1.61 | SUPPORTS |

**What this shows:**
- **OLMo's CONTRADICTS depends on the readout.** On the hedge-only readout its order flips, and its strongly negative knowledge curve becomes positive. OLMo also knows very few of these facts under the base prompt (100 known items, 15 obscure-known), so its knowledge curve is poorly estimated. There is no evidence that architecture is involved, and none against it: OLMo-2/3 differ from Llama in normalisation placement and QK-norm, and this study cannot separate architecture from training data, knowledge level and readout.
- **The onset order depends on the readout** (4 SUPPORTS / 2 CONTRADICTS here, versus 1 / 1 on the criterion's readout). That is further reason not to claim any order.
- **The base-model claim survives both readouts:** all four base models have a positive mid-depth familiarity signal under each.
- **Caveat:** " unknown" is not every model's hedge word (Qwen's is " not"; paper, Figure 5), so this readout under-measures Qwen. A per-model hedge vocabulary would be the fair version.

## Exploratory check (7 Oct): is the base-model familiarity signal a name-spelling artifact?

Item-level regression of the mid-depth lens hedge readout (relative depth 1/3-2/3) on popularity (z), with each model's own knowledge label held fixed, relation fixed effects and SE clustered by relation. It was run with and without the name surface-form controls from `scripts/unireps_confound_checks.py`: word Zipf mean/min, token count and token rank, with tokens from Llama's tokenizer as a proxy for every model.

| Model | Readout | Popularity coefficient, no surface control | With surface control |
|---|---|---|---|
| Llama-3.1-8B (base) | log-odds / hedge-only | −0.28 / −0.16 | **−0.30 / −0.17** |
| Gemma-2-9B (base) | log-odds / hedge-only | −0.14 / −0.15 | **−0.14 / −0.13** |
| Gemma-2-2B (base) | log-odds / hedge-only | −0.14 / −0.11 | **−0.12 / −0.09** |
| OLMo-3-7B (base) | log-odds / hedge-only | −0.10 / −0.21 | **−0.07 / −0.19** |
| Llama-3.1-8B-Instruct | log-odds / hedge-only | −0.89 / −1.27 | −0.93 / −1.24 |
| Gemma-2-9B-it | log-odds / hedge-only | −0.39 / −0.60 | −0.40 / −0.62 |

Every 95% CI excludes 0, so **the base-model familiarity signal survives knowledge and name-spelling controls on both readouts.**

Descriptively, the instruct models' coefficients are 3-4× their base models'. That is consistent with instruction tuning amplifying an existing familiarity signal. It is not a tested claim, because each model is decoded through its own separately fitted lens.

## What this means for the paper

- The lens study gives the UniReps paper one criterion-licensed cross-model result and one descriptive one:
  - (licensed) the familiarity signal **predates instruction tuning** in the three families with a base model (Llama, Gemma, OLMo);
  - (descriptive) on the criterion's readout, mid-depth familiarity is positive in all eight models; on the hedge-only readout it is positive in seven (Qwen's CI spans 0). Neither readout controls for name spelling, which explains part of the familiarity effect in Llama's behaviour.
- It also **retires a tempting story**: familiarity does not come before knowledge in depth. Say so, as a prediction that failed.
- Combined with the factorials, the defensible cross-model sentence is:

  > Across eight models from four families, a mid-stack signal that decodes as "I can't answer" responds to how familiar the entity is, and in the three families where we can compare, it is already present in the base model. Whether a model then *says* "I don't know" differs: Llama's hedging follows familiarity, while Qwen, Gemma and Mistral hedge rarely by our detector (judge audit pending).

  (Mistral is not in the lens study; its part of the sentence comes from the factorial.)

## Provenance

- Runs were spread over four molab sandboxes, three of which were terminated mid-study (HTTP 410). Llama ×2, Gemma-2-2B ×2 and Qwen were completed and copied off-box before their sandbox died.
- OLMo and both Gemma-2-9B models were rerun from scratch, with the identical script (sha256 `1ff2c1d7…`) and criterion, on replacement sandboxes, and synced every 10 minutes. No model was replaced or dropped.
- Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct reuse SliCK labels made on byte-identical prompts. Qwen's 37 items missing from its pool were labelled in-run. Note: `jlens_cascade_criterion.md` says "the same 2,637 PopQA candidates"; the runs actually used the 2,673-item Llama pool (37 extra items from an earlier run), identically for all eight models.


*Corrected 7 Oct 2026 after an independent audit of every number in this file: wording and figures fixed where the audit found them wrong or overstated.*
