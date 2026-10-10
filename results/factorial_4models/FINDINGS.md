# Popularity x knowledge factorial across four model families (6 Oct 2026)

Governing rules, all written before the runs they govern:

| Rule file | Covers | sha256 (prefix) | Written |
|---|---|---|---|
| `qwen_factorial_criterion.md` | Qwen | `eba35dec` | 07:26 UTC |
| `gemma_mistral_factorial_criterion.md` | Gemma, Mistral, and what the paper may claim across models | `b53d9a27` | 09:45 UTC, after Qwen, before Gemma/Mistral |

- **Data:** the same 2,637 PopQA candidates for Qwen, Gemma and Mistral (2,636 shared with Llama's 2,673-item pool), identical builder arguments, bf16, molab RTX PRO 6000.
- **Numbers:** `summary.txt`, `effects.csv`, `cells.csv` here; per-model folders `results/{qwen,gemma,mistral}_factorial/`.
- **Scripts:** `scripts/qwen_factorial_analysis.py --models … --out …`, `scripts/add_hedge_logodds.py`.

## Verdicts

| Model | Pool hedge rate | Governing hedging readout | Hedging: popularity vs knowledge | Entropy: knowledge vs popularity | Verdict |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.474 | free generation | **+0.63 vs +0.19** | **+2.14 vs +0.51** | **REPLICATES** (discovery model) |
| Gemma-2-9B-it | 0.036 (< 0.05) | first-token log-odds | **+1.79 vs −0.32** | **+1.08 vs +0.09** | **REPLICATES** |
| Qwen2.5-7B-Instruct | 0.051 | free generation | +0.047 vs +0.043 (diff CI spans 0) | **+1.23 vs −0.02** | PARTIAL |
| Mistral-7B-Instruct-v0.3 | 0.066 | free generation | +0.067 vs +0.058 (diff CI spans 0) | **+1.41 vs +0.30** | PARTIAL |

**What the rules license** (`gemma_mistral_factorial_criterion.md`, "What the paper may claim across models"):
- "Language models" in the title: **not licensed.** That needed both Gemma and Mistral to replicate.
- "N of 4 models": **2 of 4** (Llama, Gemma), with every verdict reported.
- Title: Llama-scoped (the rule's prescription when fewer than two of Gemma/Mistral replicate).

## What holds in all four families

**Entropy follows knowledge, not popularity, in every model.** Condition 2 holds in 4 of 4. The knowledge effect is at least 4 times the popularity effect (Llama 4.2×, Mistral 4.8×, Gemma 12×), and in Qwen popularity has no effect on entropy at all (−0.02). This is the robust cross-model result.

**Hedging is familiarity-driven where it can be measured.**
- In the pool regressions (relation fixed effects, knowledge held fixed), popularity predicts free-generation hedging in all four models: logit per SD −1.87 Llama, −1.19 Gemma, −1.05 Mistral, −0.92 Qwen; all p < 1e-7.
- In every model this survives controls for how the name is spelled (`results/unireps_checks/summary_*`).
- But only Llama hedges often enough (47% of items) for familiarity to dominate the behaviour. Gemma, Qwen and Mistral hedge on 4-7% of items, so on the rate scale popularity and knowledge both move hedging by only about 0.03-0.07.

**WITHDRAWN 7 Oct after the judge audit (`results/hedge_judge/FINDINGS.md`):** the regex misses 86-90% of Qwen, Gemma and Mistral hedges (recall 0.10-0.14), so the paragraph below does not hold. Those models' hedge rates on obscure unknowns are somewhere between 0.07-0.13 (regex) and 0.42-0.76 (judge), and their factorial hedging verdicts are not measurable with this detector. Original text, kept for the record: **Two kinds of model, provisionally.** By the regex detector, Llama says "I don't know" on 85% of obscure facts it doesn't know, while Gemma, Qwen and Mistral do so on 7-13% of the same items and mostly give an answer instead. The detector undercounts non-Llama hedges (caveat 0), so these labels wait on the judge audit (`hedge_judge_criterion.md`). The familiarity-over-knowledge pattern is visible in the hedger's behaviour, and in one confabulator's first-token decision (Gemma), but not in the other two confabulators' rare hedges.

## Caveats to state in the paper

0. **Gemma's REPLICATES depends on the hedge detector (sensitivity check, 7 Oct).** `behavioral_test.HEDGE_RE` misses phrasings like "not readily available in my database" (`results/unireps_checks/FINDINGS.md`, check 3). With the gap fixed:
   - Gemma's pool-level free-generation hedge rate rises from **0.036 to 0.059**, above the 0.05 floor, so free generation would govern;
   - its popularity-minus-knowledge difference is then +0.009 [−0.013, +0.034], i.e. **PARTIAL**.

   The locked verdict (original detector, as the criterion's pipeline specified) is REPLICATES and stands. The paper must report that "2 of 4" becomes "1 of 4" under the corrected detector. Llama, Qwen and Mistral are unchanged by the fix (differences ≤ 0.003). This also shows the detector, built on Llama's phrasing, undercounts other models' hedges, so a judge-model audit of the non-Llama free generations is needed before calling them "confabulators" with confidence.
1. **Gemma's REPLICATES rests on the first-token log-odds**, the readout the pre-set floor rule selected, not on free generation. Gemma's free-generation numbers alone look like Qwen and Mistral (+0.041 vs +0.030).
2. **The first-token readout does not agree across models about knowledge.** Llama +3.06, Qwen +3.63, Mistral +2.08, Gemma −0.32. The prefilled first token is a mixed signal (`results/qwen_factorial/FINDINGS.md`). Report free-generation hedging as the behavioural measure and log-odds as the mechanistic readout, never interchangeably.
3. **Cross-model item agreement is descriptive.** Hedge-rate ρ with Llama is 0.37-0.48 across the three models. The "agreement persists where knowledge differs" hint from Qwen (ρ 0.50, n = 313) does not reproduce in Gemma (0.37, n = 60) or Mistral (0.26, n = 108), whose knowledge-disagreement sets are small. Do not headline it.
4. **Mistral's chat template** places the prefill directly after `[/INST]` with no separating whitespace; this is the template's own output.

## Provenance notes

- Box one (Qwen, Mistral and Gemma pool builds) was terminated (HTTP 410) after all three pools had been copied off-box. Gemma's log-odds file was lost and recomputed on a replacement box from the same stored prompts. Its recomputed entropy matches the stored entropy to float precision on all 2,637 items (max |delta| 7e-7).
- All factorial contrast files were rebuilt locally from the pulled pools, and match the sandbox copies by sha256 (LF-normalised).


*Corrected 7 Oct 2026 after an independent audit of every number in this file: wording and figures fixed where the audit found them wrong or overstated.*
