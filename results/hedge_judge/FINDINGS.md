# Judge audit of the hedge detector (7 Oct 2026)

Governed by `hedge_judge_criterion.md` (sha256 `ae0bb807`, written before any judge call).

- **Judge:** Qwen2.5-32B-Instruct, greedy, fixed prompt.
- **Sample:** 1,515 free generations, stratified by model × popularity arm × knowledge × regex verdict, drawn on CPU with seed 0.
- **Files:** `sample.jsonl`, `stratum_counts.json`, `judgments.jsonl`, `summary.txt`, `corrected_cells.csv`.
- **Scripts:** `scripts/hedge_judge.py`, `scripts/hedge_judge_analysis.py`.

## Results under the criterion

| Model | Regex recall (adequate if ≥ 0.80) | Regex precision | Obscure & Unknown: regex → judge-corrected | Corrected pop − know on hedging | "Confabulator" label |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | **0.80 (adequate)** | 1.00 | 0.849 → 0.943 | **+0.132 [+0.027, +0.239]** | not licensed |
| Qwen2.5-7B-Instruct | **0.10** | 0.99 | 0.093 → 0.764 | −0.213 [−0.390, −0.018] | **not licensed** |
| Gemma-2-9B-it | **0.14** | 0.92 | 0.072 → 0.420 | −0.108 [−0.246, +0.029] | **not licensed** |
| Mistral-7B-Instruct-v0.3 | **0.14** | 0.96 | 0.129 → 0.736 | −0.172 [−0.346, +0.004] | **not licensed** |

**What the criterion licenses:**
- The regex is adequate for Llama only.
- The label "confabulator" is **not licensed for any model**. The "two kinds of model" (hedger vs confabulators) description in `results/factorial_4models/FINDINGS.md` is withdrawn.

## A check on the judge (not in the criterion; done after seeing the judgments)

16 judge-HEDGE / regex-answer cases (4 per model, random) were read by hand. They fall into three groups:

- **Genuine hedges the regex misses (about 6 of 16).**
  - "I can not verify the author of Patience." The regex has "cannot", not "can not".
  - "Yunguyo Province is not a recognized political division…"
  - "there might be a mismatch in the information you've provided."
- **Clarification requests (about 3 of 16).**
  - "Could you please provide more details about who Bala…"
  - "which specific Tanya…"
  - "Please provide more context!"

  Arguably a different behaviour from "I don't know".
- **Judge errors (about 7 of 16), mostly from truncation.** The free generations are only 24 tokens, and many stop before the answer appears:
  - "The producers of the 1990 film 'Ghost' include:"
  - "The co-founders of LinkedIn are:"
  - "'Goodbye' is a 2021 Chinese drama film directed by Zheng Ming and…"
  - "The author of the novel **Skin** is **يتيمه**."

  The judge reads such unfinished or committed replies as hedges.

**So:**
- The regex badly undercounts hedges for Qwen, Gemma and Mistral (recall 0.10-0.14 even allowing for judge error).
- The judge overcounts, partly because of the 24-token truncation.
- The true non-Llama hedge rates lie somewhere between the two estimates.
- **Neither instrument is good enough to decide the cross-model hedging verdicts.** Llama's regex verdict is more trustworthy (recall 0.80, precision 1.00). But even for Llama, judge-corrected rates shrink the familiarity-over-knowledge gap from +0.44 to +0.13, because the judge finds many more hedges on popular-unknown items (0.24 → 0.57).

## Consequences for earlier write-ups

1. **"Gemma, Qwen and Mistral are confabulators": withdrawn.** By the judge they hedge often on obscure unknowns (0.42-0.76); by the regex they rarely do. The measurement cannot tell which.
2. **The factorial verdicts stand as locked, but rest on an instrument that misses 86-90% of non-Llama hedges.** The Qwen and Mistral PARTIAL verdicts, and Gemma's floor switch to log-odds, were all driven by near-zero regex hedge rates. They should be presented as "not measurable with this detector", not as evidence about those models.
3. **Llama's headline survives on its regex (adequate recall), but its size is instrument-dependent:** +0.44 (regex) versus +0.13 [+0.03, +0.24] (judge-corrected). The direction holds under both.
4. **Entropy results are unaffected** (no detector involved), so "entropy follows knowledge in all four models" stands.

## What would settle it

Regenerate the free generations with a longer budget (e.g. 96 tokens, so answers are not cut off), and judge with a three-way scheme (ANSWER / DON'T-KNOW / ASK-FOR-CLARIFICATION). Use two different judge models and report their agreement, on a stratified sample per model. Write the criterion before running.
