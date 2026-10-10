# E5: a stronger third judge for abstention in all four families: criterion, written before the run

Written 2026-10-10, before any E5 run, in response to review rounds 1-2: abstention is measured in one model only, because the two judges fail κ ≥ 0.60 for Qwen, Gemma and Mistral.

Box C.

## Judge

`unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (fallback if it fails to load on this GPU: Qwen/Qwen2.5-72B-Instruct loaded in 4-bit NF4 with bitsandbytes). Greedy decoding, the SAME prompt as the two existing judges (`scripts/hedge_measure_v2.py`), labels DONT_KNOW / CLARIFY / ANSWER.

## Replies

All existing 96-token replies in `results/hedge_measure_v2/gens_{llama,qwen,gemma,mistral}.jsonl`, unchanged.

## Rules

- **Primary (new):** a reply abstains if at least 2 of the 3 judges say DONT_KNOW.
- **Reliability:** Fleiss' κ across the three judges on DONT_KNOW vs not, per family, plus pairwise Cohen's κ.
- **A family's abstention counts as MEASURED if Fleiss κ ≥ 0.60.** This is the same bar as before, applied to three judges.

## Tests per family (only where MEASURED)

Popularity − knowledge on the majority-rule abstention rate, with 95% CIs from 10,000 within-cell item bootstraps (seed 0).
- **FOLLOWS POPULARITY** if > 0 with the CI excluding 0.
- **FOLLOWS KNOWLEDGE** if < 0 with the CI excluding 0.
- Otherwise **NEITHER**.

Llama is re-reported under the majority rule as a robustness check. Its primary result stays the two-judge rule fixed on 7 Oct.

## Licensing

- "Abstention follows popularity in K families" requires MEASURED and FOLLOWS POPULARITY in each of those K families.
- A family below the κ bar stays "unmeasured".
- **The title stays scoped to Llama unless at least two other families are MEASURED and FOLLOW POPULARITY.**

## Not allowed after seeing data

Changing the judge (except the stated load fallback), the prompt, the majority rule, the κ bar or the item set.
