# Hedging measured with full-length replies and two judges (7 Oct 2026)

- **Governed by:** `hedge_measure_v2_criterion.md` (sha256 `81acfc9f`, written before anything ran; one clarification about keeping full reply text was added before anything ran, then re-hashed).
- **Scripts:** `scripts/hedge_measure_v2.py` (gen, judge), `scripts/hedge_measure_v2_analysis.py`.
- **Files:** `items_*.jsonl`, `gens_*.jsonl`, `judge_{qwen32b,gemma27b}_*.jsonl`, `summary.txt`, `item_rates.csv`.
- **Setup:** per model, up to 100 items per popularity × knowledge cell. 3 samples, T = 0.7, 96 tokens, full text. Two judges (Qwen2.5-32B-Instruct, Gemma-2-27B-it), three labels. **Primary measure:** both judges say DONT_KNOW.

## Verdicts under the criterion

| Model | κ (DONT_KNOW; bar 0.60) | Obscure & unknown: both-judge DONT_KNOW / regex / both-judge DONT_KNOW or CLARIFY | Popularity effect | Knowledge effect | Pop − know [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | **0.62** | 0.48 / 0.85 / 0.65 | +0.388 | +0.089 | **+0.300 [+0.204, +0.398]** | **FAMILIARITY > KNOWLEDGE** |
| Qwen2.5-7B-Instruct | 0.30 | 0.05 / 0.18 / 0.32 | +0.018 | +0.032 | −0.013 [−0.037, 0.000] | UNMEASURED |
| Gemma-2-9B-it | 0.47 | 0.05 / 0.07 / 0.22 | +0.025 | +0.022 | +0.003 [−0.020, +0.030] | UNMEASURED |
| Mistral-7B-Instruct-v0.3 | 0.39 | 0.10 / 0.21 / 0.25 | +0.047 | +0.050 | −0.003 [−0.010, 0.000] | UNMEASURED |

**What the criterion's advance decision means:**
- This measurement **supersedes the regex** for the hedging comparison wherever κ ≥ 0.60.
- **Llama:** saying "I don't know" follows how familiar the entity is more than whether the model knows the fact, by +0.30 [+0.20, +0.40] on the primary measure. The secondary measure (adding clarification requests, +0.31) and the regex on the same full replies (+0.43) agree in sign.
- **Qwen, Gemma, Mistral:** the hedging comparison is reported as **unmeasured**. Their earlier regex-based verdicts are reported with their known recall.

## Descriptive findings (not criterion claims)

- **With full-length replies, Qwen, Gemma and Mistral rarely say "I don't know" outright, even about obscure facts they don't know.** Both judges agree on DONT_KNOW in 5-10% of such samples. Adding clarification requests ("which one do you mean?") raises it to 22-32%.
  - The first judge audit's 42-76% was inflated by the 24-token truncation.
  - So these models do mostly commit to an answer, but the earlier "confabulator" wording is still not licensed, and the right description is "rarely says it does not know, sometimes asks for clarification".
- **Low judge agreement for these three models partly reflects the rarity of DONT_KNOW** (κ is unstable at low base rates) and partly real ambiguity between DONT_KNOW and CLARIFY.
- **Regex recall improves sharply with full replies:** Llama 0.99, Mistral 0.73, Qwen 0.58, Gemma 0.48 (versus 0.10-0.14 on the 24-token replies). The earlier recall failure was mostly truncation, not phrasing.
- **Llama's effect size depends on the instrument:**
  - regex, 24 tokens: +0.44;
  - first judge, 24 tokens: +0.13;
  - two judges, 96 tokens: **+0.30**;
  - regex, 96 tokens: +0.43.

  All are positive with CIs excluding 0. The two-judge 96-token number is the one the criterion designates.
