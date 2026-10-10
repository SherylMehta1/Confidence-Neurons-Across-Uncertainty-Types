# E17: natural-language autoencoder readout of Qwen2.5-7B-Instruct (exploratory): plan, written before the run

**Status: exploratory.** E17 is not a test of the paper's claims. The round-3 camera-ready rule ("no new claims") means its result can appear only as future work or as an appendix note labelled exploratory. It responds to the gap the round-4 referees named: no model-side measure shows whether obscure names are represented as unfamiliar (framing F1, clarity F1, methods F16). Box 8.

## Tool

The Natural Language Autoencoder pair for Qwen2.5-7B-Instruct (Fraser-Taliente et al. 2026, Transformer Circuits; `kitft/nla-qwen2.5-7b-L20-av` and `-ar`, Apache-2.0). We follow the reference recipe in `kitft/nla-inference`:
- the sidecar `nla_meta.yaml` (injection token 149705 with neighbor checks, injection scale 150, the canonical AV prompt);
- the raw residual-stream output of block 20, which is `hidden_states[21]` in Hugging Face;
- L2-rescaling to 150 before injection;
- sampling at T = 1.0, up to 200 new tokens, keeping the text inside `<explanation>` tags.

Generation uses Hugging Face `generate(inputs_embeds=...)` instead of SGLang. The AV prompt is identical across vectors, so batching needs no padding.

## Design

- **Model and items:** Qwen2.5-7B-Instruct on its own two-judge 2×2 sample (`results/hedge_measure_v2/items_qwen.jsonl`, 354 questions; Qwen's own knowledge labels).
- **Prompt:** the free prompt (chat template with the question; `format_chat_prompt`).
- **Positions:** (P1) the last prompt token, where the reply begins; (P2) the last token of the subject name in the question. Items whose subject string cannot be located are skipped at P2 only.
- **Explanations:** 3 per activation.
- **Labels:** Llama-3.3-70B-Instruct (4-bit) answers three yes/no questions about each explanation:
  - A: does it say or imply that the entity or name is unfamiliar, obscure, unrecognised or ambiguous?
  - B: does it say or imply uncertainty, lack of knowledge or inability to answer?
  - C: does it name a specific candidate answer?
  - Plus a deterministic check: does the explanation contain a gold alias that is not part of the subject name?
- **Faithfulness:** the AR critic's cosine between each explanation's reconstruction and its activation (mean per position and cell), to show the explanations carry information about the vector.

## Analysis (descriptive)

For each label and position: per-question rates (mean over 3 explanations), popularity and knowledge effects, and popularity minus knowledge, with 95% within-cell item bootstraps (10,000, seed 0).

Readings, stated in advance but **not** licensing any claim in the main text:
- If A has a popularity effect with its interval excluding 0, and A's knowledge effect is smaller, then at that position Qwen's internal state verbalises the entity as unfamiliar in a way that tracks popularity more than knowledge. This is model-side evidence for familiarity.
- If B or C follows knowledge more than popularity, the readout separates the two factors as the behavioural data do.
- Otherwise the readout does not separate them.

## Not allowed after seeing data

Changing the positions, the number of explanations, the label questions, the judge or the sampling settings.
