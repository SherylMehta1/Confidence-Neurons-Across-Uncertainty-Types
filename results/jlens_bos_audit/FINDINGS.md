# Doubled-BOS audit of the committed Jacobian-lens results (6 Oct 2026)

**Script:** `scripts/jlens_bos_audit.py`. Llama-3.1-8B-Instruct, the `familiarity` category (all 195 pairs used by the paper's Figure 4), the prefitted lens `llama3.1-8b-it/…/Llama-3.1-8B-Instruct_jacobian_lens.pt`. Run on molab. An earlier 60-pair run on a sandbox that was later terminated gave the same qualitative answer.

## The bug

`jlens` tokenizes every prompt itself and forces a BOS token. `scripts/jlens_trajectory.py` (and the other `jlens_*` scripts) pass `chat_formatted_prompt`, which already begins with the text `<|begin_of_text|>`. The Llama tokenizer then produces `[128000, 128000, …]`, so the committed lens analyses ran on a doubled BOS. Every other analysis in the paper (patching, circuits, log-odds) uses `tokenize_prompt`, which gives exactly one BOS.

- **Confirmed:** the doubled-BOS condition reproduces the committed `results/circuit_familiarity/jlens_summary.txt` to two decimals at every layer.
- **Effect on the model's own output:** final logits differ from the single-BOS forward pass by a median of 3.5 logits (max 11.8). With the BOS text stripped: median 0.03, max 0.13 (bf16 noise).

## Effect on the paper's claims

| Quantity (paper, Figure 4) | Doubled BOS (committed) | Single BOS (correct) |
|---|---|---|
| First layer where the lens hedge log-odds gap reaches 50% of final | **L16** | **L16** |
| First layer where the lens entropy gap reaches 50% of final | **L28** | **L28** |
| Final lens gaps (log-odds, entropy) | +5.05, +3.04 | +5.31, +3.31 |
| Model-output twin gaps (log-odds, entropy) | +4.88, +2.44 | +5.63, +2.76 |

**The conclusion survives unchanged:** the hedge decision is half-formed by L16, about 12 layers before the entropy gap. The curves move only slightly.

## What to do before submission

1. Strip the leading BOS text before calling `lens.apply` in `jlens_trajectory.py`, `jlens_positions.py` and `jlens_steer.py`. `scripts/jlens_cascade.py::strip_bos` shows how.
2. Regenerate Figures 4 and 5 from single-BOS runs. Figure 5 (the rank of " unknown") and the steering panel were not audited here; they use the same code path and should be rerun, not assumed unchanged.
3. Or, if there is no GPU time: keep the figures, and add one sentence plus this table to the appendix stating the doubled-BOS input and that the reported Llama-familiarity layers (L16, L28) are identical under the corrected input. The Qwen-contested curve in Figure 4 ("by L21") was not audited.


*Corrected 7 Oct 2026 after an independent audit of every number in this file: wording and figures fixed where the audit found them wrong or overstated.*
