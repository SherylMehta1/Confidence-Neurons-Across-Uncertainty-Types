# Gemma judge: re-judging of all stored replies, and the Gemma-2-9B check (24 Nov 2026, branch arr-v2)

## 1. Re-judging

- 49,521 stored replies (18 judge files) re-labelled by Gemma-2-27B with correct attention (vLLM 0.34.0; equal to transformers `eager` on all 1,161 replies where both were run). Same prompt, parser, greedy decoding. Old files are untouched.
- Prompt reconstruction check: Qwen re-run on 300 random replies per file gives its stored DONT_KNOW label on 99.9% of 5,400.
- Unparsed labels: 0. kappa with Qwen rises in every usable file (`labels_by_file.csv`), e.g. Llama 0.62 to 0.91, Llama permissive 0.57 to 0.97, OLMo permissive 0.33 to 0.82. The largest label changes are on the permissive prompt (Qwen replies 40%, OLMo replies 37%).
- `corrected/*.new`: regenerated summaries, analyses and `numbers.tex` (suffix `.new` so nothing mistakes them for the committed files). `scripts/rejudge_shadow.py` with the stored labels reproduces 37 of 39 committed outputs exactly; the other two differ only in the order of tied counts.

### Verdicts that change under the pre-set rules (`verdicts.csv`)

| Where | Stored | Corrected |
|---|---|---|
| Permissive prompt, Qwen (E12) | unmeasured (Fleiss 0.40) | measured (0.74), follows popularity, +0.16 [+0.01, +0.31] |
| Permissive prompt, Mistral (E12) | unmeasured (0.50) | measured (0.80), follows popularity, +0.17 [+0.06, +0.28] |
| Permissive prompt, OLMo-2 (E15) | unmeasured (0.46) | measured (0.86), neither, -0.09 [-0.21, +0.04] |
| Plain prompt, Mistral (E5, two judges) | unmeasured (0.54) | measured (0.75), neither |
| Plain prompt, Gemma (two judges) | unmeasured (0.47) | measured (0.95), no difference resolved |
| Plain prompt, Qwen (E5) | unmeasured (0.41) | still unmeasured (0.59, bar 0.60) |
| Steering, Llama, D1 (E4) | not met, +0.09 | met, +0.17 [+0.12, +0.22] |

Verdicts that do not change: E1, E7, E11, E13 and E14 (sizes grow: E13 +0.29 to +0.45, E14 +0.10 to +0.15; E14's relation-clustered interval still includes zero), E8 and E16 (generic abstention), E9 Llama and Gemma, E6.
E4 Gemma: obscure-minus-popular against unknown-minus-known (D3) goes from +0.00 to -0.22 [-0.29, -0.15]; unknown-minus-known now raises judged abstention (0.04 to 0.30).

## 2. Gemma-2-9B as a tested model

- Yes, it ran with the same default `sdpa` attention: stored first-token entropy and greedy answers equal an `sdpa` re-run exactly (300 of 300 pool questions).
- The effect is negligible for the 9B model. `eager` against stored: greedy text identical on 94%, correctness on 300 of 300, mean entropy 1.018 against 1.021 (correlation 0.999), cell means equal within 0.01.
- Knowledge labels of the paper's 400 questions, relabelled with correct attention and fresh samples: 379 unchanged, 21 move to the dropped band, none flips (`gemma9b_labels.csv`). Fresh samples alone would also move some.
- Abstention (regex on 96-token replies), stored against correct: permissive 0.58 / 0.42 / 0.13 / 0.00 against 0.58 / 0.39 / 0.14 / 0.00 (OU / OK / PU / PK); plain prompt all cells below 0.10 either way.
- So the Gemma-2-9B claims do not depend on this fault, as far as checked. Not checked: the lens runs (base and instruct Gemma-2 at 9B and 2B) and the steering generations, which used the same loading code. The affected numbers for Gemma-2-9B came from the 27B judge, and are corrected above.
- Any new Gemma-2 run must use `eager` or vLLM.

## 3. What this means for the papers (nothing edited)

`macro_changes.csv`: 182 of 652 number macros change. Changed and used: 41 in `srw/main.tex`, 100 in `mere/main_v2.tex`, 150 in `main.tex`. Regenerating `numbers.tex` would fix the numbers but not these sentences:

| Draft | Line | Wording that no longer holds |
|---|---|---|
| srw | 24, 61 | abstention on popular known facts is "zero" (one of 300, 0.003) |
| srw | 47 | kappa above the bar "at the point estimate but not across the whole interval" (now [0.89, 0.94]) |
| srw | 162 (caption) | two-judge measure described against a one-judge +0.13; margin range changes |
| srw | 169 | judges disagree on Qwen, Mistral and OLMo under the permissive prompt, so margins not reported: now measured (+0.16, +0.17, -0.09), and OLMo's does not follow popularity |
| srw | 88 and within-relation text | outside the three relations +0.00 becomes -0.05 (two judges) |
| mere_v2 | 16, 36, 141, 143 | "judges disagree too much to measure" three families; "cannot measure Qwen, Mistral or OLMo-2" |
| mere_v2 | 155, 160 | Gemma judges below the bar; steering wording |
| main | 28, 44, 101, 103, 216 | same family statements |
| main | 69 | popular known "never" says it |
| main | 108, 226, 243 | steering "short of the pre-specified +0.10 bar" (now met); Gemma judges below the bar |
| all | appendix | "why the judge and the regex disagree" paragraphs: with correct labels they agree |

Figures 1a (dissociation) and steering also use the old labels. Any new sentence must report OLMo alongside Qwen and Mistral.

## Files

`todo.jsonl.gz` (replies judged), `rejudge_gemma27b.jsonl` (labels; 2,400 extra rows are the fresh Gemma-2-9B replies), `rejudge_qwen32b.jsonl` (check), `labels_by_file.csv`, `qwen_check.csv`, `verdicts.csv`, `macro_changes.csv`, `corrected/`, `gemma9b_*`. Scripts: `scripts/rejudge_*.py`, `scripts/gemma9b_*.py`.

## 4. New paper versions (existing drafts unchanged)

`scripts/make_papers_v2.py` writes new files only: `paper_2.0/srw/main_v2.tex`, `paper_2.0/mere/main_v3.tex` (MERe already had a v2), `paper_2.0/main_v2.tex`, `paper_2.0/numbers_v2.tex`, `paper_2.0/numbers_x2.tex`; `scripts/make_figs_v2.py` writes `paper_2.0/figures/fig_{dissociation,steering}_v2.*`. All 55 edits with old and new text: `paper_v2_changes.csv`. X2 is added to the SRW version only. Not compiled (no LaTeX here); page limits not checked. Titles and section 4's heading of SRW are unchanged.
