# AAAI draft v2: number audit, claim fixes, limitations (25 Nov 2026)

`main.tex`, `main.bbl`, `references.bib`, `aaai2027.*` and `figures/` are the Overleaf source as supplied (IDK_Popularity_AAAI27_source.zip), unchanged. `main_v2.tex` is a copy with only the edits in `text_changes.csv`: 15 of 272 lines differ, the preamble is identical, and the only LaTeX commands added are two `\ref` inside Limitations.

## Numbers (`number_audit.csv`)

402 rows: 126 exact, 275 rounding only, 1 mismatch, 0 without a source.
- `exact`: equal to the source as stored. `rounding_only`: equal to the full-precision source at the paper's precision, or differing from another listed source by bootstrap draws only. The 10 changed interval ends (N1 to N5) are in this group.
- Not listed as numbers: LaTeX layout values, model names (for example 3.1-8B) and citation years. Figures carry axis labels only; they are the corrected X0, X2 and X3 plots (checked by eye).
- Two settings have no row in the five listed sources and were traced through them: 100 questions per cell (`hedge_measure_v2_criterion.md`, named by `kappaBar` in `numbers_v2.tex`) and the steering layer 0.55(L - 1) (`paper_2.0/PREREG/e4_steering_v3.md`).

Where two listed sources differ by draws or rounding, `results/x0_corrected/` wins (decided by the author, 25 Nov). This changes five quantities, each by 0.01. The SRW, MERe and UniReps drafts still carry the other values.

| Flagged item | Paper said | Files | v2 |
|---|---|---|---|
| 1 kappa interval | [0.89, 0.94] | `lockin_checks` 2,000 draws [0.885, 0.935], rounded up from 0.885; `A1_intervals.csv` 10,000 draws [0.8849, 0.9356] | [0.88, 0.94] |
| 2 label-corrected gap | [+0.20, +0.51] | `round4_checks` [0.196, 0.506]; `A5_unknown_gap.csv` [0.1906, 0.5036]; both 10,000 draws, different random streams | [+0.19, +0.50], 3 places |
| 3 OLMo margin | [-0.21, +0.04] | `e15_summary` prints -0.215; both -0.21 and -0.22 are roundings of it; no finer value is stored | unchanged |
| 4 steering | +0.14 and +0.16 | E16: popular-arm abstention vector and popular-to-obscure vector against 20 random vectors, majority of three judges. The 0.10 bar met at +0.17 [+0.12, +0.22] is E4's D1: the popular-to-obscure vector against 5 random vectors, two judges. Different quantities, consistent | unchanged |

Also changed: N3 margin outside three relations, upper end +0.27; N4 regex margin, lower end +0.30; N5 one judge corrected, lower end +0.02 (10,000 draws; the 5,000-draw value is +0.03); N6 "four have fewer than 300" to "three" (`sample_summary.csv`: capital of 255, color 34, mother 187), after asking. `results/x2/x2_plan.md`, `results/x2/README.md` and the SRW v2 and v3 drafts still say four.

**Draws (A6).** Cell and margin intervals: 10,000. Whole-relation resampling, the X2 and X3 fits, and the within-relation table: 5,000. Lens table: 2,000. After N1 the kappa interval uses 10,000. No quantity appears with two different draw counts in the paper.

## Text (`text_changes.csv`)

B1 to B6 as specified, with "favors" (the paper uses American spelling). B7a: in the abstract, "popularity matters more for country and less for the two smaller types" now says the point estimate favors one or the other, because the sport interval includes zero. B7b: "further" removed from "4,376 further questions". B7 search: 24 hits before, 16 left: 6 are `\label`/`\ref` names, 2 "validated" (limitations), 2 "language model" (not findings), 1 title, 1 question in the Introduction, 1 contribution title, 3 "cannot settle" statements.
C1 to C5 added at the end of Limitations. C6: no "across models" or "across datasets" without names. C7: stated limitations match the numbers.
Two C sentences are narrower than the task's wording, to stay accurate. C1: the steering results in this paper are on Llama, so it names the unreported Gemma-2 steering runs. C3: in git, 15 of 17 earlier plans were committed after their results (`results/x0_corrected/prereg_table.csv`), so it claims commit times only for the whole-range checks.

## Not done, for the author

- **PDF not built**: no LaTeX, `pdftotext` or `pdffonts` on this machine, and `anthrologos_aaai_sub.pdf` was not supplied. Page count (6) and fonts are unchecked. The text is about 180 words longer, nearly all in Limitations; if it runs over, shorten C1 to C5, not the layout.
- Build: upload this folder to Overleaf, set `main_v2.tex` as the main file, compile with pdfLaTeX. Compare with `main.tex`.
- "a model knows" in the Conclusion ("obscure facts a model knows are rare") generalizes from Llama; left as is.
- References (not edited): every citation resolves and none is unused. To check: Ferrando et al. and Kuhn et al. are cited as arXiv although published versions may exist; Kadavath, Gurnee and Lindsey use "and others".
- Title, "earlier version" wording, code link and appendix tables untouched, as instructed.
