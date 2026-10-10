# Figure plan

Project: when Llama-3.1-8B-Instruct says "I don't know" to factual questions, does that follow how familiar the entity is or whether the model knows the fact, and is that the same signal as its answer uncertainty?
Venue and page budget: UniReps 2026 extended abstract, 4 pages of main text, NeurIPS template (text width 5.5 in); references and appendix unlimited. Checked 2026-10-07.
Last updated: 2026-10-10

## The pitch (Figure 1 in one sentence)

Saying "I don't know" follows how popular the entity is, while answer uncertainty follows whether the model knows the fact, and the second pattern holds in four model families.

## Story check

| Bullet | Carried by | Gap? |
|---|---|---|
| 1. Hedging tracks familiarity, answer entropy tracks knowledge (C1, C2) | Fig 1 | none |
| 2. Pushing activations along the obscure−popular direction produces "I don't know" about known facts, more than the unknown−known direction or random directions; mostly through hedge-associated components (C4, C5) | Fig 2 | none |
| 3. The familiarity signal is already in base models' lens-decoded readout, in three families (C6) | Fig 3 | none |

CLAIMS rows marked main without a figure: C3 (spelling/guessability, one sentence), C8 (neuron negative, one sentence). Prose is enough for both.

## Figures

| Id | One-sentence claim | CLAIMS.md ids | Main or appendix | Anchor panel | Status |
|---|---|---|---|---|---|
| F1 | Hedging follows popularity; answer entropy follows knowledge, in every family tested | C1, C2 | main | a (with b beside it) | built |
| F2 | Steering along the obscure−popular direction makes models say "I don't know" about facts they know, more than the unknown−known or random directions | C4, C5 | main | a | built |
| F3 | Base models already carry a mid-depth familiarity signal in the lens-decoded hedge readout | C6 | main | a | built |
| A1 | The hedge readout half-forms ~12 layers before the entropy gap (Llama) | C7 | appendix | — | exists: repo `paper/figures_singlebos/fig4.pdf` (single-BOS rerun); not rebuilt |

## Panels

| Figure | Panel | Role | Data source (result file + fields) | Plot type | Notes |
|---|---|---|---|---|---|
| F1 | a | evidence | repo `results/hedge_measure_v2/item_rates.csv`: model == llama, `prim` (both judges DONT_KNOW), `arm`, `slick` | interaction plot, 2 × 2 cell means with item-bootstrap CIs | free generation, 96 tokens, 3 samples per item; n = 387 items |
| F1 | b | comparison (the other readout) | repo `data/familiarity_v3/all_measured.jsonl`: `entropy`, `arm`, `slick` | same layout as a | prefill "The answer is"; n = 2,303 items |
| F1 | c | validation in a second regime (4 families) | repo `results/factorial_4models/effects.csv`: readout == entropy, `pop`, `pop_lo/hi`, `know`, `know_lo/hi` | dot plot with 95% CI | CIs from the published 10,000-item bootstrap |
| F2 | a | evidence + comparison against baselines | repo `results/direction_reuse_v2/{llama,gemma9b}/steer.csv`: `cond` ∈ {none, random0-4@k*, inst_know@k*, own_fam@k*}, `hedged`; `results/direction_nohedge/llama/steer.csv`: `cond` == d_nohedge_U | grouped bars + item-bootstrap CI; random directions as 5 dots | regex hedge detector (HEDGE_RE); 150 held-out popular HighlyKnown items |
| F2 | b | comparison (entropy channel) | same files, `entropy` | grouped bars | Gemma shows the double dissociation; Llama does not |
| F3 | a | evidence | repo `results/jlens_cascade/curves.csv`: `F`, `F_lo`, `F_hi`, `rel_depth`; `outcomes.csv` `midF` (tested statistic) | line + CI band; mid-depth window shaded | 4 base models, 3 families |
| F3 | b | comparison (knowledge in the same readout) | same, `K`, `K_lo`, `K_hi` | same | shown so the reader sees familiarity is not the only signal in the readout |

## Build record

All scripts are in `paper/figures/` and use `style.py`. Each writes `.pdf`, a `.png` preview and `.provenance.json` (input mtimes plus plotted values).

| Figure | Generator script | Output | Provenance file | Last run | Headline numbers printed |
|---|---|---|---|---|---|
| F1 | fig_dissociation.py | fig_dissociation.pdf | fig_dissociation.provenance.json | 2026-10-10 | a: obscure-unknown 0.477, obscure-known 0.433, popular-unknown 0.133, popular-known 0.000; b: 3.510 / 0.951 / 2.577 / 0.858; c: know vs pop = Llama 2.14 vs 0.51, Mistral 1.41 vs 0.30, Qwen 1.23 vs −0.02, Gemma 1.08 vs 0.09 |
| F2 | fig_steering.py | fig_steering.pdf | fig_steering.provenance.json | 2026-10-10 | Llama hedge: none 0.020, random 0.025, know 0.087, fam 0.287, never-hedged fit 0.067; Gemma: 0 / 0 / 0 / 0.220; entropy Llama 0.82 / 0.90 / 1.23 / 1.26; Gemma 0.25 / 0.53 / 1.50 / 0.82 |
| F3 | fig_lens_base.py | fig_lens_base.pdf | fig_lens_base.provenance.json | 2026-10-10 | mid-window familiarity Llama +0.67, Gemma-9B +0.39, Gemma-2B +0.40, OLMo +0.29 (= tested midF); knowledge +0.36 / +0.37 / +0.11 / −0.30 |

## Captions

Numbers in captions are macros from `numbers.tex` (invariant I1).

- **F1: Saying "I don't know" follows popularity; answer uncertainty follows knowledge.**
  - (a) Llama-3.1-8B-Instruct, fraction of free-generation replies that both LLM judges label "I don't know", by entity popularity (obscure = bottom decile, popular = top decile of PopQA page views) and by whether the model knows the fact (from 10 own samples). Values: \hmLlamaObsUnk, \hmLlamaObsKnown, \hmLlamaPopUnk, \hmLlamaPopKnown.
  - (b) The same model's first-token entropy after "The answer is" on the same design.
  - (c) Popularity and knowledge effects on that entropy in four families. Knowledge dominates in every family.
  - Points are cell means with 95% item-bootstrap intervals; (a) n = \hmLlamaItems items × 3 samples, (b) n = \faLlamaN items, (c) \faGemmaN-\faQwenN items per model (\faLlamaN, \faQwenN, \faGemmaN, \faMistralN).
  - The two readouts use different prompts (Methods).
- **F2: Pushing along the obscure−popular direction makes models say "I don't know" about facts they know.**
  - (a) Hedge rate (regex detector) on 150 held-out popular questions each model answers correctly. Conditions: no steering, 5 random directions (dots: individual directions), the unknown−known direction, and the obscure−popular direction, all at equal norm (Llama 1×, Gemma 4× the natural norm, layer ℓ*).
  - The hatched bar is an obscure−popular direction fitted only on items the model never hedged on. It moves hedging by only \nhOverRand over random, so the effect runs mostly through hedge-associated components.
  - (b) First-token entropy under the same conditions. In Gemma the unknown−known direction raises entropy more than the obscure−popular direction does (\stGemmaDE \stGemmaDECI); in Llama both raise it about equally (\stLlamaDE).
  - Error bars: 95% bootstrap over items.
- **F3: Base models already carry a familiarity signal in their decoded "I don't know" readout.**
  - Lens-decoded hedge-vs-answer log-odds at the last prompt position, by relative depth, for four base models with official prefitted Jacobian lenses.
  - (a) Obscure minus popular, averaged over knowledge. (b) Unknown minus known, averaged over popularity.
  - Bands: 95% item-bootstrap intervals. Shaded: the mid-depth window fixed before the runs, where the familiarity effect is positive in all four models (\jlLlamaBaseMidF, \jlGemmaNineBaseMidF, \jlGemmaTwoBaseMidF, \jlOlmoBaseMidF; all CIs exclude 0).
  - Lenses are fitted per model, so magnitudes are not comparable across models.

## Tables

| Id | One message | Source file or generated fragment | Direction arrows | Decimals | Status |
|---|---|---|---|---|---|
| T1 (appendix) | The hedging comparison keeps its sign under four instruments (regex 24 tok, one judge 24 tok, two judges 96 tok, regex 96 tok) | numbers.tex macros \faLlamaRegexDiff, \hjLlamaDiff, \hmLlamaDiff, \hmLlamaRegexDiff | none | 2 | planned (write mode, generated from macros) |
| T2 (appendix) | Judge agreement and hedge rates per family (why hedging is measured only in Llama) | numbers.tex \hm*Kappa, \hm*ObsUnk | none | 2 | planned |

## Caption-versus-data check (2026-10-10)

| Figure | Numbers match | Directions match | Scope words match | n, interval, units match | Result |
|---|---|---|---|---|---|
| F1 | yes: plotted cells = summary.txt primary cells (0.477/0.433/0.133/0.000) and cells.csv entropy (3.510/0.951/2.577/0.858); effects = effects.csv | yes: hedging falls with popularity in both knowledge rows; entropy rises with ignorance in both popularity columns | "every family": knowledge > popularity in 4 of 4 (smallest ratio 4.2×, Llama) | yes; (a) cells n = 87/100/100/100 = 387 | pass |
| F2 | yes: = summary.txt S and E lines; never-hedged fit 0.067 = direction_nohedge summary | yes | "models" = 2 plotted; caption names both | 150 items per condition; Gemma uses `@k*` rows (the plain own_fam rows average three doses) | pass |
| F3 | yes: mid-window means = outcomes.csv midF to 2 dp | yes: all four positive in window | "all four", 4 plotted | CI bands from analysis script | pass |

## Style decisions

- Palette: Okabe-Ito. Orange = familiarity/popularity and the obscure−popular direction; bluish green = knowledge and the unknown−known direction; blue/vermillion = known/unknown items; grey = random; purple = never-hedged control. Each colour also differs in marker or line style.
- Column width and font size: 5.5 in (full NeurIPS text width), 9 pt labels, 8-8.5 pt ticks, constrained layout, saved at exact size (no rescaling in LaTeX: `\includegraphics[width=\textwidth]`).
- Shared style module: `paper/figures/style.py`.
- Palette check run with `dataviz`: not installed; Okabe-Ito rules from references/figures.md applied.

## Deferred to the appendix

- A1 (C7 timing) and the single-BOS replacements of old Figs 4/5. Reason: secondary to the three-bullet story; space.
- Instruct-model lens curves (C10, exploratory, lenses not comparable across models).
- The C9 circuit (equivocal behaviourally; "sparse abstention circuit" now owned by Nguyen et al. 2026).

## Open items

- Figure placement: three full-width figures take about 1.5 of the 4 pages. If space runs short, F3 can shrink to panel (a) alone at half width (wrapfigure). Decide at the outline checkpoint.
