# Revise round 1: response to the simulated review (2026-10-10)

Source of comments: review/round1/ (three blind referees, chair merge I1-I57, red-team P1-P5, numbers audit N1-N3 and others). These are internal simulated reviews, not venue reviews. Pre-revision manuscript: revision/round1/main_before.tex (= draft_v2.pdf). Revised: tex/main.tex (= draft_v3.pdf).

User decisions (Checkpoint 3, 2026-10-10):
- title (b), scoped to Llama;
- circularity handled by rewording (a);
- defaults for unanswered items: the cross-model result goes in as exploratory, the GPU steering controls are conceded, no human audit.

New evidence (post hoc, labelled so in the paper):
- guessability control, analysis/guessability.py;
- cross-model agreement, analysis/cross_model_agreement.py with plan PREREG/cross_model_agreement.md.

All numbers come through numbers.tex (252 macros; make_numbers.py). Two new references were verified on arXiv: kumaran2026confidence, ni2025popular.

## Comment table and traceability matrix

Actions: fix / push back with evidence / concede as limitation / defer.

| id | concern (short) | action | change made (where) | number source | check re-run | status |
|---|---|---|---|---|---|---|
| I1 | One-model abstention result stated generally | fix | Title "In Llama, …"; abstract last sentence "Where abstention could be measured"; §1 (i); conclusion "In Llama" | — | compile, prose gate | done |
| I2 | Prompt-confound defence gave Gemma only | fix | §2 two-prompts paragraph now gives Llama's forced-answer effects and says its divergence appears when it may decline; App. B table with all four families and CIs | \loLlamaPop/\loLlamaKnow, \lo*Diff(CI) | check_numbers | done |
| I6 / P2 / P3 | Obscure "known" facts may be guessable | fix (new analysis, post hoc) | §3.1 checks paragraph (\gsObsKnownGuess of \gsObsKnownTotal guessable; difference \gsDiff \gsDiffCI without them; obscure-known still \gsObsKnownRate); App. A definition, unknown-only gap, regex pool result | analysis/guessability.csv | check_numbers | done |
| I10 / P1 | Entropy and knowledge labels share a distribution | fix by rewording ([user] 2a) | §2: "The knowledge labels are sampled from this same distribution, so entropy is expected to follow knowledge; we use it as the reference readout"; §3.1 back-reference; Limitations | — | — | done |
| I11 | Alias-grading noise, no human audit | concede | Limitations: "neither the judges nor the alias grading were validated against human labels" | — | — | conceded |
| I7 | Entropy also tracks popularity | fix (partial) | §3.1 keeps "at least four times"; Ni et al. cited in Related work. The within-cell interaction is not added (space) | — | — | partly done |
| I3 / I13-chair | "Rules out name spelling" too strong | fix | §3.1 now reports the coefficient change only; App. A gives the matched-pair attenuation | \scPoolPop, \scPoolPopSurface | — | done |
| I12 | Popularity vs familiarity | fix | §2 "page-view popularity is our proxy for how familiar a subject is"; abstract "a proxy for familiarity"; Fig. 3a axis renamed "popularity effect" | — | figure rebuilt | done |
| I8 / I35 | Abstract and Fig. 2 oversell steering; control Llama-only | fix | Abstract gives the +0.04 control; Fig. 2 title "mostly via components shared with abstention"; caption "hatched (Llama only)"; §3.2 "In Llama, a version fitted…" | \nhOverRand | — | done |
| I4 / I6-fr | Base-model lens readout carries knowledge too | fix | §3.3 retitled "Base models' hedge readout already responds to popularity"; mid-window knowledge effects reported; abstract "alongside a knowledge effect"; §1 (iii) | \jl*BaseMidK | check_numbers | done |
| I5 | OLMo late reversal undisclosed | fix | Fig. 3 caption | — | — | done |
| I13 | Lindsey's misfire hypothesis not credited | fix | §1 P2 "\citet{lindsey2025biology} propose that a familiar name can trigger them without the fact" | — | — | done |
| I14 | Contributions (ii)/(iii) not positioned against Ferrando/Brzezinka | fix | §1 (ii) "As in earlier steering studies … new here, …"; Related work | — | — | done |
| I15 | Kumaran et al. 2026 not cited | fix | §1 P1 and Related work with a reconciliation (offered vs uninstructed abstention; multiple choice vs free-form) | — | bib verified | done |
| N1 / I21 | "Steering and lens use one prompt" false | fix | Sentence removed; §2 now says the prompt change matters; Fig. 2b caption "under the forced-answer prompt" | — | — | done |
| N2 / I22 / I28 | "Emerge together" overclaims; eight models and rule unnamed; App. C 12-layer lead | fix | §3.3 failed-prediction sentence without "emerge together"; App. D onset rule and per-model table (generated); App. F says the 12-layer comparison uses a different contrast | \jlOnsetRows | — | done |
| N3 | "Rarely abstain" measure-dependent | fix | §2 now justifies "Llama only" by κ alone; App. B gives all three measures and the at-most value | \hmMistralObsUnk | — | done |
| I9 | Dose chosen on evaluation questions | concede + disclose | App. C states the rule; Limitations | — | — | conceded |
| I16 / I17 / I18 / I38-40 | Projection-out steering, a control not conditioned on abstention, steered-text quality, more nulls | concede (no GPU before the deadline) | Limitations: "other norms and a direction with abstention projected out were not tested" | — | — | conceded |
| I19 | Lens readout may reflect answer-token rarity | concede | Limitations | — | — | conceded |
| I20 | Lens undefined | fix | §3.3 first sentence defines it | — | — | done |
| I23 | No cross-model analysis | fix (new analysis, exploratory) | Conclusion clause + App. E table; plan written before running; rule met for 1 of 3 pairs, stated | analysis/cross_model_agreement.* | check_numbers | done |
| I33 / D3 | Lippl misapplied | fix | Related work: "By the source-invariance criterion of Lippl et al., the abstention variable is a correlate of uncertainty rather than a representation of it" | — | — | done |
| I46 / Fr-F8 | Straw "implicit account" | fix | §1 P1 now cites Ji et al. for the divergence and asks what each follows | — | — | done |
| I24 / Fr-F11 | "Double dissociation" at 4x norm | fix | §3.2 reworded; App. C "dissociation of steering effects at four times the natural norm" | — | — | done |
| I25 / I26 / I29 | Measurement counts; "corrected" undefined; no row intervals | fix | §3.1 "four other measurements"; Table 1 caption "five measurements", "corrected" defined, CIs on every row; App. A explains the attenuated corrected row | \*DiffCI | — | done |
| I27 / I17-fr | Hashes not given | fix | App. G lists SHA-256 prefixes | \hash* | — | done |
| I52 / I56 | Table 2 Llama dashes; Fig. 1 legend overlap, truncated label | fix | Table 2 filled; Fig. 1c markers and labels moved; panel b label shortened | — | figures rebuilt | done |
| Cl-F27 | "Brzezinka distinguish" | fix | "separates" | — | — | done |
| Audit: scope words | "7-9B", "every layer", "full pool" | fix | "2-9B"; "every layer" removed; App. A states the 2,303 labelled questions | \faLlamaN | — | done |
| Audit: d_nohedge naming | which never-abstained direction | fix (partial) | App. C defines it (fitted on working-half questions with no abstaining sample) | — | — | done |
| remaining minor items (I30 cell-size table, I31/I36 regex precision, I32 ranking, I41/I43 intervals on ratios, I44 base labels, others) | | partly fixed | I32: ranking removed; I44: App. D states base labels; regex recall reported in App. B. Not done: a cell-size table, intervals on the "four times" ratio | — | — | partly done |

Read-through: every row has an action and a status. Rows marked "conceded" appear in Limitations.

## Checks after revision

- LaTeX: 0 errors, 0 undefined references or citations, 0 overfull boxes. Main text is exactly 4 pages; references start on page 5.
- check_numbers: 156 macros used; 25 literals, all design settings; 0 unsupported.
- verify_citations (offline): 35 entries, no mismatch.
- prose_gate: 2 advisory findings, both false positives ("may decline" is not a hedge; ANSWER is a quoted label).

## What a re-review should look at

- whether the Setup's circularity statement and the Llama forced-answer numbers read as honest rather than defensive;
- whether "exploratory checks find this reliably for two of three pairs, though our pre-set rule held for one" is clear;
- whether the steering section's wording matches what the never-abstained control allows.
