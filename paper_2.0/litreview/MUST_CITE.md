# Must-cite table (litreview, 2026-10-10)

Status "verified" = the authoritative page (arXiv abstract page, ACL Anthology, DOI landing page via Crossref, or the Transformer Circuits page) was opened by `build_bib.py` in this session and its title and first author are recorded in `VERIFICATION.md`. Where a published version exists, OpenAlex supplied the venue record; see VERIFICATION.md.

Budget: a 4-page extended abstract at UniReps carries roughly 20-30 references (VENUE_FIT.md). Rows marked "optional" go in only if space allows.

| citekey | why it must appear | where in the paper | status |
|---|---|---|---|
| ferrando2025iknow | Closest prior work: entity-recognition directions steer refusal in Gemma/Llama; we cross recognition with model-certified knowledge | Intro, Related work | verified |
| lindsey2025biology | Closest prior work: the "can't answer" default path inhibited by "known entity" features in Claude 3.5 Haiku | Intro, Related work | verified |
| brzezinka2026familiarity | Closest prior work: graded familiarity probes; familiarity tracks popularity only weakly in Gemma/Qwen; familiarity vs abstention policy | Related work | verified |
| che2026steering | Concurrent (5 Oct 2026): steering an unknown-entity latent moves abstention, but installing natural values transfers only 12% of the contrast. Frames our C4 qualification | Result 2 (steering caveat), Related work | verified |
| nguyen2026commit | Concurrent: commit-abstain circuit across 10 models; we separate familiarity from knowledge and hedging from entropy | Related work | verified |
| kang2025unfamiliar | Why hedging could track familiarity: fine-tuning on unfamiliar examples shapes "I don't know" behaviour | Discussion (mechanism hypothesis) | verified |
| kalai2025hallucinate | Why models guess instead of abstaining | Intro (one clause), optional | verified |
| mallen2023popqa | Dataset and the popularity variable `s_pop` | Methods | verified |
| gekhman2024slick | Knowledge labels from the model's own sampled answers (SliCK categories) | Methods | verified |
| kadavath2022know | Models' self-knowledge (P(IK)); calibration framing | Intro | verified |
| stolfo2024confidence | Confidence-regulation neurons (entropy neurons); our C8 negative result | Intro / appendix | verified |
| gurnee2024universal | Universal neurons incl. entropy/frequency neurons; C8 | Appendix | verified |
| kuhn2023semantic | Entropy-based uncertainty (semantic entropy) as the competing readout | Methods (why first-token entropy), Related work | verified |
| ji2025hedging | Verbal uncertainty as a linear feature, distinct from semantic uncertainty: closest prior on "hedging ≠ entropy" | Related work | verified |
| arditi2024refusal | Difference-of-means directions and refusal steering (method origin) | Methods | verified |
| rimsky2024caa | Contrastive activation addition (steering method) | Methods | verified |
| gurnee2026workspace | Jacobian lens (method origin; prefitted lenses from neuronpedia/jacobian-lens) | Methods, Result 3 | verified |
| reder1992 | Human FOK follows cue familiarity, not answer retrievability | Intro, Discussion | verified |
| schwartz1992 | Cue familiarity but not target retrievability enhances FOK | Intro, Discussion | verified |
| koriat2001cascade | Cue familiarity + accessibility combine; the cascade prediction (R2) | Discussion (failed prediction) | verified |
| liu2007fok | Knowing-you-know vs knowing-you-don't-know dissociate (fMRI) | Discussion, optional | verified |
| huh2024platonic | UniReps framing: convergent representations across models | Intro / Discussion | verified |
| lippl2024source | Organiser's work: source-invariant uncertainty representations; our two signals are not one uncertainty variable | Discussion | verified |
| cheng2025abstraction | Organiser's work: mid-depth abstraction phase; where our familiarity signal peaks | Result 3 (one clause), optional | verified |
| teplica2025sciurus | Shared uncertainty circuits across 8 models (closest circuit work) | Related work / appendix (C9) | verified |
| liu2026broken | Asymmetric patching in refusal; context for C9 injection asymmetry | Appendix, optional | verified |
| llama3, qwen25, gemma2, mistral7b, olmo3 | Model reports for every model used | Methods | verified |
| wang2023ioi, syed2023attribution | Patching / attribution-patching origins for the C9 circuit | Appendix only | verified |

Dropped from the earlier 26-entry list as not needed in 4 pages: none removed from references.bib; LaTeX only prints what is cited.
