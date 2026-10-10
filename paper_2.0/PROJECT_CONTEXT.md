# Project context

<!-- Memory of the project for the research-paper-pilot skill. Every mode reads it first and updates it.
     [unverified] = not opened; [user] = the user said so. Paths: repo = C:/Users/starb/Desktop/Sher/Confidence-Neurons-Across-Uncertainty-Types ;
     paper dir = C:/Users/starb/Desktop/Sher/paper (this folder, outside the repo by the user's choice, 7 Oct 2026). -->

**Status line:** 2026-10-10 | camera-ready after round-3 ACCEPT; E1-E10 folded in, humanize verified (PASS) | main.pdf: 4 pages, checks clean; C1 strengthened, C12 (Gemma, permissive prompt) and C14 (E8) added, R6-R10 retired (CLAIMS.md); collaborator guide WHAT_CHANGED.md; shared as paper_2.0/ on branch emo_attyachar_2.0 | open: author list, title scope, LLM disclosure, supplement upload; next `submit` (deadline 11 Oct 12:00 UTC)

## Identity

We study when instruction-tuned language models (Llama-3.1-8B-Instruct, plus Qwen2.5-7B, Gemma-2-9B and Mistral-7B) say "I don't know" to factual questions. The question is whether that decision tracks how familiar the question's subject is or whether the model actually knows the answer. We use the model's own sampled answers as ground truth, a popularity × knowledge factorial, causal steering, and Jacobian-lens decoding.

## Question

Work on LLM confidence usually treats "I don't know" as uncertainty made audible: one neuron, head or direction leaking output entropy into words. The project began by looking for such "confidence neurons" (Stolfo et al. 2024) across three kinds of uncertainty. It found that entropy-correlated neurons are word-frequency and temperature machinery, not uncertainty detectors, and that a small circuit (20 heads + 100 neurons) carries the hedge readout.

Since 5 Oct the question for the UniReps paper is sharper. When the model's knowledge is held fixed, does it hedge on unfamiliar names? When familiarity is held fixed, does it hedge on facts it does not know? And is that split shared across model families and present before instruction tuning? This mirrors the human feeling-of-knowing literature, where "can I answer?" judgments follow cue familiarity rather than retrievability (Reder & Ritter 1992; Schwartz & Metcalfe 1992). It matters because confidence methods that read the output distribution would then measure a different variable from the one that decides the hedge.

## Hypotheses

| | Wording | Date or round | Trigger for the change | Evidence file | Locked in advance? |
|---|---|---|---|---|---|
| Original | "do entropy-regulating neurons exist here, and are they shared across uncertainty types?" (README, project "Confidence Neurons Across Uncertainty Types") | Aug 2026 (first commit 3fc0584, 6 Aug) | n/a | README.md | no |
| Evolved 1 | Hedging is a "verdict" computed by a sparse circuit, separate from the output "spread" (entropy) ("decision vs spread") | Aug-Sep 2026; InterpScience submission 1 Sep | Neuron-level nulls; circuit patching | paper/main.tex, RESULTS.md | circuit criterion fixed in code, not registered |
| Evolved 2 | Decision-vs-spread retired as the headline; the circuit is necessary but not sufficient for hedging | 2 Oct 2026 (e1ef8dc) | Behavioural validation EQUIVOCAL against its locked bar (l5_behavioral_criterion.md, 59f5ec5) | results/circuit_familiarity_v3/behavioral* | yes (59f5ec5) |
| Current | Whether Llama says "I don't know" follows entity familiarity more than knowledge, while first-token entropy follows knowledge. The entropy split holds in 4 families; the hedging split is measurable only in Llama. A familiarity signal is decodable in base models. | 5-7 Oct 2026 | Lit review for UniReps (litreview/00_SYNTHESIS_UniReps.md); popularity × knowledge factorial (commit 1ef1c95, 19 Sep) | results/OVERVIEW_2026-10-07.md and each results/*/FINDINGS.md | yes: six criterion files at the repo root, each hashed before its run |

## Headline claim and strength

- **Headline:** In Llama-3.1-8B-Instruct, saying "I don't know" follows how familiar the entity is more than whether the model knows the fact: popularity effect +0.39 vs knowledge effect +0.09, difference +0.30 [+0.20, +0.40] (two judges, 96-token replies). Meanwhile first-token answer entropy follows knowledge, not familiarity, in all four model families.
- **Strength:** supported. The criterion was locked; the judges' κ = 0.62 only just clears its 0.60 bar, and human validation is not yet done. The entropy half is established.
- **Fallback if the hedging half is retired** (for example by the human check): the entropy-follows-knowledge result in 4 families, the base-model lens result, and the neuron-level negative.
- **Claims ledger:** `paper/CLAIMS.md` (to be written in `hypothesis` / `evidence`).

## Thesis (mode hypothesis, 2026-10-09; confirmed [user] 2026-10-09)

- **One sentence:** *When Llama-3.1-8B-Instruct says "I don't know" to a factual question, it tracks how familiar the question's subject is more than whether it knows the answer, while its answer uncertainty tracks knowledge.* Supporting results: the entropy split recurs in four model families, and the familiarity signal is already decodable in base models.
- **Thesis tests:**
  - **Single idea:** passes (one dissociation between two measures).
  - **Deletion:** passes; every result serves it (C2 makes it cross-model, C4-C6 probe its mechanism and origin).
  - **Substitution:** the rivals and what answers them:
    - name spelling → C3 (post hoc);
    - detector artefact → C1's two-judge measure;
    - post-training habit only → C6 (signal present in base models);
    - "it's hedging intent, not familiarity" → C4's no-hedge control, which only partly answers it and must be disclosed;
    - "different prompts for hedging vs entropy" → disclosed, not answered.
  - **Evidence:** every clause maps to C1, C2 or C6.
- **Scope:** PopQA single-entity factual questions; 7-9B instruct models (hedging: Llama only; entropy: 4 families); base models for C6 (3 families). Not reasoning uncertainty, not ambiguity, not long-form answers.
- **Rung of the headline:** supported. No human validation will be done ([user], 2026-10-09); disclose it in Methods/Limitations.
- **Fallback if C1 falls:** "Answer uncertainty follows knowledge in four families while a familiarity signal sits mid-stack in base and instruct models." C2 + C6, plus the neuron-level negative C8.
- **Title candidates (no "not when the fact is unknown", since knowledge has a small effect):**
  - "When Llama Says 'I Don't Know', Familiarity Matters More Than Knowledge"
  - "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge"

## Methods notes for write (2026-10-09, [user]-requested)

**Two prompt sets, each measuring a different thing, by design:**

1. **Forced-answer prompt** (chat template + assistant prefill "The answer is"). It measures **knowledge** and **answer uncertainty**:
   - The prefill commits the model to answering, so the next token is the first token of an answer. Its entropy is uncertainty over answers.
   - Sampling 10 completions (plus greedy) and grading them against the gold aliases tells us whether the model *can* produce the fact (SliCK labels).
   - Without the prefill, the first token of a reply is a preamble ("The", "I", "Sure"), and its entropy measures phrasing, not answer uncertainty.
2. **Free prompt** (chat template only, no prefill; 20 samples at T = 0.7 in the factorial pools; 3 × 96 tokens in the two-judge measurement). It measures **hedging**:
   - "I don't know" is a choice the model makes only when it is free to answer or decline.
   - The prefill removes that choice, so hedging cannot be measured under it.

**Why this is the right design, not a nuisance:** the dissociation is defined by it. The model produces the answer when made to answer ("Belorechensk", 10/10), yet says "I do not have information" when allowed to choose (20/20). Measuring both on one prompt would either suppress the hedge (prefill) or lose the answer distribution (free).

**Caveat to state:** because the prompts differ, part of any difference between hedging and entropy could come from the prompt change itself. Two counter-points:
- the first-token hedge-vs-answer log-odds at the forced-answer position also carries the familiarity signal (Gemma: popularity +1.79 vs knowledge −0.32; Llama: both, +2.93 vs +3.06);
- the steering and lens analyses (C4, C6) work at single prompts.

Repo evidence: `scripts/build_familiarity_twins.py` (PREFILL; `free_prompt` for hedging); worked example in PROJECT_EXPLAINER.md.

## Closest prior work and scoop verdict (litreview, 2026-10-10)

- **Closest:** Ferrando et al. 2025 (ICLR; entity recognition steers refusal, but "known" is confounded with fact knowledge); Lindsey et al. 2025 (Biology: known-entity features suppress default can't-answer features in Claude 3.5 Haiku); Brzezinka et al. 2026 (graded familiarity probes; weak popularity tracking in Gemma-4/Qwen3; no knowledge axis). Credit Ji et al. 2025 (EMNLP) for verbal ≠ semantic uncertainty; our addition is what each follows.
- **Concurrent:** Che & Li 2026 (arXiv 2610.07270, 5 Oct): steering an unknown-entity latent moves abstention (33% → 79%), but installing natural values transfers only 0.12 of the contrast. **Effect on us:** C4 is worded as "steering along the obscure-vs-popular direction can produce hedging", not "the model uses this direction"; cite next to our no-hedge control. Nguyen et al. 2026 (NeurIPS) owns "sparse abstention circuit across families", so C9 stays in the appendix.
- **Scoop verdict:** none for C1, C2, C6; partial (reframing only) for C4. Window 5-10 Oct checked on 2026-10-10 (litreview/QUERY_LOG.md); 15 Aug - 5 Oct cleared earlier (Sher/litreview axis I). Re-run before submission.
- **Files:** litreview/SYNTHESIS.md, MUST_CITE.md, VENUE_FIT.md, VERIFICATION.md, CITATION_REPORT.md.

## Venue, deadline, format

- **Venue:** UniReps 2026, workshop at NeurIPS 2026 (https://unireps.org/2026/call-for-papers; checked 5 and 7 Oct 2026). Paris, 12 Dec 2026.
- **Deadline:** OpenReview submission closes **11 Oct 2026, 12:00 UTC** (expiry 12:30 UTC), i.e. the end of 10 Oct AoE. Verified from the OpenReview API on 7 Oct; an earlier read and the saved memory said 10 Oct 12:00 UTC.
- **Format:** NeurIPS LaTeX template, anonymized; no NeurIPS checklist. The OpenReview form needs a TL;DR (≤ 250 chars), an abstract (≤ 5,000 chars), a reciprocal reviewer (an author's OpenReview ID), the LLM-policy checkbox, and an OpenReview profile for every author.
- **Track:** Extended abstract (non-archival), at most 4 pages of main text excluding references and appendix. [user: "this four page paper we are going for", 5 Oct]

## People and roles

| Person | Role | Commits as | Decides what |
|---|---|---|---|
| Sheryl Mehta | Lead, repo owner (GitHub SherylMehta1) | SherylMehta1 (68 commits) | Claims 1-2 per paper/DOC_MEMO.md [unverified: current roles] |
| Shivam | Co-author; the user of these sessions | shivam / shivam dubey (31) | Directs the current paper work [user] |
| Shagun Chadha | Co-author | shagunchadha (11) | Claim 4 per DOC_MEMO.md |
| Kanika Singhal | Co-author | Kanika Singhal (8) | Claim 5 per DOC_MEMO.md |

## Locked decisions

| Date | Decision | Reason | Recorded in |
|---|---|---|---|
| 2026-10-05 | Target UniReps 2026, extended-abstract track | Prior workshop rejected; UniReps fits a cross-model story; non-archival keeps a main venue open | [user]; litreview/00_SYNTHESIS_UniReps.md |
| 2026-10-06 | Title must be scoped to Llama ("language models" not licensed) | gemma_mistral_factorial_criterion.md: needs both Gemma and Mistral REPLICATES; Mistral PARTIAL | results/factorial_4models/FINDINGS.md |
| 2026-10-07 | The two-judge, 96-token measurement supersedes the regex for the hedging comparison where κ ≥ 0.60 | Fixed in advance in hedge_measure_v2_criterion.md | results/hedge_measure_v2/FINDINGS.md |
| 2026-10-07 | Decision-vs-spread is not the headline | Locked behavioural bar EQUIVOCAL (59f5ec5 / e1ef8dc) | l5_behavioral_criterion.md |
| 2026-10-07 | Causal steering claim worded as "obscure-vs-popular direction", not "familiarity per se" | No-hedge control: +0.04 vs +0.27 | results/direction_nohedge/FINDINGS.md |
| 2026-10-09 | Title: "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge"; thesis as drafted | [user] chose option (b) | LAB_LOG 2026-10-09 |
| 2026-10-09 | Methods must state that hedging and answer uncertainty are measured on different prompts, and explain why (see "Methods notes for write") | [user] instruction | LAB_LOG 2026-10-09 |
| 2026-10-07 | Paper lives in Sher/paper (new folder, outside the repo) | [user] instruction | this file |
| 2026-10-10 | Title changed to "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge" (replaces 1b) | Area chair round 2 condition 1; E2 negative (model-side familiarity measure failed), so the measured variable is named | [user] "switch" |
| 2026-10-10 | Title (b): "In Llama, Saying 'I Don't Know' Tracks Familiarity While Answer Uncertainty Tracks Knowledge" | Review round 1 I1: all three referees; abstention measured in one model | [user] "1b"; review/round1/REVIEW_REPORT.md |
| 2026-10-10 | Circularity (P1/I10) handled by rewording: state that knowledge labels and entropy share the forced-answer distribution, present entropy as the label-linked reference readout and abstention as the readout that departs from it; no relabelling run | Deadline; GPU relabel risks it | [user] "2a" |
| 2026-10-10 | Defaults (user gave no answer; recommended options): cross-model result in as one exploratory sentence + appendix table; GPU steering controls (I9, I16-I18) conceded as limitations; no human audit (2026-10-09 decision stands) | Recommended at Checkpoint 3; reversible by [user] | LAB_LOG 2026-10-10 |
| 2026-10-10 | Register: short-workshop (3 claims in main text, one figure each; negative findings stated briefly as findings); outline approved | UniReps extended-abstract call invites negative findings; 4-page limit | OUTLINE.md; LAB_LOG 2026-10-10 |
| 2026-10-10 | C4 framed as controllability, not use (cite che2026steering beside the no-hedge control) | Concurrent work shows steering success ≠ the model using a feature; consistent with our own control | LAB_LOG 2026-10-10; litreview/SYNTHESIS.md |
| 2026-10-10 | Cite published versions where they exist (13 of 33); key `rimsky2024caa` (ACL 2024 record lists Nina Rimsky; arXiv now lists Panickssery) | litreview rule: prefer the published version | litreview/VERIFICATION.md |

## Open decisions

| Decision | Options | Owner | Blocks | Status |
|---|---|---|---|---|
| Human validation of the two judges on Llama (120 blind replies) | — | [user] | — | **decided 2026-10-09: not done** [user: "leave the human check"]; C1 stays supported; the paper discloses that the judges were not human-validated |
| Commit the 5-7 Oct work to git (criteria first) | commit now / later | Shivam [user] | reproducibility; anonymity of the repo | open (asked repeatedly, no answer yet) |
| Make the GitHub repo private during review | private / leave public | Sheryl (repo owner) | anonymity | open |
| Which figures go in 4 pages | 2×2 factorial bar; four-family entropy; lens base vs instruct; steering | author at Checkpoint 2 | `write` | open |
| Use the single-BOS Figures 4/5 or drop them from the extended abstract | — | author | `figures` | open |
| Author list and order, reciprocal reviewer | — | team | submit | open |

## Glossary

| Term | Plain meaning | Where it appears | Replace with in the paper |
|---|---|---|---|
| hedge / hedging | the model says it does not know or cannot verify | everywhere | "says 'I don't know'" |
| familiarity / popularity | how well known the question's subject is (Wikipedia pageviews, PopQA `s_pop`) | factorial, lens | "how familiar the name is" |
| knowledge / SliCK class | whether the model answers correctly: HighlyKnown (≥ 9/10 samples and greedy correct) vs Unknown (0/10) | labels | "known / unknown facts" |
| factorial | 2 × 2 of obscure vs popular × known vs unknown | data/factorial_v3* | "popularity × knowledge design" |
| twin / gated twin | matched prompt pair kept only if the model's own answers confirm the labels | data/familiarity_v3 | "matched pair checked against the model's answers" |
| entropy | spread of the model's first-token probabilities after "The answer is" | all pools | "answer uncertainty" |
| arm / cell | prompt type / condition | old drafts | name the condition |
| Jacobian lens | prefitted decoder that maps a layer's activations to output words | jlens_* | "a lens that shows which words a layer pushes toward" |
| circuit | 20 attention heads + 100 MLP neurons found by patching | circuit_* | give the size every time |
| verdict / spread | old jargon for hedge decision / entropy | old drafts | drop |

## Key files map

| What | Path (repo-relative unless noted) |
|---|---|
| Results summary for this paper | results/OVERVIEW_2026-10-07.md |
| Per-experiment findings | results/{unireps_checks, qwen_factorial, factorial_4models, hedge_judge, hedge_measure_v2, jlens_cascade, jlens_bos_audit, direction_reuse, direction_reuse_v2, direction_nohedge, singlebos}/FINDINGS.md |
| Criterion files (locked before runs) | qwen_factorial_criterion.md, gemma_mistral_factorial_criterion.md, jlens_cascade_criterion.md, direction_reuse_criterion.md, direction_reuse_v2_criterion.md, hedge_judge_criterion.md, hedge_measure_v2_criterion.md, nohedge_direction_criterion.md (repo root; older: l5_behavioral_criterion.md, preregistration_commit.md) |
| Numbers file (THIS paper) | C:/Users/starb/Desktop/Sher/paper/numbers.tex, generated by make_numbers.py |
| Numbers file (old paper) | paper/numbers.tex (239 macros, generated by scripts/make_numbers_tex.py; none used by old main.tex) |
| Figure generators | scripts/make_paper_figures.py; scripts/make_fig45_singlebos.py; scripts/jlens_cascade_analysis.py (fig_cascade) |
| Old paper source | paper/main.tex (InterpScience version; the submitted PDF source is outside git: C:/Users/starb/Desktop/shshk/overleaf_submission/main.tex [unverified]) |
| Bibliography (old) | paper/references.bib (23 entries) |
| Lit review | C:/Users/starb/Desktop/Sher/litreview/ (axes A-J + 00_SYNTHESIS_UniReps.md) |
| Reviews of the prior submission | paper/reviews_interpscience.md |
| Logs and notes | RESULTS.md, git log, this folder's LAB_LOG.md |
| Criterion notes (this skill) | C:/Users/starb/Desktop/Sher/paper/PREREG/ (the project's criteria live at the repo root) |
| Inventory | C:/Users/starb/Desktop/Sher/paper/PROJECT_INVENTORY.md |
| Project understanding (another session, 6 Oct) | C:/Users/starb/Desktop/Sher/PROJECT_UNDERSTANDING.md |

## Other paper tracks and sibling projects

- **Repo `paper/main.tex`:** the InterpScience 9-page draft, "Deciding to Say 'I Don't Know'". Superseded; source of methods text and older results.
- **Submitted version:** "Doubt Is a Decision: A Sparse Circuit Commits Language Models to 'I Don't Know'", 17-page PDF. Source at `C:/Users/starb/Desktop/shshk/overleaf_submission/main.tex` [unverified]; recorded in paper/reviews_interpscience.md.
- **Working track: `C:/Users/starb/Desktop/Sher/paper/`** (this folder; new, UniReps extended abstract).

## Submission history

- **NeurIPS 2026 "Interpretability as a Science" workshop** (deadline 1 Sep 2026): **Rejected** (AC e8tZ, 29 Sep). Reviews in `paper/reviews_interpscience.md`.
  - Reviewer gbEf: Accept 3, confidence 2. Too dense and jargon-heavy; contribution is "comparison and identification"; experiments fair; doubts significance given few models.
  - Reviewer 14N6: Clear reject 1, confidence 3. Undefined jargon in the abstract; "a circuit is a post-hoc description, not itself the cause"; the title overclaims general "I don't know" while only 3 prompt types are tested; misses reasoning uncertainty.
  - A free-text line reads "substantially AI generated, abstract is extremely difficult to understand due to unsubstantiated jargon" (field attribution unverified).

## What is running now

Nothing. All GPU jobs of 5-7 Oct have finished and their outputs are copied to the repo (untracked). molab sandboxes four to eight are idle and can be shut down. **About 60 untracked paths are uncommitted** (see the inventory), including all criterion files, about 15 scripts and all 5-7 Oct results. Three modified tracked scripts (jlens_positions/steer/trajectory: an opt-in `--strip-bos` flag, default off).

## Missing documents

- **Plan documents quoted by results but absent:** l5_behavioral_validation.md, factorial_circuit_level_test.md, uu_falsification_test.md, regenerate_final_numbers.md, familiarity_regeneration_plan.md, EXPLAINER.md. Probably kept outside git or in Claude sessions; owner likely Sheryl/Shivam [unverified].
- **shshk/EXPERIMENT_PLAN.md** and the submitted Overleaf source: in `C:/Users/starb/Desktop/shshk/` (outside this workspace) [unverified].
