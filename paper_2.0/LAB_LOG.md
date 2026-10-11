# Lab log

<!-- Append only. Add new entries at the bottom. Never rewrite or delete an old entry; if it was wrong, add a dated correction below it.
     Write the entry before acting on a decision (invariant I5), then add the outcome afterward.
     One entry per event: a run, a decision, a mode that ended, a result that changed a verdict. -->

## 2026-10-07 | Mode understand

- **What ran or happened:** research-paper-pilot `understand` mode: inventory script, review of README/RESULTS/criteria/FINDINGS, the InterpScience reviews, saved project memory, and the 5-7 Oct experiment outputs.
- **What was decided:**
  - the paper folder is `C:/Users/starb/Desktop/Sher/paper`, outside the repo [user];
  - target UniReps 2026 extended abstract, 4 pages [user, 5 Oct];
  - title scoped to Llama (gemma_mistral_factorial_criterion.md);
  - the two-judge 96-token measurement is the hedging result (hedge_measure_v2_criterion.md);
  - deadline confirmed as 11 Oct 2026 12:00 UTC from the OpenReview API.
- **Why:** user instructions; criteria fixed before the runs.
- **By whom:** Shivam [user] with Claude.
- **Files:** PROJECT_INVENTORY.md, PROJECT_CONTEXT.md, PROJECT_EXPLAINER.md, LAB_LOG.md.
- **Open:** human validation of the judges; commit of the 5-7 Oct work; repo privacy; figure selection; author list and reciprocal reviewer.

Understand ran on 2026-10-07; produced PROJECT_INVENTORY.md, PROJECT_CONTEXT.md, PROJECT_EXPLAINER.md, LAB_LOG.md; decisions: paper folder, venue/track, Llama-scoped title, two-judge hedging measure, deadline; open: judge human validation, commit, repo privacy, figures, authors.

## 2026-10-09 | Mode hypothesis

- **What ran or happened:** claims ledger built from the FINDINGS files and criteria; thesis drafted and run through the single-idea, deletion, substitution and evidence tests.
- **What was decided:**
  - headline = C1 (Llama hedging follows familiarity over knowledge, +0.30), supported, with C2 (entropy follows knowledge in 4 families, established) and C6 (base-model familiarity signal, supported);
  - retired R1-R5 recorded;
  - title must not say "not when the fact is unknown", because the knowledge effect is +0.09 with CI > 0.
- **Why:** headline chosen by audit survival (C1 passed a locked criterion and survives 4 instruments; C2 passed in 4 families and was independently audited); per the criteria files.
- **By whom:** Claude draft; awaiting [user] confirmation.
- **Files:** CLAIMS.md (new), PROJECT_CONTEXT.md (Thesis section).
- **Open:** user confirmation of the thesis and title; human validation of the judges; hedging vs entropy measured on different prompts (disclose).

Hypothesis ran on 2026-10-09; headline now C1 (+0.30, supported) with C2/C6 support; retired R1-R5; open: thesis/title confirmation, judge validation.

## 2026-10-09 | Checkpoint decisions (user)

- **What was decided:**
  - title (b) "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge";
  - thesis confirmed;
  - Methods will state and justify the two prompt sets (forced-answer prefill for knowledge and answer entropy; free prompt for hedging).
- **Why:** [user] choice; reviewers would otherwise flag the prompt difference.
- **By whom:** Shivam [user].
- **Files:** PROJECT_CONTEXT.md (Locked decisions; Methods notes for write), CLAIMS.md (open item updated).

## 2026-10-09 | Mode evidence

- **What ran or happened:**
  - `make_numbers.py` written; it generates `numbers.tex` (138 macros, each citing its file and field) from the repo's result files.
  - A bug in the generator (a greedy regex captured entropy as hedge rates) was caught by inspection and fixed before any use.
  - Per-claim evidence audit added for C1, C2 and C6.
- **What was decided:** all paper numbers come from numbers.tex; verdicts unchanged from mode hypothesis (C1 supported, C2 established, C4 supported-qualified, C6 supported); C9 injection reported as 0.40.
- **Why:** invariant I1; the old draft drifted (0 of 239 macros used).
- **Files:** make_numbers.py, numbers.tex, CLAIMS.md (audit + drift list).
- **Open:** human validation of the judges; `check_numbers.py` will run once main.tex exists.

Evidence ran on 2026-10-09; produced make_numbers.py, numbers.tex (138 macros), CLAIMS audit; decisions: single numbers source, verdicts unchanged; open: judge validation.

## 2026-10-09 | Decision: no human validation of the judges

- **What was decided:** the 120-reply human check will not be done before submission. C1 stays "supported"; Methods/Limitations will state that the two LLM judges (κ 0.62) were not validated against human labels.
- **Why:** [user] "leave the human check" (time).
- **By whom:** Shivam [user].

## 2026-10-10 | Mode litreview

- **What was done:**
  - references.bib was rebuilt by litreview/build_bib.py from source pages only: the arXiv abstract pages (the API was rate-limited), Crossref, the ACL Anthology, and the Transformer Circuits page (author list read from the page). Published venues came from OpenAlex, plus OpenReview search records for 5 entries.
  - The result is 33 entries; 13 cite the published version.
  - verify_citations.py --online: 0 mismatches, 0 warnings. 13 verified directly; 20 marked "unresolved" only because the script's arXiv API calls timed out, and each was checked on its arXiv page by build_bib.py (VERIFICATION.md).
  - The wrong SCIURus hit from the arXiv title search (a squirrel paper) was caught and replaced by the ACL Anthology record.
  - Scoop check for 5-10 Oct: 10 arXiv advanced-search queries plus 3 web searches.
  - Closest prior work re-read: Ferrando (PDF), Brzezinka, Nguyen, Ji, Kang, Biology.
- **What was decided:** C4 is worded as controllability rather than use, and cites Che & Li 2026 (2610.07270) next to the no-hedge control. Published versions are cited where they exist; the key panickssery2024caa becomes rimsky2024caa.
- **Why:** concurrent evidence that steering success does not show feature use; the litreview verification rule.
- **Files:** references.bib, litreview/{build_bib.py, VERIFICATION.md, CITATION_REPORT.md, MUST_CITE.md, SYNTHESIS.md, VENUE_FIT.md, QUERY_LOG.md, notes/ (9)}, PROJECT_CONTEXT.md (closest work, scoop verdict, status), CLAIMS.md (C4 caveat).
- **Open:** Kuhn 2023's venue was not confirmable (cited as arXiv). Re-run the scoop check before submission. The optional cross-model similarity number (VENUE_FIT gap 2) is for the author to decide.

Litreview ran on 2026-10-10; produced references.bib (33 verified) and the litreview files; decisions: C4 framed as controllability, published versions cited; open: Kuhn venue, re-run scoop before submit.

## 2026-10-10 | Mode figures

- **What was done:** three main figures, each from one generator script that reads repo result files only, with a shared style (Okabe-Ito, 5.5 in, 9 pt, constrained layout, vector PDF plus PNG preview plus provenance JSON).
  - F1 fig_dissociation: Llama hedging 2 × 2 (two judges), Llama entropy 2 × 2, entropy effects in 4 families.
  - F2 fig_steering: hedge rate and entropy under no steering, 5 random directions, the unknown−known direction, the obscure−popular direction, and the never-hedged control.
  - F3 fig_lens_base: lens familiarity and knowledge effects by relative depth in 4 base models.
  - Caption-versus-data check: all plotted values equal the summary or outcomes files; scope words checked.
- **What was decided:**
  - Steering legend uses "obscure−popular" and "unknown−known" direction names, following the locked 7 Oct wording decision.
  - F3 shows the knowledge curve beside the familiarity curve, so the reader sees that familiarity is not the only signal in the readout.
  - C7 timing moves to the appendix and reuses the single-BOS fig4.
  - Instruct lens curves (C10) and the C9 circuit are not figured.
- **Why:** one claim per figure; honesty about the readout; 4-page budget.
- **Files:** paper/figures/{style.py, fig_dissociation.py/.pdf/.png/.provenance.json, fig_steering.*, fig_lens_base.*}, FIGURE_PLAN.md.
- **Open:** whether F3 shrinks to half width if space is short (outline checkpoint).
- **Incident:** an inline Python edit turned `\f` macros into form feeds in FIGURE_PLAN.md. It was caught and fixed (0 left), and a memory note was added.

Figures ran on 2026-10-10; produced 3 figure scripts + PDFs and FIGURE_PLAN.md; decisions: direction naming, knowledge curve shown in F3, C7 to the appendix; open: F3 size.

## 2026-10-10 | Checkpoint 2 passed: outline approved

- **What was decided:** OUTLINE.md approved as written ([user]: "write", after the outline was presented). This covers the three bullets, the contribution sentence, the section outline, the terminology lock and the short-workshop register.
- **Why:** [user] instruction.
- **By whom:** Shivam [user].
- **Also:** NeurIPS 2026 template downloaded (the UniReps call links it; `dblblindworkshop` option). make_numbers.py gained \stLlamaLayer(s), \stGemmaLayer(s), \stLlamaN, \stGemmaN, \poolLlama and \kappaBar (146 macros).

## 2026-10-10 | Mode write

- **What was done:**
  - tex/main.tex drafted on the official NeurIPS 2026 template (`dblblindworkshop`, linked from the UniReps call), following the approved OUTLINE.md.
  - The main text is exactly 4 pages; references start on page 5. Appendices A-D: robustness table, other families, timing/neurons/circuit, analysis timing.
  - The first draft ran 5.2 pages. The compression pass and figure heights (1.55-1.75 in) brought it to 4. No claim, macro, caveat or citation was dropped from the main text; the C8 neuron sentence moved to Appendix C.
- **Checks:**
  - LaTeX: 0 errors, 0 undefined references, 0 overfull boxes.
  - check_numbers: 83 macros used; 18 literals, all design settings (16 relations, T = 0.7, 90%, 96 tokens, 95%, 10,000 bootstraps, 20 heads/100 neurons), 0 unsupported.
  - prose_gate --academic: 0 dashes, reader load 0. 4 advisory findings kept with reasons: two passives in Limitations with no meaningful actor; ANSWER is a quoted judge label; the lens sentence carries the four tested values.
  - check_submission dry run: clean. 2 bib entries uncited (cheng2025abstraction, liu2007fok); the style hides them.
- **What was decided:**
  - Register: short workshop.
  - Verbs: "we show" only for C2.
  - Main text: the steering paragraph states "control, not use"; the Llama entropy half of the steering result is reported as a null.
- **Files:** tex/main.tex, tex/neurips_2026.sty, draft_v1.pdf; make_numbers.py (+6 macros, 146 total); figures re-rendered at the new heights.
- **Open:** fresh-reader pass; humanize; blind review; author list and reciprocal reviewer; LLM-use checkbox on OpenReview.

Write ran on 2026-10-10; produced tex/main.tex + draft_v1.pdf (4 pages main); decisions: register, verb strength, C8 to the appendix; open: 0 TODO / 0 [VERIFY] citations, no pending figures.

## 2026-10-10 | Mode review, round 1

- **What was done:**
  - Blind packet (draft_v2.pdf + text + figures).
  - Agents: three reviewers (methods: major revision; framing: major revision; clarity: minor revision), a chair merge (57 issues, 0 critical), a red-team attack plus independent adjudication (5 points: 0 answered, 3 partial, 2 unresolved), and a zero-context numbers audit (140 items; all 83 macros reproduce; 15 wording/scope mismatches, 3 critical: the "one prompt" sentence, "emerge together", "rarely abstain").
  - Post hoc guessability control on existing outputs: the two-judge difference stays +0.300 [+0.192, +0.411] without guessable items (review/round1/guessability_check.txt).
  - Verified two new references: Kumaran et al. 2026 (2603.22161) and Ni et al. (2505.17537, EMNLP 2026).
- **Decisions:** none yet; Checkpoint 3 is waiting for the user (review/round1/REVIEW_REPORT.md §9).
- **Open:**
  - title scope;
  - circularity (reframe vs relabel);
  - honest Llama forced-prompt numbers;
  - optional cross-model analysis;
  - GPU controls vs limitations.

Review round 1 ran on 2026-10-10; produced review/round1/*; decisions: pending user (Checkpoint 3); open: I1, I2, I10 and the must-fix list in REVIEW_REPORT.md.

## 2026-10-10 | Cross-model agreement (review item I23), exploratory

- **What was done:**
  - Plan written first: PREREG/cross_model_agreement.md. It is exploratory, because the point estimates already sat in repo factorial_4models/summary.txt.
  - analysis/cross_model_agreement.py: Spearman correlations with 10,000 item bootstraps and 2,000 within-relation permutations.
  - The first, loop-based version was too slow and was stopped; the vectorized rerun took 37 s with identical statistics.
- **Result, on knowledge-differs questions:**

  | Pair | n | hedge log-odds ρ | entropy ρ | Δ (hedge − entropy) | Rule |
  |---|---|---|---|---|---|
  | Llama–Qwen | 313 | +0.11 [−0.00, +0.22] | −0.17 | +0.28 [+0.14, +0.42] | not met |
  | Llama–Gemma | 60 | +0.38 [+0.13, +0.60] | +0.03 | +0.35 [−0.03, +0.74] | not met |
  | Llama–Mistral | 108 | +0.25 [+0.06, +0.42] | −0.07 | +0.32 [+0.06, +0.57] | met |

  The pre-set rule (hedge ρ > 0 and Δ > 0, both with CI excluding 0) holds for 1 of 3 pairs. Δ > 0 with CI excluding 0 holds for 2 of 3.
- **Decision:** [user] chose to run item 3, and not item 4 (GPU steering controls; recommended as limitations, not yet confirmed by [user]).
- **Open:** how to write this in (appendix table + one exploratory sentence recommended).

## 2026-10-10 | Checkpoint 3 decisions

- **[user]:** "revese 1b 2a". Title (b), scoped to Llama. Circularity handled by rewording (a).
- **Defaults taken for unanswered items (recommended, reversible):** cross-model result goes in as exploratory; GPU steering controls are conceded as limitations; no human audit.
- **Why:** the deadline (11 Oct 12:00 UTC); no GPU box available.

## 2026-10-10 | Mode revise, round 1

- **What was done:**
  - All must-fix items and numbers-audit criticals addressed (revision/round1/RESPONSE_LETTER.md has the comment table and traceability matrix).
  - Title scoped to Llama ([user] 1b). Circularity stated in Setup, with entropy as the reference readout ([user] 2a).
  - Llama's forced-answer hedge readout reported honestly: popularity and knowledge about equal.
  - Guessability control and cross-model agreement added as post hoc. Kumaran 2026 and Ni 2025 cited and verified. Lindsey, Ferrando and Brzezinka credited.
  - Lens claim narrowed: the base readout responds to popularity and to knowledge. Failed-prediction onsets in App. D. Hashes in App. G.
  - The main text was cut back to exactly 4 pages; the Effects formula became words.
- **Conceded as limitations:** steering norm chosen on the evaluation questions; no projection-out or other-norm steering; lens answer-token rarity; no human audit.
- **Files:**
  - tex/main.tex, draft_v3.pdf, numbers.tex (252 macros), make_numbers.py;
  - analysis/{guessability.*, cross_model_agreement.*}, PREREG/cross_model_agreement.md, litreview/add_bib_entries.py, references.bib (+2);
  - figures (labels), CLAIMS.md (C11 added; C2/C6 wording narrowed), revision/round1/*.
- **Checks:** compile clean; check_numbers 0 unsupported; prose gate 2 false positives.
- **Open:**
  - optional re-review (round 2);
  - author list and reciprocal reviewer;
  - repo privacy;
  - commit the 5-10 Oct work.

Revise round 1 ran on 2026-10-10; produced revision/round1/*, draft_v3.pdf; decisions: 30+ fixes, 5 concessions, 0 push-backs, 0 deferred; open: re-review optional, submission form items.

## 2026-10-10 | Review round 2 (on draft_v3)

- **What was done:**
  - [user] asked for a review round ending in an accept/reject decision.
  - Fresh blind agents ran: 3 reviewers (major, major, minor; no reject), attack + adjudication (0 unresolved, down from 2) and a numbers audit (2 critical wording errors, both introduced in revise round 1: the selective base-model K report and the cross-model rule wording).
  - **Area chair: ACCEPT, conditional on camera-ready items 1-8** (mostly rewording; title narrowed to "entity popularity", entropy framed as the guaranteed reference, steering and lens wording, OLMo).
  - Exploratory checks answer chair items 9(a) and 9(e) (analysis/round2_checks.*).
- **Open:**
  - the title conflict: the chair asks for "Tracks Entity Popularity More Than Knowledge" vs [user] 1b;
  - camera-ready items 1-8 plus audit C1/C2 before submission.

## 2026-10-10 | Round 3 experiments: setup and E2 result

- **Boxes** (all RTX PRO 6000 96 GB): A = E1 + E2; B = E3; C = E5; D = E4.
- **Criterion notes** written and hashed before any run: PREREG/e1-e5, HASHES_round3.txt, 2026-10-09T21:28Z.
- **New scripts** (repo, uncommitted): scripts/e1_relabel.py, e2_surprisal.py, e3_lens_split.py, e4_steer.py, round3_analysis.py.
- **E2 (name surprisal) result: NOT SUPPORTED** under its pre-set rule.
  - Surprisal is uncorrelated with log page views (Spearman −0.001, n = 2,673).
  - The two-judge surprisal coefficient is +0.013 [−0.026, +0.059].
  - The regex coefficient is +0.071 [+0.046, +0.096], and drops to +0.012 [−0.008, +0.033] once the arm is added.
  - Reading: this model-side exposure measure does not capture familiarity, so it cannot stand in for page views. Reported as a negative result.
  - Consequence: it supports scoping the title to "entity popularity" rather than "familiarity".

## 2026-10-10 ~22:05 UTC | E1 and E5 results (pre-set rules)

- **E1 (independent labels; sha af50f53f):**
  - **T1 SURVIVES.** With paraphrase-prompt labels, the two-judge popularity effect is +0.383 [+0.294, +0.472], knowledge +0.061 [−0.030, +0.150], difference +0.322 [+0.157, +0.490]. The obscure-known cell is 0.457 (n = 27).
  - **T2 NOT CIRCULAR.** Original-prompt entropy x paraphrase knowledge: knowledge − popularity = +1.402 [+1.020, +1.730].
  - **T4.** At the free prompt, first-token entropy follows knowledge (pop +0.053 vs know +0.160; diff −0.106 [−0.182, −0.029]), while the first-token "I" log-odds follows popularity (+8.37 vs +3.11; diff +5.25 [+4.42, +6.05]). The dissociation therefore holds within one prompt.
  - **T3.** Labels agree only moderately; the paraphrase prompt is harder: 313 of 676 popular-known stay HighlyKnown, 312 become Maybe.
- **E5 (third judge Llama-3.3-70B 4-bit; sha f35aa4d9):**
  - Llama: Fleiss κ 0.741 → measured. Majority rule: diff +0.437 [+0.322, +0.550], FOLLOWS POPULARITY. Obscure-known 0.644.
  - Gemma: κ 0.643 → measured, NEITHER (diff +0.013 [−0.030, +0.057]); it rarely abstains (obscure-unknown 0.107).
  - Qwen: κ 0.409, unmeasured. Mistral: κ 0.535, unmeasured.
  - Pairwise κ: Qwen32B–Llama70B 0.63-0.96 across families; Gemma27B is the low-agreement judge.
  - The title stays scoped to Llama (0 of 3 other families follow popularity).

## 2026-10-10 ~22:45 UTC | E3 result (base models; sha b2b916ce)

- **Integrity check:** the rerun reproduces the original per-layer F exactly (Llama, all layers).
- **Bug fixed in my analysis:** the window denominator was max lens layer; it is now n_layers − 1, as in the criterion. Window values now equal the original tested midF: +0.671, +0.387, +0.401, +0.293.
- **L1 HEDGE-SIDE (4 of 4 base models).** Hedge-term popularity effects:
  - Llama +0.548 [+0.458, +0.635]
  - Gemma-9B +0.281 [+0.212, +0.348]
  - Gemma-2B +0.098 [+0.031, +0.168]
  - OLMo +0.612 [+0.529, +0.696]
- **L2.** Window knowledge effects on the full readout:
  - Llama +0.362 [+0.282, +0.442]
  - Gemma-9B +0.375 [+0.332, +0.419]
  - Gemma-2B +0.110 [+0.040, +0.180]
  - OLMo −0.303 [−0.396, −0.203]
- **L3 (descriptive).** In the hedge term alone, knowledge ≈ popularity (Llama, Gemma-9B, OLMo), and knowledge > popularity for Gemma-2B. The full-readout popularity > knowledge asymmetry comes from the answer term.
- **Consequence for C6 wording:** a popularity-sensitive hedge signal predates instruction tuning; its dominance over knowledge is not shown in base models.

## 2026-10-10 ~23:15 UTC | E4 primary result (sha 18c74883; scorer: both original judges)

- **Llama** (k* = 1 on SELECT):
  - EVAL judge abstention: none 0.007, fam 0.100, know 0.044, abst 0.089, fam_perp 0.018, random ~0.008.
  - D1 fam − random +0.092 [+0.062, +0.125]: NOT MET (bar 0.10).
  - D2 fam_perp − random +0.010 [−0.001, +0.024]: carried by the abstention component.
  - D3 fam − know +0.056 [+0.024, +0.089]. D4 entropy know − fam +0.003.
  - cos(fam, abst) = 0.948. Regex overstates relative to the judges (fam: regex 0.233 vs judges 0.100).
- **Gemma-9B** (k* = 4):
  - fam 0.044 vs random ~0.003 (regex 0.271).
  - D1 +0.042 [+0.022, +0.063]: NOT MET. D2 +0.017: carried. D3 +0.002 (n.s.).
  - D4 entropy know − fam +0.659 [+0.577, +0.741]. cos(fam, abst) = 0.631; abst fit on only 39 abstaining items.
- No degenerate outputs in any condition.
- **Consequence:** C4 is downgraded. The steering result becomes a negative or weak result: the obscure-minus-popular direction acts mainly through a generic abstention direction. The Gemma entropy dissociation (C5, entropy half) survives.
- **Pending:** the secondary majority-of-3 scorer (70B judge of Gemma replies running on C).

## 2026-10-10 ~03:30 UTC | Round-3 results folded into the paper (draft_v4.pdf)

- **Infrastructure:**
  - Boxes B and C expired before E3's raw outputs were copied; E3's base-model runs were redone on A and D and reproduced box B's numbers exactly.
  - E's 70B Gemma scoring finished and was pulled.
  - F's descriptive instruct lens runs were abandoned (two overlapping uploads were rejected by the checksum guard; descriptive per the criterion).
  - Lesson: pull every output as soon as it exists.
- **E3 final (local lens.csv):** HEDGE-SIDE 4/4, with values as at 22:45.
- **E4 secondary scorer (majority of 3 incl. Llama-70B; amendment 7bed3826):**
  - Llama fam − random +0.173 [+0.128, +0.221]; fam_perp +0.017.
  - Gemma +0.114 [+0.078, +0.152]; fam_perp +0.049.
  - The primary (two judges) bar stays not met; the abstention-orthogonal part is small under both scorers.
- **Paper (tex/main.tex, numbers.tex 360 macros):**
  - Body rewritten. Deciding cell up front; independent labels, third judge, guessability and the one-prompt check in §3.1.
  - New §3.2 on families (Gemma neither; Qwen/Mistral unmeasured). §3.3 mechanistic tests reported as negative/partial (steering via the abstention component; base hedge term popularity ≈ knowledge).
  - New Fig. 2 (judge-scored, held-out) and Fig. 3 (window effects, hedge term).
  - Appendices A-H incl. surprisal negative (E2) and all round-3 hashes.
  - Chair camera-ready items 1-8 and audit C1/C2 addressed; title still (b) pending [user].
- **Checks:** 4 pages main; 0 errors, 0 undefined, 0 overfull; check_numbers 0 unsupported (35 design literals); prose gate 3 advisory (2 passives in Limitations, ANSWER label).

## 2026-10-10 ~04:30 UTC | E6 entity-level knowledge (sha 937b46e3)

- **Request:** [user] asked to run it ("If we boxs do entity level knowledge covariate").
- **Wikidata descriptions:** fetched for 2,662 of 2,673 items. A fetch bug (a failed batch reused the previous response) was fixed and the fetch re-run before use.
- **Probe:** Llama "{subj} is …" (greedy + 5 samples), graded by Llama-3.3-70B against the description; 15,972 grades.
- **Result: K1 SURVIVES ENTITY KNOWLEDGE.**
  - entity_known tracks page views: Spearman +0.516.
  - Two-judge obscure coefficient +0.355 → +0.337 [+0.266, +0.411] (−5%); regex +0.617 → +0.597 (−3%).
  - entity_known coefficient −0.027 [−0.057, +0.004].
  - Obscure-known entity_known 0.831 > popular-known 0.750: the model knows what these obscure entities are, yet abstains.
- **In the paper:** a §3.1 robustness sentence, an abstract clause, an App. A row and paragraph, and the hash in App. H. draft_v5.pdf, 4 pages main, clean compile.
- **Also launched (exploratory, future work, not in the paper):** full pipeline + 3 judges for Llama-3.2-3B-Instruct (F), OLMo-2-1124-7B-Instruct (D) and Phi-3.5-mini-instruct (A).

## 2026-10-10 | Title switched

- [user] "switch": the title is now "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge".
- Reason: area chair condition 1; the E2 familiarity measure failed.

## 2026-10-10 06:54 UTC | Boxes A, D, E, F expired

- The exploratory other-model runs (Llama-3.2-3B, OLMo-2-7B, Phi-3.5-mini) were lost mid-pipeline; their outputs had not been pulled.
- No paper data were lost: E1-E6 were pulled and checksum-verified locally earlier.
- **Lesson** (also for the skill): on time-limited sandboxes, sync partial outputs every 15 minutes, not only at completion.
- **Open:** rerun the exploratory models if [user] provides boxes (future work, not in the submitted paper).

## 2026-10-10 | Review round 3 (on main.pdf, commit f723070): interim

- **Referee recommendations:** methods minor, framing minor, clarity minor; no reject.
- **Adjudication:** 0 critical; 5 partial, 1 unresolved (name ambiguity).
- **Numbers audit:** 370 items; no altered or stale value; all 13 hashes match. 2 critical wording issues:
  - Gemma's pre-registered forced-readout verdict REPLICATES is unreported;
  - the base-model claim holds only for the hedge term.
- **Post-review checks (post hoc; analysis/round3_label_checks.*):**
  - Popular-unknown audit: 48% of popular-unknown questions have a gold alias in some free reply. Dropping them gives contrast +0.189 [+0.069, +0.307] (from +0.300).
  - Original vs paraphrase labels: κ 0.788, agreement 0.913. Contrast on agreeing items +0.314 [+0.147, +0.480].
  - The gold alias appears in 0% of both-judges-DONT_KNOW replies in the obscure cells.
- **Ambiguity check** (Wikidata homonym counts, 387 items): running, slowed by rate limits.
- **Date correction:** criterion notes E1-E5 and the E4 amendment say "2026-10-10" in their text, but the UTC hash records in PREREG/HASHES_round3.txt read 2026-10-09T21:28Z-22:02Z (my notes used IST dates). The recorded UTC timestamps are authoritative and precede every corresponding run. The notes are left unedited so their hashes stay valid.

## 2026-10-10 | Review round 3 decision

- **Area chair: ACCEPT, conditional on camera-ready items 1-8** (review/round3/chair_decision.md).
- **Referees:** minor, minor, minor.
- **Open:** apply items 1-8 (revise round 3); ambiguity covariate (9d) running; footer shows the main-track "Submitted to ... NeurIPS 2026" text, so check the workshop option.

## 2026-10-10 (lock-in batch, boxes 1-6)
- E7 (PREREG e7, sha 1498be91), box 1, judge Llama-3.3-70B. Semantic relabel of the 1,540 pool Unknown items: 1,193 stay unknown, 306 intermediate, 41 become known (36 of them popular). In the 2x2, popular-Unknown: 60 stay, 9 known, 31 intermediate; obscure-Unknown: 86 stay, 2 known, 12 intermediate. T7a two-judge contrast with semantic labels +0.234 [+0.119, +0.348] -> SURVIVES SEMANTIC RELABELLING. T7c DECLINE-rate contrast +0.429 [+0.312, +0.542] -> STRICT-SURVIVES; HEDGED_ANSWER share among both-judge DONT_KNOW replies 0.007 / 0.000 / 0.025 (obscure-unknown / obscure-known / popular-unknown). Note: the 70B DECLINE label is not stricter in rate than the two-judge rule (cells 0.77/0.66/0.23/0.00); what it adds is that DONT_KNOW replies almost never name a candidate. results/round3/e7_summary.txt.
- E9 (PREREG e9, sha f6459163), box 3. Permissive prompt. Llama: Fleiss kappa 0.709, pop-know +0.470 [+0.349, +0.587] -> FOLLOWS POPULARITY. Gemma: kappa 0.670, pop-know +0.297 [+0.190, +0.400] -> FOLLOWS POPULARITY; obscure-unknown 0.537 >= 0.25, so per the licensing rule the Gemma verdict replaces "cannot be estimated", labelled as a permissive-prompt result. results/round3/e9_summary.txt.
- Ambiguity check (post hoc, analysis/ambiguity_check.txt): popular items have more Wikidata homonyms (median 11-13) than obscure-known (1); adding log homonyms leaves the obscure coefficient at +0.340 [+0.267, +0.412] (from +0.355).
- E8 first run INVALID (my error): k* values from E4 were swapped when launching (Llama run at k=4, Gemma at k=1; E4's k* is Llama 1, Gemma 4). Symptom: Llama random directions produced degenerate text that the judges scored as abstention (random majority 0.51-0.88), Gemma showed no effect in any condition. Outputs kept in results/round3/e8_INVALID_swapped_kstar/ and not used. Relaunched with Llama --kstar 1 (box 2) and Gemma --kstar 4 (box 5); criterion unchanged.
- E10 (PREREG e10, sha 6af179ce), box 4. Rule for "instruction tuning increases the popularity preference in the lens readout": not met for any pair -> the paper says the lens comparison does not show an increase. Descriptive (no bar, reported as such): instruct readouts move toward knowledge, not popularity. pop - know, base vs instruct: Llama-8B full readout +0.31 vs -0.35, hedge term -0.01 vs -1.46; Gemma-9B full +0.01 vs -0.98, hedge -0.02 vs -1.28; Gemma-2B full +0.29 vs +0.15, hedge -0.27 vs -0.89. So the behavioural popularity preference of the instruct models is not visible in their window lens readout at the forced-answer prompt. results/round3/e10_summary.txt/.csv.
- Camera-ready write (2026-10-10, after round-3 ACCEPT): main.tex rewritten against chair items 1-8 with item-9 results (a: E7 regrade; b: consensus labels; c: strict declines; d: namesake covariate; e: E10 instruct lens; h: E9 permissive prompt; f: E8 pending). Main text kept to 4 pages by moving Fig. 3 (lens) to App. D and folding Related work into the Introduction. Footer fixed (UniReps notice), Fig. 1 fixed (no clipping, instruct names, popularity hue distinct from unknown), lens tokens listed (hedge: not/I/unknown/unclear/no; answer-opening: a/the/that/in/Yes/yes; the old text said "the answer token", which was wrong), onset D intervals added, App. F reports both surprisal outcomes (chair N6 checked: point-biserial with arm +0.02, partial -0.06; unresolved, stated as such), App. G names both circuit directions, App. H adds ae0bb807 and the E7-E10 hashes and notes the E4-amendment dating error. New post hoc tabulations: analysis/lockin_checks.py (known-arm intervals, bootstrapped kappa, judge-majority agreement, counts, steering overlap), analysis/surprisal_check.py. Source frozen at revision/round3/main.write_v1.tex.
- humanize ran on main.tex (main text): S13 clause-stacked sentences split in Intro, Results (labels, permissive prompt), steering; no vocabulary cluster found; no em dashes. verify_rewrite revision/round3/main.write_v1.tex main.tex: first run FAIL (dropped hedge "might" in the Intro rewrite), restored, rerun PASS (0 dropped, 0 added, 0 warnings). check_numbers: 17/17 specific literals found in results, 0 missing. verify_citations --online: all verified except lindsey2025biology (web page, as before). Disclosure: this pass edits style only; AI assistance must be disclosed per the venue policy at submission.
- E8 (PREREG e8, sha 05474954), corrected-dose rerun (Llama k=1, box 2; Gemma k=4, box 5), primary scorer majority of 3. Llama: abst_within fitted on 587 vs 55 obscure questions; cos(fam, abst_within) +0.91; fam - random +0.17 [+0.12, +0.22]; fam_perp_w - random +0.01 [+0.00, +0.03] -> CARRIED BY ABSTENTION. Gemma (secondary): cos +0.47; fam - random +0.10 [+0.07, +0.14]; fam_perp_w - random +0.03 [+0.01, +0.04]. No degenerate text. Added to Sec. 3.3 (one sentence) and App. C, with the discarded swapped-dose run disclosed. results/round3/e8_summary.txt.
- Page budget after E8: moved the guessable-answers sentence to Table 1 only, merged the steering opener, Fig. 1 height 1.52 in, Fig. 2 height 1.5 in (y-label shortened to avoid clipping). Main text ends on p.4.
- Exploratory OLMo-2-7B-Instruct (future work, not in the paper; box 6): 348 questions (obscure known only 48), two-judge kappa 0.24, Fleiss 0.45 (below 0.60 bar -> unmeasured by the E5 rule); majority-rule cells OU 0.21, OK 0.04, PU 0.06, PK 0.00; pop - know -0.02 [-0.08, +0.04]. Like Qwen/Gemma/Mistral under the plain prompt, it rarely abstains. analysis/explore_olmo.txt.
- 2026-10-10: \hashCircuit corrected from 1c99afd4 to 3e1932a9. The old value was the hash of a CRLF Windows checkout of l5_behavioral_criterion.md; the note as committed before the run (59f5ec5) hashes to 3e1932a9. Criterion notes are now marked -text in both repos so line endings cannot change their bytes. Data repo committed in 7 groups on new branch emo_attyachar_2.0 (from l0-fixes) and pushed to origin.
- 2026-10-10: CLAIMS.md updated after round 3 and E1-E10 (no rows deleted): C1 strengthened; C2 wording narrowed; C4 retired (R6), C10 retired (R7); C5 equivocal; C6 narrowed; new C12-C17; withdrawn round-2 wordings recorded as R8-R10. New WHAT_CHANGED.md for collaborators (timeline, review asks and responses per chair item, experiments, claim movement, mistakes fixed, open items). REVIEWING.md and the repo README point to both.
- 2026-10-10, revise round 4 (external readability review pasted by the user; raw text in revision/round4/REVIEWS_RAW.md): 21 comments, 18 fixed, 2 partial (hash table kept as a table; onset intervals kept with a caption note), 2 push-backs with evidence (Brzezinka 2026b ID verified; the two +0.34 rows are different models, 0.337 vs 0.340). Factual error fixed: OLMo hedge-term popularity +0.61 > knowledge +0.53, so the lens sentence and the Fig. 3 title now say "about as much" and name the OLMo exception. Provenance unified as "pre-specified" (one rule in Setup); App. G (circuit) cut; App. C lab narration cut; new Table 1 (2x2 cells with n; analysis/lockin_checks.py (1b)); Table 2 restructured with a Plan column; bib author lists > 10 truncated; the steering figure moved to App. C to keep 4 pages. Checks: check_numbers 16/16 found, 0 missing; verify_citations 33 verified (2 benign mismatches, 1 web page); verify_rewrite drops traced to moves or deliberate cuts, and 2 accidental drops (+1.40 entropy check, n = 73) restored in App. A. Response: revision/round4/RESPONSE_LETTER.md.
- 2026-10-10: style target set to Neel Nanda's published writing advice (AUTHOR_PROFILE.md; sources: his essay "Highly Opinionated Advice on How to Write ML Papers" and the structure of Ferrando et al. 2025, Arditi et al. 2024, Stolfo et al. 2024; no text copied; not the author's own measured voice). Rewrote the main text as main_nanda.tex (abstract in context-problem-contribution-evidence-standard order, intro as summary with a contributions list, finding-first headings and bold run-ins, limitations before implications). verify_rewrite main.tex -> main_nanda.tex: no fact dropped or added (one combined \cref split). Both drafts fit 4 pages. Comparison and recommendation: COMPARE_VOICE.md. main.tex unchanged; which draft goes forward is an author decision.
- 2026-10-10: author voice measured from three of the author's papers (TAIS 2026 camera-ready, first author: primary, mean 17.3 words per sentence, CV 0.66; arXiv 2605.18971 and 2509.02133, middle author: secondary). AUTHOR_PROFILE.md now has the voice first and the Nanda structure second. main_nanda.tex rewritten in that voice with the two recommended fixes (Results paragraph split; Setup trimmed): Title-Case colon headings and run-ins, "Main Contributions." numbered list, "We investigate / We find", American spelling. Results mean sentence 27.7 -> 17.9 words, passive 21% -> 9%. verify_rewrite against revision/round4/main_nanda.v1.tex: no number, citation or claim changed (flags explained in COMPARE_VOICE.md); one "suggest" that would have weakened a direct claim replaced by "mean". Fig. 1 height 1.52 -> 1.42 in (both drafts; no clipping). analysis/voice/voice_stats.py added.
- 2026-10-10: Review round 4 ran on the voiced main.tex (packet review/round4/; 3 blind referees, red-team pair, zero-context numbers audit, chair). Recommendations: methods, framing and clarity all major revision; chair ACCEPT conditional on 17 must-fix items; no critical issues. Kill argument: the obscure-known cell may be cue-based guessing (adjudication: 1 unresolved, 5 partial). Post hoc checks run during the review (analysis/round4_checks.txt, echo_check.txt): obscure-known is 92% country/capital/sport; preference relation-clustered interval [-0.10, +0.43]; all label corrections jointly +0.14 [-0.005, +0.28]; popularity gap among unknown facts +0.34 [+0.25, +0.43] (relation-clustered [+0.23, +0.46]; after all corrections +0.18 [+0.04, +0.31]). Numbers audit: 2 critical (four-family test readout misdescribed; App. A alias shares count subject-name echoes: 0.53 -> 0.25, 0.72 -> 0.44, 24 -> 9 of 87). Report: review/round4/REVIEW_REPORT.md. Open for the authors: narrow the headline or run the name-only baseline; which new experiments, if any.
- 2026-10-10 ~12:30 UTC, E11 name-cue control (PREREG e11, sha f716d396; box 7): 183 known facts probed with 3 pseudo-names each (4 untestable). Cue-recoverable (greedy correct for >= 2 of 3 pseudo-names): obscure known 34 of 86 (0.40; country 25/46, capital 2/27, sport 4/7, composer 3/3), popular known 11 of 97. Cue-recoverable obscure-known facts abstain LESS (0.33) than the rest (0.50), the opposite of the red-team prediction. Dropping them: preference +0.36 [+0.25, +0.48], n obscure known 53 -> T11 SURVIVES NAME-CUE CONTROL. Strict variant: +0.37 [+0.26, +0.47]. Relation-clustered interval [-0.03, +0.48]: the remaining obscure-known facts are mostly capital (25) and country (21), so relation-level uncertainty must be disclosed. Licensing: the title may keep "more than knowledge" with the control cited. results/round3/e11_summary.txt.
- 2026-10-10 ~12:45 UTC, E12 permissive prompt for Qwen and Mistral (PREREG e12, sha 56831ebc; box 8): both UNMEASURED by the pre-set rule (Fleiss kappa 0.40 Qwen, 0.50 Mistral < 0.60), although both now abstain often and the point estimates lean toward popularity (Qwen +0.15 [+0.004, +0.30]; Mistral +0.16 [+0.05, +0.27]). The paper may not claim a preference for them; it may report the estimates as descriptive with the judge-agreement failure. results/round3/e12_summary.txt.
- 2026-10-10: launched E13/E14 (box 7; PREREG fbbc8c25: fresh replies; new adjacent-band questions), E15 (box 9; OLMo-2 permissive prompt) and E16 (box 10; steering with a popular-arm abstention direction, 20 random directions; PREREG 2b3e665d).
- 2026-10-10 ~13:00 UTC, E15 OLMo-2-7B permissive prompt (PREREG 2b3e665d; box 9): UNMEASURED (Fleiss kappa 0.46 < 0.60); point estimate -0.09 [-0.22, +0.03], leaning toward knowledge; cells 0.89 / 0.60 / 0.70 / 0.38 (OLMo abstains heavily even on popular known facts under this prompt). With E12, the permissive-prompt preference is measured only for Llama and Gemma; Qwen and Mistral lean toward popularity but are unmeasured, OLMo leans the other way and is unmeasured. results/round3/e15_summary.txt.
- 2026-10-10: E17 (exploratory NLA readout of Qwen2.5-7B, PREREG 2d6712e1; box 8) running. First run stopped on the reference package's own check (apply_chat_template returned a dict in the newer transformers); a list-returning tokenizer proxy fixed it with no change to the template, injection or scale; explanations read as coherent English about the question.
- 2026-10-10 ~16:35 UTC: boxes 7-10 expired (about 4 h after start). Read-offs on the synced outputs:
  - E13 fresh replies (seed 1), same 387 questions: REPLICATES, +0.29 [+0.20, +0.38]; cells 0.45 / 0.41 / 0.12 / 0.00; kappa 0.61.
  - E14 new questions from the 10-30th and 70-90th percentiles (5,182 labelled, 400 sampled, 100 obscure-adjacent known): REPLICATES IN NEW QUESTIONS, +0.10 [+0.01, +0.20] (smaller, as predicted); popularity +0.25, knowledge +0.15; relation-clustered [-0.12, +0.27]; obscure-adjacent known again 85% capital/country/sport, so the concentration is structural (the model knows obscure facts mainly in these relations).
  - E16 steering with an abstention direction fitted within the popular arm (63 vs 548), 20 random directions: GENERIC ABSTENTION. cos(fam, abst_pop) 0.89; abst_pop - random +0.14 [+0.10, +0.19]; fam - random +0.17 [+0.12, +0.21]; fam_perp_p - random +0.007 [-0.003, +0.019]; no degenerate text.
  - E17 (exploratory NLA readout) LOST: the explanations finished on box 8, but box 8's sync list still pointed at E12's folder (my error), so nothing was copied before expiry. Redoing it needs about 1 h on a new box. No E17 result exists; nothing about it goes in the paper.
- 2026-10-10 (revise round 5): decision recorded before editing: keep the comparative claim because E11 (name-cue control) survives, but lead the abstract with the popularity gap among unknown facts and state the joint label correction (+0.14 [-0.01, +0.28]) and the measure range (+0.13 to +0.44); title kept pending the author's scope decision. Gold-alias counts corrected for subject-name echoes (0.53 -> 0.25; 24 -> 9 of 87). Added E11-E16 to Table 2, App. B, App. C and App. G; guessable count after filtering (30 of 87) in App. A.
- Mode revise ran on 2026-10-10; produced main.tex, numbers.tex, revision/round5/RESPONSE_LETTER.md; decisions: comparative headline kept with E11, abstract reordered; open: I5 pooled-knowledge analysis, I17 crosstab, I18 matched unknown pairs, I16 human labels, title scope.
- 2026-10-11: Appendix G (pre-specified analyses: plan files and SHA-256 prefixes) removed at the author's request; the Setup and App. C pointers to it were dropped. The plans and hashes remain in PREREG/ and PREREG/HASHES_round3.txt. Backup: revision/round5/main_before_rm_appG.tex. PDF 13 -> 12 pages.
- 2026-10-11: clarity edits at the author's request: abstract 43%/0% sentence restructured (same facts); Fig. 2 (steering) and Fig. 3 (lens) captions rewritten to explain bars, markers, dose and the judge-agreement caveat. verify_rewrite: 0 dropped; additions are facts already in the figures (doses 1x/4x). Backup revision/round5/main_before_clarity.tex.
