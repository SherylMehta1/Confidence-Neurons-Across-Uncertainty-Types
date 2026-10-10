# Review report: "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge", round 4

Manuscript version: main.tex after the voice rewrite (commit 7ac25fc; previous version revision/round4/main_before_voice.tex)
Venue assumed: UniReps 2026 (NeurIPS workshop), extended-abstract track, 4-page main text, non-archival. Venue criteria source: the call as read in earlier rounds; nothing new fetched.
Date: 2026-10-10
Reviewer agents: three blind reviewers (methods and statistics; framing and novelty; clarity and fresh reader), one chair, one red-team pair (attack, then independent adjudication), and one zero-context numbers auditor.
Independence note: all agents are the same model family as the author agent. Role separation limits shared context but does not give independent error processes. No seat ran on a different model.
Stripped from the packet: nothing (the source has no author comments; the submission is anonymous). The chair also saw review/round3/chair_decision.md, only to check the round-3 conditions.

Full texts: `reviewer_methods.md`, `reviewer_framing.md`, `reviewer_clarity.md`, `attack_memo.md`, `adjudication.md`, `numbers_audit.md`, `chair_decision.md` (68 merged issues, 10 disagreements, roadmap). Post hoc checks run during the review on existing outputs: `../../analysis/echo_check.txt`, `../../analysis/round4_checks.txt`.

## 1. Bottom line

**The chair accepts, with 17 must-fix items and no critical issues.** All three referees recommend major revision; none recommends rejection. The data are not in question, but the headline's comparative half is. "Saying 'I don't know' tracks popularity" holds under everything we tried. "More than knowledge" rests on 87 obscure-known facts that may be guessed from the name or the relation. The paper cannot yet rule that out, and our own checks in section 3 make it worse.

## 2. Kill-argument result

Attack memo (verbatim, 249 words):

> The headline claim, that Llama's "I don't know" tracks popularity rather than knowledge, rests on one cell: obscure facts Llama "knows". That cell does not measure knowledge. "Known" means the greedy answer and 9 of 10 samples match a gold alias after a forced "The answer is" prefix (lines 55-58). Under that compulsion, an obscure question whose answer follows from the relation's prior or the name's language yields consistent correct guesses, and the manuscript's own numbers show this. Only 87 of 1,351 obscure facts qualify (line 246). A paraphrased prompt keeps just 27 of them as known (lines 90-91); the reported kappa of 0.79 comes mostly from 169 questions unknown under both prompts (lines 259-260). In free generation, only 0.53 of obscure-known replies contain the answer, and on 24 of the 87 questions Llama both answers and declines (lines 292-294). Its abstention on these "known" facts, 0.43, is indistinguishable from its 0.48 on facts it never answers (Table 1). The meaning-based regrade covered only unknown facts (lines 266-267). The guessability filter was chosen post hoc, with arbitrary cutoffs that ignore cues in the name (lines 283-285). The "similar entropy" offered as evidence of equal knowledge (lines 8, 83) comes from the same forced samples that define the label, so it is low by construction (lines 62-64). Because the preference equals obscure-known minus popular-unknown abstention (lines 75-76), its simplest reading is that Llama correctly declines guesses the labeling rule miscounts as knowledge. Nothing in the paper rules that reading out.

Adjudication: none of the six points is answered by the current text, five are partially answered and one is unresolved.

| point | accusation | verdict | gap | severity |
|---|---|---|---|---|
| P1 | The relation's modal answer can yield consistent correct guesses counted as "known" | partially answered | the guessability filter is post hoc; its 20% cut is unjustified; the post-filter n is not shown | major |
| P2 | Cues in the name (language, form) can yield "known" labels | **unresolved** | no test: the paraphrase keeps the name, the regrade covers only unknown facts, and recognition is graded against descriptions the name reveals | major |
| P3 | Labels are unstable, and kappa overstates agreement | partially answered | the result holds on stable labels; kappa covers only 253 of 387 questions (the denominator is unstated) | minor |
| P4 | Free-generation data look like "unknown" | partially answered | the reply data cannot separate the readings; the 0.53 vs 0.03 conflict is unexplained | major |
| P5 | The entropy comparison is circular | partially answered | admitted at l.62-64 but still used as contrast in the abstract and l.83 | minor |
| P6 | The headline rests on one cell | partially answered | the misfire half does not (unknown-facts gap +0.34); "more than knowledge" does | major |

The adjudicator also corrected the memo: it omitted that popular known facts drop almost as much under the paraphrase (35 of 100), and kappa already corrects for chance agreement.

Net assessment (adjudicator): "A senior reader would likely accept the narrower misfire result... but not the title and Contribution 1 as written. The paper survives at the workshop only with a narrowed headline or a new control."

## 3. Post hoc checks run during the review (existing outputs; not yet in the paper)

| Check | Result | Bears on |
|---|---|---|
| Relation mix of the obscure-known cell | 87 facts from 7 relations, **92% from three: country 46, capital 27, sport 7** | P1, P2, I8, I12 |
| Preference, relation-clustered bootstrap (resampling the 16 relations) | +0.30, interval **[-0.10, +0.43]** (the within-cell item bootstrap gives [+0.20, +0.40]) | I8 |
| Leave-one-relation-out | +0.24 to +0.35 (lowest without "capital") | I8 |
| **All label corrections jointly** (semantic regrade of unknown facts, popular-unknown questions with the answer in a free reply dropped, guessable facts removed) | **+0.14 [-0.005, +0.28]**; cells 0.47 / 0.43 / 0.29 / 0.00 | I14, P1, P6 |
| Guessability inside the obscure-known cell | 30 of 87 guessable; the other 57 abstain at 0.44 (guessable ones at 0.42) | P1 |
| **Popularity gap among unknown facts** (the label-robust result) | **+0.34 [+0.25, +0.43]**; relation-clustered [+0.23, +0.46]; after all corrections **+0.18 [+0.04, +0.31]** | I23 |
| Obscure-known free replies containing the gold answer, name echoes excluded | **0.25** of replies (App. A says 0.53), 0.44 of answering replies (says 0.72), 0.00 of abstaining replies; questions with both an answer and an abstention: **9 of 87** (says 24) | I4, I15; auditor's critical 2 |

**What this means.** The methods referee said: "if the jointly corrected, relation-clustered preference stays clearly positive and the guessing baseline is low, I would move to minor revision." On our own existing data it does not stay clearly positive (+0.14 with an interval touching zero; relation-clustered interval below zero). The popularity effect among facts Llama does not know stays clearly positive under every check.

## 4. Numbers audit (zero context)

209 items checked: 88 exact, 101 rounding-ok, 16 mismatches, 4 with no evidence found. All 16 hash prefixes match. The main Llama number and the Gemma permissive-prompt number are confirmed from the raw judge files.

Critical (each changes a smaller stated conclusion):
1. **The four-family test is described wrongly (Sec. 3.2, Table 3 caption).** The governing hedging readout was the free-reply regex for Qwen, Mistral and the Llama reference. On that readout Llama shows the preference (+0.44, REPLICATES). Only Gemma's verdict used the forced-answer log-odds, where Llama shows none (-0.12). The sentence "that readout shows no preference in Llama either" is true only of Gemma's readout. It came from the round-3 chair's suggested wording, but the source file contradicts it.
2. **App. A's free-reply numbers are stale:** the name-echo issue in section 3 above.

Minor: "that readout" points to the wrong analysis (Sec. 3.2); "about as much" is wrong for Gemma-2-2B; n = 387 should be 384 for the recognition row; "five" measures should be "six"; "43% of questions" should be "of replies" and the abstract's "known" omits the greedy-answer condition; "unknown" also needs a wrong greedy answer; "every interval is a within-cell bootstrap" is not true of the regression and matched-pair intervals; the Fig. 2 caption promises three-judge rates that App. C does not give; "cells of at most 100" holds only for abstention; "often collapse" should be "2 of 8". No evidence found: T = 0.7 for forced samples, the bootstrap seeds, "abstaining questions are mostly obscure", and the hedge/answer token lists (they are in the code, not in a result file).

## 5. Referee recommendations as given

- **Methods and statistics:** major revision (22 findings, 11 major). Drivers: unstable "known" labels (F1); cue-based guessing not excluded, with checks sharing the confound (F2); inconsistent obscure-known numbers, and the abstract hiding how much the estimate shrinks (F3, F22).
- **Framing and novelty:** major revision (19 findings, 6 major). The novelty holds as far as the referee checked (Ferrando 2025, Lindsey 2025, Brzezinka 2026a/b, Madhusudhan 2024, Che & Li 2026 and others opened). Drivers: familiarity framing the body withdraws (F1); a title broader than the extreme-decile design (F2); the headline resting on the weakest-labelled cells (F3); a thin UniReps answer (F4); permissive prompt not run on Qwen and Mistral (F5).
- **Clarity:** major revision, "accept once fixed" (35 findings, 6 major). Drivers: familiarity framing (F1); the "Within One Prompt" paragraph (F2); the 0.53 vs 0.03 conflict (F3); the +0.19 variant hidden in the appendix (F4); Gemma steering reported with the scorer the paper itself calls indefensible (F5); the convergence question left unanswered (F6). It found the main text "edited prose with one consistent, careful human voice", while the appendices "read as a lab audit log".

## 6. Revision roadmap (chair; full table in chair_decision.md)

**Must fix**, headline-related, all rewording or existing outputs except I11 and I16:

| id | change | cost | status after the section 3 checks |
|---|---|---|---|
| I23 | Lead the title, abstract and Contribution 1 with the label-robust popularity gap among unknown facts; make "more than knowledge" conditional on forced-answer labels | rewording | **strongly indicated**: the joint correction is +0.14, interval touching 0 |
| I11 | Name-only guessing baseline (fabricated or swapped names, same relation) | new experiment, cheap GPU run | if not run, I23 is mandatory and Limitations names cue-based guessing |
| I12 | Obscure-known n and relation mix after guessability; preference by relation-prior bins | existing outputs | relation mix computed |
| I13 | Obscure-known abstention on questions whose free replies name the answer | existing outputs | 9 of 87 such questions after echo exclusion |
| I4, I15 | Reconcile the gold-alias numbers; name reply sets and denominators | existing outputs | **diagnosed**: App. A counts name echoes; corrected 0.25 / 0.44 / 9 of 87 |
| I1, I6, I14 | State the label-check and measure ranges in the abstract and Contribution 1; one jointly corrected estimate | rewording + existing outputs | joint estimate computed: +0.14 [-0.005, +0.28] |
| I5 | "Known" is specific to the forced-answer prompt; pooled-knowledge variant | rewording + existing outputs | |
| I8 | Per-relation counts; relation-clustered intervals; leave-one-relation-out | existing outputs | computed: [-0.10, +0.43]; LORO +0.24 to +0.35 |
| I7, I19 | Reframe around popularity, not confirmed familiarity; scope the title to the extreme-decile design | rewording | |
| I2 | "Within One Prompt": lead with the pre-specified null; units, n, intervals and post hoc status for the replacement reading | rewording + existing outputs | |
| I18 | Matched-pair name-form analysis for the preference itself, beside the regression | existing outputs | |
| I16, I17 | Human validation of abstention labels; explain the corrected-judge vs regex gap (minimum: a Limitations sentence on cell-dependent misclassification) | new data (minimum: rewording) | |

Plus the auditor's critical 1 (four-family test description), a must-fix rewording.

**Should fix:** Gemma's "same preference" made conditional on the prompt, with odds-ratio forms and plain-prompt cells (Me F8); run the permissive prompt on Qwen and Mistral (Me F9, Fr F5; new experiment); rephrase "generic abstention signal" as "not separable here" (Me F10, Fr); report Gemma's three-judge steering numbers in the main text (Cl F5); answer the convergence question in two sentences or drop it (Fr F4, Cl F6); the remaining numbers-audit minors.

**Consider:** 45 minor items (terminology, Table 2 gaps, Figure 1 legibility, appendix voice, Wilson intervals for 0/300 cells, onset definition); see chair_decision.md.

## 7. Mechanical checks (raw output)

- `check_numbers.py`: all specific literals found in the result files, 0 missing (`mechanical.txt`).
- `verify_citations.py --online`: 33 verified; 2 benign mismatches (Huh et al. ICML "Position:" prefix; Rimsky now Panickssery, DOI matches); 1 unresolved web page (Lindsey et al.). Full output in `CITATION_REPORT.md`.
- `prose_gate.py --academic`: in `mechanical.txt`.

## 8. Decisions needed from the authors before revising

1. **Headline.** (a) Narrow it: lead with the popularity gap among unknown facts, with "more than knowledge" stated as conditional on forced-answer labels. This changes the title, abstract and Contribution 1. Or (b) keep it and run the name-only guessing baseline (I11) first, accepting the risk that it confirms the attack.
2. **New experiments.** Run any of: I11 (name-only baseline), the permissive prompt on Qwen and Mistral, a human audit of the abstention labels? Each needs a GPU box or people, and the deadline is 11 Oct 12:00 UTC.
3. **Title wording**, if (a): the title is the authors' call.
