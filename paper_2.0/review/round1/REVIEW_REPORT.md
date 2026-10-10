# Review report: "Saying 'I Don't Know' Tracks Familiarity…", round 1

Manuscript version: draft_v2.pdf (2026-10-10, after humanize)
Venue assumed: UniReps 2026 extended abstract (4 pages main text). Venue criteria source: https://unireps.org/2026/call-for-papers (checked 2026-10-10). No public reviewer form.
Date: 2026-10-10
Reviewer agents: three blind reviewers (methods and statistics; framing and novelty; clarity and fresh reader), one chair, one red-team pair, one zero-context numbers auditor.
Independence note: all agents are the same model family as the author agent. Role separation limits shared context but does not give independent error processes. No seat ran on a different model.
Stripped from the packet: nothing (the PDF carries no author notes).
Raw files: reviewer_methods.md, reviewer_framing.md, reviewer_clarity.md, chair_merge.md (full 57-issue table, disagreements D1-D9, chair notes), attack_memo.md, adjudication.md, numbers_audit.md (140 items), mechanical.txt, guessability_check.py/.txt.

## 1. Kill-argument result

The attack memo (verbatim in attack_memo.md) makes two arguments:
- **Circularity:** "entropy follows knowledge" restates the labelling rule, because knowledge labels and entropy come from the same forced-answer distribution.
- **Guessability:** obscure "known" facts may be guesses (Mallen et al. 2023 §4.2, which the adjudicator verified), so the +0.39 popularity effect is labelling error.

Adjudication (adjudication.md): five points, 0 answered, 3 partially answered, 2 unresolved, none critical.

| point | accusation | verdict | severity if unresolved | fix |
|---|---|---|---|---|
| P1 | Entropy and knowledge labels share one forced-answer distribution, so "entropy follows knowledge" is near-tautological | unresolved | major | State it; present entropy as a positive control, or relabel knowledge from an independent prompt |
| P2 | Obscure "known" facts may be guessable; only 55 pairs and a spelling regression address it | unresolved | major | Guessability control on the full pool |
| P3 | The popularity effect is labelling error | partially answered (the unknown-arm gap 0.48 vs 0.13 cannot come from mislabelled guesses, but the paper never says so) | major | Report the unknown-only popularity gap |
| P4 | One model, 387 questions | partially answered | minor | Scope the title |
| P5 | Judges never checked against humans | partially answered | minor | Report regex precision; discuss the +0.13 row |

**Post-review analysis on existing outputs (exploratory, post hoc; guessability_check.txt).**
- Definition: an answer counts as guessable if a gold alias appears in the subject's name, or if it is the relation's modal answer covering at least 20% of the pool (relations with such an answer: sport, country, occupation, religion, color).
- Prevalence: 30 of 87 obscure-known items (34%), against 4-9% in the other cells.
- Two judges, all items: popularity +0.388, knowledge +0.089, difference +0.300 [+0.204, +0.398].
- Two judges, non-guessable items only (n = 335):
  - popularity +0.373 [+0.307, +0.440], knowledge +0.074 [+0.008, +0.140], difference +0.300 [+0.192, +0.411];
  - the obscure-known cell still abstains 0.439 (n = 57).
- Unknown-only popularity gap: +0.343 [+0.257, +0.430] on all items, +0.308 [+0.216, +0.398] on non-guessable items.
- Regex measure on the full pool (2,106 non-guessable items): difference +0.480 [+0.361, +0.592].
- Reading: the guessability alternative predicts that the effect shrinks once guessable items are removed. It does not shrink. P2 and P3 are answerable with this analysis; it is not yet in the paper.

## 2. Overall assessment (chair, abridged)

No referee recommends rejection. Methods and framing recommend major revision; clarity recommends minor revision, and would move to major if Llama's forced-prompt numbers show a knowledge effect at least as large as the popularity effect (they do; see §6).

All three referees:
- value the popularity × model-certified knowledge design and the result 0.43 vs 0.00 on known facts;
- credit the honest reporting of negative results;
- find that the title, abstract and conclusion claim more than the evidence supports. Abstention is measured in one model, the steering effect runs mostly through abstention-associated components, and the base-model lens readout moves with knowledge as much as with popularity.

The framing referee also finds uncredited prior work: Lindsey et al. already state the misfire hypothesis, Ferrando et al. already steer refusal and trace it to the base model, and Kumaran et al. 2026 is contrary evidence. I verified Kumaran et al. 2026 (arXiv 2603.22161) and Ni et al. (arXiv 2505.17537, EMNLP 2026) on their arXiv pages on 2026-10-10.

## 3. Must-fix items (chair roadmap; full table in chair_merge.md)

| id | issue | cost | status after this round |
|---|---|---|---|
| I1 | Title, abstract and conclusion state a one-model abstention result generally | rewording | open: needs your title decision |
| I2 | The prompt-confound defence gives Gemma only. Llama's forced-prompt hedge log-odds are popularity +2.93 vs knowledge +3.05, difference −0.12 [−0.72, +0.45] | existing outputs + rewording | open: must be reported honestly |
| I6 | Guessability of obscure-known facts | existing outputs | **analysis done** (§1); needs writing in |
| I10 | Knowledge labels and entropy come from the same distribution | new GPU run, or reframing | open: decision |
| I11 | Alias-grading noise; no human audit | new data | open: you declined human checks on 2026-10-09 |
| I7 | Entropy also tracks popularity among unknown facts (~0.9 nats); state relative weights | existing outputs + rewording | open |
| I3 | "Rules out name spelling" contradicts Appendix A (0.76 → 0.43) | rewording | open |
| I12 | "Popularity" (measured) vs "familiarity" (construct) | rewording | open |
| I8 | The abstract and the Fig. 2 title oversell steering | rewording | open |
| I4 | The base-model lens readout carries knowledge too; "the familiarity signal predates instruction tuning" overstates it | rewording + existing outputs | open |
| I5 | OLMo's late-layer reversal is undisclosed | rewording | open |
| I13-I15 | Credit Lindsey (misfire hypothesis); position (ii)/(iii) against Ferrando and Brzezinka; cite and reconcile Kumaran 2026 | rewording | open |

Numbers-audit criticals to fold in:
- **N1:** "steering … use one prompt" is false; steering entropy is measured at the forced-answer prompt.
- **N2:** "emerge together at 0.45-0.74" overclaims. The verdict was "mixed"; OLMo D = −5, Gemma-2-9B D = +1; 0.74 is Qwen-instruct, which is not plotted.
- **N3:** "rarely abstain" depends on the measure (both-judges 0.05-0.10; with CLARIFY 0.22-0.32).

## 4. Should fix (chair) and consider

- **Should fix:**
  - I9: dose chosen on the evaluation questions;
  - I16/I17: a steering control that separates the accounts (new experiment);
  - I18: steered replies' quality;
  - I19: the lens readout may reflect answer-token rarity;
  - I20: define the lens;
  - I21 = N1;
  - I22: name the eight models and the onset rule;
  - I23: one cross-model analysis, to test the closing prediction on existing per-item data.
- **Consider:** about 30 minor items (counts "three/four/five measurements", undefined "corrected" row, CLARIFY undefined, cell-size table, Fig. 1 legend overlap, "Brzezinka distinguish" → "distinguishes", and others). See chair_merge.md.

## 5. Disagreements (chair)

D1-D9 are in chair_merge.md. The main ones:
- **D2:** whether +0.04 can be read as "the size of the familiarity component". Favours methods: no, the control conditions on abstention.
- **D3:** the Lippl sentence. Favours framing: by Lippl's criterion the abstention variable is a correlate of uncertainty, not a representation of it.
- **D4:** whether OLMo dips inside the window. Unresolved from the figure; the per-layer values settle it.

## 6. Numbers audit (zero context)

Counts: exact 76, rounding-ok 41, mismatch 15, no-evidence-found 8. All 83 macros reproduce their result files. The mismatches are wording, scope, counts and configuration (N1-N3 above).

Other mismatches:
- Llama's forced-prompt values were omitted (selected value);
- measurement counts are inconsistent;
- d_nohedge_U is not named (d_nohedge_both gives +0.06);
- "full Llama pool" means 2,303 of 2,673;
- "7-9B" ignores Gemma-2-2B;
- "every layer" is wrong (the curves stop before the last layer);
- the regex recall for the three families is not reported, which the criterion requires;
- the instruct mid-depth values are not reported, which the lens criterion requires.

No evidence in the permitted files:
- 10,000 bootstraps outside the hedge measure (the lens criterion uses 2,000);
- the criterion hashes;
- the individual random-direction dots.

## 7. Mechanical checks (raw output)

See mechanical.txt.
- check_numbers: 83 macros used, 18 literals (design settings), 0 unsupported.
- verify_citations (offline): no mismatches.
- prose_gate: 4 advisory findings.

## 8. Revision roadmap (feasible before 11 Oct 12:00 UTC)

| item | issue ids | obligation | location | cost | depends on |
|---|---|---|---|---|---|
| R1 | I1, I12, I8, I4, N3 | must | title, abstract, §1, §5 | rewording | title decision (§9) |
| R2 | I2, N1, I21 | must | §2 two-prompt paragraph → end of §3.1; appendix table of forced-prompt log-odds for all four families | existing outputs + rewording | none |
| R3 | I6, I3, P3 | must | §3.1 one sentence + App. A paragraph + guessability numbers (new macros from a committed script) | existing outputs (done) | none |
| R4 | I10 / P1 | must | §2 + §3.1: state the shared distribution; present entropy as the expected readout and abstention as the readout that departs from it | rewording (or GPU relabel, §9) | decision |
| R5 | I13, I14, I15, I7 | must | §1, §4: credit Lindsey, Ferrando, Brzezinka; add Kumaran, Ni | rewording + 2 bib entries | none |
| R6 | I5, N2, I22, I28 | must | §3.3, Fig. 3 caption, App. C | rewording + per-model onsets in the appendix | none |
| R7 | I23 | should | §4/§5 + appendix: cross-model per-item agreement of popularity effects (cross_model_items*.csv) | existing outputs | time |
| R8 | I9, I16-I19, I11 | should | — | new experiments / data | GPU boxes; state as limitations otherwise |
| R9 | minor items | consider | various | rewording | page budget |

## 9. Decisions for the user

| item | options | cost of each | user decision | reason |
|---|---|---|---|---|
| Title | (a) keep; (b) "In Llama, Saying 'I Don't Know' Tracks Familiarity While Answer Uncertainty Tracks Knowledge"; (c) a question form | (b)/(c) cost nothing; (a) leaves I1 open, raised by all three referees | | |
| P1 circularity | (a) reframe entropy as the expected, label-linked readout (wording only); (b) relabel knowledge from an independent prompt (GPU, about 2-4 h per model; boxes may be gone) | (a) concedes part of the title's second half; (b) risks the deadline | | |
| Llama forced-prompt result (I2) | report it honestly: at the forced-answer position Llama's hedge readout weighs popularity and knowledge equally, so for Llama the divergence appears only when the model may decline | wording; weakens the prompt-confound defence | | |
| New steering controls (I16/I17), dose sweep (I9) | run on GPU / list as limitations | GPU time vs a limitation sentence | | |
| Cross-model analysis (I23) | run on existing per-item data / skip | about 1 h; strengthens venue fit | | |
| Human audit (I11) | keep the 2026-10-09 decision (no) / reopen | — | | |

## 10. Chair notes

See chair_merge.md "Chair notes" (C1-C6), including:
- C2: the base and instruct model sets in §3.3 differ from the eight in the criterion;
- C5: Appendix C's 12-layer lead uses a different contrast.
