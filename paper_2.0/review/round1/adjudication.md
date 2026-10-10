# Area-chair adjudication: round 1 attack memo

Manuscript: "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge" (UniReps 2026, extended abstract).
Sources used: manuscript.txt and manuscript.pdf (pages 2, 3, 10), plus Mallen et al. 2023 (arXiv 2212.10511v4, HTML full text) to check the memo's citation.

**Citation check.** The memo's description of Mallen et al. Sec. 4.2 holds up [verified]. The paragraph "Relationship types affects memorization" says that "certain relationship types (e.g., nationalities) allow models to exploit surface-level artifacts in subject entity names". It also says that models "often output the most dominant answer entities for questions about relationship types with fewer answer entities (e.g., red for the color relationship type)", and that for low-correlation relations "(e.g., country, sport) ... models may exploit surface-level clues". PopQA has exactly 16 relations, so "16 relations" (line 39) means the manuscript keeps all of them, color, country and sport included.

---

## P1. The entropy result is circular

- **Point:** "Known" means the greedy answer and at least 90% of T=0.7 forced-answer samples are correct. Entropy is measured on that same forced-answer distribution, so "entropy follows knowledge" largely restates how the facts were selected.
- **Verdict:** Unresolved.
- **Evidence:** The labels come from samples under the forced-answer prompt (lines 41-44), and entropy is the first-token entropy of the same prompt (lines 47-50). I found no passage that acknowledges this dependence. I checked Section 2, Section 3.1 (lines 76-78), Limitations (lines 131-132) and Appendix D. Line 29 ("as answer entropy does") presents the entropy pattern as an expected benchmark, but lines 6-7, 32-33 and the title present it as a finding. Two further features of the design make the contrast larger by construction:
  - Items with 1 to 9 correct samples are dropped (lines 43-44). That removes exactly the intermediate-entropy items.
  - Known items have their entropy pinned low, so popularity has little room to move it there. This alone favours "knowledge effect at least four times the popularity effect" (lines 77-78).
- **What is not circular:** The unknown arm (no sample correct) does not force high entropy, because a model could be confidently wrong. So the 3.5 / 2.6 nat levels in Fig. 1b carry some real information. The text does not make this argument.
- **Severity:** Major. Half of the title, and the four-family claim in the abstract, rest on a quantity that is partly fixed by the labelling rule. The abstention half does not depend on it. If anything, the circularity makes that half sharper: abstention fails to follow a variable defined by the model's own answer distribution.
- **Fix:** State the dependence and present entropy as a positive control rather than a finding. Alternatively, recompute the entropy contrast with knowledge labelled from an independent prompt (a paraphrase, or the free-prompt answers), and keep the dropped middle band as a continuous accuracy covariate.

## P2. Obscure "known" facts may be guesses, and the checks do not test this

- **Point:** For obscure subjects, a consistently correct forced answer may be a guess: the relation's dominant answer, or a cue in the name. The authors test answer-in-name only on 55 matched pairs, and their full-pool regression controls spelling, not guessability.
- **Verdict:** Unresolved. The text acknowledges the problem only in a narrow sub-analysis.
- **Evidence:**
  - Lines 415-418 (Appendix A) admit that "some obscure names also contain their answer". They exclude such pairs only within the 55-pair spelling analysis (gap +0.79 over 36 pairs).
  - The full-pool regression (lines 79-82) controls word frequency, token count and token rarity, plus relation fixed effects. Fixed effects absorb relation-level differences in abstention base rates. They do not test whether individual "known" labels are real. The text also does not say whether that regression includes the knowledge label.
  - I found no per-relation breakdown, no majority-answer baseline and no name-cue baseline anywhere in the 2,303- or 387-question analyses.
  - The paper does not cite Mallen et al.'s caveat, although it uses PopQA.
- **Severity:** Major. The obscure-known cell is where the paper makes its signature claim: the model says "I don't know" "even for facts the model always answers correctly" (line 30). That cell is the one most exposed to guessing. It is not critical, for the reason given under P3.
- **Fix:** On the full Llama pool, flag items that a relation-majority answer or a name-string or nationality cue would get right. Drop or stratify the low-correlation relations Mallen et al. name (color, country, sport, and others like them). Then recompute the 2x2 effects for both the 387-question judge measure and the 2,303-question regex measure.

## P3. The +0.39 popularity effect could be labelling error

- **Point:** Read the other way, Llama saying "I don't know" on 0.43 of obscure "known" facts is a correct report that it lacks them. On that reading, the +0.39 popularity effect measures labelling error, not a familiarity heuristic.
- **Verdict:** Partially answered.
- **Evidence:**
  - **What answers it:** The cell rates in lines 70-72 limit how far the memo's reading can go. The popularity effect averages two contrasts: known facts (0.43 vs 0.00) and unknown facts (0.48 vs 0.13). Mislabelled guesses can only move items into the "known" cells. They cannot create the unknown-arm gap, 0.48 obscure against 0.13 popular. That gap, about +0.35 on its own, is several times the knowledge effect (+0.09).
  - **What is missing:** The authors never make this argument and never report the within-unknown contrast with an interval. The within-known half (0.43 vs 0.00) is exactly what the memo's reading targets, and it stays exposed until P2 is fixed. So the magnitude of +0.39 is not secure.
- **Severity of the missing part:** Major. The direction of the headline survives on the text's own numbers. Its magnitude, and the "facts it knows" framing, do not survive without new analysis.
- **Fix:** Report the popularity effect separately within unknown facts (with a bootstrap interval) as the test that labelling error cannot produce. Report the within-known effect only after the guessability filter from P2.

## P4. The abstention result rests on one model and 387 questions

- **Point:** The abstention dissociation is shown in one model (Llama-3.1-8B-Instruct) on 387 judged questions, yet the title states it generally.
- **Verdict:** Partially answered.
- **Evidence:**
  - **Sample size:** This part is answered.
    - The primary difference has an interval of +0.30 [+0.20, +0.40] (line 73).
    - A regex detector on 2,303 questions with 20 replies each gives +0.44 (Table 1).
    - Five measurements agree in sign (Table 1).
  - **Model count:** This part is disclosed and explained.
    - The one-model limit is stated at lines 61-62 and 128-129.
    - Appendix B, Table 2 shows that the other families rarely abstain at all (0.05-0.10 in the obscure-unknown cell).
    - Gemma-2 contributes steering evidence (lines 95-96) and four base models contribute lens evidence (Section 3.3). These support the familiarity signal, not the behavioural 2x2.
  - **What is missing:** The title and the opening of the conclusion (line 133) state the abstention claim without scoping it to one model.
- **Severity of the missing part:** Minor. The limit is disclosed and the statistics are adequate. The remaining gap is wording that claims more than one model shows.
- **Fix:** Scope the title and line 133 to Llama-3.1-8B-Instruct, or to "an instruction-tuned model", and keep the cross-family claims for entropy, steering and the lens.

## P5. The judges were never checked against human labels

- **Point:** Abstention is labelled by two LLM judges whose agreement barely clears the pre-set bar (kappa = 0.62 against 0.60). Neither judge was validated against human labels.
- **Verdict:** Partially answered.
- **Evidence:**
  - **What is disclosed:** The lack of human validation is stated twice (lines 62 and 129-130).
  - **Judge-free check:** The effect keeps its sign under a judge-free regex detector, at +0.43 on the same replies and +0.44 on 2,303 questions (Table 1, line 75).
  - **What is missing:** The regex's "recall 0.99" (line 75) does not say what reference it was measured against. Its precision is not reported. No human-labelled sample exists for either judges or regex. One measurement in Table 1 drops to +0.13 (one judge, corrected, 24 tokens), and the text does not discuss it.
- **Severity of the missing part:** Minor. A judge-free measure agrees in sign at larger n, so the direction does not hinge on the judges. The magnitude does.
- **Fix:** Hand-label about 150 replies stratified across the four cells. Report precision and recall against those labels for both the judges and the regex. Explain the +0.13 row.

---

## Summary

| Class | Count | Points |
|---|---|---|
| Answered by current text | 0 | none |
| Partially answered | 3 | P3 (major), P4 (minor), P5 (minor) |
| Unresolved | 2 | P1 (major), P2 (major) |
| Critical | 0 | none |

**Would the paper survive a senior reader?** Not in its current form, though the memo's verdict ("the title's two-variable claim is not established") is stronger than the text warrants.

The memo is right that the entropy half of the title is largely fixed by the labelling rule. A senior reader will see that within a minute, and the manuscript says nothing about it. The memo is also right, and correctly cites Mallen et al. Sec. 4.2, that obscure "known" labels in PopQA are open to guessing. The authors' only defences are a 55-pair spelling check and a spelling regression, and neither tests guessing.

Where the memo overreaches is the claim that the whole +0.39 is labelling error. The manuscript's own cell rates show that obscure unknown facts draw "I don't know" far more often than popular unknown facts (0.48 against 0.13). Mislabelled guesses cannot produce that gap. So the core behavioural finding probably survives in direction: Llama's abstention tracks subject popularity more than knowledge. But the paper never makes this argument.

Partial and unresolved points touch the headline in three places:
- P1 undercuts "answer uncertainty tracks knowledge", which is half of the title.
- P2 and P3 undercut the size of the popularity effect and the most quotable claim: "I don't know" about facts the model always answers correctly (abstract lines 8-9, line 30).
- P4 leaves the title broader than the one-model evidence.

None of these is fatal, and all are repairable with analyses on data the authors already have. As written, though, the dissociation is less firmly shown than the abstract and title claim.

## Action items (priority order)

1. **Guessability control (P2, P3).** Flag full-pool items answerable by a relation-majority answer or a name cue, and drop or stratify the low-correlation relations (color, country, sport and similar). Recompute the 2x2 under both the judge and regex measures. Report the within-unknown popularity effect (0.48 vs 0.13) with its interval as the contrast that labelling error cannot produce, and cite Mallen et al. Sec. 4.2.
2. **Circularity (P1).** State in Section 2 that the knowledge labels and the entropy come from the same forced-answer distribution, and present entropy as a positive control. Better still, relabel knowledge from an independent prompt and treat the dropped 1-9 band as a continuous covariate. Rescope lines 6-7 and 32-33 accordingly.
3. **Validation and scope (P5, P4).** Hand-label about 150 stratified replies to validate the judges and the regex. Scope the title and line 133 to the one model in which abstention was measured.
