# Area-chair adjudication, round 4

Manuscript: "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge" (UniReps 2026, extended abstract). Line numbers are the printed numbers in manuscript.txt. Table 2 row alignment was checked against manuscript.pdf, p. 7.

## Fact-check of the memo

Every line reference in the memo points to the passage it describes. The numbers are quoted correctly, but three of them are framed selectively.

| Memo statement | Check |
|---|---|
| "Known" = greedy answer plus at least 9 of 10 samples correct, after the forced "The answer is" prefix (55-58) | Accurate. |
| 87 of 1,351 obscure facts qualify (246) | Accurate (87 + 85 + 1179 = 1,351). |
| The paraphrase keeps just 27 of the 87 as known (90-91) | Accurate, but selective. Popular known facts drop almost as much: 35 of 100 stay known (262). The 27 that survive abstain at 0.46, and the preference on stable labels is +0.32 [+0.16, +0.49] (91-92). The memo leaves both facts out. |
| kappa 0.79 "comes mostly from" the 169 questions unknown under both prompts (259-260) | The numbers are right but the reasoning is not. Kappa is already corrected for chance, so the large unknown-unknown block does not inflate it the way it would inflate raw agreement. There is a real problem the memo misses: kappa is computed on the 253 questions labeled under both prompts. The other 134 of the 387 fell into the dropped band (261), and the main text at line 90 does not say so. |
| 0.53 of obscure-known free replies contain the answer; 24 of 87 questions both answer and decline (292-294) | Accurate. |
| 0.43 is indistinguishable from 0.48 (Table 1) | Accurate (the intervals overlap). "Facts it never answers" should read "never answers correctly". |
| The meaning-based regrade covered only unknown facts (266-267) | Accurate. |
| The guessability filter was post hoc, with arbitrary cutoffs, and ignores cues in the name (283-285) | Post hoc: accurate. Cutoffs (3 characters, 20%): no justification is given in the text. "Ignores cues in the name" is overstated. The rule does catch a gold alias that appears literally in the name. It does not catch indirect cues such as the name's language or morphology. |
| Similar entropy at lines 8 and 83; low by construction per 62-64 | Accurate. Lines 62-64 admit the entropy and the labels come from the same distribution. |
| Preference = obscure-known minus popular-unknown abstention (75-76) | Accurate. |
| "Nothing in the paper rules that reading out" | Mostly fair (see P2 and P6). The memo does skip evidence it should have engaged with: recognition of 0.83 (100-102), the unique-name subset at 0.43 (105-106, 290-291), the +0.30 preference with guessable answers removed (Table 2), and a popularity gap of +0.34 among unknown facts alone (286). |

**The manuscript has an internal inconsistency that the memo did not catch.** Line 293 says that, in the obscure-known cell, 0.53 of free replies contain a gold alias, as do 0.72 of the replies that do not abstain. With abstention at 0.43, about 0.57 x 0.72 = 0.41 of replies are non-abstaining replies that contain an alias. That leaves about 0.12 of all replies that abstain and still contain an alias, which is roughly 28% of the abstaining replies. Lines 97-98 instead say that "at most 0.03 of the 'I don't know' replies in any cell contain the gold answer", and line 275 gives the same bound for hedged answers. The text does not say which set of replies or which abstention measure line 293 uses, so the two passages cannot be reconciled from the page.

## Points

### P1. The relation prior can produce "known" labels
- **Point:** Under a forced "The answer is" prefix, an obscure question whose answer is the relation's typical value gets consistently correct guesses. The 9-of-10 rule then labels these guesses as knowledge.
- **Verdict:** Partially answered.
- **Evidence:** The post hoc guessability rule removes a fact when its answer is the relation's most common answer and covers at least 20% of the relation's pool (283-285). With those removed, the preference is +0.30 [+0.19, +0.41] (Table 2). Relation fixed effects leave the obscure-arm coefficient at +0.35 (Table 2).
- **What is missing:**
  - The rule is post hoc, and the 20% threshold and "most common only" are not justified. A second- or third-most-common answer can still be guessed reliably.
  - The text never says how many of the 87 obscure-known facts survive the filter (the n column of Table 2 is blank for that row). It also does not give the relation mix of the 87.
- **Severity of the remaining gap:** Major. It touches the headline.
- **Fix:** Report the obscure-known n and the relation mix after filtering. Then replace the single threshold with a graded per-question prior, meaning the base rate of the gold answer in its relation, and show the preference across prior bins.

### P2. Cues in the name can produce "known" labels
- **Point:** An obscure name's language or form signals answers such as country, place of birth or genre. Consistent correct forced answers may therefore be inferred from the name rather than retrieved.
- **Verdict:** Unresolved.
- **Evidence:** I looked in four places and none of them addresses this:
  - The guessability rule only catches a gold alias that appears verbatim in the name (283-284).
  - The paraphrase relabel (257-265) keeps the subject name, so a name-driven guess survives it by design.
  - The meaning regrade covers only unknown facts (266-267), so it cannot detect false "known" labels.
  - Recognition (100-102, 276-282) is graded against Wikidata descriptions, which are typically generic (entity type plus country). Those descriptions are just as inferable from the name. Recognition is also higher for obscure-known (0.83) than for popular-known facts (0.75) (101, 111). That is what one would expect if obscure facts become "known" when the name carries the information.
- **Severity:** Major. This is the specific mechanism behind the memo's alternative reading of the headline cell, and the text has no control that could rule it out.
- **Fix:** Add a name-only guessing baseline: ask Llama, or a second model, for the most likely answer given only the relation and a matched nonce or swapped subject name of the same form. Report obscure-known abstention, with its n, on the questions this baseline cannot guess.

### P3. The knowledge labels are unstable
- **Point:** Only 27 of the 87 obscure-known facts stay known under a paraphrased prompt. The reported kappa of 0.79 overstates how well the two labelings agree.
- **Verdict:** Partially answered.
- **Evidence:** The pre-specified relabel re-estimates the result on stable labels. The 27 facts that stay known abstain at 0.46, and the preference is +0.32 [+0.16, +0.49] (90-92, 262-265). Attrition is similar on the popular side (35 of 100 popular-known facts stay known, 262), so the instability is not specific to the obscure arm.
- **What is missing:** Line 90 reports kappa = 0.79 without saying it covers only 253 of the 387 questions. The other 134 dropped into the intermediate band (259-261).
- **Severity of the remaining gap:** Minor (presentation).
- **Fix:** At line 90, state that the agreement covers 253 of 387 questions, and give the share of each cell that lost its label.

### P4. The free-generation data point to "unknown"
- **Point:** Only 0.53 of obscure-known free replies contain the answer, and 24 of the 87 questions both answer and decline. Obscure-known abstention (0.43) also matches obscure-unknown abstention (0.48), so the cell behaves like unknown facts.
- **Verdict:** Partially answered.
- **Evidence:**
  - The data are reported openly (292-296).
  - The paper's reading is that the matching 0.43 and 0.48 rates are the finding itself: within obscure entities, knowledge does not move abstention.
  - A linear probability model gives knowledge -0.05 [-0.14, +0.03] (295).
- **What is missing:**
  - None of these observations separates the two readings. Both "knows but declines" and "mislabeled guess, correctly declined" predict them, and the text does not argue otherwise.
  - Line 293 conflicts with lines 97-98 (see the fact-check above), which weakens the "refusals, not hedged answers" defence in exactly this cell.
  - The text does not report obscure-known abstention restricted to questions where Llama produces the gold answer unprompted, in held-out free replies.
- **Severity of the remaining gap:** Major. It overlaps with P2 and touches the headline.
- **Fix:** Reconcile line 293 with lines 97-98. Then report obscure-known abstention on fresh replies, restricted to questions where independent free samples name the gold answer.

### P5. The entropy comparison is circular
- **Point:** The similar forced-answer entropy in obscure-known and popular-known cells (lines 8 and 83) is offered as if it showed equal knowledge. It comes from the same samples that define the label, so it is low by construction.
- **Verdict:** Partially answered.
- **Evidence:** Lines 62-64 admit that entropy is a control implied by the labeling.
- **What is missing:** The abstract (7-8, "although its answer entropy is similar") and line 83 still present the similarity as a contrast that carries information.
- **Severity of the remaining gap:** Minor (presentation).
- **Fix:** In the abstract and at line 83, state that the similar entropy follows from the labeling rule, or drop the "although" clause.

### P6. The headline rests on one cell, and the alternative reading is not excluded
- **Point:** The preference equals obscure-known minus popular-unknown abstention. Its simplest reading is that Llama correctly declines guesses the label counts as knowledge, and nothing excludes that reading.
- **Verdict:** Partially answered.
- **Evidence:** The misfire half of the result does not depend on the obscure-known cell:
  - Popular-unknown abstention is 0.13, against 0.48 for obscure-unknown (Table 1).
  - The popularity gap among unknown facts alone is +0.34 [+0.26, +0.43] (286).
  - Regrading by meaning leaves +0.23 [+0.12, +0.35] (94-95).
  - The abstract (6-7) defines "known" operationally.
- **What is missing:**
  - The comparative claim "more than knowledge" in the title (1-2), abstract (5) and contribution 1 (44-45) depends entirely on the obscure-known cell: 87 facts, 6% of the obscure pool. P2 and P4 leave that cell's interpretation open.
  - The paper does not present the unknown-only gap as the result that survives without that cell.
  - Popular-unknown labels are also leaky: 0.48 of those questions contain a gold alias in some free reply (270), and dropping them lowers the preference to +0.19.
- **Severity of the remaining gap:** Major. It touches the headline.
- **Fix:** Lead with the label-robust finding (the popularity gap among unknown facts). State the comparative "more than knowledge" claim as conditional on the forced-answer label, pending the P2 control.

## Summary

| Class | Count | Points |
|---|---|---|
| Answered by current text | 0 | none |
| Partially answered | 5 | P1, P3, P4, P5, P6 |
| Unresolved | 1 | P2 (major) |

The memo is accurate on facts but selective. It ignores the stable-label re-estimate, the similar attrition on the popular side, and the popularity gap among unknown facts, and it misreads kappa. Its central alternative is still coherent, and the current text does not defeat it.

The paper is transparent and has many robustness checks. But almost all of them vary the abstention measure or audit the unknown side of the labels. The one known-side audit (guessability) is post hoc and catches only verbatim name cues and the single most common answer. Nothing tests whether the 87 obscure-known facts are retrieved or inferred from the name and relation. The paper's own numbers are at least as consistent with the memo's reading as with the authors': half of these facts go unstated in free replies, recognition is higher for obscure-known than for popular-known facts, and obscure-known abstention matches obscure-unknown abstention.

A senior reader would probably accept the narrower result: Llama abstains far less on popular entities whose facts it does not know (misfires), and this holds across judges and relabelings. That reader would not accept the title and contribution 1 as written, because "more than knowledge" stands or falls with the contested cell. P2, P4 and P6 all touch the headline claim, and so does P1 more weakly. As written, the paper survives at a workshop only if the headline is narrowed or the P2 control is added. The unreconciled numbers at lines 293 and 97-98 would also lower a careful reader's trust in the "refusals, not hedged answers" defence.

## Action items, in priority order

1. **Add a name and prior guessing control for the obscure-known cell (P1, P2).** Build a per-question guessability baseline from the relation plus a matched nonce or swapped name of the same form. Report obscure-known n and abstention on the questions this baseline cannot guess, and on the existing guessable-removed subset, together with the relation mix of the 87.
2. **Narrow the headline (P5, P6).** Retitle and reword the abstract and contribution 1 so that the label-robust misfire gap among unknown facts (+0.34) is the lead result. State "more than knowledge" as conditional on forced-answer labels, and remove the entropy "although" clause or mark it as implied by the labeling.
3. **Repair the numbers (P3, P4).** Reconcile line 293 with lines 97-98 and 275, naming the reply set and abstention measure each one uses. Report kappa and agreement over all 387 questions, including those that dropped to the intermediate band.
