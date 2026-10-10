# Referee report: methods and statistics

**Manuscript:** "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge" (UniReps 2026, extended-abstract track)
**Lens:** methods and statistics

## Summary

**What the paper claims.** In Llama-3.1-8B-Instruct, whether the model says "I don't know" depends more on how popular the entity is (PopQA page-view decile) than on whether the model knows the fact (its own forced-answer correctness). Gemma-2-9B shows the same preference when the prompt allows abstention. Steering and lens analyses do not isolate a familiarity mechanism.

**What the paper shows.** It reports a 2x2 table on 387 sampled questions. Llama abstains on 0.43 of replies for obscure-entity facts it answers correctly under a forced "The answer is" prefix, and on 0.13 for popular-entity facts it never answers correctly. The sign of this gap holds under many rescorings of the same replies. The size of the gap shrinks substantially under several label audits (to +0.19 to +0.23). It also depends on what "known" means, and in the key cells that label is unstable across prompts.

**Gaps between the abstract and the body:**
- The abstract says the gap "holds" under regrading but does not say that it shrinks by a quarter to a third.
- The abstract reports only the primary measure. That measure was designed after the first pre-specified measure returned the smallest estimate (+0.13).
- Contribution 3 asserts a "generic abstention signal". The steering design cannot separate a generic signal from a familiarity signal.

**Verified prior work:**
- Ferrando et al. (arXiv 2411.14257) define known entities as "those where the model gets at least two attributes correct" and unknown entities as those where the model gets every attribute wrong, discarding the middle. This matches the manuscript's characterisation at lines 30-31.
- Brzezinka 2026a (arXiv 2607.07670) crosses well-known, obscure and fabricated entities with answer correctness in representations. It reports that its models almost never abstain (2 refusals in 2,520 answers), so it does not cross familiarity with knowledge in abstention behaviour. This is consistent with the novelty claim at lines 32-35.

---

## Findings

### F1. Knowledge labels are unstable in the cells that drive the result, and the reported agreement hides this
- **Severity:** major
- **Location:** Sec. 3.1, lines 89-92; App. A, lines 257-263; Table 2
- **Evidence:** "agrees with the original labels at κ = 0.79. Of the 87 obscure known facts, 27 stay known"
- **Problem:**
  - The kappa of 0.79 is computed only on the 253 questions that receive a known or unknown label under both prompts. The 134 questions that fall into the dropped band under the paraphrase are excluded.
  - In the cells that define the preference, only 27 of 87 obscure known facts (31%) and 35 of 100 popular known facts stay known under a paraphrase.
  - "Known" (greedy answer plus at least 9 of 10 samples correct) is therefore largely prompt-specific. Presenting kappa = 0.79 as agreement "with the original labels" overstates how stable the labels are.
  - Line 261 says "most other changes move popular known facts to the dropped band". This understates that 69% of obscure known facts also fail to stay known.
- **Fix:**
  - Report the full 3x3 cross-tabulation (known, intermediate, unknown) of original against paraphrase labels over all 387 questions, with three-category kappa.
  - Report the preference with knowledge defined as the fraction of correct samples pooled over both prompts, and the cell rates on the 27/99/70/35 subset in Table 1 style.
  - Drop the sentence at line 261 or give the per-cell counts behind it.

### F2. "Known" under a forced prefix may be cue-based guessing, and the checks share this confound
- **Severity:** major
- **Location:** Sec. 2, lines 55-58; Sec. 3.1, lines 100-102; App. A, lines 276-279, 283-285, 292-294
- **Evidence:** "In the obscure known cell, 0.53 of free replies contain a gold alias, and 0.72 of the replies that do not abstain"
- **Problem:**
  - The best alternative explanation is this. For obscure entities, correct answers under "The answer is" can come from relation priors or name cues (for example, the country inferred from name phonology, the sport from a relation prior) without any knowledge of the entity. The free-prompt model then correctly treats the entity as unknown.
  - The data fit this explanation:
    - When Llama does answer freely in this cell, it is wrong 28% of the time.
    - For 24 of 87 questions, one free reply answers and another abstains.
  - The checks that are meant to rule it out share the same weakness:
    - The guessability rule catches only an alias inside the name or a single answer covering at least 20% of a relation.
    - The paraphrase relabeling is still a forced-answer prompt.
    - The recognition score is graded against generic Wikidata descriptions (for example, "village in X"), which are themselves guessable from the name.
  - No gold-alias rate for non-abstaining replies in the popular known cell is given for comparison.
- **Fix:** Add a counterfactual-name baseline. Replace each obscure subject with a fabricated or shuffled name of the same relation and entity type, and measure forced-answer accuracy under the same rule. Define "known" as accuracy above that baseline, or report the preference on facts whose answers are not recovered from the fabricated name. Also report the gold-alias rate among non-abstaining free replies for popular known facts.

### F3. Numbers in the obscure known cell are mutually inconsistent
- **Severity:** major
- **Location:** App. A, lines 274, 292-293; Sec. 3.1, lines 81-82, 97-99
- **Evidence:** "Declines occur in 0.66 of obscure known replies and 0.00 of popular known ones"
- **Problem:** Three sets of numbers do not reconcile:
  - **Strict declines against the primary rate.** "Strict declines" (declining while naming no candidate) are presented as a stricter criterion (lines 98-99), yet they occur in 0.66 of obscure known replies against 0.43 under the primary two-judge rule.
  - **Strict declines against gold-alias replies.** 0.66 of these replies decline naming no candidate, and 0.53 contain a gold alias. Since 0.66 + 0.53 = 1.19, at least 19% of replies must do both.
  - **The gold-alias decomposition.** With 0.43 abstaining, at most 0.03 of those containing gold (line 97), and 0.72 of non-abstainers containing gold, the overall gold-alias rate can be at most about 0.42, not 0.53.
  - Either the reply sets or denominators differ without being stated, or alias matching produces false positives. Alias-matching false positives would also contaminate the knowledge labels, which use the same matching.
- **Fix:**
  - State the reply set and denominator for every rate in App. A.
  - Reconcile the three numbers in a small cross-tabulation (70B label against two-judge label against gold-alias presence) for the 261 obscure known replies.
  - Hand-audit alias matches in replies labeled as declines.

### F4. The headline estimate comes from a measure designed after the first pre-specified one gave +0.13, and the abstract does not say so
- **Severity:** major
- **Location:** Abstract, lines 5-8; Contribution 1, lines 44-45; Sec. 3.1, lines 86-87; App. A, lines 251-254; Table 2
- **Evidence:** "We designed the primary measure after the smallest of these was seen."
- **Problem:**
  - The originally pre-specified measure (one corrected judge) gave +0.13 [+0.03, +0.24]. The two-judge measure was then designed in response, labelled "pre" in Table 2, and made primary.
  - The abstract and contributions report only +0.30 and 43%. The magnitude ranges from +0.13 to +0.44 depending on the measure.
  - The "pre" label in Table 2 means only that a plan existed before the run. It does not mean the measure was chosen independently of the outcome.
- **Fix:**
  - State in the abstract and in Contribution 1 that estimates range from +0.13 to +0.44 across measures, and that the primary measure was adopted after the first one.
  - Add a third Plan category in Table 2, such as "pre, after seeing earlier estimate", for the two-judge rows.

### F5. The outcome variable has no human ground truth, and two measures on identical replies disagree threefold without explanation
- **Severity:** major
- **Location:** Limitations, line 154; Sec. 2, lines 67-70; App. A, lines 251-253; Table 3
- **Evidence:** "why it and the regex disagree on the same replies is not established"
- **Problem:**
  - Abstention is the dependent variable, yet on the same 24-token replies two measures give +0.13 and +0.44.
  - In the obscure unknown cell, rates range from 0.48 (both judges) to 0.85 (regex).
  - The kappa that clears the pre-specified bar, 0.62 [0.57, 0.67], has a lower bound below that bar.
  - The relevant threat is misclassification that differs by cell. For example, replies about obscure entities may phrase uncertainty differently from wrong confident answers about popular entities. No validation can detect this.
- **Fix:**
  - Hand-label a stratified set of replies: at least 50 per cell, from both the 24-token and 96-token runs.
  - Report each measure's precision and recall by cell, and recompute the preference with those error rates.
  - Explain the 24-token disagreement using that set.

### F6. Intervals ignore relation structure, and relation composition of the key cells is not reported
- **Severity:** major
- **Location:** Sec. 2, lines 53-55 and 76-77; App. A, lines 246-248; Table 2 (regression rows)
- **Evidence:** "Intervals are 95% percentile bootstraps over questions within cells, not clustered by relation."
- **Problem:**
  - Arms are defined within each of 16 relations, but the 87 obscure known facts are 6% of the obscure pool. They are probably concentrated in a few guessable relations, and Appendix A gives no per-relation counts despite line 61's promise of "cell counts".
  - It is not stated what the stratified sample is stratified by.
  - The relation fixed-effects row reports the arm coefficient (+0.35, with no interval), not the preference.
  - If one or two relations carry the obscure known cell, the effective sample size is much smaller than 387, and the claim is about those relations.
- **Fix:**
  - Give per-relation cell counts and per-relation preferences.
  - Report a relation-clustered or relation-stratified bootstrap for the preference, and a leave-one-relation-out range.
  - Add the missing interval for the fixed-effects row.

### F7. "Popularity" bundles name form and entity type, and the strongest control removes 43% of the gap
- **Severity:** major
- **Location:** Sec. 3.1, lines 106-112; App. A, lines 297-303; App. F
- **Evidence:** "the regex abstention gap of +0.76 shrinks to +0.43 [+0.18, +0.69] once the names' word frequency, token count and token rarity"
- **Problem:**
  - The matched-pair design is the more direct test of whether surface form drives the gap, and it removes 43% of the gap among known facts.
  - The text instead emphasises that the full-sample regression coefficient "barely changes". That regression adjusts for noisy covariates and is prone to residual confounding.
  - Neither model-side familiarity measure tracks page views (lines 109-111). The page-view arm therefore stands for an unmeasured bundle of entity type, name form and relation mix, and the design cannot attribute the effect to popularity as such.
- **Fix:**
  - Report the matched-pair analysis for the preference itself, matching unknown facts as well as known.
  - Add Wikidata entity type (P31) as a stratifier.
  - Present the matched result alongside the regression in Sec. 3.1 rather than below it.

### F8. Gemma's plain-prompt null is set aside while the permissive-prompt positive is headlined as "same preference"
- **Severity:** major
- **Location:** Contribution 2, lines 46-47; Sec. 3.2, lines 119-122; Table 3 and line 307; App. B, lines 314-315
- **Evidence:** "abstains in at most 0.11 of replies in any cell, so its preference cannot be estimated"
- **Problem:**
  - Table 3 does estimate Gemma's plain-prompt preference: +0.01 [-0.03, +0.06]. This is a precise near-zero estimate, not a missing one.
  - A floor effect bounds the absolute difference, but a scale-free contrast (an odds ratio of obscure known against popular unknown) is estimable and is not reported.
  - Without it, "Same Preference in Gemma" rests on a prompt that explicitly cues abstention, adopted after the plain prompt failed.
- **Fix:**
  - Report all four plain-prompt cell rates for Gemma, and the odds-ratio form of the preference under both prompts.
  - Retitle Contribution 2 to condition on the prompt, for example "Under a prompt that invites abstention, Gemma shows the same preference".

### F9. Permissive-prompt results are reported for two of four families, with no explanation
- **Severity:** major
- **Location:** Sec. 1, line 42; Sec. 3.2, lines 125-128; App. B, lines 310-314; App. G (e9_permissive_prompt.md)
- **Evidence:** "on each model's existing two-judge question set, with three replies scored by the three judges and a majority rule"
- **Problem:**
  - The plan's wording ("each model") implies all four families. Results appear only for Llama and Gemma.
  - If the permissive prompt was run on Qwen and Mistral and not reported, the cross-family claim is selectively reported. If it was not run, the reason is not given, and line 158 ("whether other families share the preference is open") could have been resolved with existing infrastructure.
- **Fix:** State whether Qwen and Mistral were run under the permissive prompt. If they were, report their cell rates and preferences. If not, run them, since the question sets and judges already exist.

### F10. The steering design cannot distinguish a "generic abstention signal" from a familiarity signal
- **Severity:** major
- **Location:** Contribution 3, line 48; Sec. 3.3, lines 132-144; App. C, lines 317-348
- **Evidence:** "In Llama, the direction is nearly parallel to the abstention direction (cosine 0.95), partly by construction"
- **Problem:**
  - The abstain-minus-answer direction is fitted on questions that split mostly by popularity. Even the within-obscure refit has cosine 0.91, using 587 abstaining against 55 answering questions, and may capture residual within-decile familiarity.
  - Projecting a component with cosine 0.95 out of the obscure-minus-popular direction leaves a residual of about 31% of its norm, which is then rescaled by about 3.2 times. The null for that residual says little about familiarity.
  - Both hypotheses predict these results. The positive wording "moves a generic abstention signal" is not supported.
  - Five random directions is also a thin null distribution.
- **Fix:**
  - Rephrase Contribution 3 as "familiarity and abstention directions are not separable here".
  - To test genericness, fit an abstention direction from a source orthogonal to entity popularity, such as abstain against answer within the popular unknown cell, or refusals on fabricated entities, and test whether it reproduces the effect.
  - Use at least 50 random or activation-matched directions.

### F11. The "within one prompt" section asserts a result its pre-specified test did not find
- **Severity:** minor
- **Location:** Sec. 3.1, lines 113-117; Table 3
- **Evidence:** "(-0.12 [-0.72, +0.45]; Appendix B). After seeing this, we specified a reading at the free prompt"
- **Problem:**
  - The pre-specified within-prompt readout shows no preference in Llama. The replacement reading (log-odds of "I": +8.4 against +3.1; entropy: +0.16 against +0.05 nats) was specified afterwards and is reported without intervals or sample size.
  - This section is the only one that addresses the prompt-mismatch alternative in F2, yet its heading reads "The Split Remains".
- **Fix:**
  - Give intervals and n for the post hoc reading, and mark it post hoc in the text.
  - Retitle the paragraph to report the pre-specified null first.

### F12. "Answer uncertainty alone predicts its abstention poorly" has no predictive analysis behind it
- **Severity:** minor
- **Location:** Sec. 1, lines 50-51; Abstract, line 8; Table 1 caption; App. A, lines 295-296
- **Evidence:** "Llama's answer uncertainty alone predicts its abstention poorly"
- **Problem:**
  - No AUROC, R² or comparison between predictors is reported.
  - The only relevant model shows +0.14 per standard deviation of entropy, which is not small.
  - The baseline is first-token entropy, the weakest available measure of answer uncertainty.
  - The abstract's "entropy is similar" comparison uses full-pool entropy, while abstention uses the 387-question sample (Table 1 caption).
- **Fix:**
  - On the 387 questions, report AUROC for predicting abstention from first-token entropy, from semantic entropy or answer agreement over the 10 existing samples, and from the popularity arm.
  - Compute Table 1 entropy on the same 387 questions.

### F13. Degenerate bootstrap intervals are reported and used as decision inputs
- **Severity:** minor
- **Location:** Table 1 (popular known); Table 7; App. D, lines 360-366
- **Evidence:** "OLMo-3-7B 32 19 14 -5 [-5, -5] contradicts"
- **Problem:**
  - A percentile bootstrap of a 0/300 proportion gives [0.00, 0.00], which understates uncertainty.
  - Integer onset differences with collapsed intervals are fed into a rule that declares "contradicts". The caption itself says they are "not a precision estimate".
- **Fix:**
  - Use Wilson or Clopper-Pearson intervals at the question level for proportions at 0 or 1.
  - Define onset on an interpolated curve, or use a permutation test across layers, before applying the support/contradict rule.

### F14. Sign robustness rests on rescorings of the same replies, not independent replications
- **Severity:** minor
- **Location:** Sec. 3.1, lines 85-86; Table 2 (top block)
- **Evidence:** "Its sign holds under five other measures of abstention, from +0.13 to +0.44 (Table 2)."
- **Problem:**
  - Five of the six alternative measures rescore the same 387 x 3 replies. Regex recall against the judges is 0.99, and one judge agrees 1.00 with the majority.
  - Agreement in sign therefore shows robustness to scoring, not to sampling. It adds little beyond the primary interval.
- **Fix:** Say so explicitly. Treat a fresh set of replies, or the 2,303-question regex run, as the replication, and report its cell rates.

### F15. The provenance of the pre-registered amendment is irregular, and Gemma's steering result relies on it
- **Severity:** minor
- **Location:** App. G footnote a; App. C, lines 333-336
- **Evidence:** "its internal date is one day later than its recorded hash time."
- **Problem:**
  - The amendment that adds the third judge is said to precede judging (line 334), but its internal date postdates its hash.
  - Line 336 declares the secondary scorer "the defensible one" for Gemma. The hashes are self-recorded (line 388).
- **Fix:**
  - Explain the discrepancy.
  - Deposit the plans with an external timestamp (for example, OSF or a signed commit).
  - Report Gemma's steering result with the pre-amendment scorer as primary, alongside the secondary scorer.

### F16. The recognition null comes from an unvalidated proxy, and the heading overstates it
- **Severity:** minor
- **Location:** Sec. 3.1, lines 100-102; App. A, lines 276-282
- **Evidence:** "Recognition & Ambiguity: Neither Explains It."
- **Problem:**
  - Recognition is an LLM-graded match to generic Wikidata descriptions. Its coefficient on abstention is about zero (-0.03 [-0.06, +0.00]).
  - A noisy proxy under-adjusts, so a 5% shrinkage cannot rule out recognition as a mediator.
  - Recognition is higher for obscure known facts (0.83) than for popular known facts (0.75), which suggests it measures guessable descriptions rather than familiarity (see F2).
- **Fix:**
  - Validate the recognition score against fabricated-name controls; it should be near 0 for invented entities.
  - Retitle the paragraph to "a recognition proxy does not explain it".

### F17. The ambiguity check measures label collisions, not ambiguity as the model experiences it
- **Severity:** minor
- **Location:** Sec. 3.1, lines 103-106; App. A, lines 288-291
- **Evidence:** "popular names are shared with more Wikidata entities (median 11–13, against 1 for obscure known facts)"
- **Problem:** Popular names collide with many Wikidata labels, but the model resolves them to the dominant referent. An obscure name with one Wikidata label may still be unresolvable for the model. The statistic therefore does not "point the other way" on model-side ambiguity.
- **Fix:** Measure ambiguity from the model, for example the entropy of the entity type it assigns in the recognition completions. Alternatively, soften the claim to "Wikidata label collisions do not explain it".

### F18. Lens results are stated more uniformly than Table 5 shows, and the readout is not validated against behaviour
- **Severity:** minor
- **Location:** Sec. 3.3, lines 147-152; Table 5; Table 6
- **Evidence:** "Knowledge raises it about as much (Appendix D, Figure 3)."
- **Problem:**
  - In Gemma-2-2B the hedge-term effects are +0.10 for popularity and +0.37 for knowledge, which is not "about as much".
  - In Llama-Instruct, which shows the behavioural preference, the readout leans strongly toward knowledge (-1.46). The readout therefore does not track the abstention it is meant to explain, so its null is uninformative.
- **Fix:**
  - Report how well the lens hedge term predicts free-prompt abstention in Llama-Instruct before interpreting base-model effects.
  - Change the sentence to "comparably in three of four models".

### F19. An unresolved regression anomaly in the surprisal analysis
- **Severity:** minor
- **Location:** App. F, lines 377-381
- **Evidence:** "although surprisal hardly correlates with the arm (point-biserial +0.02; -0.06 given relation and knowledge), which we have not resolved"
- **Problem:** Adding a regressor that is nearly uncorrelated with surprisal should not move its coefficient from +0.07 to +0.01. This points either to an interaction or Simpson structure, or to an analysis error. If it is an error, it may also affect the other regex regressions in Table 2.
- **Fix:** Re-run with relation-by-arm interactions and report surprisal-abstention slopes within each arm. If the anomaly persists, check the design matrix.

### F20. Counts and units do not agree across sections
- **Severity:** minor
- **Location:** Abstract, line 6; Contribution 1, line 45; Sec. 3.1, lines 82 and 85-86; Table 2
- **Evidence:** "It survives four label and measurement checks."
- **Problem:**
  - The abstract lists three checks, Contribution 1 says four, line 86 says "five other measures", and Table 2 lists six alternative measures.
  - The abstract says "43% of questions", while the body reports "0.43 of replies", which is an average over three replies per question.
- **Fix:** Use one count consistently, and state the unit ("of replies") in the abstract.

### F21. Table 2 is missing sample sizes and intervals
- **Severity:** minor
- **Location:** Table 2
- **Evidence:** "Relation fixed effects and knowledge only ... 387 pre +0.35"
- **Problem:**
  - The relation fixed-effects row has no interval.
  - The n column is blank for most rows in the "Other knowledge labels" block and for "plus namesake count".
  - Readers cannot judge precision for the regraded (+0.23) or alias-dropped (+0.19) estimates, which are the most consequential robustness rows.
- **Fix:** Fill in n and 95% intervals for every row, including the cell sizes after regrading and alias dropping.

### F22. Label-error audits are applied one at a time, and the abstract hides how much the estimate shrinks
- **Severity:** major
- **Location:** Abstract, lines 8-10; Sec. 3.1, lines 92-95; App. A, lines 266-271; Table 2
- **Evidence:** "Dropping instead every popular unknown question with a gold alias in any free reply (0.48 of them) gives +0.19 [+0.07, +0.31]."
- **Problem:**
  - The "misfire" cell (popular unknown) has substantial label error:
    - The 70B regrader moves 40% of it out of "unknown".
    - 48% of these questions produce the gold alias in a free reply.
  - Each correction alone cuts the preference to +0.23 or +0.19, and the corrections are never combined with each other or with guessability removal.
  - The abstract's "The gap holds" conveys none of this shrinkage.
- **Fix:**
  - Report a single estimate with all label corrections applied jointly (regrade, alias drop and guessability, on both the popular unknown and obscure known cells).
  - Quote that estimate, or the range, in the abstract.

---

## Strengths

1. **Cell-level reporting.** The 2x2 crossing of popularity and model-specific knowledge is clean, and every cell is reported with its n and interval (Table 1, Sec. 2, lines 53-61). Readers can recompute every effect; the main arithmetic checks out (0.43 - 0.13 = 0.30; the entropy effects of 2.14 and 0.51).
2. **Transparent forking-paths disclosure.** The paper labels which analyses are pre-specified and which are post hoc, and admits that the primary measure followed the +0.13 result (lines 86-87, 251-254; Table 2 Plan column; App. G).
3. **Held-out discipline in steering.** Directions are fitted, the dose is selected and the result is scored on three disjoint sets, with the dose rule kept separate from the confirmatory bar (lines 133-137, 325-329).
4. **Appropriate resampling unit.** The bootstrap resamples questions rather than replies, which respects the dependence among the three replies per question (lines 76-77, 248-250).
5. **Restraint about mechanism.** The paper states that its familiarity measures do not track page views, and it does not claim a mechanism (lines 109-112; Limitations).

## Questions for the authors

1. How are the 87 obscure known facts spread across the 16 relations, and what is the preference within relations? (F6)
2. Are the 0.66 strict-decline rate and the 0.53 gold-alias rate for obscure known facts computed on the same 261 replies? If so, how can both hold? (F3)
3. What is Llama's forced-answer accuracy on obscure-cell questions when the subject is replaced by a fabricated name of the same type? (F2)
4. Among non-abstaining free replies to popular known facts, what share contain a gold alias? (F2)
5. Was the permissive prompt run on Qwen and Mistral? (F9)
6. What is the full 3x3 agreement between original and paraphrase labels over all 387 questions? (F1)

## Recommendation: major revision

The behavioural dissociation is probably real in sign: Llama abstains more on obscure-entity facts it answers under forcing than on popular-entity facts it misses. The paper is unusually candid about its process. The headline framing, however, outruns what the labels and measures support.

**Findings that drove the recommendation:**
- **F1:** the "known" label is unstable in the key cells, and the reported agreement hides this.
- **F2:** forced-answer correctness on obscure entities may be cue-based guessing, and the checks share this confound.
- **F3, together with F22:** the key cell's numbers do not reconcile, and the corrected estimates shrink to +0.19 to +0.23 without the abstract saying so.

**What would change my view:** Most of the fixes use data the authors already have.
- Reconcile the obscure known numbers.
- Report the full label cross-tabulation, and a preference computed with relation-clustered intervals and all label corrections applied jointly.
- Add a fabricated-name guessing baseline for "known".
- Restate the abstract and contributions with the range of estimates and the conditional nature of the Gemma result.

If the jointly corrected, relation-clustered preference stays clearly positive and the guessing baseline is low, I would move to minor revision.
