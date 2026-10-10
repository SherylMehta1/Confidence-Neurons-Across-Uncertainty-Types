# Area-chair adjudication: round 3 attack memo

Venue: UniReps 2026 (NeurIPS workshop), 4-page extended abstract. Basis: manuscript.txt only. Line numbers below refer to its left margin. I checked every line reference in the memo and each one matches the text. The memo has one small overstatement: at line 487 the authors offer short common-word names ("New York" as a book title) as the reason surprisal fails. They do not say low-view subjects are *often* such names.

## Rejection points

### P1. Manipulated variable is popularity, not familiarity to the model (construct and framing)
**Point.** The paper calls itself a factorial test of the familiarity-misfire hypothesis. Its factor, though, is Wikipedia page views. Its only model-side familiarity measure (name surprisal) is null, and entity recognition is *higher* in the decisive cell.
**Verdict: partially answered.**
**Evidence.** The title and abstract claim only "popularity" (lines 1-2, 4). Line 41 calls page views "a proxy for familiarity, and we report it as popularity". Lines 86-88 concede "page views are not the model's own familiarity: name surprisal under the model did not even correlate with them". Limitations repeat the proxy caveat (line 131). Against this, the contribution is still stated as "this factorial test of the misfire idea" (lines 34-35), and Related work says "the misfire hypothesis we test" (line 123). The paper never discusses the implication of its own entity-knowledge result: Llama correctly describes the entity for 0.83 of obscure-known facts against 0.75 for popular ones (lines 79-80, 414). The misfire account (lines 27-28) says an unrecognised name fails to suppress a default "can't answer" response. These entities are recognised, and the text shows no model-side unfamiliarity anywhere. So the paper reports the right caveats but keeps a contribution claim that those caveats undercut. That is a deliberate framing choice, and the text does not justify it.
**Severity: major.** The descriptive headline in the title and abstract survives. The stated contribution, a test of the misfire idea, does not survive as worded.
**Fix.** Restate the contribution as a dissociation of abstention from knowledge along a page-view axis, which the misfire idea predicts. Say explicitly that neither model-side measure (null surprisal, higher entity recognition in the obscure-known cell) shows the obscure names are unfamiliar to Llama.

### P2. The decisive cell is thin
**Point.** The paper rests on the obscure-known cell: n = 87, and only n = 27 under independent labels.
**Verdict: partially answered.**
**Evidence.** The paper's main quantities are not single-cell numbers. The popularity effect is averaged over both knowledge levels: +0.39 [+0.33, +0.45] on 387 questions (lines 68-69). The popularity-minus-knowledge difference keeps its sign under nine measurements and labellings (Table 1). Under independent labels it is +0.32 [+0.16, +0.49] (line 76). Among unknown facts alone, the popularity gap is +0.34 [+0.26, +0.43] (line 421), so the effect does not depend on the obscure-known cell. What is missing: the abstract leads with the cell value (0.43 vs 0.00, lines 5-7), which rests on n = 87 and has no interval in the text (Figure 1a has one, but its value is not stated). The paper also calls this cell "the deciding observation" (line 33). The text does not explain why it shrinks to 27 under independent labels. Line 404-405 explains only the loss of *popular* known facts.
**Severity: minor.**
**Fix.** Give the 95% interval for the 0.43 cell value in the text. Explain the fall from 87 to 27 in one clause, or lead the abstract with the design-level effect.

### P3. Referential ambiguity (homonymy) is an unexcluded alternative cause
**Point.** Low-view subjects may be obscure bearers of common names. "I don't know" could then reflect referential ambiguity rather than familiarity. No homonymy control was run, and the CLARIFY judge label is not human-validated.
**Verdict: unresolved.**
**Evidence.** The text never names referential ambiguity as an alternative. It runs no homonymy control (for example, number of Wikidata entities sharing the label, or disambiguation-page status). The CLARIFY label (line 55) is a judge category that no human checked (lines 57, 130). The text does contain indirect evidence against a crude version of this alternative, but the authors never use it for that purpose:
- Llama describes the intended entity for 0.83 of obscure-known subjects from the bare query "What is {subject}?" (lines 411-414). A name that resolved mostly to a more famous homonym would score poorly here.
- Counting CLARIFY replies as abstentions moves the difference only from +0.30 to +0.31 (Table 1), so explicit "which one do you mean" replies are rare.
- Under the forced prompt, the same question yields the correct answer in the greedy reply and in at least 90% of samples (lines 43-44). So the model's dominant reading is the intended referent.

None of this excludes ambiguity-aware hedging: a model that knows both referents may say "I don't know" without asking for clarification. Line 487 confirms that common-word subject names exist in the item set.
**Severity: major.** The title's correlational claim would still hold if ambiguity explained the effect. But the interpretation of the abstentions as misplaced, and the misfire reading, would not.
**Fix.** Before submission, add an ambiguity covariate and re-estimate, at least on the two-judge measure. Candidates: count of Wikidata items with the same English label, or a flag for subjects with a Wikipedia disambiguation page. If that cannot be done, name ambiguity as an unexcluded alternative in the limitations and cite the three pieces of indirect evidence above.

### P4. Prompt asymmetry: the bare free prompt lets ambiguity surface, while "The answer is" forces commitment
**Point.** Knowledge is labelled under a forced-answer prompt and abstention is measured under a bare prompt. A cell can therefore be "known" under commitment yet hedged when the model is free.
**Verdict: partially answered.**
**Evidence.** The asymmetry is a design choice, and it is justified: "Abstention needs the option to decline" (line 51). The one-prompt check (lines 83-85, 409-410) shows that within the free prompt itself, the first token's lean towards "I" follows popularity (+8.4 against +3.1), while its entropy follows knowledge. So the dissociation does not come from comparing two prompts. What is missing: the check runs on the same bare prompt, so it cannot separate familiarity from ambiguity. The free-reply analysis (lines 423-425) also leaves an opening: only 0.72 of non-abstaining obscure-known replies contain a gold alias, and the text does not say whether the other 0.28 address a different referent.
**Severity: minor.** The ambiguity part is covered by P3.
**Fix.** Report what the non-abstaining obscure-known free replies without a gold alias are about: the intended entity answered wrongly, or a different referent.

### P5. Equating surface form removes 43% of the matched-pair gap
**Point.** At equal surface form, the matched-pair abstention gap falls from +0.76 to +0.43. Much of the popularity effect therefore lies in the name itself.
**Verdict: partially answered.**
**Evidence.** The memo's arithmetic is correct ((0.76 - 0.43)/0.76 = 43%, lines 431-432). The residual gap's interval excludes zero (+0.43 [+0.18, +0.69]). In the full-sample regression, adding word frequency, token count and token rarity moves the popularity coefficient only from -1.87 to -1.79 (lines 428-430). What is missing: the result is post hoc, appears only in Appendix A, and is absent from the main text's list of checks. Nothing reconciles the 4% attenuation in the regression with the 43% attenuation in the 55 matched pairs. The sentence at lines 430-432 is also unclear about which pairs the +0.76 refers to.
**Severity: minor.** The popularity effect survives both analyses, but a sizeable name-form share goes unreported in the body.
**Fix.** Add one main-text sentence stating both estimates and why they differ (sample, matching versus covariate adjustment). Note that a name-form share is consistent with both familiarity and ambiguity.

### P6. No identified cause: what remains is a page-view correlate of one model's hedging
**Point.** Steering and the lens isolate nothing, and the effect appears in one model. The paper therefore delivers a correlate, not a mechanism.
**Verdict: partially answered.**
**Evidence.** The paper claims a correlate and says so. The title says "Tracks". Section 3.3 is titled "Mechanistic tests narrow the reading". Lines 106-107 say steering "moves a generic abstention signal rather than isolating a familiarity cause". Line 95 says "The popularity preference is thus Llama's, not general", and lines 132-133 say "Neither steering nor the lens isolates a familiarity cause". The call explicitly invites early-stage results and negative findings, so a controlled single-model dissociation with honest negative mechanistic results is in scope. What is missing: the conclusion slips into causal language. It calls popularity abstention's "main driver" (line 135), and that wording, together with the misfire framing in P1, claims more than a correlate.
**Severity: minor.**
**Fix.** Replace "misses its main driver" (line 135) with correlational wording, such as "misses the variable it tracks most strongly in Llama".

## Summary

| Class | Count | Points |
|---|---|---|
| Answered by current text | 0 | none |
| Partially answered | 5 | P1 (major), P2 (minor), P4 (minor), P5 (minor), P6 (minor) |
| Unresolved | 1 | P3 (major) |

No point is critical.

**Does the paper survive?** In my reading it survives as a 4-page workshop extended abstract only if it changes what it says it contributes. The descriptive headline holds up against the memo: in Llama-3.1-8B-Instruct, uninstructed "I don't know" tracks page-view popularity more than sampled knowledge, while forced-answer entropy tracks knowledge across four families. It is carefully measured and survives nine measurements and labellings. The authors say plainly that it does not generalise (Gemma) and that the mechanistic tests fail. These are exactly the early-stage and negative results the call invites. The memo does land two blows. First, the paper presents itself as a *test of the familiarity-misfire hypothesis* (lines 34-35, 123), yet its own data never show that the obscure names are unfamiliar to Llama, and on entity recognition they point the other way (P1). Second, referential ambiguity is an obvious alternative that the text neither names nor controls (P3). It leaves useful indirect evidence on that question unused. Neither blow overturns the title claim. Both strike at what the finding is taken to mean, and the "deciding observation" in the abstract also rests on a thin, unintervalled cell (P2). So partial and unresolved points do touch the headline: P2 touches the abstract's lead number, and P1 touches the stated contribution. A senior reader who takes the contribution sentence at face value would score this below the bar. A reader who takes the title would accept, provided the framing is corrected.

## Action items (priority order)

1. **Reframe the contribution (P1, P6).** Rewrite lines 34-35 and 123 so the paper reports a popularity-versus-knowledge dissociation that the misfire idea predicts, not a test that the idea passed. State in Section 3.1 that neither model-side measure shows unfamiliarity: surprisal is null, and entity recognition is 0.83 for obscure-known against 0.75 for popular. Remove "main driver" (line 135).
2. **Deal with referential ambiguity (P3, P4).** Preferably add a homonymy covariate (same-label Wikidata count or disambiguation flag) and re-estimate the two-judge difference. At minimum, name ambiguity in the limitations as an unexcluded alternative and cite the indirect evidence: entity description at 0.83 from a bare query, CLARIFY adding +0.01, and forced-prompt correctness of at least 90%.
3. **Firm up the numbers behind the headline (P2, P5).** Give the interval for the obscure-known 0.43 and explain the fall from n = 87 to n = 27. Move the surface-form result (+0.76 to +0.43 in matched pairs, against a 4% shift in the regression) into the main text with one sentence reconciling the two.
