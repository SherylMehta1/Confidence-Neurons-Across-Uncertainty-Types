# Area-chair adjudication: attack memo vs. round-2 manuscript

Venue: UniReps 2026 workshop (NeurIPS), 4-page extended-abstract track. Sources: manuscript.txt line numbers (L), manuscript.pdf pp. 9-10 for Table 1 and Table 2 layout. No other files opened. No outside references were needed. A note on quoting: the memo puts "by construction" in quotation marks as if the authors wrote it. The text says entropy "is expected to follow knowledge" (L52) and "this is expected from how the labels are built" (L79-80). The substance is the same, but the wording is the memo's own.

## Points

### P1. The entropy half of the title is circular
**Point.** Knowledge labels are sampled from the same forced-answer distribution whose first-token entropy is the "answer uncertainty" readout. So "answer uncertainty tracks knowledge" is a consequence of the definition, not a finding, and the title's dissociation is half tautology.

**Verdict: partially answered.**

**Evidence.** The text concedes the circularity and recasts entropy as a "reference readout" (L49-53, L79-80, L130). The abstract flags that entropy is "measured on the prompt that defines the knowledge labels" (L7). The circularity is not complete. "Unknown" means zero of ten samples are correct (L45), which still allows a confidently wrong, low-entropy answer, so the size of the knowledge effect is partly empirical. The abstention half is also not definitional. It is measured on a different prompt and labelled by judges who are independent of the knowledge labels (L54, L58-61), so the memo's claim that "the headline dissociation comes from how knowledge is defined" overreaches. The remaining problems:
- The title and abstract still present entropy-follows-knowledge as a co-equal result, extended "as it does in three other families" (L8; also L33).
- The text reports no uncertainty readout that is independent of the labels.
- L55-57 shows that a hedge readout on the label prompt follows popularity and knowledge about equally (+2.93 vs +3.05). Part of the contrast between abstention and entropy is therefore a change of prompt, not a change of variable.

**Does the deliberate choice survive?** Using entropy as a reference readout is defensible. Putting it in the title as a finding is not.

**Severity:** major. It touches the headline wording.

**Fix.** Recast the title and abstract so that entropy is an expected reference readout, not a finding, or replace it with a readout independent of the labels, such as entropy over held-out samples or on a paraphrased prompt.

### P2. "Known" is not "stored", and the guessability filter is narrow
**Point.** For obscure entities, a consistent correct forced answer may be a confident guess from the name or the relation. The post hoc filter catches only alias-in-name answers and modal answers covering at least 20% of a relation, and removes 30 of 87. On the remaining 57, abstention could be an accurate report that nothing is stored. No independent test of knowledge is given.

**Verdict: partially answered.**

**Evidence.** The filter and its effect are described accurately by the memo (L81-84, L405-407). After filtering, the difference stays at +0.30 [+0.19, +0.41] (n = 335), and obscure known facts still draw "I don't know" in 0.44 of replies. The strongest reply to the memo is in Appendix A, not the main text: among unknown facts alone, the popularity gap in abstention is +0.34 [+0.26, +0.43], "which mislabelled guesses cannot produce" (L407-408). The memo's alternative ("abstention accurately reports that no fact is stored") predicts equal abstention for popular-unknown and obscure-unknown facts. The data show 0.13 vs 0.48 (L72-74). So the alternative cannot explain the popularity effect, and the memo's "opposite of the paper's conclusion" overreaches.

However:
- The rebuttal still depends on alias grading, which the authors say is unvalidated (L129-130).
- The memo is right that the filter does not catch answers inferable from the name, such as nationality or language guessed from a name that does not contain the answer.
- The claim that most catches the eye ("I don't know" about facts it knows: 0.43/0.44, L73, L84; "even for facts the model always answers correctly", L31) rests on 57 items with no paraphrase or related-fact test. I searched the text and found none.
- The name-spelling check (L411-413) is post hoc and also small (36 to 55 pairs).

**Does the deliberate choice survive?** The operational definition of "known" (L44-45) survives as an operational label, but not as "stored". The text's own wording drifts toward "stored" (L21-22).

**Severity:** major. It touches the headline, specifically the "IDK about known facts" reading.

**Fix.** Move the unknown-only gap into Section 3.1 as the test that guessing cannot produce, call "known" "consistently answered correctly under the forced prompt", and add a paraphrase or related-fact consistency check on the 57 remaining obscure-known facts.

### P3. The deciding result comes from one model
**Point.** The behavioural dissociation, abstention following popularity, is measured in Llama only.

**Verdict: answered by current text.**

**Evidence.** The scope is stated in the title ("In Llama"), in the abstract (L4, L13-14: "Where abstention could be measured"), and in the limitations (L129: "Abstention is measured reliably in one model"). The restriction is justified rather than simply chosen. The other three families fall below the pre-set judge-agreement bar (κ = 0.30, 0.47, 0.39; L62-64), and they barely abstain (≤ 0.10 of replies even in the obscure-unknown cell; L418-419, Table 2). A single-model behavioural result, honestly scoped, is acceptable for an early-stage workshop abstract. One residual overreach is minor: L134 ("calibration work should measure both") generalises beyond Llama.

**Severity:** n/a. At most minor wording at L134.

**Fix.** Scope L134 to "in Llama" or phrase it as a hypothesis for other models.

### P4. Judges at κ = 0.62, never checked against human labels
**Point.** The abstention labels in the deciding cell come from two LLM judges with moderate agreement and no human validation.

**Verdict: partially answered.**

**Evidence.** The text discloses both facts (L62-64, L129-130). It mitigates them with measurements that converge. A non-LLM regex detector on the same 96-token replies gives +0.43 [+0.31, +0.55], and on 20 replies × 2,303 questions it gives +0.44 [+0.34, +0.53] (Table 1, p. 10). The regex has 0.99 recall against the two-judge labels (L403-404). All five measurements are positive, with intervals that exclude zero. This makes it unlikely that the direction is an artefact of the judges.

It does not validate any detector against people. The detectors also disagree sharply on levels: in the obscure-unknown cell, regex flags 0.85 and both judges flag 0.48 (Table 2). This bears on the magnitudes the abstract quotes (+0.39, +0.09, "about a quarter", L5, L75).

**Severity:** minor for the directional claim, major for the quoted magnitudes.

**Fix.** Have humans label a stratified sample of free replies (about 25 per cell), report precision and recall for each detector, and state which magnitude the human labels support.

### P5. The effect size depends on the measurement (+0.13 to +0.44)
**Point.** Alternative measurements place the popularity-minus-knowledge difference anywhere from +0.13 to +0.44, so the effect is not pinned down.

**Verdict: partially answered.**

**Evidence.** The paper claims sign robustness only ("keeps its sign under four other measurements", L76). The two-judge, 96-token measurement was designed and made primary before it ran (L401-403), and its criterion note is hashed (L466-467). That answers any charge of selecting the measurement after the fact. The explanation for the +0.13 outlier is incomplete, though. It is attributed to 24-token replies being "cut off before the model commits" (L401-402). Yet the regex detector on the same 24-token replies gives +0.44 (Table 1). The low value therefore comes from the judge-based correction, not from truncation, and the text does not explain this. The abstract's "far more" (L5) is also not true under every measurement.

**Severity:** minor. The sign holds everywhere, and the primary measurement was pre-specified.

**Fix.** Report the +0.13 to +0.44 range alongside the primary estimate, soften "far more" to "more", and explain why the corrected judge and the regex disagree on identical replies.

### P6. The cue-familiarity conclusion is not established
**Point.** Given P1 to P5, the claim that abstention follows familiarity, resembling the human cue-familiarity heuristic, is not established.

**Verdict: partially answered.**

**Evidence.** For the behavioural correlation, the text supports more than the memo allows:
- the popularity effect within unknown facts (L407-408);
- the effect survives guessability filtering (L83-84);
- the effect is nearly unchanged after controlling for the name's word frequency, length and rarity (L85-86, L414-415).

The interpretive step is weaker. "Familiarity" is a page-view proxy (L42). The text never separates familiarity from general knowledge about the entity, meaning how much else the model knows about it. The one mechanistic prediction taken from the human account failed (onsets 1 support / 1 contradict / 6 undecided, L116-119). The supporting mechanisms are weak:
- In Llama, a steering direction fitted only on questions it never abstained on adds only +0.04 (L103-105).
- The lens readout may reflect answer-token rarity (L131).
- The circuit was dropped for missing its bar (L459-460).

The text hedges with "resembling" (L15) and "a correlate of uncertainty rather than a representation" (L127). But the title asserts "Tracks Familiarity", and L87-88 says the account "predicts" the result.

**Does the position survive?** As an analogy, yes. As an established mechanism, no.

**Severity:** major. It touches the headline (the title's wording).

**Fix.** Say "entity popularity" in the title and results, and keep cue-familiarity as an interpretation the data resemble, stating that the timing prediction failed and that popularity is not separated from general entity knowledge.

## Summary

| Class | Count | Points |
|---|---|---|
| Answered by current text | 1 | P3 |
| Partially answered | 5 | P1 (major), P2 (major), P4 (minor/major), P5 (minor), P6 (major) |
| Unresolved | 0 | none |

**Survival judgement.** A senior reader would find that the memo overreaches on its two strongest rhetorical claims. First, the dissociation is not "from how knowledge is defined": abstention is measured on a separate prompt by labellers independent of the knowledge labels. Second, the guessing alternative cannot produce the +0.34 popularity gap among unknown facts (L407-408), and that gap is guess-proof as long as alias grading holds. The core empirical observation survives. In Llama-3.1-8B-Instruct, free-reply "I don't know" varies with entity popularity more than with forced-answer correctness, in the same direction under five measurements, one of them a non-LLM detector. That is a legitimate early-stage result for a 4-page UniReps abstract, and the paper is unusually candid about its post hoc checks, its failed prediction and its missing validations.

The same reader would side with the memo on three things that all touch the headline:
- The entropy half of the title is a reference readout expected from the label construction, presented as a finding (P1).
- "Known" is operational, and the most striking reading, "I don't know" about facts it knows, rests on 57 filtered items with no independent knowledge test (P2).
- "Familiarity" is an interpretation of a popularity proxy whose one mechanistic prediction failed and whose supporting mechanisms are weak (P6).

On the current text I would expect acceptance as a workshop abstract only if the title and abstract are narrowed. As written, the title claims more than the evidence licenses on both of its halves.

## Action items (priority order)

1. **Narrow the headline.** Retitle and re-abstract around what is measured, for example: "In Llama, saying 'I don't know' follows entity popularity more than knowledge". Present entropy as the expected reference readout, keep cue-familiarity as a resemblance, and drop "far more".
2. **Promote the test guessing cannot produce, and tighten "known".** Move the unknown-only popularity gap (+0.34 [+0.26, +0.43]) into Section 3.1, define "known" operationally throughout, and if any compute is available, run a paraphrase or related-fact consistency check on the 57 remaining obscure-known facts.
3. **Validate the abstention measurement.** Human-label a stratified sample of free replies against both judges and the regex. Report the range across measurements (+0.13 to +0.44, sign constant) and explain why the corrected judge and the regex disagree on the same 24-token replies.
