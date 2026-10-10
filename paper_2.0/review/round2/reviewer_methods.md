# Referee report: "In Llama, Saying 'I Don't Know' Tracks Familiarity While Answer Uncertainty Tracks Knowledge"

UniReps 2026 extended-abstract track. Lens: methods and statistics.

## What it claims and what it shows

**Claims.** In Llama-3.1-8B-Instruct, uninstructed "I don't know" follows entity popularity (a familiarity proxy) far more than whether the model knows the fact, while first-token answer entropy follows knowledge in four families. A steering direction that separates obscure from popular entities causes abstention on known facts. Base models' lens-decoded hedge readout already responds to popularity.

**Shows.** A balanced 2x2 (popularity decile x self-sampled knowledge) for one model, where obscure *known* facts draw abstention at 0.43 against 0.00 for popular known facts. The sign holds across five abstention measurements and two post hoc guessability checks. The steering and lens results are weaker than the abstract suggests. The steering effect largely disappears once abstaining items are removed from the fit. The lens readout is built against the answer token, so it is confounded with answer-token rarity, a confound the paper admits but does not test.

## Findings

**F1 (major). The cell that carries the claim has knowledge labels from a different prompt.**
- *Location:* Sec. 2, l.42-45 and l.49-54; Sec. 3.1, l.72-74.
- *Evidence:* "A fact is known if the greedy answer and at least 90% of samples are correct" (forced-answer prompt); abstention is measured with "the chat template alone".
- *Problem:* The only comparison that separates familiarity from answer uncertainty is in the known arm: obscure known 0.43 vs popular known 0.00, with nearly equal entropy (Fig. 1b, about 0.95 vs 0.85 nats). That cell has only 87 questions, 57 after removing guessable ones. "Known" means the model completes "The answer is ..." correctly. The best alternative is not ruled out: for obscure entities, the prefix may draw out an answer that the model does not retrieve, or would not commit to, under the free prompt. If so, abstention would track free-prompt retrievability, which is knowledge, and the result would be an artefact of the prompt. The guessability filter covers only two shortcut types: an alias inside the name, and the relation's modal answer. It does not cover name-based inference such as nationality from a surname.
- *Fix:* Report how accurate the non-abstaining free replies are in the obscure-known cell. Alternatively, relabel knowledge from free-prompt samples, for example answer-forcing only after a free reply, and recompute the 2x2. If obscure-known free answers are mostly correct, the claim is much stronger.

**F2 (major). The test against "abstention tracks answer uncertainty" uses the 2x2 knowledge contrast, not item-level uncertainty, and popularity is confounded with entropy in the unknown arm.**
- *Location:* Abstract l.13-15; Sec. 3.1 l.77-80; App. A l.407-408; Fig. 1b.
- *Evidence:* "Among unknown facts alone the popularity gap in abstention is +0.34 [+0.26, +0.43], which mislabelled guesses cannot produce."
- *Problem:* Within the unknown arm, popular items have markedly lower answer entropy than obscure ones (Fig. 1b: about 2.6 vs 3.5 nats). This is the "popular but wrong" overconfidence that Ni et al. describe. The +0.34 gap is therefore what an answer-uncertainty account predicts too, so it cannot support "abstention follows a different variable from answer uncertainty". The binary knowledge label is a coarse stand-in for uncertainty, and the entropy reference is tied to that label by construction, as the paper itself notes at l.79-80.
- *Fix:* Fit an item-level model of abstention on popularity plus a same-item uncertainty measure, with relation fixed effects. Candidate measures are first-token entropy, or self-consistency of the 10 forced samples, which is informative within the unknown arm. Alternatively, compare entropy-matched obscure and popular items. Restrict the abstract's final sentence to what survives.

**F3 (major). Relation composition is uncontrolled in the headline 2x2 effects.**
- *Location:* Sec. 2 l.40-45 ("Within each relation, subjects in the bottom and top deciles..."); App. A l.414-415.
- *Evidence:* Relation fixed effects are reported only for the "pool-level popularity coefficient"; the +0.39/+0.09 effects are raw cell means.
- *Problem:* Knowledge filtering (0/11 or at least 10/11 correct) leaves different relation mixes in each cell. Obscure-known facts plausibly concentrate in easy relations such as country or sport. Abstention rates differ strongly by relation type (people vs. films vs. places). The cell-mean effects, and their item bootstrap, ignore this structure. The regression reports no knowledge coefficient, so it cannot back the comparison between the two effects.
- *Fix:* Tabulate relation counts per cell. Report within-relation (stratified or reweighted) popularity and knowledge effects, plus the knowledge coefficient from the fixed-effects regression. Use a bootstrap clustered by relation, or report its sensitivity.

**F4 (major). The primary abstention measure is not validated against humans, and the only externally anchored estimate is a third of it.**
- *Location:* Sec. 2 l.62-64; Table 1; App. A l.401-403.
- *Evidence:* "One LLM judge, corrected ... +0.13 [+0.03, +0.24]" vs primary "+0.30 [+0.20, +0.40]"; "The judges were not checked against human labels."
- *Problem:* Agreement between the two judges (kappa = 0.62, just over a 0.60 bar) measures how consistent they are with each other, not whether they are right. The "corrected" row is adjusted for "recall and precision ... measured on a labelled subset", but the paper never says who labelled that subset. If humans did, it is the paper's only human-anchored estimate, and it is the smallest. The 96-token two-judge measure was designed after this result was seen, even if it was hashed before it ran. That is a fork chosen after looking at the data. The truncation explanation also sits awkwardly with the regex row on the same 24-token replies, which gives +0.44.
- *Fix:* State who produced the labelled subset. Have humans label about 100-150 of the 96-token replies, stratified by cell, and report judge precision and recall against them. Then report the primary effect corrected for those error rates, as for the 24-token row.

**F5 (major). The steering comparison is made at a norm tuned for one direction on the evaluation set.**
- *Location:* Sec. 3.2 l.93-99; App. C l.426-433.
- *Evidence:* "the smallest multiple k ... at which that direction beat the mean of five random directions by 0.10 ... on the 150 evaluation questions; all directions were then scaled to that norm."
- *Problem:* The headline contrast is "far more than an unknown-minus-known direction". It is evaluated at a norm chosen so that the obscure-minus-popular direction succeeds, and on the same questions. The natural norms of the two directions are not reported. If unknown-minus-known has a larger natural norm, equal-norm steering under-doses it. In Gemma, k=1 and k=2 failed, so the reported effect only exists at 4x, which is outside the natural range (the paper cites Che and Li). With five random directions, the null is coarse.
- *Fix:* Report each direction's natural norm. Show dose-response curves for both directions at k in {1,2,4,8}, each scaled to its own natural norm and to a common norm, on questions not used to choose k. Report the Gemma 1x and 2x results.

**F6 (major). The steering outcome in Gemma uses a detector the paper rejects for Gemma elsewhere, and steered replies are not checked for coherence.**
- *Location:* Fig. 2a caption ("Abstention rate (regex detector)"); Table 2.
- *Evidence:* Table 2: "Regex recall vs two judges" is 0.48 for Gemma; in Llama the regex fires on 0.85 of obscure-unknown replies against 0.48 by both judges.
- *Problem:* The regex has high recall and unreported, evidently low precision: in Llama, about 40% of regex hits are not DONT_KNOW by the judges. In Gemma it misses half of the judge-labelled abstentions, and the judges themselves fail the paper's reliability bar. At 4x the natural norm, steered text may be degenerate or hedged-but-correct and still match the regex. The abstract's "makes ... Gemma-2 say 'I don't know' about facts they answer correctly" therefore rests on a measurement the paper does not trust for Gemma.
- *Fix:* Run the two-judge labelling, or a human spot-check, on steered replies. Report a coherence or perplexity check and the answer accuracy of non-abstaining steered replies. Otherwise, drop Gemma from the abstract's steering sentence.

**F7 (major). The never-abstained control selects on the outcome, so it cannot show that steering works "mostly through components shared with abstention".**
- *Location:* Intro l.36-37; Fig. 2 title; Sec. 3.2 l.102-105; App. C l.430-431.
- *Evidence:* "refits the obscure-minus-popular direction on working-half questions with no abstaining sample".
- *Problem:* Excluding abstaining items removes about half of the obscure items, namely the ones the model treats as least familiar. The refit direction therefore loses familiarity content as well as abstention content. A +0.04 result is explained equally well by "familiarity does not drive abstention" and by "the subset no longer carries the familiarity contrast". The paper's caveat at l.102 is right, but the intro and the Fig. 2 title assert the decomposition anyway.
- *Fix:* Fit an abstention direction (abstaining minus non-abstaining replies, within the obscure cell). Report its cosine with the obscure-minus-popular direction, and steer with the component orthogonal to it at matched norm. The limitations section (l.133) already names this test. Until it is done, soften the intro and the figure title.

**F8 (major). The lens readout is confounded with the answer token in both effects.**
- *Location:* Sec. 3.3 l.109-115; Limitations l.131-132; Fig. 3b.
- *Evidence:* "log-odds of a fixed set of hedge tokens against the answer token"; "the lens readout may partly reflect answer-token rarity".
- *Problem:* If "the answer token" is the gold answer's first token, two things follow. Obscure entities have rarer answer tokens, which raises hedge-vs-answer log-odds whether or not anything hedge-like is encoded. Unknown facts have a low answer-token logit by definition, so a positive knowledge effect is built in. The admitted confound could therefore produce claim (iii) entirely. OLMo's knowledge effect is strongly negative inside the pre-fixed window (Fig. 3b, about -1.5 near 0.55 depth). Under this readout that is hard to reconcile with a hedge signal, and it is not discussed. The paper also does not say whether "the answer token" is gold or model-predicted.
- *Fix:* Define the answer token. Decompose the readout into its hedge-token term and its answer-token term, each against a fixed baseline such as the mean logit over vocabulary. Show that the popularity effect lives in the hedge term, or add answer-token frequency as a covariate. Report and discuss the knowledge effects for all four models.

**F9 (minor). The effect comparison depends on scale and hides an interaction.**
- *Location:* Abstract l.19-21; Sec. 3.1 l.72-76.
- *Evidence:* "0.48 ... 0.43 ... 0.13 and 0.00"; "at about a quarter of the popularity effect".
- *Problem:* On the probability scale, with a cell at the floor, the knowledge effect is 0.05 among obscure items and 0.13 among popular ones. On the logit scale the popular-arm knowledge effect is large. The ordering popularity > knowledge plausibly survives, but "far more" and "a quarter" are specific to the probability scale.
- *Fix:* Report the conditional effects and the interaction with intervals, plus a logit-scale (or GLMM) check of the comparison.

**F10 (minor). The main text reports the name control that barely moves the estimate and leaves out the one that halves it.**
- *Location:* Sec. 3.1 l.85-86 vs App. A l.411-413.
- *Evidence:* "the abstention gap of +0.76 falls to +0.43 [+0.18, +0.69] at equal surface form."
- *Problem:* About 40% of the matched-pair gap goes away at equal surface form. That is relevant to how much of "familiarity" is lexical. Several things are undefined: "equal surface form"; which outcome the +0.76 gap and the -1.87 regression use; and whether +0.79 should be compared with +0.76 or +0.43.
- *Fix:* Report both checks in the main text, define the matching, and name the outcome measure for each.

**F11 (minor). Section 3.3 points to intervals and values that the appendix does not contain.**
- *Location:* Sec. 3.3 l.111-115; Table 3.
- *Evidence:* "all intervals excluding zero (Figure 3a; values in Appendix D)". Table 3 gives point estimates only, has no knowledge-effect column, and lists D without intervals even though the onset rule (l.436-438) needs them.
- *Fix:* Add window means with intervals for both effects and all models, and intervals for D. Report the number of labelled items per base model.

**F12 (minor). The conclusion over-reads the cross-model check and mislabels the readout.**
- *Location:* l.135-137; App. E.
- *Evidence:* "exploratory checks find this reliably for two of three pairs, though our pre-set rule held for one".
- *Problem:* "Two of three" counts only the hedge-agreement component; the pre-set rule, which combines hedge agreement and gap, holds for one pair. The readout used is the forced-answer hedge log-odds, which Sec. 2 (l.55-57) shows is *not* the abstention readout, yet l.136 calls it "abstention readouts".
- *Fix:* State that the pre-set rule held for 1/3 and call the readout "forced-answer hedge log-odds".

**F13 (minor). The Lippl et al. criterion is invoked without the test it requires.**
- *Location:* l.126-127.
- *Problem:* Source invariance (verified: Lippl et al. 2024, arXiv 2404.08101) asks whether a representation encodes uncertainty across *different sources* of uncertainty. The paper has a single source and a behavioural readout.
- *Fix:* Rephrase as motivation ("in the spirit of"), or drop the conclusion drawn from it.

**F14 (minor). Pre-registration claims cannot be checked from the page.**
- *Location:* App. G l.464-468.
- *Evidence:* "the SHA-256 prefix of each note was recorded before the run".
- *Problem:* A hash prefix without the note, and without a timestamped public record, does not let a reader check the timing. It is still good practice and should be credited.
- *Fix:* Deposit the notes, for example in an anonymous repository, or quote each criterion in one line. Also say whether the 150 held-out steering questions overlap the 387-question abstention sample, and list the hedge-token set.

## Strengths

1. **Design.** Crossing popularity with model-specific knowledge labels (Sec. 2, l.27-31) addresses the confound in definitions where an entity counts as known when the model answers about it. The obscure-known cell (0.43, Fig. 1a) is a genuinely informative observation.
2. **Honest scoping.** A reliability bar was set in advance and applied, abstention is reported for Llama only, and the title says "In Llama" (l.62-64).
3. **Robustness reported with the unflattering result included.** Table 1 includes the +0.13 corrected row alongside the +0.44 regex row.
4. **Negative results kept and labelled.** The failed onset prediction (l.116-119), the circuit injection that fell short at 0.40 (App. F, l.458-460), and the deflationary never-abstained control (l.102-105) all appear in the paper.
5. **Transparency about circularity.** Entropy is explicitly described as an "expected" reference readout (l.51-53, 79-80), and post hoc analyses are labelled as such throughout.

## Questions for the authors (decision-relevant)

1. In the obscure-known cell, when Llama does answer on the free prompt, what fraction of replies are correct? (F1)
2. Who labelled the "labelled subset" behind Table 1's corrected row? Can that labelling be used to score the 96-token two-judge labels? (F4)
3. Do the popularity and knowledge effects keep their ordering within relations, and does popularity predict abstention once item-level entropy or self-consistency is controlled in the unknown arm? (F2, F3)
4. In the lens readout, does the popularity effect sit in the hedge-token term or in the answer-token term? (F8)

## Prior-work checks

- **Verified (pages opened):**
  - Gurnee et al. 2026 (arXiv 2607.15495) introduces the Jacobian lens and says J-lens readouts for open models are on Neuronpedia. Which base models have prefitted lenses is [unverified].
  - Kumaran et al. 2026 (2603.22161) offers abstention explicitly ("choose '5'") on 4-way multiple-choice SimpleQA, matching l.123-125.
  - Brzezinka 2026 (2607.13568) separates familiarity readouts from the refusal policy and steers refusal, using real vs. fabricated entities, which is consistent with "without a knowledge axis".
  - Che and Li 2026 (2610.07270) state that steering can push a feature outside its natural range.
  - Ni et al. (2505.17537) find that confidence tracks answer popularity; the arXiv comment lists EMNLP 2026 Main.
  - Lippl et al. 2024: see F13.
- **[unverified]:**
  - Lindsey et al. 2025, that known-entity features suppress a default "can't answer" response.
  - Mallen et al. 2023 as the source for the claim that answers are guessable from the relation's commonest answer.

## Recommendation: major revision

**What supports the core claim.** The behavioural result for one model is supported at workshop standard, is honestly scoped, and is unusually transparent about which analyses were post hoc and which predictions failed.

**What drives the recommendation.**
- F1 and F2: the claim that abstention "follows a different variable from answer uncertainty" rests on one small cell whose knowledge labels come from a different prompt. In the unknown arm, popularity is confounded with answer entropy.
- F3 and F4: the 2x2 effects lack relation control and human validation.
- F5-F8: the steering claim (ii) and the lens claim (iii), as worded in the abstract and intro, go beyond what the designs can distinguish.

**What would make it acceptable.** Most fixes are analyses on outputs the authors already have: F1 (free-reply accuracy), F2 (item-level model), F3 (stratification), F8 (readout decomposition). The rest are rewording: F6, F7 and F12. With F1-F3 addressed and claims (ii) and (iii) softened, I would support acceptance.
