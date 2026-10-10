# Referee report: framing and novelty lens

**Manuscript:** "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge" (UniReps 2026, extended-abstract track)

## Claim vs shown (first pass: abstract, figures, conclusion)

The paper claims that Llama-3.1-8B-Instruct's free-form "I don't know" follows entity popularity more than whether it knows the fact, while first-token answer entropy follows knowledge in four families. It also claims that the abstention pattern is Llama-specific and is not inherited from base models. The Llama dissociation is well supported: Fig. 1a/b, Table 1 and five robustness checks back it. The cross-family abstention contrast and the base-versus-instruct reading go further than the data. The first rests on a model that almost never abstains. The second rests on a summary of Fig. 3 that does not match panel (a).

## Closest prior work (verified)

- **Lindsey et al. 2025** (transformer-circuits.pub, opened). This paper proposes that "known entity" features suppress a default "can't answer" circuit and that hallucinations come from a misfire of that circuit (the Karpathy example). The manuscript attributes the idea correctly.
- **Ferrando et al. 2025** (ICLR 2025, arXiv 2411.14257, opened). It finds SAE entity-recognition directions that steer refusal versus hallucination. The manuscript's description is fair, but the reference omits the venue.
- **Brzezinka 2026a** (arXiv 2607.13568, opened). It uses page-view-tiered familiarity probes across 12 models and steers refusal in Gemma-4. Its "known" means a high page-view tier, not answer correctness, so the manuscript's "without a knowledge axis" is fair.
- **Brzezinka 2026b, "Does Bielik Know What It Doesn't Know? Activation Dispersion Separates Entity Familiarity from Factual Reliability"** (arXiv 2607.07670, opened). This one is not cited. It separates familiarity from factual reliability at the representation level, but its models almost never abstain (2 refusals in 2,520 answers).
- Also verified and fairly cited: Kumaran et al. 2026 (2603.22161), Ni et al. (2505.17537, listed as EMNLP 2026 Main), Che & Li 2026 (2610.07270), Nguyen et al. 2026 (2609.32964) and Gurnee et al. 2026 (2607.15495, Jacobian lens). Possibly relevant and uncited: Kang et al. 2024, "Unfamiliar finetuning examples control how language models hallucinate" (arXiv 2403.05612). It traces "I don't know"-type behaviour to how fine-tuning treats unfamiliar examples, which bears on the origin of the Llama-specific pattern.

The core novelty survives this search. I found no prior work that crosses popularity with fact-level knowledge labels and measures behavioural abstention in the off-diagonal cell (obscure entity, known fact).

## Findings

**F1 (major): the cross-family abstention contrast rests on a floor.**
- **Location:** Abstract l.16-17; §3.2 heading and l.90-95; Conclusion l.136-137; Table 2.
- **Evidence:** "at most 0.11 even for obscure, unknown facts" (l.93); "That only the first holds in every family suggests the two are shaped separately" (l.136-137).
- **Problem:** Gemma's highest-abstention cell is 0.05 with two judges and 0.11 with three. A popularity-minus-knowledge difference is therefore bounded near zero by construction. "Shows no popularity preference" really means "rarely abstains". Qwen and Mistral are unmeasured. So the claim "different abstention" across families rests on one comparison that sits near the floor. That comparison is the paper's main cross-model, UniReps-relevant claim.
- **Fix:** Reword the abstract and §3.2 as "Gemma rarely abstains (at most 0.11), so its preference cannot be estimated; Qwen and Mistral could not be measured." Retitle §3.2, for example "Shared answer uncertainty; abstention measurable only in Llama". Drop "shaped separately" from the conclusion or mark it as speculation. Alternatively, lift the floor with a prompt that explicitly permits "I don't know" and report Gemma's preference there.

**F2 (major): the base-model lens summary contradicts Fig. 3a, and the base-versus-instruct inference has no matched instruct readout.**
- **Location:** Abstract l.14-15; §3.3 l.114-118; Fig. 3 caption; Tables 4, 5 and 2.
- **Evidence:** "Llama-Instruct's preference for popularity over knowledge is not seen in base models" (l.117-118), against Table 4, where base Llama-3.1-8B's full readout gives popularity +0.67 [+0.59, +0.75] and knowledge +0.36 [+0.28, +0.44].
- **Problem:**
  - On the full readout, popularity exceeds knowledge with non-overlapping intervals in three of four base models: Llama +0.67 vs +0.36, Gemma-2-2B +0.40 vs +0.11, OLMo +0.29 vs −0.30.
  - "About as much" holds only for the hedge-term panel (3b). Even there, Gemma-2-2B goes the other way (+0.10 vs +0.37).
  - The contrast pairs a base-model lens readout with instruct behavioural abstention. The paper's own forced-answer hedge readout in Llama-Instruct shows no preference either (Table 2: −0.12 [−0.72, +0.45]). So the readout, rather than instruction tuning, may be what lacks the preference.
  - Table 5 reports an instruct mid-depth popularity effect (+1.58 for Llama-Instruct) but no matching knowledge effect, so the comparison that would decide the question is missing.
- **Fix:** State which readout was pre-specified as primary. Add Table 4 rows for the instruct models (the data appear to exist). Rewrite l.14-15 and l.116-118 to describe both panels accurately. Remove the implication that the preference arises in instruction tuning unless the instruct rows show it.

**F3 (major): the "main driver" significance claim generalises beyond an extreme-tails design, and the limitations omit that design.**
- **Location:** Conclusion l.134-136; Setup l.39-45; Appendix A l.426-427; Limitations l.130-133.
- **Evidence:** "predicting abstention from answer uncertainty misses its main driver; calibration work should measure both" (l.134-136).
- **Problem:** The design keeps only the bottom and top page-view deciles. It keeps only facts that are known (at least 90% correct) or unknown (0% correct), drops the rest, and balances cells at up to 100 each. Effect sizes from such a design say nothing about which factor dominates in the natural question distribution. The one direct comparison is in Appendix A: popularity +0.29 as a decile contrast against entropy +0.14 per SD. These are not on a common scale, and the entropy term also shows that abstention partly does track answer uncertainty. Neither the extreme design nor the dropped intermediate band appears in Limitations.
- **Fix:**
  - Add both design restrictions to Limitations.
  - Rephrase as "in this extreme-contrast design, popularity accounts for more abstention than knowledge".
  - Put both predictors on a standardised scale.
  - Optionally, show the gradient across all page-view deciles using the regex detector, whose recall against the judges is 0.99.

**F4 (major): the fit to UniReps is asserted rather than developed.**
- **Location:** Introduction l.23-24; §3.2; Conclusion; Appendix E.
- **Evidence:** "and whether different models follow the same variable, is less clear" (l.23-24) is the only explicit link to the venue's question.
- **Problem:** UniReps asks whether different models learn the same representations. The paper's cross-model evidence is behavioural: per-model entropy effects and abstention in two families. The lens magnitudes are "not comparable across models" (Fig. 3 caption). The only item-level cross-model analysis is in Appendix E (Spearman agreement of entropy and hedge readouts between models), and it is post hoc and absent from the main text. No work on representational convergence is cited.
- **Fix:**
  - State the convergence question in the introduction and the conclusion: models converge on what drives answer uncertainty but not on the abstention policy.
  - Bring one sentence of Appendix E into §3.2, labelled post hoc. For example: entropy agreement is +0.46 to +0.69 on same-label items.
  - Position the result against one or two works on representational convergence.

**F5 (minor): the novelty sentence overgeneralises about prior tests.**
- **Location:** Introduction l.28-30; Related work l.124.
- **Evidence:** "Tests of this idea define an entity as known when the model answers about it correctly … so familiarity and knowledge move together".
- **Problem:** Brzezinka 2026a, which the paper cites, defines familiarity by page-view tier rather than by correctness. Brzezinka 2026b (arXiv 2607.07670, uncited) separates familiarity from factual reliability, though only in representations and in models that almost never abstain. The paper's real novelty is narrower and stronger: a fact-level factorial measured on behavioural abstention.
- **Fix:** Cite 2607.07670. Rephrase along the lines of "representation-level work separates familiarity from correctness, but no study has crossed them in abstention behaviour".

**F6 (minor): which misfire prediction each cell tests is left implicit.**
- **Location:** Introduction l.27-34.
- **Evidence:** "a familiar name can trigger them without the fact" (l.28) vs "The deciding observation is … facts about obscure entities" (l.33-34).
- **Problem:** Lindsey et al.'s misfire is the popular/unknown cell, where a familiar name with an unknown fact leads to hallucination. The "deciding" cell is the converse prediction: without familiarity the default stays on despite knowledge. Readers may think the paper tests the misfire cell directly.
- **Fix:** One sentence mapping the obscure/known cell to "default not released" and the popular/unknown cell to "misfire" (abstention 0.13 there). Report the hallucination rate in that cell.

**F7 (minor): referent ambiguity is an unaddressed alternative label for "obscure".**
- **Location:** §3.1 l.86-88; Appendix F l.486-487.
- **Evidence:** "Short names made of common words ("New York" as a book title) are a likely reason."
- **Problem:** Obscure PopQA subjects often share a name with a more popular referent. "I don't know" may then express uncertainty about which entity is meant, not missing familiarity. The CLARIFY category and the entity-knowledge check (0.83) argue against this but do not rule it out.
- **Fix:** Report or control for the number of Wikidata items sharing the subject label. Otherwise, name the alternative in Limitations.

**F8 (minor): the count of robustness measurements is inconsistent with Table 1.**
- **Location:** §3.1 l.69-70; Table 1.
- **Evidence:** "keeps its sign under four other measurements, which range from +0.13 to +0.44".
- **Problem:** Table 1 lists five alternative measurements besides the primary one (regex at 24 tokens, corrected judge, DONT_KNOW or CLARIFY, regex at 96 tokens, three judges), plus three relabelling or control rows.
- **Fix:** Correct the count or say which rows are meant.

**F9 (minor): the abstract implies independent labels in all four families.**
- **Location:** Abstract l.10-11.
- **Evidence:** "in four model families and under the independent labels".
- **Problem:** The relabelling was done only for Llama (§3.1, App. A).
- **Fix:** "in four families, and in Llama under independent labels".

**F10 (minor): "only through its abstention component" holds for Llama only.**
- **Location:** Abstract l.13-14; Appendix C l.453-457.
- **Evidence:** Gemma "with abstention removed … +0.02 [+0.01, +0.03]" (three judges: +0.05 [+0.03, +0.07]).
- **Problem:** Gemma's residual interval excludes zero. The Appendix C sentence also repeats "with abstention removed" for two different scorers, which makes it hard to read.
- **Fix:** Scope the abstract to Llama, or say "mostly". Split the Appendix C sentence by scorer.

**F11 (minor): the depth prediction is attributed to Koriat & Levy-Sadot.**
- **Location:** §3.3 l.118-120.
- **Evidence:** "The human-memory prediction that familiarity acts at a shallower depth than knowledge [Koriat and Levy-Sadot, 2001]".
- **Problem:** That work concerns how cue familiarity and accessibility combine in feeling-of-knowing judgements. Layer depth is the authors' analogue, not a stated prediction of the cited work. [unverified whether a timing claim appears in that paper]
- **Fix:** "our depth analogue of the claim that familiarity precedes or gates accessibility".

**F12 (minor): the one-prompt entropy effects are small relative to the headline.**
- **Location:** §3.1 l.83-85; Appendix A l.410.
- **Evidence:** Free-prompt entropy: knowledge "+0.16 against +0.05"; difference −0.11 [−0.18, −0.03].
- **Problem:** The abstract says entropy "follows knowledge" at this prompt on a 0.11-nat difference, against 2.14 nats at the forced-answer prompt.
- **Fix:** Give the magnitudes in the abstract or in §3.1, so readers do not equate the two.

**F13 (minor): reference metadata.**
- **Location:** References.
- **Problem:** Ferrando et al. is cited by OpenReview link only (it is ICLR 2025). Ni et al. is dated 2025 but annotated EMNLP 2026. Kuhn et al. lacks its venue (ICLR 2023).
- **Fix:** Complete the entries.

## Strengths

1. A clean factorial design that isolates the decisive off-diagonal cell (obscure entity, known fact). As far as I can verify, nobody has tested this behaviourally before.
2. The title and abstract scope the headline claim to Llama, and negative results are reported plainly: steering falls short of its bar, the depth prediction failed, and name surprisal is a negative result.
3. Robustness is unusually thorough for a 4-page abstract (Table 1). It includes independent labels and an entity-knowledge covariate that directly target the obvious confounds.
4. Analyses are separated into pre-specified and post hoc (Appendix H), and the steering appendix discloses that an earlier version overstated the effect.
5. The contributions are stated with section pointers (l.34-37), and the setup admits that no human validated the judges.

## Questions (decision-relevant)

1. Do the instruct-model lens data behind Table 5 give knowledge effects in the mid-depth window? If Llama-Instruct also shows popularity roughly equal to knowledge on the lens, the base-versus-instruct sentence (F2) has to go.
2. With a prompt that explicitly permits "I don't know", does Gemma abstain at a measurable rate, and does a popularity preference then appear? The answer decides whether the cross-family claim (F1) holds.
3. Which base-model readout, full or hedge term, was the pre-specified primary?

## Recommendation: minor revision

The core contribution is well supported, novel against the closest verified prior work, and a good fit for the workshop's call for early-stage results: Llama's abstention follows popularity, its answer entropy follows knowledge, and the obscure/known cell shows it. The decision is driven by F1, F2 and F3. Each overstates a secondary claim: the cross-family abstention difference, the base-versus-instruct origin, and the "main driver" significance. F4 is needed to make the UniReps link explicit. For a 4-page non-archival abstract, all four can be fixed mainly by re-scoping text, plus one table built from data the authors appear to have (instruct lens rows). If those claims are re-scoped, I would support acceptance.
