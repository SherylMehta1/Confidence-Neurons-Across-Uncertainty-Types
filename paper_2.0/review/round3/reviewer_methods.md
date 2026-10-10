# Referee report: methods and statistics

**Manuscript:** "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge" (UniReps 2026, extended-abstract track)

## What it claims and what it shows

**Claim.** In Llama-3.1-8B-Instruct, uninstructed abstention follows entity popularity more than whether the model knows the fact. First-token answer entropy follows knowledge in four families. Steering and lens analyses show no isolated familiarity cause, and Gemma-2-9B shows no popularity preference.

**What it shows.** A 2x2 PopQA design (387 questions, 3 replies each, two LLM judges) gives 0.43 abstention for obscure-known facts against 0.00 for popular-known facts. The popularity-minus-knowledge contrast is +0.30 [+0.20, +0.40]. It holds across several measurement and labelling variants (Table 1).

Overall, the Llama behavioural result is well supported for a workshop abstract. The weaker parts are:
- how far the headline contrast rests on two label-sensitive cells;
- several abstract-level secondary claims (the one-prompt split, "only through its abstention component", the Gemma null, the base-model comparison) whose evidence is selective or built in by construction.

## Findings

**F1 (major): the headline contrast reduces to two off-diagonal cells, and the popular-unknown cell is unaudited**
- **Location:** Sec. 2 l.58-60, Sec. 3.1 l.68-69, App. A l.420-421.
- **Evidence:** "Among unknown facts alone the popularity gap in abstention is +0.34 [+0.26, +0.43], which mislabelled guesses cannot produce."
- **Problem:** Each effect averages over the other factor. So popularity effect minus knowledge effect equals exactly P(abstain | obscure-known) minus P(abstain | popular-unknown): 0.43 - ~0.13 = +0.30.
  - Every "difference" in Table 1 therefore depends only on those two cells. These are the cells where label error is most plausible.
  - The obscure-known cell gets several checks (guessability, relabel, free replies containing the alias). The popular-unknown cell gets none.
  - Knowledge is graded by matching gold aliases. Popular entities plausibly have more valid answers outside PopQA's alias lists (occupation, genre, multiple places), so "unknown popular" can contain facts the model knows. That would lower abstention in that cell and inflate both the +0.30 and the within-unknown +0.34.
  - The defence "mislabelled guesses cannot produce" covers false-known errors only, not false-unknown errors. The relabel check uses the same alias grader, so it inherits the same confound.
- **Fix:**
  - State the identity in Sec. 2.
  - Audit the popular-unknown cell: have a stronger LLM judge or a human grade a sample of forced answers for correctness beyond the alias list, and report the share of free replies there that contain a gold alias (as is done for obscure-known, l.423).
  - Recompute the contrast without the items found to be mislabelled.

**F2 (major): knowledge-label reliability is unquantified, and the deciding cell mostly fails to replicate**
- **Location:** Sec. 3.1 l.74-76; App. A l.402-405.
- **Evidence:** "Agreement with the original labels is moderate, mainly because the paraphrase moves many popular known facts to the intermediate band"
- **Problem:**
  - Popularity is measured without error, but the knowledge label is noisy, and its agreement is given only as "moderate". Noise in one factor attenuates that factor's effect, which biases the comparison towards "popularity dominates".
  - Under the paraphrase labels the obscure-known cell has n = 27 of the original 87. Intermediate-band items are dropped and unknown-to-known flips are unlikely, so at most about 31% of the deciding facts keep their "known" label under a paraphrase.
  - The surviving 27 abstain at 0.46, which supports the claim. But the main text does not say that most obscure-known facts are prompt-fragile, and fragility is itself the main alternative to "the model knows it".
- **Fix:**
  - Report the 2x2 cross-tabulation of original against paraphrase labels, with kappa.
  - Report the abstention contrast on consensus labels (known or unknown under both prompts), with the per-cell n.
  - Say in Sec. 3.1 how many obscure-known facts survive.

**F3 (major): the "I don't know" measure appears to include hedged answers that contain the correct answer**
- **Location:** App. A l.423-425; headline 0.43 in the abstract l.5-6.
- **Evidence:** "In the obscure known cell, 0.53 of free replies contain a gold alias, and 0.72 of the replies that do not abstain"
- **Problem:** My arithmetic, assuming the same abstention definition: 0.57 x 0.72 ≈ 0.41 of replies are non-abstaining and contain the alias. That leaves ≈0.12 of all replies, or about 28% of the abstaining replies in the deciding cell, that both abstain and contain the gold alias. Either:
  - the judges label hedged answers ("I'm not sure, but it may be X") as DONT_KNOW, which is verbal uncertainty rather than declining and would cut the 0.43 to roughly 0.31; or
  - the alias matches are echoes of the subject name, which weakens l.423 as evidence that the model knows the fact in free form.
  
  No human checked the judges (l.57), so the reader cannot tell which.
- **Fix:**
  - Report the share of DONT_KNOW replies that contain the gold alias in each cell.
  - Add a strict "declines without giving any candidate answer" measure.
  - Hand-check a sample (for example 50 replies from the obscure-known cell).

**F4 (major): the one-prompt split in the abstract uses a prompt chosen after the forced-answer readout showed no split**
- **Location:** Abstract l.11-12; Sec. 3.1 l.83-85; App. B l.439-440; Table 2 bottom.
- **Evidence:** "Llama's hedge readout follows popularity and knowledge about equally, which is why the one-prompt check in Section 3.1 uses the free prompt."
- **Problem:**
  - At the forced-answer position, Llama's first-token hedge lean shows popularity +2.93 against knowledge +3.05 (difference -0.12 [-0.72, +0.45]). The free prompt was used because of this result. That is post hoc selection, but the check is labelled only "descriptive" and is missing from the post hoc list in App. H.
  - Table 2 also shows that Gemma's forced-position hedge readout strongly prefers popularity (+2.11 [+1.79, +2.42]) while its abstention does not. So the first-token lean tracks the behavioural preference in neither measured model at that position.
  - The abstract sentence "within one prompt, the first token's lean … follows popularity" therefore reports the favourable prompt only.
- **Fix:** Mark the check as post hoc. Report the forced-position result next to it in Sec. 3.1. Either drop the sentence from the abstract or qualify it ("at the free prompt; not at the forced-answer prompt").

**F5 (major): "only through its abstention component" is built in by how the abstention direction was made**
- **Location:** Abstract l.13-14; Sec. 3.3 l.103-106; App. C l.445-447, l.461-462.
- **Evidence:** "abstain minus answer (items whose stored free-generation regex abstention rate is at least 0.5 minus items with rate 0)"
- **Problem:**
  - The abstention direction contrasts abstaining items, which are mostly obscure (as the authors note, l.462), with answering items, which are mostly popular. So it largely is an obscure-minus-popular direction (cosine 0.95).
  - Projecting it out removes the familiarity signal by construction. The remainder is about 0.31 of the original norm, then rescaled about 3x, so it is likely dominated by estimation noise.
  - A null for that remainder cannot separate "familiarity acts only via abstention" from "the two directions are the same estimate".
- **Fix:** Fit the abstention direction within one popularity level (abstain against answer among popular items only, or within each popularity x knowledge cell and averaged), so it is orthogonal to popularity by design. Then repeat the projection. Otherwise remove the clause from the abstract and keep l.106's weaker wording.

**F6 (minor): the abstract generalises the "abstention component only" result that Gemma does not show**
- **Location:** Abstract l.13-14; App. C l.453-456.
- **Evidence:** "with abstention removed: +0.01 [-0.00, +0.02] and +0.02 [+0.01, +0.03]"
- **Problem:** In Gemma the residual direction keeps about half of the effect, and its interval excludes zero: +0.02 of +0.04 with two judges, +0.05 of +0.11 with three judges. The abstract states the result without naming a model.
- **Fix:** Write "in Llama" in the abstract and Sec. 3.3, and report the Gemma residual in the main text.

**F7 (major): the Gemma null sits at a floor and does not show the absence of a popularity preference**
- **Location:** Abstract l.16-17; Sec. 3.2 l.92-95; Sec. 5 l.131, l.136-137; Table 2.
- **Evidence:** "no preference for popularity over knowledge (difference +0.01 [-0.03, +0.06]; at most 0.11 even for obscure, unknown facts)"
- **Problem:**
  - Gemma abstains in at most 0.05-0.11 of replies in any cell, so a probability-scale difference of about 0 is expected whatever drives its abstention.
  - The kappa of 0.64 also rests on few positive cases, which makes it unstable at this prevalence. The three-judge kappa was adopted after the two-judge bar failed.
  - The interval rules out a Llama-sized absolute effect, but not a proportional preference.
  - The conclusion "the two are shaped separately" (l.136) depends on this null.
- **Fix:**
  - Report all four Gemma cell rates with intervals and a log-odds contrast.
  - Or raise Gemma's base rate (for example with a mild invitation to abstain, held fixed across cells) and retest.
  - Otherwise phrase it as "Gemma rarely abstains; no preference detectable at this rate".

**F8 (major): the base-model claim depends on which lens panel is used**
- **Location:** Sec. 3.3 l.114-118; Fig. 3a; Table 4; Table 5.
- **Evidence:** "while Llama-Instruct's preference for popularity over knowledge is not seen in base models"
- **Problem:**
  - On the full readout (Table 4), base models do show popularity above knowledge: Llama-3.1-8B +0.67 against +0.36, Gemma-2-2B +0.40 against +0.11, OLMo +0.29 against -0.30. Only the hedge-term panel (Fig. 3b) supports "about as large", and even there Gemma-2-2B gives +0.10 against +0.37.
  - The comparison sets a lens log-odds readout in base models against behavioural abstention in the instruct model, which are different readouts.
  - Table 5 gives the instruct models' mid-depth popularity effect (Llama-Instruct +1.58) but leaves out their knowledge effect, which is the matched comparison needed.
- **Fix:**
  - State which readout the criterion note made primary.
  - Add the instruct models' mid-depth knowledge effect to Table 5, and compare base and instruct models on the same readout.
  - Rewrite l.116-118 to match both panels.

**F9 (major): name surface form and ambiguity are the best alternative to familiarity, and the strongest test of it stays in the appendix**
- **Location:** Intro l.34-35 ("factorial test of the misfire idea"); Sec. 3.1 l.86-88; App. A l.428-432; App. F l.486-487.
- **Evidence:** "Among 55 matched pairs of known facts that differ in popularity, the abstention gap of +0.76 falls to +0.43 [+0.18, +0.69] at equal surface form."
- **Problem:**
  - Obscure PopQA subjects often share names with more prominent referents (the authors' own "New York" as a book title example). A model may decline because it cannot tell which entity is meant, not because a familiarity signal misfires.
  - The matched-pair analysis, the closest the paper gets to a design-level control, removes about 43% of the gap. The main text reports only the covariate results, where linear adjustment shrinks the effect by 4-5%.
  - The CLARIFY variant does not settle this, because an ambiguity-driven reply can be phrased as "I don't know".
- **Fix:**
  - Report the matched-pair result in Sec. 3.1, and state what "+0.76" is computed over.
  - Add a direct ambiguity covariate, such as the number of Wikidata items sharing the subject's label.
  - Present the result as popularity-linked abstention whose mechanism, familiarity or ambiguity, is open.

**F10 (minor): the entity-knowledge covariate is weakly validated, so "controlling for what the model knows about the entity" overstates it**
- **Location:** Abstract l.8-9; Sec. 3.1 l.79-82; App. A l.411-417.
- **Evidence:** "It averages 0.83 for obscure known facts and 0.75 for popular known facts."
- **Problem:** The measure is higher for obscure than popular known facts, its coefficient on abstention is about zero (-0.03 [-0.06, +0.00]), and its LLM grading against Wikidata descriptions is unvalidated. A noisy covariate under-adjusts, so the small 5% shrink is weak evidence that entity knowledge does not explain the effect.
- **Fix:** Validate the grader on a hand-labelled sample. Say "adjusting for one LLM-graded measure of entity knowledge" in the abstract.

**F11 (minor): the entropy-follows-knowledge half is largely true by construction, and is checked independently in one family only**
- **Location:** Abstract l.9-11; Sec. 2 l.49-51; Sec. 3.1 l.70-72; Fig. 1c.
- **Evidence:** "Because the knowledge labels are sampled from the same distribution, entropy is expected to follow knowledge"
- **Problem:**
  - Known means at least 90% of samples agree, which forces low first-token entropy.
  - The "independent" relabel uses the same forced-answer format with a paraphrase. It is closer to a replication than an independent measure, and it was run on Llama only.
  - The "in four families" result and the at-least-4x ratio, which has no interval, therefore add little to the dissociation. The informative half is abstention.
- **Fix:** Say so in Sec. 3.2. Give the knowledge/popularity ratio with an interval, or drop the ratio.

**F12 (minor): judge-reliability reporting**
- **Location:** Sec. 2 l.56; Table 2.
- **Evidence:** "On Llama the judges agree at κ = 0.62, above a pre-set bar of 0.60"
- **Problem:**
  - The kappa that clears the bar is a point estimate with no interval, and it is 0.02 above the bar.
  - Two of the three judges come from evaluated families (Qwen, Gemma; the third is Llama-3.3-70B), and the family-matched judging is not discussed.
  - Regex "corroboration" comes from a detector with low precision: 0.85 against 0.48 in the obscure-unknown cell.
- **Fix:** Bootstrap the kappa. Report agreement between each judge and the majority, broken down by family. Note the regex detector's precision where it is used as corroboration.

**F13 (minor): the lens onset test reports no intervals**
- **Location:** Sec. 3.3 l.118-120; App. D l.470-473; Table 5.
- **Evidence:** "Gemma-2-9B 42 24 25 +1 supports"
- **Problem:**
  - The rule needs the interval on D = K - F to exclude zero, but Table 5 gives no intervals. A one-layer difference counting as "supports" needs one.
  - "Half its positive maximum" is not well defined for OLMo's knowledge effect, which is negative on the full readout.
  - With 1 supporting, 1 contradicting and 6 undecided models, "failed" is too strong; "not supported" fits.
- **Fix:** Add a bootstrap interval for D in Table 5, say which readout the onsets use, and change "failed" to "was not supported".

**F14 (minor): sampling and design counts are missing**
- **Location:** Sec. 2 l.45, l.52-53.
- **Evidence:** absent. Expected: the full 2x2 cell counts per model, how up to 100 items per cell were drawn, and whether the intervals account for clustering by the 16 relations. Looked in Sec. 2, App. A, App. B and the captions.
- **Problem:**
  - Relation mix probably differs across knowledge cells (obscure-known may be enriched for low-entropy relations), and the primary contrast is not relation-adjusted. The fixed-effects LPM in l.426 helps but is post hoc.
  - The number of random steering directions (5) and whether their norms were matched is unstated.
- **Fix:** Add a cell-count table and one sentence on how items were drawn. Report the primary contrast with relation fixed effects, or a relation-stratified average, next to the raw one.

**F15 (minor): consistency and pre-registration transparency**
- **Location:** Sec. 3.1 l.69-70; Table 1; App. A l.398-400; App. H.
- **Evidence:** "keeps its sign under four other measurements, which range from +0.13 to +0.44"
- **Problem:**
  - Table 1 lists five alternative measurements besides the primary one.
  - "Keeps its sign" hides a 3.4x spread in size (+0.13 to +0.44).
  - The two-judge 96-token measurement was made primary after the single-judge result (+0.13) had been seen. App. A discloses this, but Sec. 3.1 does not.
  - The SHA-256 prefixes cannot be checked without the notes.
- **Fix:** Correct the count. Note in Sec. 3.1 when the primary measurement was chosen. Release the criterion notes as supplementary material so the hashes can be checked.

## Strengths

1. **The design targets the right cell.** Crossing popularity with model-derived knowledge labels, and naming the obscure-known cell as decisive (l.31-35), is a clean operationalisation of the misfire hypothesis that earlier work, where known meant answered correctly, could not test.
2. **Held-out steering, reported candidly.** Directions are fitted on half the questions, the dose is chosen on a second set and results are scored on a third. The superseded regex-scored version, which overstated the effect (0.02 to 0.29), is disclosed (App. C l.458-460; App. H).
3. **Pre-set bars are reported even when missed.** This covers the kappa bar, the steering bar (+0.09 against 0.10, l.102), the circuit behaviour bar (0.40, l.497), and families declared unmeasured rather than forced into the comparison (l.94).
4. **Every robustness variant has an interval, and post hoc analyses are labelled** (Table 1; App. H l.509-511).
5. **The limitations section is honest.** It covers unvalidated judges, a single model showing the effect, popularity being a proxy, and no isolated cause (l.130-133).

## Questions (decision-relevant)

1. In the popular-unknown cell, what share of forced answers are correct but outside the alias list, and what share of free replies contain a gold alias? If it is large, the +0.30 shrinks (F1).
2. What is the full cross-tabulation of original against paraphrase knowledge labels, and what is the abstention contrast on consensus labels (F2)?
3. Was the free prompt for the one-prompt check chosen before or after the forced-position hedge readout was computed (F4)?

## Recommendation: minor revision

**Why not rejection.** The core result is supported at the standard of a 4-page UniReps extended abstract. In Llama-3.1-8B-Instruct, obscure-known facts draw substantial abstention (0.43) where popular-known facts draw none, and this survives relabelling (n = 27), a third judge and covariate checks. The paper is unusually candid about held-out design, missed bars and post hoc status.

**What drove the recommendation:**
- **F1 and F2.** The headline contrast equals obscure-known minus popular-unknown, and only the first of those cells has been audited. The fix is cheap with existing data.
- **F3.** The abstention measure may count hedged correct answers.
- **F4, F5, F7 and F8.** Four abstract or main-text secondary claims rest on post hoc prompt choice, a projection that removes familiarity by construction, a floor-level null, and a selectively reported lens panel. These need narrower wording, or the matched analyses described above.
- **F9.** Name ambiguity and surface form deserve main-text treatment as the leading alternative to familiarity.

**What would change this.** If the popular-unknown audit (Q1) or the consensus-label analysis (Q2) shrank the contrast's interval to include zero, I would move to major revision.

## Prior-work checks

- **Lindsey et al. 2025 (misfire, known-entity features suppress "can't answer"):** verified on transformer-circuits.pub.
- **Ferrando et al. 2025 (directions from SAEs trained on the base model steer chat-model refusal):** verified on arXiv 2411.14257.
  - Their definition of "known" as answered correctly: [unverified; the abstract does not state it].
- **Gurnee et al. 2026 (Jacobian lens):** verified on arXiv 2607.15495.
  - Availability of prefitted lenses for these base models: [unverified].
- **Kumaran et al. 2026 (implicit confidence threshold for abstention, causal steering):** verified on arXiv 2603.22161.
  - "Multiple-choice prompt": [unverified].
- **Brzezinka 2026 (familiarity readouts separated from the abstention policy, refusal steering):** verified on arXiv 2607.13568.
- **Che and Li 2026 (steering compared with use):** verified on arXiv 2610.07270.
- **Ni et al. 2025 (confidence tied to answer popularity; EMNLP 2026):** verified on arXiv 2505.17537.
- **Ji et al. 2025 (verbal uncertainty only moderately correlated with semantic uncertainty):** verified on ACL Anthology.
- **Not opened:** Gekhman et al. 2024, Reder and Ritter 1992, Schwartz and Metcalfe 1992, Koriat and Levy-Sadot 2001, Mallen et al. 2023 (guessable answers): [unverified].
