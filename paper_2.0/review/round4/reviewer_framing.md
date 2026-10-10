# Referee report: framing and novelty lens

Manuscript: "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge" (UniReps 2026, extended-abstract track)

## What I believe the paper claims vs. what it shows (written after abstract, figures and limitations)

- **Claims:** In Llama-3.1-8B-Instruct, whether the model says "I don't know" depends more on how popular the entity is than on whether the model knows the fact. Gemma-2-9B behaves the same way when the prompt invites abstention, and steering and lens readouts do not pin down a familiarity mechanism.
- **Shows:** The design takes the top and bottom page-view deciles of PopQA. Each cell has at most 100 questions, and the critical "obscure known" cell has 87. Llama abstains on 0.43 of free replies to obscure-entity facts it reliably produces under a forced prefix, and on 0.00 for popular known facts. On popular facts it never produces it abstains only 0.13. The sign of this contrast survives many relabelings and re-scorings. However, the two cells that define the headline statistic are also the cells where the knowledge labels are least stable. The paper's own model-side measures do not show that "popularity" means "familiarity" to the model.

## Closest prior work opened, and positioning verdict

| Work | What it does (from the page I opened) | Does the manuscript position fairly? |
|---|---|---|
| Ferrando, Obeso, Rajamanoharan, Nanda, "Do I Know This Entity?" ICLR 2025 (arXiv 2411.14257) | Uses SAE entity-recognition latents. An entity counts as "known" if the model gets at least 2 attributes right and "unknown" if it gets all wrong. Page views are not used. Steering the unknown-entity latent gives "almost 100% refusal" in Gemma 2 2B. | Fair on labels: line 30-31 correctly says known = answered correctly. Not fair on steering: the manuscript never says that its own steering result (the effect is carried by the abstention component) qualifies this kind of finding. |
| Lindsey et al., "On the Biology of a LLM" (2025) | Describes a default "can't answer" circuit that known-entity features inhibit, with misfires on familiar names whose facts are missing. Does not discuss unfamiliar entities whose facts the model knows. | Misfire is described accurately. The "converse failure" (lines 27-29) is the authors' extension, but the sentence can be read as Lindsey's claim. |
| Brzezinka 2026a, "Does Bielik know what it doesn't know?" (arXiv 2607.07670) | Activation dispersion separates known from fabricated entities. Separates entity familiarity from factual reliability. Models almost never abstain (2 refusals in 2,520 answers). | Cited fairly (line 32-33) as a representational, not behavioural, separation. |
| Brzezinka 2026b, "Graded entity-familiarity readouts..." (arXiv 2607.13568) | Uses page-view deciles. In Gemma-4-12B, a familiarity direction moves refusal on well-known entities from 0.24 to 1.00. Its controls are random directions only, with no generic-refusal control. It does not cross refusal with behavioural correctness. | Novelty claim holds against it. However, the manuscript's main methodological advance over it (projecting out an abstention direction) is not stated as one. |
| Madhusudhan et al., "Do LLMs Know When to NOT Answer?" (arXiv 2407.16221) | Uses PopQA page views only to balance popular and unpopular splits in an abstention benchmark. Reports no breakdown by popularity tier and no crossing with knowledge. | Not cited. It is the nearest behavioural use of PopQA popularity in abstention evaluation and helps place the novelty. |
| Che and Li 2026, "Does the Model Use the Feature?" (arXiv 2610.07270) | Reports that a published unknown-entity latent "strongly steers abstention but transfers only a small share of the natural known-unknown contrast". | Cited only in passing for a Gemma entropy point (line 146). It directly supports Contribution 3 and should be used there. |

On novelty: in the works I opened, I found nothing that crosses entity popularity with model-specific knowledge labels in abstention behaviour. The claim in line 34-35 ("this crossing is what is new here") therefore looks fair as far as I could check, but it should be scoped "to our knowledge". I also opened Pan et al., "Can LLMs Refuse Questions They Do Not Know?" (arXiv 2510.01782, ICLR 2026). At the abstract level it correlates refusal with error but does not stratify by popularity [full text unverified].

## Findings

**F1 (major): the introduction frames the result as familiarity, but the body's own evidence undercuts popularity as a proxy for familiarity.**
- **Location:** Sec. 1, lines 22-29 and 38-40; Sec. 2, line 54; Sec. 3.1, lines 109-112; App. F.
- **Evidence (both sides):** "If abstention follows familiarity, two cells show it ... In Llama, they do" (l. 38-40) against "Neither model-side measure shows the obscure names to be unfamiliar to Llama" (l. 109-110). Recognition is also higher for obscure known facts (0.83 vs 0.75, l. 101), and name surprisal correlates with page views at Spearman -0.001 (l. 377).
- **Problem:** The motivation, the predictions and the "misfire" vocabulary all rest on familiarity, and line 40 says the predictions are confirmed. The body then shows that the model-side familiarity measures do not separate the arms. So the paper does not establish what "popularity" stands for in the model, and the introduction promises more than Sec. 3.1 delivers.
- **Fix:**
  - Rewrite lines 38-40 so the two cells test whether abstention follows popularity, with familiarity offered as one candidate explanation that the evidence does not confirm.
  - Move the "Not shown directly" result into the introduction's framing.
  - Name the alternatives the data leave open, for example surface or prompt features of obscure-entity questions, or instruction-tuning policy.

**F2 (major): the title and abstract state a design-relative "more than" comparison as a general property of Llama.**
- **Location:** Title; abstract lines 4-5; Sec. 1, lines 50-51; Sec. 2, lines 53-58.
- **Evidence (both sides):** Title: "Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge". Body: "in this extreme-contrast design, Llama's answer uncertainty alone predicts its abstention poorly" (l. 50-51).
- **Problem:** The "popularity preference" compares two effects whose sizes depend on the contrasts the authors chose: top vs bottom page-view decile, and ≥9/10 vs 0/10 correct. A different range on either axis would change which factor "wins". Only line 50 carries the scope qualifier, and the title and abstract do not.
- **Fix:**
  - Scope the title and the abstract sentence, e.g. "...between the most and least popular entities".
  - Better, add abstention for intermediate page-view deciles in the known cell, so the reader can see whether the effect is graded or confined to the extremes.

**F3 (major): the knowledge labels are least stable in exactly the two cells that define the headline statistic, and the abstract reports only that the gap "holds".**
- **Location:** Abstract lines 8-10; Sec. 3.1, lines 89-95; App. A, lines 257-271; Table 2.
- **Evidence:**
  - "Of the 87 obscure known facts, 27 stay known" (l. 90-91).
  - "Dropping instead every popular unknown question with a gold alias in any free reply (0.48 of them) gives +0.19" (l. 269-271).
- **Problem:** By the paper's definition, the preference equals obscure-known abstention minus popular-unknown abstention (l. 75-76).
  - Under a paraphrased prompt, about two-thirds of the obscure-known cell does not stay known.
  - Nearly half the popular-unknown questions produce the gold answer in a free reply.
  - The checks shrink the estimate from +0.30 to +0.19-0.23. The relabel check rests on 27 questions with interval [+0.16, +0.49].
  - The abstract's "The gap holds" hides how much the core quantity depends on prompt-specific labels.
- **Fix:**
  - In the abstract, give the range under the label checks (+0.19 to +0.32) and the shrunken cell size.
  - In Sec. 3.1, say plainly that "known" and "unknown" are properties of the forced-answer prompt.
  - Add label instability to the Limitations.

**F4 (major): the UniReps question the paper raises is not answered in the main text.**
- **Location:** Sec. 1, lines 40-42; Sec. 3.2, line 130; App. E.
- **Evidence:** "do models that agree on answer uncertainty also agree on what makes them abstain? Only Llama and, under a permissive prompt, Gemma let us ask." (l. 41-42)
- **Problem:** This is the paper's only bridge to the venue. The answer it receives is one post hoc sentence (l. 130) that points to an appendix analysis:
  - The appendix analysis uses the forced-answer hedge readout, which "shows no preference in Llama either" (l. 124), not abstention.
  - It met a post hoc rule for one of three pairs, and that pair is Llama-Mistral (App. E table), not the Llama-Gemma pair the sentence implies.
  - Lens magnitudes "are not comparable across models" (Fig. 3 caption).
  
  In its current form this is a behavioural and interpretability study of one model with a single-model replication. That fits the UniReps theme ("when and why different models learn similar representations") only weakly.
- **Fix:** Either:
  - Add a main-text, like-for-like cross-model comparison: same prompt, same scorer, per-question agreement of Llama and Gemma abstention under the permissive prompt, beyond shared labels; or
  - Reframe the venue contribution honestly. For example: "two instruct models trained differently converge on the same popularity-keyed abstention policy despite answer uncertainty that follows knowledge in all four", supported by Fig. 1c and the Gemma permissive result.

**F5 (major): the permissive prompt was run on only two of the four families, which leaves the generality question open by choice.**
- **Location:** Sec. 3.2, lines 125-130; Limitations, line 158; App. B, lines 310-315.
- **Evidence:** "Both models abstain often" (l. 126); "whether other families share the preference is open" (l. 158).
- **Problem:** Qwen and Mistral were excluded for abstaining too rarely under the plain prompt. The permissive prompt was the fix for exactly that problem, yet it was not applied to them, and no reason is given. Contribution 2 and the cross-family framing therefore rest on one extra model.
- **Fix:**
  - Run the pre-specified permissive-prompt protocol on Qwen2.5-7B and Mistral-7B, and report them in Sec. 3.2 whatever the outcome; or
  - State why they were excluded.

**F6 (major): Contribution 3 is not positioned against prior steering results it qualifies.**
- **Location:** Sec. 3.3, lines 132-146; Contribution 3, lines 48-49.
- **Evidence:** "Like work steering refusal along recognition directions [Ferrando et al., 2025, Brzezinka, 2026b]" (l. 132-133)
- **Problem:** Prior work reports strong refusal steering from entity-recognition or familiarity directions:
  - Ferrando et al.: "almost 100% refusal" in Gemma 2 2B.
  - Brzezinka 2026b: 0.24 to 1.00 on well-known entities, controlled only by random directions.

  This paper shows that its obscure-minus-popular direction adds +0.01 once its overlap with an abstain-minus-answer direction is removed. That is a substantive qualification of such claims, and Che and Li 2026 make a related point. The manuscript presents its own result as a null ("No Isolated Familiarity Mechanism") instead of saying what it adds beyond prior work.
- **Fix:** Add one or two sentences stating:
  - that prior steering studies used random-direction controls only;
  - that this study adds an abstention-overlap control and finds the effect carried by that overlap;
  - how this bears on reading Ferrando et al. and Brzezinka 2026b.
  
  Cite Che and Li here for the steering-vs-use distinction.

**F7 (minor): the abstract's first statistic differs from the body's unit and leaves out the cell condition.**
- **Location:** Abstract lines 6-7; Sec. 3.1, lines 81-82.
- **Evidence:** Abstract: "Llama says 'I don't know' to 43% of questions it answers correctly"; body: "says 'I don't know' in 0.43 of replies".
- **Problem:** The rate is a share of replies (3 per question), not of questions. "On popular entities the rate is 0%" holds only for popular *known* facts (popular unknown is 0.13).
- **Fix:** "...in 43% of replies to questions it answers correctly...; on popular entities it knows, the rate is 0%".

**F8 (minor): the "misfire" label for the popular-unknown cell is not supported by what those replies contain.**
- **Location:** Sec. 1, lines 27 and 38-39; App. A, lines 269-271.
- **Evidence:** "popular unknown facts should draw few 'I don't know' replies (misfires)" (l. 38-39)
- **Problem:** Lindsey et al.'s misfire is a hallucinated answer. Here non-abstention on popular "unknown" facts includes correct answers on 0.48 of questions (l. 270). The cell therefore mixes misfires with knowledge the forced prompt missed.
- **Fix:** Drop "misfires" as the cell label, or report the share of non-abstaining popular-unknown replies that are wrong.

**F9 (minor): Contribution 2 implies a numerical match ("Same Preference") that is not like-for-like.**
- **Location:** Contributions lines 44-47; Sec. 3.2, lines 127-128.
- **Evidence:** Llama "+0.30 [+0.20, +0.40]" (two judges, plain prompt) against Gemma "+0.30 [+0.19, +0.40]" (three-judge majority, permissive prompt).
- **Problem:** The two numbers come from different prompts and scorers. Under the same permissive prompt and scorer, Llama is +0.47 [+0.35, +0.59].
- **Fix:** In the contributions, compare Gemma with Llama's +0.47 under the same conditions, or say "same sign" rather than "same preference".

**F10 (minor): Gemma's plain-prompt preference is said to be unestimable while a table reports an estimate.**
- **Location:** Sec. 3.2, lines 121-122; Table 3; line 307.
- **Evidence:** "its preference cannot be estimated" (l. 122) against "Three judges: popularity - knowledge ... +0.01" with interval [-0.03, +0.06] (l. 307).
- **Problem:** A reader will read +0.01 with a tight interval as "no preference". The authors mean that floor effects make it uninformative.
- **Fix:** Report the number in the text and say why it is uninformative (floor; pre-specified 0.25 criterion, l. 315).

**F11 (minor): unclear referent in the one cross-model sentence of the main text.**
- **Location:** Sec. 3.2, lines 129-130.
- **Evidence:** "Cross-model agreement on that readout, beyond shared labels, met a post hoc rule for one of three pairs (Appendix E)."
- **Problem:** "That readout" follows a sentence about permissive-prompt abstention. Appendix E analyses forced-answer hedge log-odds and entropy, and the one pair that met the rule is Llama-Mistral.
- **Fix:** Name the readout and the pair explicitly, or remove the sentence (see F4).

**F12 (minor): "generic abstention signal" claims more than the steering analysis shows.**
- **Location:** Contribution 3, line 48; Sec. 3.3 heading, line 132.
- **Evidence:** "steering moves a generic abstention signal"
- **Problem:** The abstain-minus-answer direction is fitted on the same entity questions. Its cosine with obscure-minus-popular is 0.95 (0.91 within the obscure level). No test shows it is generic, for example that it induces abstention outside entity QA.
- **Fix:** Say "an abstain-minus-answer direction fitted on these questions", or add an out-of-domain test of that direction.

**F13 (minor): novelty sentence and attribution need scoping.**
- **Location:** Sec. 1, lines 27-29 and 34-35.
- **Evidence:** "It has not been crossed with knowledge in abstention behavior; this crossing is what is new here."
- **Problem:**
  - The universal claim should be hedged.
  - The "converse failure" (abstaining on known facts about unfamiliar names) reads as Lindsey et al.'s proposal. Their page does not discuss it.
  - Nearby behavioural uses of PopQA popularity in abstention (Madhusudhan et al.) and the popularity-accuracy link with popularity-gated retrieval (Mallen et al., cited only as a data source) are not used to place the contribution.
- **Fix:**
  - Add "to our knowledge".
  - Write "we extend this to a converse case".
  - Add one sentence placing the work relative to popularity-based abstention benchmarks and adaptive retrieval.

**F14 (minor): there is no conclusion or discussion, and the significance claim is a single sentence that does not engage the literature it contradicts.**
- **Location:** Main text ends at Sec. 4 (Limitations), lines 153-158; significance at lines 50-51.
- **Evidence:** absent: expected a closing section stating implications. I looked in Sec. 3.3, Sec. 4 and the end of Sec. 1.
- **Problem:**
  - The one significance claim ("answer uncertainty alone predicts its abstention poorly") is in tension with Kumaran et al. 2026, whose abstract reports that models use internal confidence to decide whether to abstain. The paper cites it only as background (l. 20).
  - Answer uncertainty is also measured on a different prompt (forced) than abstention (free).
- **Fix:** Add three to five sentences of discussion:
  - how the result relates to confidence-driven abstention;
  - what follows for abstention training and retrieval gating;
  - the caveat that the two readouts come from different prompts. The within-prompt check in lines 113-117 is post hoc.

**F15 (minor): the abstract and contributions do not disclose that the originally planned measure gave a much smaller effect.**
- **Location:** Sec. 3.1, lines 86-87; App. A, lines 251-254; Table 2; App. G footnote a.
- **Evidence:** "We designed the primary measure after the smallest of these was seen." (l. 86-87); "One LLM judge, corrected ... +0.13 [+0.03, +0.24]"
- **Problem:**
  - The body discloses this honestly, but the headline 43%/0% comes from a measure designed after the pre-specified one returned +0.13. The abstract presents the result without this history.
  - The footnote that an amendment's "internal date is one day later than its recorded hash time" further weakens the pre-registration framing.
- **Fix:**
  - Add a clause to the abstract or to Contribution 1 noting the range across measures (+0.13 to +0.44).
  - Explain the amendment date discrepancy in one sentence.

**F16 (minor): within-prompt numbers have no units or plan status.**
- **Location:** Sec. 3.1, lines 116-117.
- **Evidence:** "the log-odds of a first token 'I' follow popularity (+8.4 vs. +3.1)"
- **Problem:**
  - +8.4 log-odds is an implausibly large effect for a reader to interpret without units or intervals.
  - "After seeing this, we specified" leaves unclear whether the analysis counts as pre-specified.
  - This is the paper's only same-prompt test of the dissociation.
- **Fix:** Give intervals, state that the numbers are popularity and knowledge effects on log-odds, and mark the analysis pre or post hoc. Also list it in Table 2 or App. B.

**F17 (minor): Limitations omit relation composition and clustering, which matter for a cell of 87 questions.**
- **Location:** Sec. 2, lines 76-77; Sec. 4.
- **Evidence:** "Intervals are 95% percentile bootstraps over questions within cells, not clustered by relation." absent: expected the distribution of the 87 obscure-known facts across the 16 relations; I looked in App. A and Tables 1-2.
- **Problem:** Obscure facts the model knows are likely concentrated in a few relations with guessable answers. Unclustered intervals may then be too narrow. The relation fixed-effects regression in Table 2 has no interval in its first row.
- **Fix:**
  - Report per-relation counts for the obscure-known cell.
  - Add relation-clustered bootstrap intervals for the headline preference.
  - Add both points to the Limitations.

**F18 (minor): Figure 1's title presents a by-construction control as a co-equal finding.**
- **Location:** Figure 1 caption.
- **Evidence:** "In Llama, saying 'I don't know' tracks popularity, while answer uncertainty follows knowledge."
- **Problem:** The paper itself says entropy follows knowledge "as the labeling implies" (abstract l. 13). The informative contrast is that entropy is similar in obscure-known and popular-known cells while abstention differs.
- **Fix:** Retitle along the lines of "Llama abstains on obscure facts it knows, at answer uncertainty matching popular known facts."

**F19 (minor): contributions lack precise evidence pointers.**
- **Location:** Contributions, lines 44-49.
- **Evidence:** "It survives four label and measurement checks." (l. 45)
- **Problem:** The four checks are not named. Section 3.1 runs more than four checks, and line 86 says "five other measures", so the reader cannot map the claim to its evidence.
- **Fix:** Name the checks and point to Table 2 rows; point Contributions 2 and 3 to App. B and App. C rows.

## Strengths

1. **Clear estimand (Sec. 2, l. 71-76).** The preference is defined as popularity effect minus knowledge effect, with the identity that it equals obscure-known minus popular-unknown. I checked the Llama numbers: (0.43 + 0.48)/2 - (0.00 + 0.13)/2 = 0.39; 0.39 - 0.09 = 0.30 = 0.43 - 0.13. The entropy effects (+2.14, +0.51) and the permissive-prompt preferences (+0.47, +0.30) also reproduce from the cell rates.
2. **Transparent robustness accounting (Table 2, App. G).** Every variant is marked pre-specified or post hoc. The adverse +0.13 is reported in the main table rather than hidden.
3. **Unusually honest scoping of mechanism.** The abstract's last sentence ("shows what abstention tracks, not why") and "Familiarity: Not Shown Directly" (l. 109-112) say plainly what the mechanistic analyses do not show.
4. **A stronger steering control than prior work (App. C, l. 317-348).** The analysis projects out the abstention component, refits the abstention direction within the obscure level, and holds out the dose. This goes beyond the random-direction-only controls in Brzezinka 2026b.
5. **The ambiguity alternative is addressed in the right direction (l. 103-106, 288-291).** Namesake counts run against the ambiguity explanation, and unique-name obscure-known facts still abstain at 0.43.

## Questions for the authors (answers would change my view)

1. Across intermediate page-view deciles, restricted to known facts, does abstention fall gradually, or only at the extreme? (F2)
2. How are the 87 obscure-known facts spread over the 16 relations, and what is the relation-clustered interval for +0.30? (F17)
3. Were Qwen and Mistral run under the permissive prompt? If yes, what happened; if no, why not? (F5)
4. Given that Llama recognizes obscure-known entities at least as well as popular ones, what do you think page views stand for in the model? (F1)
5. Under the permissive prompt with the same scorer, do Llama and Gemma agree per question on which known facts they abstain on, beyond what popularity and knowledge labels predict? (F4)

## Recommendation: major revision

The empirical core is novel as far as I could verify, carefully controlled, and honestly reported. The framing does not yet match it.

- **F1:** The familiarity narrative in the introduction is contradicted by the paper's own measures.
- **F3:** The headline statistic rests on the two least stable label cells, and the abstract reports only that the gap "holds".
- **F4:** The UniReps connection is raised and then left unanswered in the main text, so venue fit is weak.

**What would change my mind:**
- A title and abstract scoped to the extreme-decile design, with label-check magnitudes reported (F2, F3).
- An introduction reframed around popularity rather than confirmed familiarity (F1).
- One main-text, like-for-like cross-model abstention comparison, or an honest restatement of the venue contribution (F4).

With those changes, plus permissive-prompt runs on Qwen and Mistral (F5), I would recommend acceptance as an extended abstract.

## Prior works opened

- Ferrando et al. 2025, ICLR: https://arxiv.org/abs/2411.14257 and https://arxiv.org/html/2411.14257v2
- Lindsey et al. 2025, "On the Biology of a Large Language Model": https://transformer-circuits.pub/2025/attribution-graphs/biology.html (first ~100k characters read; the entity-recognition and hallucination section was covered)
- Brzezinka 2026a: https://arxiv.org/abs/2607.07670
- Brzezinka 2026b: https://arxiv.org/abs/2607.13568 and https://arxiv.org/html/2607.13568
- Madhusudhan et al. 2024: https://arxiv.org/html/2407.16221v2
- Pan et al. 2025/2026, Refusal Index: https://arxiv.org/abs/2510.01782 (abstract only)
- Kumaran et al. 2026: https://arxiv.org/abs/2603.22161 (abstract only)
- Che and Li 2026: https://arxiv.org/abs/2610.07270 (abstract only)
- Gurnee et al. 2026 (Jacobian lens): https://arxiv.org/abs/2607.15495 (abstract only; confirms the citation exists and introduces the lens)
- I-CALM: https://arxiv.org/abs/2604.03904 (abstract only; no popularity analysis visible; not used in findings)
