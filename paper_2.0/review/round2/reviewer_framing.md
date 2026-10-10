# Referee report: framing and novelty

**Manuscript:** "In Llama, Saying 'I Don't Know' Tracks Familiarity While Answer Uncertainty Tracks Knowledge" (UniReps 2026, extended-abstract track)
**Lens:** framing and novelty, positioning against prior work, whether claims match the evidence, venue fit

## Claim vs what is shown

The paper claims four things: in Llama-3.1-8B-Instruct, uninstructed "I don't know" follows entity popularity (a proxy for familiarity) more than item-level knowledge; answer entropy follows knowledge in four families; an obscure-minus-popular direction causally produces abstention; and base models already carry the popularity signal. What the body establishes firmly is the 2x2 behavioural result in one model (n = 387, judge kappa = 0.62, robust across five measurements). The other three legs are weaker than the framing suggests. The entropy contrast holds by construction. The steering ranking mostly re-expresses the abstention asymmetry: the never-abstained fit adds +0.04. The base-model lens reads a forced-answer hedge readout that, by the paper's own Section 2, does not dissociate popularity from knowledge in Llama-Instruct.

## Closest prior work (verified)

- **Lindsey et al. 2025, "On the Biology of a LLM"** (opened, transformer-circuits.pub). It describes a default "can't answer" circuit inhibited by known-entity features, and a "misfire" in which a recognised name suppresses refusal without the answer. The manuscript states the hypothesis fairly (l.22-24). The direct test of that hypothesis is the popular-unknown cell (0.13). The obscure-known cell (0.43) is the new complementary observation. The paper never says that this cell is its main new contribution.
- **Ferrando et al. 2025, ICLR, "Do I Know This Entity?"** (arXiv 2411.14257, opened). Entity-recognition SAE directions steer the chat model to refuse questions about known entities, and the directions come from base-model SAEs. The paper acknowledges this precedent for steering and base-model carry-over (l.34, l.121), so the positioning is fair. Ferrando et al. define "known" at the entity level, from answer correctness. The present paper's "known" is item-level, which matters for finding F1.
- **Brzezinka 2026a, arXiv 2607.13568** (opened). It reports graded familiarity probes tracking page-view deciles and refusal steering in Gemma-4, judges sampled answers for real entities, and reports that familiarity readouts and abstention dissociate across models. The manuscript says this work has no knowledge axis (l.122-123). That is roughly fair: it does not cross familiarity with item-level knowledge. But it does run a behavioural correctness check, and the wording should be more precise.
- **Ji et al. 2025, EMNLP** (arXiv 2503.14477, opened). Verbal and semantic uncertainty are only moderately correlated, and their mismatch predicts hallucination. This already establishes that verbal uncertainty and answer uncertainty diverge. The new contribution here is identifying the variable that abstention follows instead (popularity at fixed item-level knowledge).
- Also opened, and relevant to specific findings below: Kumaran et al. 2026 (2603.22161: multiple-choice SimpleQA with an explicit abstain option "5", so the paper's contrast at l.124-125 is accurate); Ni et al. 2025 (2505.17537, EMNLP 2026 per arXiv comment); Che and Li 2026 (2610.07270); Lippl et al. 2024 (2404.08101); Gurnee et al. 2026 (2607.15495, which introduces a Jacobian lens; the released model list was not verifiable from the abstract page [unverified]); Brzezinka 2026b (2607.07670); Luo et al. 2023/2024 (2309.02654); Kang et al. 2024 (2403.05612).

## Findings

### F1 (major): "Familiarity" is confounded with entity-level knowledge; the accessibility account fits the data as well as cue familiarity

- **Location:** title; abstract l.14-15; §2 l.42; §3.1 l.87-88.
- **Evidence:** "page-view popularity is our proxy for how familiar a subject is." (l.42)
- **Problem:** The knowledge label is item-level: one relation per subject. Popular subjects are ones the model knows many other facts about; obscure subjects are ones it knows few facts about. Popularity therefore mixes surface or cue familiarity with entity-level knowledge, meaning how much partial information about the subject is retrievable. Entity-level knowledge is Koriat's accessibility heuristic (cited at l.117). It is also the notion of "known entity" used by Ferrando et al. The surface-form control (name frequency, token count, rarity barely move the coefficient, l.85-86) argues against a purely lexical cue and so, if anything, favours accessibility. Under the accessibility reading, abstention still reports uncertainty, at the entity level rather than the item level. That weakens the headline contrast "familiarity vs uncertainty" and the closing appeal to the cue-familiarity heuristic.
- **Fix:** Add entity-level knowledge as a covariate. For example, use the model's accuracy on the same subject's other PopQA/Wikidata relations, or Ferrando-style entity labels. Report whether the popularity effect on abstention survives. If the analysis cannot be run, name the accessibility alternative in §3.1 and §5, and say "entity popularity" rather than "familiarity" in the title.

### F2 (major): Half of the title's dissociation holds by construction

- **Location:** title; abstract l.6-8; contribution (i) l.33-34; §3.1 l.77-80; conclusion l.134.
- **Evidence:** "The knowledge labels are sampled from this same distribution, so entropy is expected to follow knowledge" (l.52)
- **Problem:** The paper concedes this point (l.52, l.80), yet the title, contribution (i) and the conclusion ("As the two readouts follow different variables, calibration work should measure both") present the entropy result as one arm of a finding. The reference readout could not have failed. It is also measured on a different prompt from abstention. On the same forced-answer prompt, Llama's hedge readout follows popularity and knowledge about equally (+2.93 vs +3.05, l.56). The result is therefore a single dissociation (abstention against a near-definitional baseline), not two measured readouts that diverge.
- **Fix:** Either retitle around the abstention result alone (e.g., "In Llama, saying 'I don't know' tracks entity popularity more than knowledge") and present Fig. 1b/c as a sanity check, or make the entropy arm falsifiable. To do the latter, compute knowledge labels from held-out samples or a paraphrased prompt, and entropy on another prompt.

### F3 (major): Contribution (ii)'s "new here" comparison follows from (i), and the paper's own controls undercut it

- **Location:** intro l.35-37; §3.2 l.91-104; App. C l.426-433.
- **Evidence:** "new here, it does so far more than an unknown-minus-known direction, and mostly through components shared with abstention" (l.36-37)
- **Problem:** Both directions are differences of cell means on the free prompt. From Fig. 1a, the obscure and popular cells abstain at about 0.46 vs 0.07, while the unknown and known cells abstain at about 0.31 vs 0.22. A direction that contrasts groups differing by 0.39 in abstention is expected to carry more abstention signal, so the ranking re-expresses (i). The informative test is the never-abstained fit, and it adds only +0.04 [+0.01, +0.07] (l.104).
  - The dose was chosen on the evaluation questions by a rule requiring the obscure-minus-popular direction to beat random by 0.10 (App. C), which favours that direction.
  - Gemma requires 4x the natural norm. That is the regime Che and Li 2026 warn about: a published latent steers knowledge abstention, but installing its observed values transfers little (verified).
  - The paper's own circuit-injection result (0.40, App. F) shows the same gap.
- **Fix:** Reframe (ii) as a partly negative result: the direction produces abstention, but mostly via abstention components. Drop "new here, far more than". Alternatively, run the direction with abstention projected out, which §5 lists as untested.
  - Gemma deserves its own emphasis. Its base abstention is near zero (0.05 in the obscure-unknown cell, Table 2), so its fitted direction is less contaminated by abstention. Its entropy dissociation (+0.68 nats the other way, l.100) is the paper's cleanest double dissociation, though it holds only at 4x.

### F4 (major): The base-model lens result does not bear on abstention as framed

- **Location:** contribution (iii) l.37-38; §3.3 title and l.108-115; abstract l.11-13.
- **Evidence:** "at the forced-answer position, Llama's log-odds of hedge tokens against answer tokens follow popularity and knowledge about equally" (l.55-56)
- **Problem:** §3.3 reads hedge log-odds at "Answer: The answer is", the forced-answer position. §2 itself shows that this readout does not dissociate in Llama-Instruct, and that the divergence appears only "when it may decline". The finding that base models' forced-answer hedge readout responds to both variables is therefore what one would expect if it is unrelated to the abstention bias. The word "already" in the heading and abstract implies the abstention bias is inherited from pretraining, which the analysis cannot show.
  - Table 3 gives the base-vs-instruct comparison the criterion required, but the main text omits it. Instruction tuning roughly doubles the mid-depth popularity effect (Llama +0.67 base vs +1.58 instruct; Gemma-2-9B +0.39 vs +0.71). That cuts against "already" as the main message.
- **Fix:** Reword (iii) as "a forced-answer hedge readout in base models is sensitive to popularity as well as knowledge". Report the base/instruct numbers in §3.3. If possible, test whether the base readout predicts the instruct model's free-prompt abstention item by item; that test would connect (iii) to (i).

### F5 (minor): The abstract's "alongside a knowledge effect" in three families goes beyond what the body reports

- **Location:** abstract l.12-13; §3.3 l.111-115; Fig. 3b; Table 3.
- **Evidence:** "all intervals excluding zero (Figure 3a; values in Appendix D)" (l.112). Table 3 lists only mid-depth popularity means, with no intervals and no knowledge effects.
- **Problem:** In-window knowledge effects are given for 2 of 4 base models. In Fig. 3b, OLMo-3-7B's knowledge curve dips well below zero inside the shaded window (to about -1.5 near 0.55 depth). OLMo is the third family, so "alongside a knowledge effect" may have the wrong sign for it. The pointer to intervals in App. D leads to a table without them.
- **Fix:** Add in-window popularity and knowledge effects with intervals for all four base models to Table 3, and qualify the abstract for OLMo.

### F6 (major): Venue fit is never argued; the cross-model material is either by construction, not comparable, or post hoc

- **Location:** intro l.17-38 (absent); Fig. 1c; Fig. 3 caption; App. E; conclusion l.134-137.
- **Evidence:** "Lenses are fitted per model, so magnitudes are not comparable across models" (Fig. 3 caption). Also: "exploratory checks find this reliably for two of three pairs, though our pre-set rule held for one" (l.136-137).
- **Problem:** UniReps asks when and why different models learn the same representations. The intro never poses a convergence question. The cross-family evidence has three parts, and none of them is a convergence test:
  - Entropy (Fig. 1c), which holds by construction (F2).
  - The base-model lens, which per the caption cannot be compared across models.
  - App. E, which is post hoc and meets the pre-set rule for 1 of 3 pairs. "Reliably for two of three" is a generous summary: only Mistral passes the rule, and Gemma's gap interval includes zero.
  
  As written, this is an interpretability and metacognition paper with a small cross-model coda.
- **Fix:** Add two or three sentences to the intro that state the UniReps question, for example: "Is the familiarity sensitivity of hedge readouts shared across families because it reflects shared training-text exposure?" Bring App. E's knowledge-differs vs same-knowledge contrast into the main text as the test of that question, labelled exploratory, and summarise it plainly ("rule met for 1 of 3 pairs").

### F7 (minor): A conclusion drawn from Lippl et al. sits in Related Work without the test it requires

- **Location:** §4 l.126-127.
- **Evidence:** "By the source-invariance criterion of Lippl et al. [2024], the abstention variable we find is a correlate of uncertainty rather than a representation of it."
- **Problem:** Source invariance (verified, arXiv 2404.08101) requires that uncertainty be encoded the same way whatever its source. No experiment varies the source of uncertainty. The argument the paper actually makes is that a non-uncertainty variable (popularity) moves abstention at fixed knowledge. That is closer to Lippl et al.'s "representational indeterminacy" point than to a source-invariance test. Entity-level knowledge (F1) is also a source of uncertainty, so the inference does not follow until F1 is resolved.
- **Fix:** Spell out the mapping in one sentence and soften the claim to "suggests". Alternatively, move it to the conclusion with the F1 caveat.

### F8 (minor): The novelty statement does not say what is new relative to Ji et al. and Lindsey et al.

- **Location:** intro l.20-26, l.32-38; abstract l.13-15.
- **Evidence:** "Where abstention could be measured, it follows a different variable from answer uncertainty" (abstract l.13-14)
- **Problem:** Ji et al. 2025 already show that verbal and answer-level uncertainty diverge. Lindsey et al.'s misfire hypothesis predicts low abstention for popular-unknown items. Neither has the crossed design, and neither reports that a model abstains on 43% of obscure facts it reliably answers correctly. That cell is the paper's most distinctive observation, but the abstract and contribution list never single it out.
- **Fix:** Add one sentence to the intro, along the lines of: "Prior work shows verbal and answer uncertainty diverge [Ji] and predicts familiar-but-unknown failures [Lindsey]; new here is the crossed design and the complementary failure: abstention on obscure facts the model knows (0.43 vs 0.00 for popular)." Put the 0.43 figure in the abstract.

### F9 (minor): Some close work is missing or characterised loosely

- **Location:** §4 l.121-127.
- **Evidence:** absent. Expected: Brzezinka 2026b (arXiv 2607.07670, "...Separates Entity Familiarity from Factual Reliability"); Luo et al. (arXiv 2309.02654, SELF-FAMILIARITY, familiarity-gated abstention modelled on human behaviour); Kang et al. 2024 (arXiv 2403.05612, hedging on unfamiliar inputs mirrors fine-tuning targets). Looked in §1, §4, App. F and the reference list.
- **Problem:** Brzezinka 2026b's title claims the same separation the intro claims ("We separate the two", l.27), although its contrast is real vs fabricated entities, and its models barely abstain. Luo et al. propose familiarity-based abstention as a desirable design, which the present result recasts as a bias when knowledge is present. Kang et al. offer a fine-tuning account of the instruct-over-base increase in Table 3. Separately, Ni et al. find that question-entity popularity relates only weakly to answer-probability confidence (verified). That corroborates Fig. 1b/c and could be cited as such.
- **Fix:** Add one or two sentences distinguishing item-level knowledge crossing from real-vs-fabricated familiarity, and cite Luo and Kang.

### F10 (minor): The main text and Appendix A disagree on how much surface form matters

- **Location:** §3.1 l.85-86; App. A l.411-415.
- **Evidence:** "the abstention gap of +0.76 falls to +0.43 [+0.18, +0.69] at equal surface form" (l.412)
- **Problem:** The main text cites the regression change (-1.87 to -1.79), but Appendix A does not report that regression, its outcome measure (presumably regex on 2,303 items), or its covariates in a table. The matched-pair analysis it does report shows a roughly 43% drop at "equal surface form", which is undefined, and the measure behind +0.76 is not given. A reader cannot tell whether surface form matters a little or a lot. A large drop would also bear on F1, since a lexical cue is the cue-familiarity reading.
- **Fix:** Report the regression with its measure, and define "equal surface form" and the +0.76 measure. Mention the matched-pair reduction in §3.1.

### F11 (minor): Abstention is reported for steered Gemma with a detector the paper does not trust for Gemma

- **Location:** abstract l.9-10; §3.2 l.98-99; Fig. 2a caption; Table 2.
- **Evidence:** "Regex recall vs two judges ... 0.48" (Table 2, Gemma). Also: "Abstention rate (regex detector)" (Fig. 2 caption).
- **Problem:** §2 withholds Gemma abstention because the judges fail the kappa bar. The abstract still says steering "makes ... Gemma-2 say 'I don't know'", based on regex alone. That regex has 0.48 recall for Gemma, its precision is unknown, and it is applied to 4x-norm outputs that may be off-distribution.
- **Fix:** Report regex precision on a hand-checked sample of steered Gemma outputs. Otherwise, flag the measurement caveat in the abstract and caption.

### F12 (minor): The headline cell's size is not visible where the claim is made

- **Location:** Fig. 1a caption; §3.1 l.72-84.
- **Evidence:** "as for 30 of 87 obscure known facts" (l.82-83)
- **Problem:** The key observation (obscure-known abstention, 0.43) rests on 87 items, 57 of them non-guessable, in one model. That is the smallest cell, and the reader learns its size only from the guessability aside.
- **Fix:** Give per-cell n in the Fig. 1a caption.

## Strengths

1. **The 2x2 design is clean and well motivated** (§1 l.27-31, §2 l.40-48). Crossing page-view popularity with item-level, self-sampled knowledge labels tests the Lindsey et al. hypothesis directly. The obscure-known vs popular-known contrast (0.43 vs 0.00, l.72-74) is a crisp and, to my knowledge, new observation.
2. **Scoping is unusually candid.** The title says "In Llama". Abstention is reported only where judge agreement passes a pre-set bar (l.62-64). The paper states that entropy follows knowledge by construction (l.52, l.80) and that the judges were not validated against human labels (l.64).
3. **Analyses were declared in advance with hashed criterion notes** (App. G), post hoc analyses are labelled as such, and a failed prediction is reported in the main text (l.116-119, Table 3).
4. **The steering control that undercuts the paper's own result is in the abstract** (+0.04 never-abstained fit, l.10-11, l.103-104). This is the kind of negative finding the call invites.
5. **The abstention result is robust across five measurement schemes** (Table 1), and the guessability check is restricted to unknown items (+0.34, l.407-408).

## Questions for the authors (decision-relevant)

1. If you add entity-level knowledge (accuracy on the subject's other relations) as a covariate, does the popularity effect on Llama's abstention survive? If yes, the familiarity framing stands. If no, the headline should read "abstention follows entity-level rather than item-level knowledge" (F1).
2. Does any result connect the base-model lens to free-prompt abstention? For example, does the base model's mid-depth hedge readout predict, item by item, which questions the instruct model abstains on (F4)?
3. Which result do you put forward as the UniReps contribution, that is, as evidence that different models share this representation (F6)?

## Recommendation: major revision

The core behavioural finding is interesting, honestly reported, and well suited to an extended abstract: in Llama, "I don't know" follows entity popularity, including on facts the model reliably knows. The framing claims more than that, and four findings drive the recommendation:

- **F2:** the dissociation in the title is half by construction.
- **F3:** the steering novelty claim re-expresses (i), and the paper's own control undercuts it.
- **F4:** the base-model lens reads a position that does not bear on abstention.
- **F1:** "familiarity" is not separated from entity-level knowledge, which also weakens the cue-familiarity conclusion.

F6, the unargued venue fit, adds to these. Most fixes are rewording: the title, the contribution list, §3.3 and a UniReps paragraph. Only F1 needs a new analysis, and the data for it appear to be at hand. With those changes this would be a clear accept for the workshop.
