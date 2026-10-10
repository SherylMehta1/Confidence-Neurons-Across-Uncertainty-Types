# Referee report: Methods and statistics

**Manuscript:** "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge" (UniReps 2026, extended abstract track)
**Lens:** methods and statistics: whether the evidence supports the claims

## What I believe the paper claims and shows

**Claims.** In LLMs, verbal abstention ("I don't know") is driven by how familiar the question's subject is, not by uncertainty about the answer. Answer uncertainty (first-token entropy) is driven by knowledge. The familiarity signal is causal for abstention (steering) and is already present in base models (lens).

**Shows.** For one model (Llama-3.1-8B-Instruct, 387 questions), judge-labelled abstention differs much more between bottom- and top-decile-popularity entities than between facts the model reliably answers and facts it never answers. First-token entropy under a different prompt follows those knowledge labels in four models. A mean-difference "obscure-minus-popular" direction raises abstention on held-out known popular facts, but by the authors' own control it does so mostly through its abstention component. A lens readout of hedge-vs-answer-token log-odds in four base models also differs between obscure and popular entities at mid-depth.

The Llama abstention result is the solid core. The entropy contrast, the steering section and the lens section each have a design weakness that could produce the reported pattern without the claimed mechanism.

## Findings

### F1: The knowledge labels are built from the same sampling distribution that the entropy readout measures
- **Severity:** major
- **Location:** Section 2 "Questions and knowledge labels" (lines 41–44); Section 3.1 (lines 76–78); Figure 1b, 1c
- **Evidence:** "A fact is known if the greedy answer and at least 90% of samples are correct and unknown if none is; the rest are dropped."
- **Problem:** "Known" means that ≥9 of 10 samples under the forced-answer prompt agree on the correct answer. First-token entropy under that same prompt must therefore be low for known facts. Part of the "knowledge effect" on entropy is true by construction, and it is not an independent finding. The construction also clips entropy variation within the known cell, which suppresses the popularity effect on entropy there. Read off Figure 1b, the popularity effect on entropy is about 0.1 nats within known facts but about 0.9 nats within unknown facts (3.5 vs 2.6). The averaged +0.51 and the "knowledge ≥4× popularity" contrast that sets up the title's second half are therefore shaped by the labelling rule. The abstention readout uses a different prompt and is not tied to the labels in this way, so the two readouts are not on equal footing.
- **Fix:** Derive knowledge labels from a disjoint set of samples, and preferably from a different prompt (paraphrased question or a non-chat template). Then recompute the entropy effects. Also report the popularity effect on entropy separately within the known and unknown cells with intervals, and state the title's second claim in terms of the cell where entropy is free to vary.

### F2: The prompt confound is answered with numbers for the wrong model
- **Severity:** major
- **Location:** Section 2 "Two prompts, by necessity" (lines 51–55)
- **Evidence:** "(Gemma: popularity +1.79, knowledge −0.32; in Llama both effects are large)"
- **Problem:** Abstention (free prompt) and entropy (forced prompt) are measured under different prompts, and the authors concede that "the prompt change could contribute to the divergence". The within-prompt check is reported numerically only for Gemma, whose abstention is not measured. For Llama, the only model with measured abstention, the text says that both effects are "large". That wording suggests the forced-prompt hedge readout also follows knowledge in Llama, which is the case the confound threatens. No intervals are given, and the analysis is not labelled confirmatory or post hoc.
- **Fix:** Report popularity and knowledge effects with bootstrap intervals on the forced-prompt hedge log-odds for all four models, Llama first. Add the complementary check: entropy of the first answer token under the free prompt, restricted to replies that answer. Label the analysis's status as in Appendix D.

### F3: The obscure-known cell may be dominated by guessable, low-cardinality answers
- **Severity:** major
- **Location:** Section 3.1 (lines 70–73); Appendix A (lines 417–418)
- **Evidence:** "Some obscure names also contain their answer (the capital of a district named after its town)."
- **Problem:** The headline contrast rests on the obscure-known cell (0.43 abstention). A fact counts as known when samples are ≥90% correct. For obscure entities, that can happen because the answer can be guessed from the name or from a strong prior over a small answer set. Examples are a capital named after its district, a nationality implied by the name, or a modal answer for relations with few possible values. PopQA has 16 relations. I could verify only "occupation" as one of them; the others I recall with small answer sets are [unverified]. If that happens, the model's low entropy reflects a confident prior and its abstention reflects genuine lack of entity knowledge, which is an uncertainty-based account and not a familiarity heuristic. The authors found the name-contains-answer case themselves but excluded it only in a 55-pair side analysis, not in the primary 2×2. Relation composition per cell is not reported, and relation fixed effects appear only in the post hoc regex regression.
- **Fix:** Report the number of questions per relation in each cell. Rerun the primary two-judge contrast within relation (or with relation-stratified weights). Repeat it restricted to relations with large answer sets (e.g., director, author, father, mother, composer, screenwriter) and excluding items whose answer string appears in the question. If the obscure-known abstention rate survives these restrictions, the familiarity reading is much stronger.

### F4: Popularity is measured almost exactly while knowledge is a noisy label, which biases the comparison toward popularity
- **Severity:** major
- **Location:** Section 2 (line 42); Section 3.1 (lines 70–74)
- **Evidence:** "are graded against the gold aliases" … "for popular entities, in 0.13 and 0.00"
- **Problem:** Popularity comes from page-view metadata and has essentially no error. Knowledge comes from string matching against PopQA aliases, which misses correct paraphrases and can be wrong in the gold data. Errors push facts the model knows into the "unknown" cell, and that attenuates the knowledge effect but not the popularity effect. The popular-unknown cell carries most of the knowledge effect (0.13 vs 0.00, compared with 0.48 vs 0.43 among obscure facts). Popular entities the model "never" answers correctly are exactly where alias-grading failures concentrate. The comparison "popularity effect exceeds knowledge effect by +0.30" depends on how clean that cell is. The judges' labels are not checked against humans (line 62), and neither are the knowledge labels.
- **Fix:** Hand-audit a random sample of unknown labels (at least 50 per popularity arm). Report the grading error rate per cell, and either correct the labels or report a sensitivity bound on the knowledge effect under the observed error rate.

### F5: The title and conclusion generalise a single-model abstention result
- **Severity:** major
- **Location:** Title; Section 5 (lines 128, 133–134); Appendix B, Table 2
- **Evidence:** Title: "Saying “I Don’t Know” Tracks Familiarity". Line 128: "Abstention is measured reliably in one model; the other three families rarely say “I don't know”"
- **Problem:** The behavioural half of the claim rests on one model. In the other three, the pre-set judge-agreement bar failed, and abstention in the cell where it should be highest is 0.05–0.10 (Table 2). The conclusion, "abstention follows familiarity, so calibration work should measure both", is stated without that qualifier. The abstract correctly names Llama; the title and conclusion do not.
- **Fix:** Scope the title and conclusion to Llama-3.1-8B-Instruct, or add a model that abstains often enough to measure. Also report the full 2×2 for Qwen, Gemma and Mistral with the regex detector (which needs no judge agreement), with intervals, even at low base rates. That would show whether the popularity effect has the same sign there.

### F6: The steering direction contains the abstention signal by construction, so the steering result repeats Section 3.1
- **Severity:** major
- **Location:** Section 3.2 (lines 87–89, 94–101); Figure 2a
- **Evidence:** "obscure-minus-popular (difference of cell means, averaged over knowledge)" … "so most of the effect runs through components shared with abstaining."
- **Problem:** At the free-prompt last position, the obscure cells contain about 45% eventual abstainers and the popular cells about 6% (Section 3.1). A difference of cell means therefore includes the model's upcoming-abstention state. The unknown-minus-known difference includes much less of it, because knowledge barely moves abstention (+0.09). So "obscure-popular steers abstention more than unknown-known" is what you would expect if each direction were just a scaled abstention direction. The steering adds no independent evidence for familiarity. The authors' own control supports this reading. With abstention components reduced, the direction gives 0.07, about the same as unknown-minus-known (0.09) in Figure 2a. The abstract still presents the steering as a finding ("makes Llama and Gemma-2 say 'I don't know' about facts they answer correctly").
- **Fix:** Add an abstain-minus-answer direction as a baseline at equal norm. Then project the abstain direction out of the obscure-minus-popular direction and steer with the residual. If the residual still raises abstention above random and above unknown-minus-known, the familiarity claim gains causal support. Otherwise, reframe Section 3.2 and the abstract sentence as a negative result.

### F7: The never-abstained control cannot distinguish the two accounts
- **Severity:** major
- **Location:** Section 3.2 (lines 99–101)
- **Evidence:** "A version fitted only on questions Llama never abstained on raised abstention by just +0.04 [+0.01, +0.07] over random directions"
- **Problem:** Under the familiarity hypothesis, the obscure questions on which Llama never abstains are the ones whose familiarity signal happens to be high. Restricting the fit to them removes the familiarity contrast as well as the abstention component. A small effect is therefore predicted both by "the direction is just abstention" and by "familiarity drives abstention". The control does not discriminate between them, yet the text uses it to draw a conclusion about mechanism.
- **Fix:** Use a control that removes abstention without conditioning on it. Options are projecting out a separately fitted abstain-minus-answer direction (see F6), or matching obscure and popular items on abstention rate while keeping their popularity difference. Alternatively, state explicitly that this control cannot separate the accounts.

### F8: Sample size and estimation noise of the never-abstained fit are not reported
- **Severity:** minor
- **Location:** Section 3.2 (line 100); Figure 2a, hatched bar
- **Evidence:** absent: expected the number of questions per cell used for the never-abstained fit, and a size-matched control; looked in Section 3.2, Figure 2 caption, Appendices A–D
- **Problem:** The fit is restricted to a subset, so the mean difference is estimated from fewer items and is noisier. At a fixed norm, a noisier direction projects less onto any true signal. Part of the drop from 0.29 to 0.07 could be estimation noise.
- **Fix:** Report n per cell for this fit. Add a control that fits the full obscure-minus-popular direction on a random subsample of the same size, repeated over several subsamples.

### F9: Dose and layer are tuned only for the treatment direction, and only one dose per model is shown
- **Severity:** major
- **Location:** Section 3.2 (lines 90–93); Figure 2
- **Evidence:** "had the norm of a dose chosen on the obscure-minus-popular direction alone (1× and 4× its natural norm)"
- **Problem:** The layer (17/32, 23/42) and the dose were chosen for the obscure-minus-popular direction, and different doses were chosen for the two models. The text does not say how, or on which questions. The controls inherit the treatment's dose and are never tuned, so the baseline is not tuned the same way as the treatment. If the 150 evaluation questions informed the choice, the held-out claim does not hold. The other dose for each model is not shown, which is a selective report.
- **Fix:** State the selection procedure and the data it used. Show dose-response curves (for example 0.5×–8×) for every direction, including unknown-minus-known at its own natural norm, at both doses for both models. Report a layer sweep or justify the layer choice with data independent of the evaluation set.

### F10: Five isotropic random directions are a weak control
- **Severity:** minor
- **Location:** Section 3.2 (line 92); Figure 2a
- **Evidence:** "All directions, including five random ones, had the norm of a dose chosen on the obscure-minus-popular direction alone"
- **Problem:** Random isotropic directions mostly fall outside the high-variance subspace where mean-difference directions lie, so they move behaviour less at equal norm. Beating them is a low bar.
- **Fix:** Add structured nulls: mean-difference directions between random splits of the same questions, and between unrelated attributes (for example relation A minus relation B), at equal norm. Use at least 20 of each so the null distribution has a usable spread.

### F11: High-dose steering may produce degenerate text that the regex detector counts as abstention
- **Severity:** major
- **Location:** Section 3.2 (lines 92–96, 102); Figure 2a caption
- **Evidence:** "Steering can also push a feature outside its natural range [Che and Li, 2026]."
- **Problem:** Gemma's effect appears only at 4× natural norm, and steering abstention is scored by the regex detector rather than the pre-specified two-judge measure. Its precision on steered, possibly off-distribution text is unknown. The cited Che and Li (2026; verified, arXiv 2610.07270) argues that effects at such magnitudes need not reflect the model's own computation. The authors cite it, but they do not check whether their outputs stay in range.
- **Fix:** Report fluency/perplexity and accuracy on the non-abstaining steered replies. Score steered replies with the two-judge measure, and give 10–20 random steered outputs per condition in an appendix. Report the projection of steered activations onto the direction relative to its natural range.

### F12: The lens readout confounds hedge propensity with the answer-token logit
- **Severity:** major
- **Location:** Section 3.3 (lines 104–109); Figure 3
- **Evidence:** "reading the log-odds of a fixed set of hedge tokens against the answer token at the end of"
- **Problem:** The readout is a difference of the hedge logit and the gold answer's token logit. Answers to questions about obscure entities are likely rarer tokens with lower prior logits, so obscure-minus-popular can be positive with no change in hedge propensity. The same applies to the knowledge effect, where the gold-token logit is low for unknown facts by definition. The section therefore cannot show that a "do I know this" signal exists in base models.
- **Fix:** Plot the hedge-token logit (or hedge mass against a fixed reference token set) and the answer-token logit separately by depth. Add answer-token frequency as a covariate, or match obscure and popular items on answer-token frequency.

### F13: Section 3.3 does not show that the base-model signal is familiarity rather than knowledge
- **Severity:** major
- **Location:** Abstract (lines 10–11); Section 1 (lines 35–37); Section 3.3 title; Figure 3a vs 3b
- **Evidence:** "A Jacobian lens finds the familiarity signal at mid-depth in base models of three families."
- **Problem:** The paper's thesis is a dissociation, but the base-model analysis reports only that the popularity effect is positive in the window. Figure 3b shows that the knowledge effect is as large or larger in the same readout. For Gemma-2-9B it peaks around +1.7 against about +1.0 for familiarity. Calling this "the familiarity signal" and concluding that it "predates instruction tuning" are not supported. That would need the base readout to follow popularity more than knowledge, and a link to the instruct-model abstention variable (the instruct-versus-base comparison is post hoc and unreported).
- **Fix:** Report the popularity effect minus the knowledge effect within the pre-set window per base model, with intervals. Either show the same instruct-model lens curves alongside, or reword to: "a hedge readout that differs by popularity (and by knowledge) is present in base models".

### F14: "Rules out" is stronger than a post hoc regression with crude proxies supports
- **Severity:** minor
- **Location:** Section 3.1 (lines 79–82); Appendix A (lines 415–416)
- **Evidence:** "A post hoc check rules out name spelling as the main driver."
- **Problem:** The regression gives no standard errors and does not say whether knowledge is a covariate. Its controls (word frequency, token count, token rarity) are rough proxies for surface familiarity. The paper's own matched-pairs analysis shows the gap falling from +0.76 to +0.43 at equal surface form, so surface form accounts for over 40% of that contrast.
- **Fix:** Report coefficients with intervals and the full covariate list, including knowledge. Replace "rules out … as the main driver" with "surface-form proxies reduce the popularity coefficient only slightly (−1.87 to −1.79), while in matched pairs surface form explains about 40% of the gap".

### F15: The matched-pairs analysis is underspecified
- **Severity:** minor
- **Location:** Appendix A (lines 415–418)
- **Evidence:** "Among 55 matched pairs of known facts that differ in popularity, the abstention gap of +0.76 falls to +0.43 [+0.18, +0.69] at equal surface form"
- **Problem:** The measure and unit of the "abstention gap" are not stated, and +0.76 is far larger than any contrast in Figure 1a. "Equal surface form" is not defined, and neither is how the pairs were chosen. It is unclear whether the +0.79 over 36 pairs corresponds to the +0.76 or to the +0.43.
- **Fix:** State the measure, the matching procedure and the selection criterion for the 55 pairs. Report both raw and matched gaps, with intervals, for the 36-pair subset.

### F16: The robustness table lacks intervals, contains an undefined "corrected" row, and its count does not match the text
- **Severity:** minor
- **Location:** Appendix A, Table 1; Section 3.1 (line 74); Section 5 (line 131)
- **Evidence:** "One LLM judge, corrected … +0.13" and "The difference keeps its sign under three other measurements"
- **Problem:** Only the primary row has an interval. The +0.13 row, under a third of the others, has no interval, so "keeps its sign" cannot be checked, and "corrected" is undefined. The caption says "four measurements", the table has five rows, the text says "three other" and Section 5 says "four measurements agree in sign".
- **Fix:** Add bootstrap intervals to every row and define the correction. Make the count consistent across the caption, Section 3.1 and Section 5.

### F17: The regex detector's validation is one-sided, and a statement about it conflicts with Table 2
- **Severity:** minor
- **Location:** Section 3.1 (line 75); Figure 2 caption; Appendix B, Table 2
- **Evidence:** "a regex detector with recall 0.99 on these replies" and "(regex detector, which undercounts Gemma's abstentions)"
- **Problem:** Recall is reported against the judges (presumably), without precision. Precision matters most for steered text (F11). In Table 2, the regex rate for Gemma (0.07) is higher than the both-judges rate (0.05), which contradicts "undercounts".
- **Fix:** Report precision and recall against the judges and against a small human-labelled set, separately for unsteered and steered replies. Support or remove the "undercounts" remark.

### F18: Cell sizes for the primary abstention sample are not given
- **Severity:** minor
- **Location:** Section 2 "Measuring 'I don't know'" (line 56); Figure 1a caption
- **Evidence:** "For up to 100 questions per cell (387 in total)"
- **Problem:** At least one cell has fewer than 100 questions. Readers cannot tell which cell, or whether the items were randomly sampled within cells. The popular-known cell sits at 0.00, so its bootstrap interval collapses to a point.
- **Fix:** Give n per cell and the sampling rule. For the 0.00 cell, report an exact binomial or Wilson upper bound rather than a degenerate bootstrap interval.

### F19: The steering intervals ignore variability in direction sampling and data split
- **Severity:** minor
- **Location:** Figure 2 caption; Section 3.2 (lines 87, 100)
- **Evidence:** "Error bars: 95% bootstrap over questions."
- **Problem:** The comparison "+0.04 over random directions" treats the mean of five random directions as fixed. The directions come from a single random half-split. The number of generations per question under steering is not stated.
- **Fix:** Use a hierarchical bootstrap over random directions and questions, and refit the directions over several random splits (at least five). State the generations per question and the temperature.

### F20: The pre-registration claims cannot be checked from the manuscript
- **Severity:** minor
- **Location:** Section 2 (lines 65–66); Appendix D
- **Evidence:** "Each confirmatory analysis followed a criterion note hashed before its run (Appendix D)"
- **Problem:** Neither the hashes nor the notes are given, and the confirmatory criterion for Section 3.1 (what result would have counted against the hypothesis) is not stated. Several analyses are not labelled either way: the forced-prompt hedge check (F2) and the Appendix C timing. A reader cannot tell which results were fixed in advance.
- **Fix:** List each hash with its criterion text in an appendix or an anonymised supplement, and state the pre-set decision rule for Section 3.1. Mark every reported analysis as confirmatory or post hoc.

### F21: "At least four times" rests on point estimates and barely holds for Llama
- **Severity:** minor
- **Location:** Section 3.1 (lines 77–78); Figure 1c
- **Evidence:** "In every family the knowledge effect is at least four times the popularity effect (Figure 1c)."
- **Problem:** For Llama the ratio is 2.14/0.51 ≈ 4.2 from point estimates, and the lower bound of a ratio interval is likely below 4. The threshold looks chosen after the fact.
- **Fix:** Report the knowledge-minus-popularity difference (or the ratio) per model with bootstrap intervals, and drop the "four times" wording unless it was pre-specified.

### F22: "Double dissociation" overstates the steering result
- **Severity:** minor
- **Location:** Section 3.2 (lines 96–98); Figure 2b
- **Evidence:** "a double dissociation;"
- **Problem:** In Gemma, the obscure-minus-popular direction also raises entropy (about 0.27 to 0.83, against about 0.55 for random directions). In Llama, both directions raise entropy equally (−0.03 [−0.11, +0.05]). The data show a crossover interaction in one model, not two clean single dissociations.
- **Fix:** Report every condition's change relative to the random-direction mean on both readouts, and call it an interaction in Gemma.

### F23: Cross-model lens magnitudes are compared after being declared incomparable, and sign reversals go unmentioned
- **Severity:** minor
- **Location:** Section 3.3 (lines 107–109); Figure 3 caption and panels
- **Evidence:** "Lenses are fitted per model, so magnitudes are not comparable across models."
- **Problem:** The text still ranks the models, "from +0.29 … in OLMo-3-7B to +0.67 … in Llama-3.1-8B". The window average hides OLMo's negative familiarity effect late in the network and its strongly negative knowledge effect (about −2.3 at depth 0.75), which the text does not discuss. Lens faithfulness per layer is not reported. Gurnee et al. (2026; verified, arXiv 2607.15495) introduces the Jacobian lens, but the pages I could open discuss Claude models and mention open-model readouts on Neuronpedia without naming these four. That official prefitted lenses exist for these models is [unverified].
- **Fix:** Drop the cross-model ranking and report each model against its own null. Discuss the OLMo reversals. Report a per-layer lens-fidelity metric, and give the source and version of each lens.

### F24: "Emerge together" in Section 3.3 sits beside a 12-layer gap in Appendix C
- **Severity:** minor
- **Location:** Section 3.3 (line 114); Appendix C (lines 421–423)
- **Evidence:** "reaches half its final size at layer 16, while the entropy gap does so at layer 28, 12 layers later"
- **Problem:** The two analyses compare different quantities (hedge familiarity vs knowledge effects, and abstention readout vs entropy gap), but readers will see them as contradictory timing claims. The Appendix C analysis has no stated pre-registration status and no interval on the half-size layers.
- **Fix:** State in one sentence how the two timing analyses differ. Give bootstrap intervals on the half-size layers and the status of the analysis.

### F25: The averaged effects hide an interaction and depend on how extreme each contrast is
- **Severity:** minor
- **Location:** Section 2 "Effects" (lines 63–64); Section 3.1 (lines 70–74); Abstract (line 5)
- **Evidence:** "The abstentions of Llama-3.1-8B-Instruct follow popularity (effect +0.39) far more than knowledge (+0.09)"
- **Problem:** The knowledge effect is 0.05 among obscure entities and 0.13 among popular ones, and the popular-known cell is at floor. The relative sizes of the two main effects also depend on choosing deciles for popularity and ≥90%/0% for knowledge, so "far more" is partly a property of the design. The quantity that does not depend on scale is the crossover: abstention vs entropy × popularity vs knowledge.
- **Fix:** Report the popularity × knowledge interaction with an interval. Present the readout × factor crossover as the main statistic, or show how the effects change with less extreme popularity bins (for example quartiles).

### F26: Knowledge labels for the base models are not defined
- **Severity:** minor
- **Location:** Section 2 (lines 41–44); Section 3.3 (lines 104–109)
- **Evidence:** absent: expected how known/unknown was assigned for the four base models (own samples under which prompt, or the instruct model's labels); looked in Section 2, Section 3.3, Figure 3 caption, Appendix D
- **Problem:** The labelling procedure uses the chat-template forced-answer prompt, which base models do not have. The knowledge effect in Figure 3b cannot be interpreted without knowing whose labels are used.
- **Fix:** State the labelling prompt and source for each base model, and the resulting cell sizes.

## Strengths

1. **The 2×2 crossing of popularity with self-derived knowledge labels** (Section 1, lines 26–30; Section 2). It removes the confound the authors identify in prior "known entity" work: Ferrando et al. label entities known when the model gets at least two attributes right (verified, arXiv 2411.14257). Without it, there is no way to separate familiarity from knowledge.
2. **Large effect within facts the model knows** (Section 3.1, lines 70–72; Figure 1a). Abstention is 0.43 for obscure known facts against 0.00 for popular known ones. Several measurements agree in sign (Table 1), and the primary measure is conservative (both judges must agree). This is the one result that could not be removed without losing the paper.
3. **Pre-set bars, with failures reported** (Section 2, lines 59–62; Section 3.3, lines 110–114; Appendix C, lines 427–430). The judge bar failed in three families, the depth-ordering prediction failed (1/1/6), and the circuit fell short of its behavioural bar. All three are reported, and the circuit is kept out of the main text. Without this, readers would have no basis for trusting that the reported effects were not selected.
4. **Equal-norm controls and a held-out evaluation in steering, plus a control that limits the authors' own claim** (Section 3.2, lines 87–101). The line "This shows control, not that the model uses familiarity as such" keeps the section honest even though the control is weak (F7).
5. **A lens depth window fixed before the runs** (Figure 3 caption). It blocks layer cherry-picking in an analysis with dozens of layers per model.

## Questions for the authors

1. What is the relation composition of each cell? Does the obscure-known abstention rate (0.43) hold when restricted to relations with large answer sets and to items whose answer does not appear in the question? (F3)
2. What are the Llama numbers, with intervals, for popularity and knowledge effects on the forced-prompt hedge log-odds? (F2)
3. How, and on which questions, were the steering layer and dose chosen? What do the other dose and the unknown-minus-known direction at its own natural norm give? (F9)
4. What happens to abstention if you steer with the obscure-minus-popular direction after projecting out an abstain-minus-answer direction? (F6)
5. In the base-model lens, does the obscure-minus-popular effect come from the hedge-token logit or from the answer-token logit? (F12)
6. When knowledge is labelled from a separate prompt or sample set, how large is the knowledge effect on entropy, and how large is the popularity effect within unknown facts? (F1)

## Recommendation: major revision

The Llama abstention result in Section 3.1 is interesting, measured conservatively, and suitable for an early-stage workshop paper. The supporting claims are another matter. "Answer uncertainty tracks knowledge", the causal reading of steering, and "the familiarity signal predates instruction tuning" each rest on a design that could produce the reported pattern without the claimed mechanism.

**Findings that drove this:**
- **F1:** the entropy-knowledge link is partly built into the labels.
- **F3:** obscure "known" facts may be guessable, which would make abstention on them an uncertainty response.
- **F6:** the steering direction contains abstention by construction.

**What would change my mind:**
- **For F3:** a relation-stratified or high-cardinality-only rerun of the two-judge contrast in which obscure-known abstention persists.
- **For F1:** entropy recomputed with knowledge labels from a disjoint prompt or sample set.
- **For F6:** either a projection-out steering result above the random and unknown-minus-known baselines, or a rewrite of Section 3.2 and the abstract as a negative steering result.

With those analyses, plus a title scoped to the model where abstention was measured (F5), I would move to accept.
