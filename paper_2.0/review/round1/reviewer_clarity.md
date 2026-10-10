# Referee report: clarity and the fresh reader

UniReps 2026 extended abstract: "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge"

## Step 1: what a skim leaves behind (abstract, figures, conclusion only)

**What I believe it claims:** When language models say "I don't know", they respond to how familiar the question's subject is, not to whether they know the answer. The model's uncertainty over its answer tracks knowledge instead, so the two are separate variables. A direction in activation space causally drives the familiarity response, and that response already exists in base models.

**What I believe it shows:** In one model (Llama-3.1-8B-Instruct), "I don't know" rates depend on Wikipedia popularity much more than on a knowledge label from sampling (Fig. 1a). First-token entropy depends on knowledge in four families (Fig. 1b,c). Adding an obscure-minus-popular direction raises "I don't know" on known facts to 0.29 (Llama) and 0.22 (Gemma) (Fig. 2a). A lens readout of hedge tokens versus the answer token separates obscure from popular at mid-depth in four base models (Fig. 3a).

**Where the skim misleads.** The title and the conclusion's sentence on line 133 state the abstention result as general, but abstention was measured in one model, with judge agreement only just above its bar. The abstract's steering sentence ends "mostly through components shared with abstention itself". A skimmer reads this as mechanistic detail, but the body (lines 99-101) says it means the familiarity-specific part of the direction adds only +0.04. Fig. 3's title says base models "carry a familiarity signal", but neither the caption nor the abstract says that the knowledge signal emerges at the same depth (Fig. 3b) or that OLMo's readout reverses sign in late layers. The pre-registered prediction about depth failed, and a skim would never learn that.

## Findings

**F1. Major. Title; abstract lines 11-13; conclusion line 133.**
Evidence: "Answer uncertainty follows knowledge while abstention follows familiarity, so calibration work should measure both."
Problem: Abstention was measured in one model only (lines 61-62, 128). That model's judge agreement is κ = 0.62 against a 0.60 bar, and the judges were never checked against human labels. The title, the last sentence of the abstract and the conclusion all state the dissociation without that qualifier, so a skimmer comes away believing it holds across models.
Fix: Line 133: "In Llama-3.1-8B-Instruct abstention follows familiarity, while answer uncertainty follows knowledge in all four families, so calibration work should measure both." Abstract line 11: "...are different variables in the one model where abstention could be measured, and...". Title: add "in Llama" or "in an LLM".

**F2. Major. Abstract lines 9-10; Fig. 2 caption title; Section 3.2 lines 99-101.**
Evidence: "mostly through components shared with abstention itself"
Problem: The abstract does not define "components shared with abstention". Only in the body do I learn what it means: a familiarity direction fitted without any abstaining questions raises abstention by just +0.04 over random. So steering does not isolate a familiarity variable. It mostly re-injects the abstention difference that obscure questions already carry. The abstract and the Fig. 2 title ("makes models say 'I don't know' about facts they know") present the result as support for the main claim. Line 99 ("This shows control, not that the model uses familiarity as such") is the honest reading, and it is invisible to a skimmer.
Fix: Abstract: "Steering along an obscure-minus-popular direction raises Llama's 'I don't know' rate on known facts from 0.02 to 0.29, but a version fitted without abstaining questions adds only +0.04, so the effect runs mostly through abstention itself." Fig. 2 title: "...mostly via components shared with abstention."

**F3. Major. Abstract line 4; lines 40-41; used throughout.**
Evidence: "separates familiarity with a question's subject from knowledge of the fact."
Problem: The measured variable is the decile of Wikipedia page views ("popularity"). "Familiarity" is an interpretation of it: the model's exposure to the name in training. The paper uses "familiarity", "popularity" and "obscure-minus-popular" interchangeably. It never says that popularity stands in for familiarity, or why page views should track training exposure (line 134 even makes this conditional: "If familiarity reflects training-text exposure"). A reader cannot tell what was measured and what was inferred.
Fix: At line 41, add "; we treat page-view popularity as a proxy for the model's familiarity with the subject". Then use "popularity" for every measured quantity in Sections 2-3 and keep "familiarity" for interpretation. To pay for the clause, cut line 102 ("Steering can also push a feature outside its natural range [Che and Li, 2026].").

**F4. Major. Abstract line 10; Section 3.3 line 104.**
Evidence: "We decoded every layer of four base models with the Jacobian lens and its official prefitted lenses"
Problem: The Jacobian lens is never defined, yet all of Section 3.3 and Fig. 3 depend on it. A non-specialist cannot tell what "lens-decoded log-odds" at layer L means: a learned probe, a projection through the unembedding, or something else. "Decodable" (line 36) suggests a probe.
Fix: Replace the clause with "with the Jacobian lens [Gurnee et al., 2026], which maps a layer's hidden state to next-token logits by <one-clause mechanism>, using the released per-model fits". Change "decodable" in line 36 to "readable through a lens".

**F5. Major. Fig. 2b caption; lines 54-55, 97-98.**
Evidence: "the steering and lens analyses each use one prompt."
Problem: Steering is applied at the free prompt (line 88). Section 2 says first-token entropy is only an answer-uncertainty measure under the forced-answer prompt, because at the free prompt it spreads over preambles like "Sure" (lines 49-50). The Fig. 2b caption says only "First-token entropy". If 2b uses the free prompt, it is not answer uncertainty, and the "double dissociation" on line 97 is between abstention and preamble entropy. If 2b uses the forced-answer prompt, then line 54 is wrong. I cannot tell which.
Fix: In the Fig. 2b caption, write "(b) First-token entropy under the <forced-answer / free> prompt, with the same steering". If it is the forced-answer prompt, change line 54 to "the lens analysis uses one prompt".

**F6. Major. Section 2, lines 51-55.**
Evidence: "(Gemma: popularity +1.79, knowledge -0.32; in Llama both effects are large)"
Problem: This is the only defence against the prompt confound. For Llama, the only model whose abstention is reported, it gives no numbers, and "both effects are large" does not say which is larger, which is exactly the question. The numbers also arrive in Setup, before the reader knows what hedge tokens are, what the main result is, or what these numbers should show. Gemma's negative knowledge effect (known facts hedge more) goes unexplained.
Fix: Move the sentence to the end of Section 3.1 and give both models: "At the forced-answer position, the hedge-versus-answer log-odds show popularity and knowledge effects of +X and +Y in Llama and +1.79 and -0.32 in Gemma, so the dissociation does not depend on the prompt change." If the Llama knowledge effect is the larger one, say so and soften line 83.

**F7. Major. Section 3.1 line 79 vs. Appendix A lines 415-416.**
Evidence: "A post hoc check rules out name spelling as the main driver."
Problem: The main text cites only the regression (-1.87 to -1.79). Appendix A reports that surface form cuts the matched-pair gap from +0.76 to +0.43, which is about 43% of it. That is "part, not most", not "ruled out". A reader who checks the appendix finds the main text overstated.
Fix: "A post hoc check finds name spelling explains part but not most of the effect: the regression coefficient moves from -1.87 to -1.79 with surface controls, and a matched-pair gap falls from +0.76 to +0.43 (Appendix A)."

**F8. Major. Section 3.3 lines 112-114; Appendix C lines 421-423.**
Evidence: "Across eight base and instruct models our criterion returned 1 supporting, 1 contradicting and 6 undecided outcomes"
Problem: The paper never says which eight models these are (Fig. 3 shows four base models), and "our criterion" is never stated, so the failed prediction cannot be checked. Appendix C also reports that in Llama the lens abstention readout "reaches half its final size at layer 16, while the entropy gap does so at layer 28, 12 layers later". That reads as support for the very prediction declared failed, and nothing reconciles the two.
Fix: Name the models and the criterion in one clause, for example "(Fig. 3's four base models and the four instruct models of Section 3.1; onset = first layer whose interval excludes zero; undecided = onsets within one layer)". In Appendix C, add one sentence explaining why the 12-layer lead against entropy does not bear on the knowledge-effect criterion.

**F9. Major. Fig. 3; Section 3.3.**
Evidence: absent. Expected: a comment on OLMo-3-7B's late-layer curves. Looked in Section 3.3, the Fig. 3 caption, and Appendices C-D.
Problem: In Fig. 3a, OLMo's obscure-minus-popular effect turns negative after a relative depth of about 0.75. In Fig. 3b, its knowledge effect falls to about -2.3, meaning known facts get higher hedge log-odds than unknown ones. The figure title ("Base models already carry a familiarity signal in their decoded 'I don't know' readout") rests only on the window average. A reader looking at the figure sees a readout that, in one model, does not behave like "I don't know" at the layers nearest the output, and nothing explains it.
Fix: Add to the caption: "OLMo's effects reverse sign after depth 0.75; we test only the pre-set window." To pay for this, shorten lines 107-109 as in F21.

**F10. Major. Section 3.2 lines 92-93; Fig. 2 x-labels.**
Evidence: "had the norm of a dose chosen on the obscure-minus-popular direction alone (1× and 4× its natural norm)"
Problem: "Dose" and "natural norm" are coined and not defined. The paper does not say how the dose was chosen (by what criterion, and on which questions), or why Llama gets 1× and Gemma 4×. If the dose was tuned to maximise abstention for the obscure-minus-popular direction, then equal-norm controls are not a level comparison, and I cannot read Fig. 2a without knowing this. The steering layers (17/32, 23/42) are also unexplained.
Fix: "All directions were scaled to the norm of the obscure-minus-popular difference vector (its natural norm) times a dose of 1× (Llama) or 4× (Gemma), the smallest dose that <criterion> on the fitting half; layers were chosen by <rule>."

**F11. Minor. Abstract lines 5-6.**
Evidence: "follow popularity (effect +0.39) far more than knowledge (+0.09)"
Problem: The numbers appear before the reader knows their scale or what they compare. They are differences in the proportion of replies, but nothing says so.
Fix: "says 'I don't know' 39 percentage points more often about obscure than popular subjects, but only 9 points more often about facts it lacks than facts it knows".

**F12. Minor. Section 2 line 42.**
Evidence: "under the forced-answer prompt below are graded against the gold aliases."
Problem: The term is used before it is defined; the definition comes in the next paragraph.
Fix: Swap the paragraphs "Questions and knowledge labels" and "Two prompts, by necessity", which costs no length.

**F13. Minor. Section 2 line 56.**
Evidence: "For up to 100 questions per cell (387 in total)"
Problem: "Cell" is not defined. I inferred that it means the four popularity × knowledge groups.
Fix: "For up to 100 questions in each popularity × knowledge cell (387 in total)".

**F14. Minor. Section 2 line 58; Table 2.**
Evidence: "label each reply DONT_KNOW, CLARIFY or ANSWER"
Problem: CLARIFY is never defined, yet Table 2 shows the other families' abstention rate rising from 0.05-0.10 to 0.22-0.32 when CLARIFY is included. So the statement "these models rarely abstain" depends on what CLARIFY means.
Fix: Add a parenthesis such as "CLARIFY (asks which entity or sense is meant)".

**F15. Minor. Fig. 3a y-label; abstract line 10; Section 2 line 63; line 81.**
Evidence: "familiarity effect (obscure − popular)"
Problem: One quantity has three names: "popularity effect" (Section 2), "familiarity effect" (Fig. 3) and "familiarity signal" (abstract). Its sign is positive when the subject is unfamiliar. The regression on line 81 then uses the opposite sign (-1.87 per SD of popularity). I had to re-derive the direction each time.
Fix: Label Fig. 3a "popularity effect (obscure − popular)", matching Section 2. Line 81: "each SD of log page views lowers the abstention log-odds by 1.87 (1.79 after controlling for...)".

**F16. Minor. Line 74; Table 1 caption; line 131.**
Evidence: "The difference keeps its sign under three other measurements"
Problem: The counts disagree. Table 1 has five rows, the primary measurement plus four others. Its caption says "four measurements", and line 131 says "four measurements agree in sign".
Fix: Line 74: "four other measurements". Caption: "under five measurements". Line 131: "five measurements".

**F17. Minor. Table 1 row 2; line 75.**
Evidence: "One LLM judge, corrected"
Problem: The paper does not say what was corrected or how. The regex detector's "recall 0.99" does not say what it is measured against, and no precision is given. The regex also gives +0.43 against the judges' +0.30, which suggests it over-counts.
Fix: "One LLM judge, corrected for <bias, e.g. its false-positive rate on N hand-labelled replies>". Line 75: "a regex detector with recall 0.99 and precision P against the two-judge labels".

**F18. Minor. Table 2, Llama column.**
Evidence: "Both judges DONT_KNOW or CLARIFY   –" (Llama)
Problem: The Llama column shows dashes for the CLARIFY and regex rows, but Table 1 reports both measurements on Llama's same replies. The gap is not explained.
Fix: Fill in Llama's values for the obscure-unknown cell, or add the footnote "Llama: see Table 1".

**F19. Minor. Section 2 line 52; Section 3.3 lines 105-106.**
Evidence: "reading the log-odds of a fixed set of hedge tokens against the answer token"
Problem: The hedge tokens are never listed, and "the answer token" is ambiguous: it could be the gold answer's first token or the model's greedy token. Both the prompt-confound check and Fig. 3 rest on this readout.
Fix: Add one line to Appendix C: "Hedge tokens: {...}; answer token: first token of the gold answer."

**F20. Minor. Section 3.2 line 94; Fig. 2a caption.**
Evidence: "(regex detector, which undercounts Gemma's abstentions)"
Problem: Section 3.1 measures abstention with two judges. Section 3.2 switches to regex, and only the caption says so. Readers will compare 0.29 with Section 3.1's 0.48 and 0.43, and the reason for using a detector known to undercount Gemma is not given.
Fix: Line 94: "raised Llama's abstention rate (regex detector) from 0.02 to 0.29". Add a half-clause on why regex was used, for example "(judges were too costly for N steered generations)".

**F21. Minor. Section 3.3 lines 107-109 vs. Fig. 3 caption.**
Evidence: "from +0.29 [+0.19, +0.38] in OLMo-3-7B [Olmo et al., 2025] to +0.67 [+0.59, +0.75] in Llama-3.1-8B"
Problem: The caption says magnitudes "are not comparable across models", but the text sets them out as a range from smallest to largest, which invites exactly that comparison.
Fix: "In all four models the obscure-minus-popular effect is positive in the pre-set mid-depth window, with every interval excluding zero (Figure 3a; values in Appendix C)." This saves about a line, which can go to F9.

**F22. Minor. Section 3.2 lines 96-98.**
Evidence: "There, the unknown-minus-known direction instead raised answer entropy more (difference +0.68 [+0.59, +0.77])"
Problem: "There" and "difference" make the reader reconstruct what is being compared. The "double dissociation" is also relative: in Gemma, the obscure-minus-popular direction still raises entropy above random (about 0.82 vs 0.55 in Fig. 2b).
Fix: "In Gemma the unknown-minus-known direction raised answer entropy 0.68 nats [0.59, 0.77] more than the obscure-minus-popular direction but did not raise abstention; in Llama the two raised entropy equally (−0.03 [−0.11, +0.05])."

**F23. Minor. Section 3.2 line 99.**
Evidence: "This shows control, not that the model uses familiarity as such."
Problem: In a paragraph about control directions, "control" reads as "experimental control".
Fix: "This shows the direction can cause abstention, not that the model uses familiarity as such."

**F24. Minor. Fig. 1.**
Evidence: Fig. 1c legend text "knowledge / popularity" placed inside the axes.
Problem: In the PDF the inline legend overlaps the Qwen row. Panels a-b use circle = known and square = unknown, while panel c uses circle = popularity and square = knowledge, so the same marker shapes change meaning within one figure. The panel b y-label is cut off ("first-token entropy (nat").
Fix: Use different markers in panel c (for example triangle and diamond), move its legend above the axes, and widen panel b so its label fits.

**F25. Minor. Section 4 lines 124-126.**
Evidence: "a strict uncertainty representation should not depend on its source [Lippl et al., 2024], yet the abstention variable depends on one"
Problem: This is the paper's only link to the workshop's representational theme, and an outsider cannot follow it: "strict" and "its source" are undefined.
Fix: "If a representation encoded uncertainty as such, its readout would not depend on what caused the uncertainty [Lippl et al., 2024]; the abstention variable instead depends on one cause, familiarity." To make room, cut lines 119-120 (Nguyen; Kang), which the argument does not use.

**F26. Minor. Appendix D lines 437-438.**
Evidence: "the instruct-versus-base comparison of lens magnitudes were decided after seeing data"
Problem: No instruct-versus-base lens comparison appears anywhere in the manuscript. The hashes that line 65 relies on are not reported either, so "hashed before its run" cannot be checked.
Fix: Delete the clause, or report the comparison in one sentence. List the hashes, or give an anonymised link, in Appendix D.

**F27. Minor. Section 4 line 117.**
Evidence: "Brzezinka [2026] distinguish graded familiarity probes"
Problem: The cited work has a single author, so the verb should be "distinguishes".
Fix: Change "distinguish" to "distinguishes".

## Numbers checked

- Fig. 1a vs. text: 0.48 / 0.43 / 0.13 / 0.00 give a popularity effect of (0.35 + 0.43) / 2 = 0.39 and a knowledge effect of (0.05 + 0.13) / 2 = 0.09. The difference is 0.30. All consistent.
- Fig. 1b vs. text: (3.5 − 0.95 + 2.6 − 0.85) / 2 ≈ 2.15 nats against the stated +2.14, and the popularity effect ≈ 0.5 against +0.51. Consistent.
- Fig. 1c, "at least four times": Llama 2.14 / 0.51 ≈ 4.2 is the closest case. Consistent.
- Fig. 2a: Llama 0.02 / ~0.03 / 0.09 / 0.29, and never-abstained ≈ 0.065 − 0.025 ≈ +0.04 over random. Gemma 0 / 0 / 0 / 0.22. Consistent.
- Fig. 2b: Gemma 1.50 − 0.82 = 0.68 and Llama 1.23 − 1.26 = −0.03. Consistent.
- Table 2 Llama 0.48 matches Fig. 1a. The κ values match line 60. The measurement counts do not match (F16).

## Strengths

1. **The design crosses two factors (lines 26-30).** Popularity is crossed with a knowledge label taken from the model's own samples, and the intro states what each hypothesis predicts before any result. This is the clearest part of the paper.
2. **Fig. 1a-b carry the central dissociation.** The opposite slopes are visible at a glance, and every number in Section 3.1 checks against them.
3. **Negative results are reported honestly.** The failed depth prediction (lines 110-114), the circuit excluded for missing its pre-set bar (Appendix C), the judges never checked against humans (line 62) and the κ bar are all stated.
4. **The never-abstained control (lines 99-101) and the sentence on line 99** spell out what the steering result does not show. Few papers include such a control, and the body interprets it correctly. The abstract does not (F2).
5. **The limitations paragraph (lines 128-132)** is specific and matches what the body shows.

## Questions for the authors

1. In Llama, what are the popularity and knowledge effects on the hedge-versus-answer log-odds at the forced-answer position? If the knowledge effect is the larger one, the prompt change could explain the main dissociation (F6).
2. Under which prompt is the Fig. 2b entropy measured (F5)?
3. How was the steering dose chosen: by what criterion, and on which questions? Was it tuned to maximise abstention for the obscure-minus-popular direction (F10)?
4. Which eight models enter the depth criterion? How does Appendix C's 12-layer lead fit with "the two signals emerge together" (F8)?
5. Why does OLMo's late-layer knowledge effect reach about −2.3? Does the lens readout mean "I don't know" in that model (F9)?

## Recommendation

**Minor revision.** The core result (Fig. 1a-b, Section 3.1) is easy to follow and internally consistent, and the paper is unusually candid about its limits. The verdict rests on gaps between what a skim conveys and what the body shows: generality (F1), the steering control (F2) and name spelling (F7). It also rests on four places where a non-specialist cannot follow or check the argument: the undefined lens (F4), the prompt behind the Fig. 2b entropy (F5), the missing Llama numbers in the prompt-confound check (F6), and the unspecified depth criterion together with the Appendix C timing result that seems to contradict it (F8). Two more figure-level gaps matter: the unexplained OLMo reversal (F9) and the undefined steering dose (F10). Except for reporting the Llama numbers in F6 and the dose criterion in F10, which the authors presumably already have, all of these are text edits within the page limit. The cuts suggested in F3, F21 and F25 free the lines they need. If the F6 Llama numbers show a knowledge effect larger than the popularity effect, I would move to major revision.
