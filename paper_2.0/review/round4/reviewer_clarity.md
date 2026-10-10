# Referee report: "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge"

Lens: clarity and the fresh reader. Line numbers are the manuscript's printed line numbers.

## Skim first (abstract, figures, limitations)

What I believe the paper claims: in Llama-3.1-8B-Instruct, whether the model says "I don't know" depends more on how popular (Wikipedia page views) the entity is than on whether the model actually knows the fact. Gemma-2-9B shows the same pattern when the prompt explicitly permits abstention, and the authors could not find the mechanism.

What I believe it shows: in a 2x2 of obscure/popular entities and known/unknown facts, Llama abstains on 43% of replies about obscure entities whose facts it knows and on 0% about popular entities whose facts it knows, while its first-token answer entropy is about the same in those two cells. Steering and lens readouts do not isolate a familiarity signal.

What a skim would leave a reader believing, and where the full paper disagrees:
- That obscure entities are *unfamiliar* to the model. The abstract's first sentence and the introduction frame the question as familiarity, but Section 3.1 (l.109-112) reports that Llama recognizes obscure-known entities *more* often than popular-known ones and that name surprisal does not track page views. The title says "popularity", which is accurate. The abstract and introduction say "familiar", which the paper's own evidence does not support.
- That answer uncertainty following knowledge is a second, independent finding (Figure 1 title). The labels are built from the same forced-answer distribution, so it is largely true by construction. The abstract says so (l.13); the Figure 1 caption does not.
- That the result survives every label check with preferences of +0.23 or more. The smallest label variant (+0.19), and the fact behind it (48% of popular "unknown" questions give the gold answer in some free reply), appear only in Appendix A and Table 2.
- Nothing about representational convergence, although the introduction raises it as the workshop question (l.40-42).
- There is no conclusion section; the paper ends on a five-line Limitations section.

## Findings

**F1. major. Framing: "familiarity" in the abstract and introduction, contradicted by Section 3.1**
- Location: Abstract l.1-2; Introduction l.22-29; Setup l.54; Section 3.1 l.109-112.
- Evidence: "Neither model-side measure shows the obscure names to be unfamiliar to Llama" (l.109-110), against the abstract's "or how familiar the entity is" (l.2).
- Problem: The paper opens by asking whether abstention tracks familiarity and builds the misfire story on unfamiliar names failing to suppress a default. Its own measurements then show the obscure names are recognized (0.83 vs 0.75, l.101) and that surprisal does not track popularity. A reader who stops at the abstract comes away with a familiarity conclusion the body explicitly withholds.
- Fix: Recast the abstract's first sentence and l.22-35 in terms of entity *popularity* (the measured variable). Present familiarity as the motivating hypothesis that popularity was meant to proxy. Then move the l.109-112 result into the abstract as one sentence: "Popularity is not explained by recognition or name surprisal, so the familiarity account is not supported directly."

**F2. major. "Within One Prompt: The Split Remains": cannot be followed, and the header overstates it**
- Location: Section 3.1, l.113-117.
- Evidence: "where the log-odds of a first token 'I' follow popularity (+8.4 vs. +3.1), while its entropy follows knowledge (+0.16 vs. +0.05 nats)"
- Problem: I could not state this paragraph's purpose until I worked out that it addresses the mismatch between prompts (abstention is measured on the free prompt, entropy on the forced prompt). The pre-specified within-prompt readout shows *no* preference (-0.12). The readout that shows the split was specified after seeing that result, and the paragraph does not say whether it is labelled post hoc. "+8.4 vs +3.1" has no unit and no interval, and log-odds differences of 8.4 are implausibly large unless the unit is something else. "its entropy" has an unclear referent. The header "The Split Remains" claims more than a post hoc readout can support when the pre-specified one failed.
- Fix: Open with one sentence giving the purpose ("Abstention and entropy come from different prompts; here we read both at one prompt"). State the pre-specified null plainly. Label the free-prompt reading post hoc. Give units and 95% intervals for +8.4/+3.1 and +0.16/+0.05, and say which token's entropy is meant. Retitle to something like "Within One Prompt: Mixed".

**F3. major. The reply-level figures for gold answers inside "I don't know" replies do not reconcile**
- Location: Section 3.1 l.97-98; Appendix A l.274-275 and l.292-294.
- Evidence: "At most 0.03 of the 'I don't know' replies in any cell contain the gold answer" (l.97-98) vs "0.53 of free replies contain a gold alias, and 0.72 of the replies that do not abstain" (l.292-293).
- Problem: Take the obscure-known abstention rate as 0.43. Under either reading of the ambiguous l.292-293 sentence, roughly 0.09-0.12 of all obscure-known replies would both abstain and contain a gold alias. That is about a fifth to a quarter of the "I don't know" replies, not at most 3%. The two statements may refer to different reply sets (24-token pool vs 96-token sample), but the text does not say. The claim "The 'I don't know' replies are refusals, not hedged answers" (l.95-96) depends on this number.
- Fix: In l.292-294, name the reply set (prompt, token budget, replies per question). Rewrite the sentence so that "0.72" has one reading. Report the share of abstaining obscure-known replies that contain a gold alias on the same 387-question, 96-token sample used in l.97. If the numbers differ, reconcile them in one sentence.

**F4. major. The main text's robustness summary leaves out the weakest label variant and the fact behind it**
- Location: Section 3.1 l.89-99; Table 2 ("Popular unknown with the answer in a reply dropped", +0.19); Appendix A l.269-271.
- Evidence: "Dropping instead every popular unknown question with a gold alias in any free reply (0.48 of them) gives +0.19 [+0.07, +0.31]."
- Problem: Nearly half of the popular "unknown" facts, the cell that defines a misfire, produce the gold answer in at least one free reply. That bears directly on what "unknown" means for the cell that drives the preference (preference = obscure-known minus popular-unknown, l.75-76). The main text reports +0.32 and +0.23 and says "The Preference Survives". The smallest label variant and the 48% figure appear only in an appendix paragraph and a table row.
- Fix: Add one sentence after l.95: "Nearly half (0.48) of popular unknown questions name the gold answer in some free reply; dropping them gives +0.19 [+0.07, +0.31]." Give the full range of label variants (+0.19 to +0.32) in contribution 1 (l.45) and in the abstract's robustness sentence (l.8-10).

**F5. major. Gemma's steering results in the main text and Figure 2a use the scorer the authors call indefensible for Gemma**
- Location: Section 3.3 l.138-142; Figure 2a; Appendix C l.333-337.
- Evidence: "For Gemma the secondary scorer is the defensible one, since its two judges agree below the bar."
- Problem: The main text reports Gemma's two-judge numbers (0.00 to 0.04; removed +0.02). With the scorer the paper itself calls defensible, Gemma's obscure-minus-popular effect is +0.11 and the residual after removing the abstention component is +0.05 [+0.03, +0.07] (within-level: +0.03 [+0.01, +0.04]). That is small but not nil, and it qualifies contribution 3's "steering moves a generic abstention signal" for Gemma. A reader of the four pages sees only the weaker numbers.
- Fix: Report the three-judge Gemma numbers in l.139-142, and either plot them in Figure 2a or add a three-judge panel. State in one clause whether the Gemma residual changes the conclusion of contribution 3.

**F6. major. The representational-convergence question posed for UniReps is never answered**
- Location: Introduction l.40-42; Section 3.2 l.129-130; Appendix E.
- Evidence: "do models that agree on answer uncertainty also agree on what makes them abstain? Only Llama and, under a permissive prompt, Gemma let us ask."
- Problem: This is the paper's only connection to the workshop theme. The single follow-up sentence (l.129-130) refers to "that readout", which grammatically points to permissive-prompt abstention. Appendix E actually analyses forced-answer hedge log-odds and entropy, a different readout, and calls the analysis exploratory and post hoc. I cannot state what the paper concludes about convergence.
- Fix: Either add a two-sentence answer in Section 3.2: "Llama and Gemma show the same direction and similar size of preference under the permissive prompt (+0.47 vs +0.30). Question-level agreement on forced-answer hedging beyond shared labels met our rule only for Llama–Mistral (Appendix E)." Or remove the convergence framing from l.40-42. In either case fix the referent of "that readout".

**F7. minor. "Popularity preference" and its numbers appear in the contributions before the term is defined**
- Location: Contributions l.44-47; definition l.71-76.
- Evidence: "Llama's popularity preference is +0.30 [+0.20, +0.40] (Section 3.1)."
- Problem: The reader meets a signed number and interval with no scale or meaning; the definition (obscure-known minus popular-unknown abstention) comes 30 lines later.
- Fix: In contribution 1, gloss it inline: "Llama abstains 0.30 more on obscure facts it knows than on popular facts it does not (the popularity preference, Section 2)."

**F8. minor. "Plain prompt", "free prompt" and "permissive prompt": one name too many, and one used before it is defined**
- Location: Abstract l.11; l.42, l.46-47; Setup l.65; Section 3.2 l.119, l.125-126.
- Evidence: "Gemma-2-9B rarely abstains under a plain prompt." (l.10-11) vs "Abstention is measured with a free prompt (the chat template alone)" (l.65)
- Problem: "Plain" and "free" name the same prompt and are never equated. "Permissive prompt" appears at l.42 and l.46 but its wording comes only at l.126.
- Fix: Use "free prompt" everywhere (or "plain" everywhere). In Setup, add one sentence defining the permissive prompt with its exact suffix.

**F9. minor. Section 3.2's "four-family test" paragraph relies on terms defined only in an appendix table**
- Location: Section 3.2 l.122-124; Table 3 caption.
- Evidence: "found both of its conditions in Gemma and one in Qwen and Mistral, but that readout shows no preference in Llama either"
- Problem: "Four-family test", its "conditions" and the "hedge readout" are defined only in the Table 3 caption. The sentence also undercuts itself (the readout fails on the reference model), so I could not tell why it is in the main text.
- Fix: Either delete the sentence from the main text and leave it in Appendix B, or state the two conditions inline and say what the reader should conclude ("this readout is not a valid proxy for abstention, so we do not use it to compare families").

**F10. minor. "Logit-lens", "Jacobian lens" and "hedge-token term" are used without definition**
- Location: Abstract l.14; Section 3.3 l.147-152; Appendix D l.350-353.
- Evidence: "With prefitted Jacobian lenses [Gurnee et al., 2026], read at the forced-answer position ... popularity raises the hedge-token term"
- Problem: The abstract says "logit-lens" and the body says "Jacobian lens". Neither is explained, and "hedge-token term" is defined only in Appendix D. An outside reader cannot follow the Lens paragraph.
- Fix: Use one name. Add one clause defining it ("a learned map from a middle layer's activations to next-token log-probabilities") and one clause defining the hedge-token term ("log-sum-exp probability of five refusal-opening tokens").

**F11. minor. Counts of robustness checks disagree across abstract, contributions, text and table**
- Location: Abstract l.8-10 (three checks); Contribution 1 l.45 ("four label and measurement checks"); l.85-86 ("five other measures"); Table 2 (six non-primary abstention measures).
- Evidence: "Its sign holds under five other measures of abstention, from +0.13 to +0.44 (Table 2)."
- Problem: Table 2 lists six non-primary measures of abstention (+0.31, +0.44, +0.43, +0.43, +0.44, +0.13). The reader cannot match the counts to the table.
- Fix: Make l.86 say "six" (or say which row is excluded and why). Name the four checks in contribution 1.

**F12. minor. The abstract's headline figure is the wrong unit and slightly misstates the known criterion**
- Location: Abstract l.6-7; Section 3.1 l.81-82; Table 1.
- Evidence: "Llama says 'I don't know' to 43% of questions it answers correctly in at least 9 of 10 forced samples"
- Problem: Table 1 and l.82 give 0.43 as a share of *replies* (three per question), not of questions. The known criterion also requires a correct greedy answer (l.57).
- Fix: "in 43% of replies to questions it answers correctly greedily and in at least 9 of 10 samples".

**F13. minor. The Figure 1 caption presents a by-construction result as a finding**
- Location: Figure 1 caption l.144-145; compare Abstract l.13 and Setup l.63-64.
- Evidence: "In Llama, saying 'I don't know' tracks popularity, while answer uncertainty follows knowledge."
- Problem: The knowledge labels are derived from the same forced-answer sampling distribution whose first-token entropy is plotted, so panel (b) is expected to show a knowledge effect. The informative part of panel (b) is that obscure-known and popular-known entropies are similar (0.95 vs 0.86). The caption does not say this, and a skim reader sees two independent dissociated signals.
- Fix: Retitle the caption to say what (b) adds: "...while its answer entropy on known facts is the same for obscure and popular entities." Add "(entropy follows knowledge by construction; it serves as a control)".

**F14. minor. The Gemma permissive-prompt result (contribution 2) has no main-text display**
- Location: Section 3.2 l.125-130; Appendix B l.310-315.
- Evidence: "Gemma says 'I don't know' to 0.41 of obscure known questions and 0.00 of popular known ones."
- Problem: The four cell rates for both models appear only in appendix prose. The reader cannot see the 2x2 that a contribution rests on.
- Fix: Add Gemma-permissive and Llama-permissive rows to Table 1, or a small panel beside Figure 1a.

**F15. minor. The permissive prompt was run on two models only, and the paper does not say why**
- Location: Section 3.2 l.125-127; Limitations l.158.
- Evidence: absent: expected a reason why Qwen and Mistral were not run with the permissive prompt; looked in Section 3.2, Appendix B l.310-315, Limitations.
- Problem: The Limitations say "whether other families share the preference is open", yet the experiment that would test it appears to have been run on only two of the four models.
- Fix: Run Qwen and Mistral under the permissive prompt, or state the reason they were omitted.

**F16. minor. The judge-agreement kappas for the permissive prompt are unlabelled**
- Location: Section 3.2 l.127.
- Evidence: "three judges agree (κ = 0.71, 0.67)"
- Problem: The text does not say which kappa belongs to which model, and κ is never named (Cohen's or Fleiss').
- Fix: "Fleiss κ = 0.71 (Llama) and 0.67 (Gemma)". At first use (l.69) write "Cohen's κ".

**F17. minor. Llama's agreement bar is passed by the point estimate but not by the interval**
- Location: Setup l.69-70.
- Evidence: "On Llama the judges agree at κ = 0.62 [0.57, 0.67], above a pre-specified bar of 0.60"
- Problem: The interval includes values below the bar, so a reader wonders whether the bar applies to the point estimate or to the interval.
- Fix: State "(the bar applies to the point estimate)" or report the bar as passed by the point estimate only.

**F18. minor. "Llama" and "Gemma" each name several different models**
- Location: Throughout; e.g., l.93, l.96, l.98 (Llama-3.3-70B judge), l.149 (Llama-3.1-8B base), l.67 (Gemma-2-27B judge), l.149 (Gemma-2-9B and 2-2B base).
- Evidence: "With Llama-3.3-70B as a third judge and a majority rule, κ = 0.74"
- Problem: "Llama" is the subject model, a judge and a base model; "Gemma" is a subject, a judge and two base models. When I read "Llama" I had to keep checking which one was meant.
- Fix: Define short names once in Setup ("Llama" = Llama-3.1-8B-Instruct; "70B judge"; "Gemma judge") and use the full name for every base model.

**F19. minor. The Lens section introduces new models without warning**
- Location: Section 3.3 l.149-150; Setup l.58-61.
- Evidence: "Llama-3.1-8B, Gemma-2-9B, Gemma-2-2B and OLMo-3-7B [Olmo et al., 2025]"
- Problem: Setup lists four instruct models. Base models, Gemma-2-2B and OLMo-3-7B appear without explanation, and Appendix D adds Qwen2.5-7B-Instruct onsets but drops Mistral.
- Fix: Add one sentence to Setup listing the base models used for the lens and why these (availability of prefitted lenses, presumably).

**F20. minor. Figures 2 and 3 are cited as main evidence but sit in the appendix**
- Location: Section 3.3 l.139, l.145, l.150; Figure 2 (Appendix C, p.9); Figure 3 (Appendix D, p.10).
- Evidence: "and Gemma's from 0.00 to 0.04 at 4× the norm (Figure 2a)"
- Problem: Section 3.3 is a dense run of numbers, and the figure that organises them is five pages away. A four-page reader cannot evaluate contribution 3.
- Fix: Mark the citations "(Appendix C, Figure 2a)" so readers know where to look. If space allows, replace Figure 1c (which repeats the entropy control across models) with Figure 2a.

**F21. minor. Two different "0.10 bars" in the steering analysis**
- Location: Section 3.3 l.138; Appendix C l.325-327.
- Evidence: "k was chosen ... as the smallest value at which that direction beat five random directions by 0.10 in regex abstention, a rule separate from the confirmatory bar"
- Problem: A dose-selection threshold of 0.10 (regex) and a confirmatory bar of +0.10 (two judges) are both called bars. The reader has to work out which one "+0.09 ... short of the pre-specified +0.10 bar" refers to.
- Fix: Name them ("dose-selection threshold", "confirmatory bar") and use the names in l.138.

**F22. minor. The comparison in "+0.66 nats more" has no stated referent, and the sentence's purpose is unclear**
- Location: Section 3.3 l.145-146.
- Evidence: "In Gemma, unknown-minus-known raises entropy +0.66 [+0.58, +0.74] nats more (Figure 2b; cf. Che and Li, 2026)."
- Problem: "More" than which direction? (From Figure 2b, it is obscure-minus-popular.) I could not tell why the result matters or what the Che and Li citation contributes.
- Fix: "...raises forced-answer entropy 0.66 nats more than obscure-minus-popular does, so the knowledge direction moves answer uncertainty while the popularity direction does not."

**F23. minor. The Figure 3 caption and l.150 say "about as much", which Gemma-2-2B contradicts**
- Location: Section 3.3 l.150; Figure 3 caption; Table 5.
- Evidence: "Knowledge raises it about as much (Appendix D, Figure 3)."
- Problem: For Gemma-2-2B the hedge-term effects are +0.10 (popularity) vs +0.37 (knowledge), almost a fourfold gap. The caption's "about as large or larger" is accurate; the main text's "about as much" is not.
- Fix: l.150: "Knowledge raises it at least as much in three of four models."

**F24. minor. Table 3 prints an estimate that the text says cannot be estimated**
- Location: Section 3.2 l.121-122; Table 3 row "Three judges: popularity - knowledge".
- Evidence: "abstains in at most 0.11 of replies in any cell, so its preference cannot be estimated"
- Problem: Table 3 gives Gemma +0.01 [-0.03, +0.06]. A reader sees a number for something declared not estimable.
- Fix: Replace it with a dash and a footnote ("below the pre-specified abstention floor of 0.25"), or say in the text that the estimate exists but falls below the floor.

**F25. minor. "Pre-specified" is applied inconsistently**
- Location: l.77-78, l.86-87; Table 2 (primary row "pre"); Appendix A l.253-254; Appendix E l.370-373.
- Evidence: "We designed the primary measure after the smallest of these was seen."
- Problem: The primary measure is labelled "pre" although it was designed after a related estimate (+0.13) had been seen. Appendix E, by contrast, labels an analysis post hoc because its point estimates had been seen. Readers will take "pre-specified" in the usual sense (fixed before any outcome data), and the two cases are treated differently.
- Fix: Define the term once in Section 2 as "plan hashed before this analysis ran; related results may have been seen". Mark the primary measure "pre (after +0.13 seen)" in Table 2.

**F26. minor. The Appendix G footnote raises a provenance question and leaves it open**
- Location: Appendix G, footnote a.
- Evidence: "its internal date is one day later than its recorded hash time."
- Problem: A reader takes this to mean the amendment (which adds the third judge, used for every three-judge number in the paper) may have been edited after its hash was recorded. No explanation follows.
- Fix: Explain the discrepancy in one sentence (e.g., time-zone or a re-saved file with unchanged content, verifiable from the supplied hash), or mark the third-judge scorer as post hoc.

**F27. minor. Table 2 has gaps and cryptic entries**
- Location: Table 2.
- Evidence: "Regex detector 20, 24 tokens 2,303 pre +0.44"
- Problem: "20, 24 tokens" and "3, 96 tokens" must be decoded as replies x token budget. n is blank for four knowledge-label rows (e.g., "where both labelings agree" should be 231). The first regression row has no interval. The "regex detector" is never described anywhere.
- Fix: Rename the column "Replies x max tokens" and fill in n and the missing interval. Add one sentence to Section 2 or Appendix A describing the regex (or listing its patterns).

**F28. minor. It is unclear which subset the +0.32 relabel estimate refers to**
- Location: Section 3.1 l.90-92; Table 2.
- Evidence: "Of the 87 obscure known facts, 27 stay known; these abstain at 0.46. The preference is +0.32 [+0.16, +0.49]"
- Problem: Placed right after the 27-fact subset, +0.32 reads as the agreement-only estimate. Table 2 shows it is the full relabel (n = 253) and that the agreement-only estimate is +0.31.
- Fix: "Under the new labels (253 questions) the preference is +0.32 [...]; where both labelings agree, +0.31."

**F29. minor. Sign and unit conventions change in the name-spelling paragraph**
- Location: Appendix A l.299-301.
- Evidence: "the popularity coefficient on the regex measure moves from -1.87 to -1.79 logits per standard deviation"
- Problem: Elsewhere the popularity effect is positive (obscure minus popular, an arm indicator). Here it is negative and "per standard deviation" of an unnamed variable (presumably log page views), on a logit scale.
- Fix: Name the regressor and the model (logistic, standardized log page views), and say that negative means more abstention for less-viewed entities.

**F30. minor. Appendix F's conclusion does not follow from what precedes it**
- Location: Appendix F l.379-383.
- Evidence: "which we have not resolved. Surprisal therefore does not stand in for popularity. Short names made of common words ... are a likely reason."
- Problem: "Therefore" follows an admitted unresolved puzzle. "A likely reason" has no clear referent: is it the reason surprisal fails to track page views, or the reason for the puzzle?
- Fix: Split the paragraph: (1) surprisal does not correlate with page views (Spearman -0.001), so it does not proxy popularity; (2) a likely reason is X; (3) separately, the regex coefficient behaves oddly when the arm is added, which we have not explained.

**F31. minor. The onset analysis is never mentioned in the main text**
- Location: Appendix D l.360-366; Table 7.
- Evidence: "The prediction was not supported (mixed): 1 supporting, 1 contradicting and 6 undecided"
- Problem: A pre-specified test of a timing prediction, with its own table, is never mentioned in the main text, so its role in the argument is unclear.
- Fix: Add one clause to the Lens paragraph (l.152): "and a pre-specified prediction that popularity acts at shallower layers than knowledge was not supported (Appendix D)." Otherwise drop Table 7.

**F32. minor. "Extreme-contrast design" and "balanced 2x2" are used loosely**
- Location: l.50 and l.75.
- Evidence: "in this extreme-contrast design, Llama's answer uncertainty alone predicts its abstention poorly"
- Problem: "Extreme-contrast design" is never defined (it presumably means top/bottom deciles crossed with the 9-of-10 / 0-of-10 knowledge criterion). "Balanced" is wrong for cell sizes 87/100/100/100; the identity holds because cells are equally weighted, not equally sized.
- Fix: "in this design, which contrasts only extreme deciles and clear-cut knowledge". "In this equally weighted 2x2".

**F33. minor. No conclusion section**
- Location: absent: expected a short Conclusion or Discussion; looked in Sections 1-4 and after Limitations.
- Evidence: absent: expected 2-4 sentences restating the result and its implication; looked after l.158.
- Problem: The paper ends on a five-line Limitations section. A skim reader gets no closing statement of the result, the failed familiarity account or the convergence answer.
- Fix: Add a three-sentence conclusion, for instance by moving the "Together, these findings mean..." sentence (l.50-51) there and expanding it.

**F34. minor. Figure 1 is hard to read at print size, and panel (c) has no legend**
- Location: Figure 1 (p.3 of the PDF).
- Evidence: absent: expected a marker legend in panel (c); looked in Figure 1c and its caption.
- Problem: At the printed width the tick and label text is very small. Panel (c) identifies popularity and knowledge only by coloured header words, not by marker shape. Its model order is not explained.
- Fix: Enlarge Figure 1 to full text width with a minimum 7 pt font. Add a marker legend to (c). Order the models by knowledge effect or by name.

**F35. minor. Voice: the main text is edited prose, but the appendices read as a lab audit log**
- Location: Appendix A l.251-254; Appendix B l.314-315; Appendix C l.320, l.333-334; Appendix G table and footnote; DONT_KNOW/CLARIFY codes in l.68-69 and Table 2.
- Evidence: "abstain minus answer (questions whose stored free-generation regex abstention rate is at least 0.5 minus those with rate 0)"
- Problem: The main text has a single consistent, careful voice, though it is compressed to the point of being telegraphic in Section 3.3. The appendices carry pipeline provenance: plan filenames with experiment codes (e1-e10, "v2", "v3", "jlens_cascade"), "stored", "seed 0", "made primary before it ran", "added by amendment before judging", "the criterion let Gemma's verdict replace 'cannot be estimated'". This narrates decisions rather than reporting results. It is honest, but it makes Appendices A-C slower to read than necessary.
- Fix: Move the decision history into Appendix G and keep Appendices A-D as methods and results. Write judge labels in plain words ("don't know", "clarify", "answer") in prose. Drop experiment codes from the Appendix G table and keep the descriptive analysis names and hashes.

## Strengths

1. The title and abstract (l.1-15) state a scoped, checkable claim with concrete numbers and say plainly what the paper does not show ("what abstention tracks, not why").
2. The metric is interpretable. The identity "preference = obscure-known minus popular-unknown abstention" (l.75-76) lets a reader verify the headline from Table 1, and every main-text Llama number I checked (+0.39, +0.09, +0.30, +2.14, +0.51, steering differences, permissive +0.47/+0.30, Table 3 hedge differences, Table 6 differences) reproduces from the tables.
3. Table 2 gathers every alternative measure with a pre/post hoc label, and l.86-87 openly discloses that the primary measure was designed after the smallest estimate was seen.
4. The claim-style run-in headings in Section 3.1 (l.89, l.100, l.109) let a reader skim the argument.
5. The negative and mixed results are reported rather than hidden: familiarity not shown (l.109-112), steering residual near zero (l.141-143), surprisal null (Appendix F), onset prediction unsupported (Appendix D).

## Questions for the authors

1. In l.292-293, which reply set is meant, and what share of the two-judge "I don't know" replies in the obscure-known cell contain a gold alias? If it is well above 0.03, the "refusals, not hedged answers" claim needs revising (F3).
2. Do you hold that obscure entities are less familiar to Llama? If not, will you reframe the abstract and introduction around popularity (F1)?
3. What unit is "+8.4 vs +3.1", what are the intervals, and is that analysis post hoc (F2)?
4. With the three-judge scorer you call defensible for Gemma, does the +0.05 [+0.03, +0.07] residual change contribution 3 for Gemma (F5)?
5. What, in one sentence, is your answer to the convergence question in l.40-42 (F6)?

## Recommendation

**Major revision.** For a workshop that means accept once the issues below are fixed; no new experiments are needed beyond reconciling one number.

The findings that drove this:
- **F1**: the abstract and introduction frame the result as familiarity, and Section 3.1 says that is not shown.
- **F3 and F4**: the main text's case that the "I don't know" replies are clean refusals and that the knowledge labels are sound depends on a number that does not reconcile with Appendix A. The main text also omits the weakest label variant and the 48% figure behind it.
- **F2 and F6**: the within-prompt paragraph and the convergence question, both central to how a UniReps reader would place the work, cannot be followed or are left unanswered.

The core Llama result is clearly presented, internally consistent, and reproduces from the tables; Sections 1, 2 and the first half of 3.1 read well. What would change my mind: reframe the abstract and introduction around popularity (F1); reconcile the gold-alias figures (F3); report the +0.19 variant in the main text (F4); rewrite l.113-117 with units, intervals and the post hoc label (F2); answer the convergence question in two sentences or drop it (F6). With those fixes I would recommend accept.
