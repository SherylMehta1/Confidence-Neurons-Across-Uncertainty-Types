# Referee report: framing and novelty

Manuscript: "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge" (UniReps 2026, extended-abstract track)

## Two-sentence reading (after abstract, figures, conclusion)

**What it claims:** When an LLM says "I don't know", it is reporting familiarity with the question's subject, not uncertainty about the answer. Answer uncertainty is a separate variable that tracks knowledge, and the familiarity variable is causal for abstention, already present in base models, and analogous to the human cue-familiarity heuristic.

**What it shows:** In one model (Llama-3.1-8B-Instruct, 387 questions, two LLM judges never checked against human labels, kappa = 0.62), the abstention rate depends more on PopQA popularity than on a sampling-based knowledge label. First-token entropy under a forced-answer prompt depends mostly on that knowledge label in four families. A mean-difference "obscure-minus-popular" direction can induce abstention in two models, but its familiarity-specific part adds only +0.04 over random. A lens readout in four base models responds to both popularity and knowledge at mid-depth.

## Closest prior work (verified)

1. **Ferrando, Obeso, Rajamanoharan, Nanda, "Do I Know This Entity?" (ICLR 2025; arXiv 2411.14257).** Opened. Known and unknown entities are defined by model correctness ("at least two attributes correct" vs "all wrong"), so the manuscript is right that familiarity and knowledge are confounded there. The paper also states that the directions "are causally relevant: capable of steering the model to refuse to answer questions about known entities", and that "despite the sparse autoencoders being trained on the base model, these directions have a causal effect on the chat model's refusal behavior, suggesting that chat finetuning has repurposed this existing mechanism". It covers Gemma-2 2B/9B base and chat, and Llama-3.1-8B.
2. **Lindsey et al., "On the Biology of a Large Language Model" (2025).** Opened. It describes a default "can't answer" circuit that is suppressed by "known entity/answer" features. It also proposes explicitly that a natural hallucination (the Andrej Karpathy paper example) happens when the model "activates 'known answer / entity' features (on account of recognizing Andrej Karpathy's name) despite not knowing the answer". That is the familiarity-without-knowledge hypothesis, stated as a mechanism.
3. **Brzezinka, "Graded Entity-Familiarity Readouts..." (arXiv 2607.13568, July 2026).** Opened (abstract). A pageview-graded familiarity probe, plus a familiarity direction in Gemma-4-12B that moves refusal "0.24 to 1.00 on well-known entities". It has no knowledge axis, so the manuscript's description of it is accurate. Steering refusal on well-known entities is nonetheless prior work.
4. **Ji et al., "Calibrating Verbal Uncertainty as a Linear Feature..." (EMNLP 2025; arXiv 2503.14477).** Abstract verified. Verbal uncertainty is a single linear feature with "only moderate correlation" to semantic uncertainty. This already establishes that verbal and answer-level uncertainty are different variables.
5. **Kumaran, Daw, Osindero, Velickovic, Patraucean, "Causal Evidence that Language Models use Confidence to Drive Behavior" (arXiv 2603.22161, 2026).** Opened. Models "apply an implicit threshold to internal confidence when deciding to abstain". The confidence effect is about ten times larger than competing predictors, and steering confidence changes abstention. The setting differs from the manuscript's (SimpleQA as 4-way multiple choice, instructed or invited abstention, GPT-4o/Gemma-3-27B/DeepSeek/Qwen-80B). It is the closest contrary result, and the manuscript does not cite it.
6. **Ni, Bi, Guo, Cheng, "Popular but Wrong" (arXiv 2505.17537; arXiv comment says EMNLP 2026).** Opened. Probability-based confidence is tied to popularity, especially among incorrect answers. This is relevant to the claim that answer uncertainty "tracks knowledge".

Other citations checked and found to exist with the stated content: Nguyen et al. 2026 (arXiv 2609.32964, commit-abstain circuit, ten LMs, no familiarity axis); Che and Li 2026 (arXiv 2610.07270; it notes the "published unknown-entity latent strongly steers knowledge abstention, yet installing observed values" transfers only a small part of the contrast); Gurnee et al. 2026 (arXiv 2607.15495, Jacobian lens; I could not verify which prefitted lenses are released [unverified]); Lippl et al. 2024 (arXiv 2404.08101, source invariance).

**Novelty verdict.** One thing in the paper is new: the factorial crossing of popularity with a knowledge label from the model's own samples, applied to abstention behaviour. Neither Ferrando et al. (knowledge defined by correctness) nor Brzezinka (no knowledge axis) did this, and Lindsey et al. only hypothesise it from a single case. Contributions (ii) and (iii), steering abstention on known entities and a base-model origin, largely replicate Ferrando et al. and Brzezinka. The manuscript does not say so.

## Findings

**F1. Major. Title and abstract generalise a one-model result.**
- Location: title; abstract l.11-13; l.133.
- Evidence: title "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge" vs l.128 "Abstention is measured reliably in one model; the other three families rarely say 'I don't know'".
- Problem: The title and the closing sentences ("Abstention and answer uncertainty are different variables"; l.133 "abstention follows familiarity") state a general law about language models. The abstention half rests on one 8B model, 387 questions and judges with kappa = 0.62 that were never validated against humans. The generalisation in the abstract ("holds in four model families") applies only to the entropy half, but the sentence reads as if it covered both.
- Fix: Scope the title to the model, e.g. "In Llama-3.1-8B, saying 'I don't know' tracks familiarity...", or phrase it as a question. In the abstract and conclusion, state that the abstention result is single-model.

**F2. Major. Lindsey et al. are mischaracterised, which overstates the novelty of the hypothesis.**
- Location: l.22-25.
- Evidence: "these studies call an entity known when the model answers questions about it correctly, so familiarity and knowledge move together"
- Problem: This holds for Ferrando et al. (verified). It does not hold for Lindsey et al., who explicitly propose that recognising a familiar name ("Andrej Karpathy") switches on "known entity" features "despite not knowing the answer", causing a hallucination. The familiarity-not-knowledge hypothesis is therefore already in the literature. The manuscript's contribution is a quantitative, factorial test of it, which is still valuable.
- Fix: Credit Lindsey et al. for the misfire hypothesis. Present the paper as the first factorial behavioural test of that hypothesis, crossing popularity with model-specific knowledge.

**F3. Major. Contributions (ii) and (iii) are not positioned against Ferrando et al. and Brzezinka, who already did them.**
- Location: l.33-37 (contributions ii, iii); l.116-118 (related work).
- Evidence: "Entity-recognition features gate refusal in Gemma-2 and Claude 3.5 Haiku [Ferrando et al., 2025, Lindsey et al., 2025]."
- Problem: Ferrando et al. steer chat models into refusing about known entities and show that the directions come from the base model ("chat finetuning has repurposed this existing mechanism"). Brzezinka steers refusal on well-known entities from 0.24 to 1.00. The manuscript presents "makes Llama and Gemma-2-9B say 'I don't know' about popular facts they know" and "predates instruction tuning" as findings without saying that both are replications. The genuinely new parts of 3.2 (the comparison against an unknown-minus-known direction and the never-abstained control) are not the parts being claimed.
- Fix: In the intro, recast (ii) and (iii) as replications with a twist, e.g. "consistent with Ferrando et al., ...; new here is that the familiarity-specific component contributes only +0.04". In related work, state what Ferrando et al. and Brzezinka already showed about steering and base models.

**F4. Major. The closest contrary evidence is not cited.**
- Location: §1, §4.
- Evidence: absent. Expected a discussion of Kumaran et al. 2026 (arXiv 2603.22161), which finds abstention is driven by internal confidence with an effect "approximately ten times larger" than competing factors. Looked in §1, §4, §5 and the reference list.
- Problem: The headline (abstention follows something other than answer confidence) conflicts directly with a recent causal study. Without a reconciliation, the reader cannot tell whether the dissociation is general or specific to spontaneous, uninstructed abstention in a small model on free-form PopQA. Plausible reconciling factors include instructed vs spontaneous abstention, multiple-choice vs open answers, and model scale.
- Fix: Cite it and say in one or two sentences which design difference you think explains the disagreement. That turns the conflict into a sharper claim: "uninstructed abstention follows familiarity".

**F5. Major. An alternative reading of the key obscure-known cell is not ruled out.**
- Location: §2 l.42-44; §3.1 l.70-72; Appendix A l.417-418.
- Evidence: "Some obscure names also contain their answer (the capital of a district named after its town)."
- Problem: The headline rests on the obscure-known cell (0.43 abstention). "Known" means the greedy answer and at least 90% of samples are correct under a forced-answer prompt. Several of the 16 PopQA relations (country, occupation, genre, sport, religion, colour, capital) can be answered from relation-level priors or from the name itself. A model that abstains on such items may be correctly reporting that it has no stored entity-specific fact, and then guessing well when forced. On that reading abstention would track knowledge in the stored-fact sense, not a familiarity heuristic. The low entropy of this cell (Fig. 1b, about 0.95 nats) is what a confident prior-based guess would produce. Relation fixed effects (l.80) do not address guessability within a relation, and the Appendix A exclusion covers only one form of it.
- Fix: Report accuracy on obscure-known items when the entity name is replaced by a fabricated name with the same relation template, as a guessing baseline. Re-estimate the popularity and knowledge effects after removing items the baseline answers correctly, or report them by relation. Add this threat to the limitations.

**F6. Major. The lens result does not isolate familiarity, so "predates instruction tuning" is more than it shows.**
- Location: l.36-37, l.103-114, Fig. 3; abstract l.10-11.
- Evidence: "so it predates instruction tuning" (l.36-37) vs "the two signals emerge together, at relative depth 0.45-0.74" (l.114).
- Problem: In the same pre-set window, the base-model hedge readout also carries a knowledge effect of comparable size (Fig. 3b: Llama about +0.3 to +0.7, Gemma-2-9B rising to about +1). The base readout therefore mixes both variables and is not the familiarity-dominated abstention variable seen in the instruct model. OLMo's familiarity curve dips below zero inside the window and is clearly negative after about 0.7 (Fig. 3a), and a window mean hides this. Nothing links this base-model readout to the instruct model's behavioural abstention: the instruct-vs-base comparison is post hoc (App. D) and not shown.
- Fix: Retitle §3.3 and abstract l.10-11 to something like "base models' hedge readout already responds to popularity (and to knowledge)". Report the familiarity-to-knowledge ratio per base model, and show or drop the instruct-vs-base comparison.

**F7. Major. Fit to the venue is thin, and the one cross-model prediction is untested.**
- Location: l.124-126; l.134-136.
- Evidence: "then two models' abstentions should agree on obscure names even where only one of them knows the fact."
- Problem: UniReps asks "why, when and how distinct learning processes yield similar representations". Its mechanistic-interpretability topic is "whether circuits, features, and computations can be related across models". The paper runs per-model analyses side by side but never relates representations across models. The single sentence linking it to Huh et al. (l.124-126) is a non sequitur: convergence of representations says nothing about whether abstention depends on familiarity. The data already in hand seem enough to test the closing prediction: four families on overlapping PopQA questions, a forced-prompt hedge readout in all four (l.52-54), and lens readouts in four base models.
- Fix: Add one cross-model analysis. Options: per-item agreement of the familiarity readout across models, conditioned on items where models disagree on knowledge; or cosine and CKA alignment of the obscure-minus-popular directions after a learned map. Then rewrite the related-work sentence around that result.

**F8. Minor. The "different variables" conclusion is framed against a straw account that the cited literature already rejects.**
- Location: l.18-19; abstract l.11-12; l.122-124.
- Evidence: "The implicit account is that abstention is uncertainty made audible."
- Problem: Ji et al. (cited) already show verbal uncertainty is only moderately correlated with semantic uncertainty, and Ferrando et al. and Lindsey et al. (cited) already tie refusal to entity recognition. That abstention and answer uncertainty are different variables is therefore not the new finding. The new finding is what each variable follows, as l.123 itself says.
- Fix: Move "we add what each variable follows" into the abstract and intro as the stated contribution. Drop or soften the "implicit account" sentence.

**F9. Minor. "Answer uncertainty tracks knowledge" understates its popularity component.**
- Location: title; l.76-78; Fig. 1b.
- Evidence: "entropy rises by +2.14 nats when Llama lacks the fact and by +0.51 when the entity is obscure."
- Problem: Among unknown items, entropy in Fig. 1b falls by about 0.9 nats from obscure to popular, so answer uncertainty also tracks popularity, as Ni et al. (arXiv 2505.17537) report for probability-based confidence. The title presents a difference in weighting as a clean dichotomy.
- Fix: Phrase the contrast as relative weights ("mainly", "4x"). Cite Ni et al. and note the interaction in Fig. 1b.

**F10. Minor. The steering mechanism is generalised beyond the model where the control was run.**
- Location: l.33-35; abstract l.8-10.
- Evidence: "makes Llama and Gemma-2-9B say 'I don't know' about popular facts they know, largely through components shared with abstention"
- Problem: The never-abstained control (l.99-101) was run only on Llama. Nothing in §3.2 supports "largely through components shared with abstention" for Gemma.
- Fix: Run the control on Gemma, or restrict the clause to Llama.

**F11. Minor. "Double dissociation" is claimed from a 4x-norm intervention.**
- Location: l.96-97.
- Evidence: "There, the unknown-minus-known direction instead raised answer entropy more (difference +0.68 [+0.59, +0.77]), a double dissociation"
- Problem: In Gemma the effect needs four times the natural norm, and Gemma barely abstains naturally (Table 2: 0.05). The paper itself cites Che and Li on out-of-range steering. Calling the result a double dissociation of the model's variables overstates it. It is a dissociation between two steering vectors.
- Fix: Call it "a dissociation of steering effects at 4x norm" and add a dose-response curve if space allows.

**F12. Minor. Lippl et al. are misapplied.**
- Location: l.124-126.
- Evidence: "a strict uncertainty representation should not depend on its source [Lippl et al., 2024], yet the abstention variable depends on one, familiarity."
- Problem: Lippl et al. define source invariance as encoding uncertainty "regardless of the source of the uncertainty". Your result (abstention responding to familiarity when there is no answer uncertainty) fits their "representational indeterminacy" better: a variable merely related to uncertainty. "Depends on its source" does not follow from their definition.
- Fix: Reword to "by Lippl et al.'s criterion, the abstention variable is a correlate of uncertainty rather than a representation of it". Or drop the sentence.

**F13. Minor. The main text "rules out" what the appendix says explains part of the effect.**
- Location: l.79 vs l.415-416.
- Evidence: "A post hoc check rules out name spelling as the main driver." vs "the abstention gap of +0.76 falls to +0.43 [+0.18, +0.69] at equal surface form"
- Problem: The matched-pair analysis attributes about 40% of the gap to surface form. "Rules out" is too strong for a post hoc regression.
- Fix: "Name surface form explains part, but not most, of the popularity effect", citing both numbers.

**F14. Minor. The robustness count is inconsistent, and one row is unexplained.**
- Location: l.74-75; Table 1; l.131.
- Evidence: "The difference keeps its sign under three other measurements"
- Problem: Table 1 has five rows (four besides the primary), its caption says "four measurements", and l.131 says "four measurements agree in sign". The "One LLM judge, corrected" row is not defined and shrinks the effect to +0.13, a third of the primary estimate, with no comment. Its replies are 24 tokens long, but its question count is not given.
- Fix: Make the counts agree, define "corrected", report the question count and an interval for every row, and comment on the attenuation.

**F15. Minor. Cell sizes are not reported.**
- Location: §2 l.44-46, l.56; Fig. 1.
- Evidence: "For up to 100 questions per cell (387 in total)"
- Problem: The headline cell (obscure-known) is likely the smallest, and its n is never given, either for the 387 abstention items or for the 2,303-item entropy pool.
- Fix: Add a 2x2 n table per model, in the appendix if needed.

**F16. Minor. The only abstention-side evidence beyond Llama is buried and anomalous.**
- Location: l.52-54.
- Evidence: "(Gemma: popularity +1.79, knowledge -0.32; in Llama both effects are large)"
- Problem: This forced-prompt hedge readout is the only evidence that the abstention-side pattern extends past Llama, yet it has no intervals. The Llama values are not given, Qwen and Mistral are not reported, and the negative knowledge effect (more hedging on known facts) goes unexplained.
- Fix: Add a small table with this readout for all four families, with CIs, and comment on the sign.

**F17. Minor. The preregistration claims cannot be checked.**
- Location: l.65-66; Appendix D.
- Evidence: "the note's SHA-256 hash was recorded before the run"
- Problem: The paper reports neither the hashes nor the criteria (other than the kappa bar), so a reader cannot verify what was confirmatory.
- Fix: List the hashes and the decision criteria, or link an anonymised timestamped repository.

**F18. Minor. Appendix C's timing result is not reconciled with "emerge together".**
- Location: l.421-423 vs l.114.
- Evidence: "reaches half its final size at layer 16, while the entropy gap does so at layer 28, 12 layers later"
- Problem: Read naively, this supports the "familiarity first" prediction that §3.3 reports as failed. The two analyses use different contrasts (matched pairs, abstention readout vs entropy), but the text does not say so.
- Fix: Add one sentence explaining why the two results differ and which one bears on the prediction.

## Strengths

1. **Clean factorial design (§2, l.26-30, 39-44).** Crossing pageview deciles with model-specific, sampling-based knowledge labels is a real advance over correctness-defined "known entities" (Ferrando et al.) and popularity-only probes (Brzezinka).
2. **Honest reporting of negatives (l.110-114; App. C l.427-430; App. D).** The failed depth-ordering prediction and the circuit that missed its pre-set bar are reported, and post hoc analyses are labelled as such.
3. **The never-abstained control (l.99-101).** It is the most informative result in §3.2 and is interpreted cautiously ("This shows control, not that the model uses familiarity as such"), consistent with Che and Li.
4. **Numbers agree with the figures.** The cell means in Fig. 1a (0.48/0.43/0.13/0.00) reproduce the +0.39 and +0.09 effects, Fig. 1b reproduces +2.14 and +0.51, and the steering values in §3.2 match Fig. 2.
5. **Contributions carry section pointers (l.31-37), and the two-prompt confound is stated openly (l.47-55).**

## Questions for the authors

1. When the entity name in obscure-known items is replaced by a fabricated name of the same relation, how often is the forced answer still correct? If often, the headline cell is mostly guessable facts, and my reading of F5 stands.
2. What are the 2x2 cell sizes for the 387-question abstention set and the 2,303-question pool?
3. With your existing data, do per-item familiarity readouts (hedge log-odds or lens) agree across models on the same PopQA items, and does that agreement survive conditioning on the models disagreeing about knowledge? This would settle F7.
4. How do you reconcile your Llama result with Kumaran et al. (2026), who find that confidence dominates abstention?
5. Why is Gemma's forced-prompt knowledge effect on hedge log-odds negative (-0.32)?

## Recommendation: major revision

The findings that drove this recommendation are F1, F3 (together with F2) and F5.

The crossed design is a good idea and the paper is unusually candid about what failed. As an early-stage UniReps abstract its core observation is worth showing. As written, though:
- The title and abstract state a general law from one model (F1).
- Two of the three listed contributions largely replicate Ferrando et al. and Brzezinka without saying so, and the familiarity-without-knowledge hypothesis is attributed away from Lindsey et al. (F2, F3).
- The key obscure-known cell has an uncontrolled guessability explanation (F5).
- The cross-model angle the venue asks for is only gestured at (F7).

What would change my mind to accept:
- a scoped title and abstract;
- related work that credits Ferrando et al. (steering and base-model origin), Lindsey et al. (misfire hypothesis) and Kumaran et al. (contrary evidence);
- a guessing-baseline analysis showing the popularity effect survives on non-guessable known facts;
- ideally, one cross-model agreement analysis that tests the paper's own closing prediction.
