# Literature Review: Uncertainty, Hedging, and Confidence Mechanisms in LLMs

*Current as of 30 August 2026. Covers every entry in `paper/references.bib` (23 works) plus three recommended-but-uncited items surfaced by the final novelty sweep. Each entry gives the citation, what the work shows, and how our paper relates to it. Bib keys are in brackets for the team.*

*Where we stand: no prior or concurrent work combines a model-certified uncertainty dataset, a hedging circuit verified at head-plus-neuron granularity against size-matched random sets, a decision-versus-spread dissociation, and a layer-resolved account of when the hedge becomes a word. The neighborhood is crowded in 2026, especially around two-channel dissociations, so positioning has to be explicit.*

---

## 1. The single-component account of confidence

This is the literature our paper deflates. Its shared move is to select individual neurons by correlation with output entropy and interpret them as confidence machinery.

**Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, Nanda (NeurIPS 2024). Confidence Regulation Neurons in Language Models.** [stolfo2024confidence]
Identifies two neuron families in the final layers: entropy neurons (high weight norm, near-zero direct logit effect, acting through the final normalization by writing into the unembedding null space, so they scale the output distribution without reordering tokens) and token-frequency neurons (which push logits toward or away from the unigram distribution). Validated by targeted ablation on the prompts where the correlation is strongest.
*Relation:* the direct object of our claims 1 and 2. We reproduce the taxonomy causally (temperature-matched frequency test, frozen-RMSNorm decomposition) and show these units are the spread's plumbing: no single one is uncertainty-specific under an identified null, and no set of them moves hedging behavior. We do not dispute their local effect on entropy; we dispute the reading that they decide anything.

**Gurnee, Horsley, Guo, Kheirkhah, Sun, Hathaway, Nanda, Bertsimas (TMLR 2024). Universal Neurons in GPT-2 Language Models.** [gurnee2024universal]
Finds neurons that recur across independently trained GPT-2 seeds and sorts them into functional families, including prediction, suppression, and entropy-modulating neurons.
*Relation:* establishes that entropy neurons are a stable, cross-seed phenomenon, which is exactly why they are tempting to over-interpret. Cited as part of the single-component tradition.

**Du, Li, Cai, Saraipour, Zhang, Lakkaraju, Sun, Zhang (COLM 2025). How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence.** [du2025posttraining] arXiv:2504.02904
A mechanistic comparison of base and post-trained models, reporting that post-training leaves knowledge storage largely intact while altering truthfulness, refusal, and confidence behavior through a comparatively small set of components.
*Relation:* our temperature-neuron finding (L31_N2477, monotonic dose-response, RMSNorm-mediated) is consistent with their picture of confidence being adjusted through distribution-scaling machinery. Cited as supporting context in Results.

---

## 2. Circuit-level accounts of uncertainty and confidence

The closest prior work. These papers move from single units to circuits; our paper is positioned relative to them by rigor (nulls, held-out faithfulness, granularity) and by the decision-versus-spread split.

**Teplica, Liu, Cohan, Rudner (NAACL 2025 Long). SCIURus: Shared Circuits for Interpretable Uncertainty Representations in Language Models.** [teplica2025sciurus]
Across eight models, uses causal tracing and zero-ablation to show that the components carrying uncertainty overlap heavily with those carrying answer factuality: uncertainty machinery is not a separable module.
*Relation:* the closest prior work and cited first in Related Work. We inherit its non-modularity lesson (our cross-arm overlap is Jaccard 0.11 with head-importance correlation 0.61: a shared variable, not a shared circuit). What we add: the gated-twin data standard, twin-based activation patching (reference-free counterfactuals instead of zero-ablation), faithfulness against size-matched random sets on held-out pairs, single-neuron granularity inside the circuit, and the decision-versus-spread dissociation. We claim no priority on "an uncertainty circuit exists."

**Zhao, He, Zheng, Zhang, Chen (COLM 2026). Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs.** [zhao2026wired] arXiv:2604.01457 (v3, July 2026, camera-ready)
Localizes "confidence-mover" components responsible for inflated numeric verbalized confidence in two models, reports faithfulness above 85%, and demonstrates a practical payoff: large calibration-error reductions from ablating the identified components.
*Relation:* the nearest neighbor on the affirmative side. Differences: their target is a verbalized confidence score, ours is the hedge decision; they report no random-set baselines; they deliver a deployment payoff we deliberately do not attempt. Their calibration result is the kind of practical significance a main-track reviewer will ask us for (see section 8).

**Singha Roy, Jhaveri, Triantafyllopoulos (2025). Interpreting and Mitigating Unwanted Uncertainty in LLMs.** [roy2025unwanted] arXiv:2510.22866
Identifies attention heads that drive answer-flipping under uncertainty and shows that masking them mitigates the behavior.
*Relation:* head-level uncertainty mechanism in the same family as our verified heads, but selected for a different behavior (flipping, not hedging) and without a faithfulness protocol.

**Vazhentsev, Rvanova, Kuzmin, Fadeeva, Lazichny, Panchenko, Panov, Sachan, Nakov, Baldwin, Shelmanov (ICML 2026). Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads.** [vazhentsev2026heads] arXiv:2505.20045
Uses attention-head patterns as cheap features for hallucination detection.
*Relation:* evidence that head-level signals carry uncertainty information usable downstream; a detection application rather than a mechanistic account.

**Basu, Morariu, Wang, Rossi, Zhao, Feizi, Manjunatha (2025). On Mechanistic Circuits for Extractive Question-Answering.** [basu2025extractiveqa] arXiv:2502.08059
Builds circuits that distinguish answering from context versus from parametric memory.
*Relation:* structurally adjacent to our contested arm (two sources versus one) and to the context-versus-memory routing our Qwen mover heads show (attending to the conflicting passage).

**Arora, Wu, Steinhardt, Schwettmann (ICML 2026, spotlight). Language Model Circuits Are Sparse in the Neuron Basis.** [arora2026sparse] arXiv:2601.22594
Shows that task circuits are sparse when expressed over MLP neurons.
*Relation:* supports our finding that a hedging circuit is expressible in roughly 100 neurons. Note for the team: our verified description is "sparse in the neuron basis (MLP neurons only)". The current draft's phrase "and in learned SAE features" is not in our verified notes; confirm against the paper before keeping it.

---

## 3. Directions, features, and two-channel dissociations

Where the 2026 neighborhood is densest. Several concurrent papers carve adjacent two-channel splits; none carves ours (decision versus spread, with the rank-1 direction as the spread carrier that can switch hedging on but not off).

**Ferrando, Obeso, Rajamanoharan, Nanda (ICLR 2025). Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models.** [ferrando2025iknow] arXiv:2411.14257
Using sparse autoencoders, finds entity-familiarity directions (known versus unknown entity) that causally steer refusal, and traces their effect through downstream attention heads.
*Relation:* the closest mechanism on our familiarity arm; both mock reviewers flagged it. Our early-layer entity reading (L0 to 7) is plausibly the same signal. What we add on that arm: the verified head-plus-neuron circuit the signal feeds, held-out faithfulness against random sets, and the on/off asymmetry of the direction (switches the hedge on at 0.97, cannot switch it off at -0.10). Must be an explicit contrast paragraph, not a list citation.

**Anthropic Interpretability Team (2025). On the Biology of a Large Language Model.** [anthropic2025biology] Transformer Circuits Thread
Attribution-graph case studies in Claude 3.5 Haiku, including a known-versus-unknown entity pathway in which a default "cannot answer" behavior is suppressed by known-entity features.
*Relation:* independent evidence, in a different model family, for an early entity-recognition signal gating refusal. Cited among directions-and-features work.

**Ji, Yu, Koishekenov, Bang, Hartshorn, Schelten, Zhang, Fung, Cancedda (2025). Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations.** [ji2025hedging] arXiv:2503.14477
Shows verbal uncertainty is captured by a linear feature and that steering it reduces hallucinations.
*Relation:* the rank-1 tradition our direction analysis extends. Our contribution is the asymmetry: the linear feature suffices to induce hedging, not to remove it.

**Patel, Chen, Wei, Papalexakis, Chen (April 2026). Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders.** [patel2026dissociation] arXiv:2604.19974
On Llama-3.1-8B and Gemma-2-9B, separates pure-uncertainty features (functionally critical) from pure-incorrectness features (little causal impact) and confounded features; removing confounded features improves accuracy and reduces entropy.
*Relation:* same base model as ours and a two-channel dissociation, but the axes are uncertainty versus correctness, not decision versus spread; feature-level and probe-based, not a verified circuit. Threatens the framing of our claim 7 more than its substance; cited with the distinction stated.

**Xiros, Zoumpoulidi, Paraskevopoulos (July 2026). Knowledge Knows, Verbalization Tells: Disentangling Latent Directions for Mathematical Solvability in LLMs.** [xiros2026knowledge] arXiv:2607.05013
Finds knowledge and verbalization encoded as distinct linearly decodable directions in the math-solvability domain; steering the verbalization direction improves refusal of unsolvable problems.
*Relation:* another adjacent two-channel split (knowledge versus verbalization). Domain and axes differ from ours; cited.

**Kumaran, Conmy, Barbero, Osindero, Patraucean, Velickovic (March 2026, updated May 2026). How do LLMs Compute Verbal Confidence?** [kumaran2026verbal] arXiv:2603.17839
In Gemma 3 27B, Qwen2.5-7B and Magistral Small, shows via patching, steering and attention blocking that verbal confidence is a cached self-evaluation: confidence representations emerge at answer-adjacent positions before appearing at the verbalization site, and explain variance beyond token log-probabilities.
*Relation:* the closest conceptual neighbor to our verbalization-moment result. They show where confidence lives before it is said; we show when the hedge commits to being said (rank of "unknown" 1,000 to 10 across L13 to 17) and that each uncertainty type verbalizes its own vocabulary. Complementary, and cited as such.

**Mazzaccara, Bertolazzi, Bernardi (August 2026). Different Facets of Verbalised Overconfidence: an Interpretability Study.** [mazzaccara2026facets] arXiv:2608.18106
In Qwen3-4B, uses transcoder features across verbal markers, abstention and numeric confidence; finds certainty carried by a broad coalition of shared features while uncertainty is a sparse override by a few dedicated features, with interventions that generalize across expression modes and languages.
*Relation:* the one result a careless reader will see as contradicting our claim 2 ("no dedicated uncertainty unit"). It does not: theirs is feature-level in a different model without a gated dataset; ours is a neuron-level negative under an identified null. Their sparse-override picture is consistent with our circuit-level sparsity. The reconciliation must be explicit in the paper.

---

## 4. Ground truth: defining "the model does not know"

Our gated-twin standard is built on these.

**Kadavath et al. (2022). Language Models (Mostly) Know What They Know.** [kadavath2022know] arXiv:2207.05221
Introduces P(True) and P(IK): models can self-evaluate answer correctness and predict whether they know an answer.
*Relation:* the origin of model-relative knowledge labels. Our gate operationalizes "knows" by sampled correctness rather than by self-report, and that is what makes the twins model-certified.

**Gekhman, Yona, Aharoni, Eyal, Feder, Reichart, Herzig (EMNLP 2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?** [gekhman2024slick]
Introduces the SliCK categories (HighlyKnown, MaybeKnown, WeaklyKnown, Unknown) from sampled correctness, and shows fine-tuning on Unknown examples increases hallucination.
*Relation:* our gate uses the SliCK extremes directly (control HighlyKnown at 9/10 or better with greedy correct; uncertain Unknown at 0/10 with greedy incorrect). Reviewer note: this restricts us to the tails; the MaybeKnown band is untested, which matters for uncertainty-estimation relevance.

**Mallen, Asai, Zhong, Das, Khashabi, Hajishirzi (ACL 2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.** [mallen2023popqa]
Introduces PopQA, entity-centric questions with popularity metadata, and shows accuracy tracks entity popularity.
*Relation:* the familiarity arm's source; obscure-versus-famous subjects in the same relation template are our one-variable manipulation.

**Xie, Zhang, Chen, Lou, Su (ICLR 2024). Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.** [xie2024conflictqa]
Introduces ConflictQA and characterizes how models behave when presented evidence conflicts with parametric memory.
*Relation:* the contested arm's source. Note the memory-guard step in our gate (6 pairs dropped where the parametric answer already matched gold).

---

## 5. Methods we build on

**Syed, Rager, Conmy (2023). Attribution Patching Outperforms Automated Circuit Discovery.** [syed2023attribution] arXiv:2310.10348
Gradient-based linear approximation to activation patching, cheap enough to screen every component.
*Relation:* our head and neuron screen. Because the approximation is first-order, we verify the top 30 heads and all neuron sets by real patching; reviewer 1 notes the screen bounds what the pipeline can find, so minimality claims are conditional on it.

**Wang, Variengien, Conmy, Shlegeris, Steinhardt (ICLR 2023). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.** [wang2023ioi]
The IOI circuit and the faithfulness, completeness and minimality criteria for circuit claims.
*Relation:* the source of our faithfulness protocol; we add held-out evaluation, both patch directions, and size-matched random-set nulls.

**Anthropic Interpretability Team (2026). Verbalizable Representations Form a Global Workspace in Language Models.** [anthropic2026workspace] Transformer Circuits Thread; Jacobian Lens library and prefitted lenses (neuronpedia/jacobian-lens)
Introduces the Jacobian lens, a prefitted per-layer decoder from residual state to output vocabulary, and the workspace framing of verbalizable representations.
*Relation:* our independent, observational witness (paper Figures 4 and 5). We use the released prefitted lenses for both models. Not the first external application of the lens (see JADR below), but the first to uncertainty verbalization as far as the sweep found.

---

## 6. Recommended, not yet cited

Surfaced by the 30 August novelty sweep. Authors not yet verified; confirm before adding to the bib.

- **arXiv:2602.02132 (February 2026). There Is More to Refusal in LLMs than a Single Direction.** Multiple geometrically distinct refusal directions that act as one behavioral knob. Structurally parallel to our direction-versus-set asymmetry, in the refusal domain. Useful for sharpening the refusal-versus-uncertainty boundary.
- **arXiv:2507.16199 (July 2025). Abstention as a prompt artifact.** Motivation for why gating on model-certified uncertainty matters: annotated "uncertain" prompts can elicit abstention for reasons unrelated to knowledge.
- **arXiv:2607.12792 (July 2026). JADR: J-space danger recognition.** A third-party safety application of the Jacobian lens. Footnote-level: it means we should not claim to be the first external users of the lens.

---

## 7. Positioning at a glance

| Our claim | Closest prior / concurrent | Status after the sweep |
|---|---|---|
| (i) Gated-twin, model-certified dataset standard | Gekhman (SliCK), Kadavath (P(IK)); 2507.16199 as motivation | Novel as a dataset standard; nothing comparable found |
| (ii) No uncertainty-specific single neuron under identified nulls | Stolfo, Gurnee; Mazzaccara (sparse dedicated features, transcoder level) | Novel as a neuron-level, null-tested negative; must reconcile with Mazzaccara explicitly |
| (iii) Verified sparse head-plus-neuron hedge circuit, held-out, vs random sets, two types, two models | SCIURus, Zhao, Roy, Arora | Novel at this granularity and rigor; SCIURus owns "uncertainty circuits exist" |
| (iv) Decision vs spread: direction switches hedge on, not off; set does both | Patel, Xiros, Kumaran, Ji; 2602.02132 | Specific form unclaimed elsewhere; the two-channel neighborhood is crowded, so state the axes precisely |
| (v) Verbalization moment with arm-specific vocabularies | Kumaran (cached confidence before verbalization); Anthropic workspace | Novel as a method application; conceptually foreshadowed by Kumaran, cite and differentiate |

## 8. What the review changes for the paper

- Lead Related Work with SCIURus, then the Ferrando contrast paragraph, then the four concurrent 2026 two-channel papers with one clause each on how the axes differ.
- Keep the Mazzaccara reconciliation sentence: feature-level sparsity is consistent with, not contrary to, a neuron-level negative.
- Do not claim to be the first uncertainty circuit or the first external Jacobian-lens application.
- The main-track follow-up the reviewers asked for (LM-Polygraph baselines plus a verdict-channel probe on a reinstated MaybeKnown band) is where this literature meets the uncertainty-estimation literature; Zhao's calibration payoff is the bar to clear.
