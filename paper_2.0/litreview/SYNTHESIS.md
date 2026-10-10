# Literature synthesis for the extended abstract (litreview, 2026-10-10)

Extends `C:/Users/starb/Desktop/Sher/litreview/00_SYNTHESIS_UniReps.md` (5 Oct; axes A-J). Numbers for our results come from `../numbers.tex` macros (names in backticks); nothing here is a new number. Citekeys match `../references.bib`.

## Closest prior work (the paragraphs reviewers read hardest)

**1. Ferrando et al. 2025 (ICLR), "Do I Know This Entity?"** [ferrando2025iknow]
- *What they did:* sparse-autoencoder latents in Gemma-2 detect whether the model recognises an entity, and steering them makes the chat model refuse questions about known entities or hallucinate about unknown ones. Recognition is defined from the model's own correctness on the entity's attributes.
- *What they did not show:* entity recognition and fact knowledge are confounded in their design, because "known" entities are those whose attributes the model gets right. They never hold knowledge fixed while varying familiarity, and they do not compare the refusal signal with the model's answer uncertainty.
- *What we add:* a popularity × knowledge factorial on PopQA with knowledge certified per question from 10 samples. Hedging follows popularity (difference `\hmLlamaDiff` `\hmLlamaDiffCI`, Llama) while first-token entropy follows knowledge in four families (C1, C2).
- *Together:* the "recognition" signal Ferrando et al. found is mostly a familiarity signal, and it is dissociable from whether the fact is retrievable.

**2. Lindsey et al. 2025, "On the Biology of a Large Language Model"** [lindsey2025biology]
- *What they did:* attribution graphs in Claude 3.5 Haiku show a default "can't answer" path that "known entity" features inhibit. Misfiring recognition, a familiar name with no fact behind it, produces hallucination.
- *What they did not show:* one model, case studies, no population-level test separating familiarity from knowledge.
- *What we add:* the population-level dissociation in open models (C1, C2), base-model lens evidence that the familiarity signal predates instruction tuning (C6), and steering (C4).
- *Together:* their mechanism predicts our behavioural pattern. Our steering caveat (no-hedge control `\nhOverRand`) says the direction that moves hedging is entangled with hedge-associated components.

**3. Brzezinka et al. 2026, "Graded Entity-Familiarity Readouts"** [brzezinka2026familiarity]
- *What they did:* probes at the last prompt token in 12 instruction-tuned models (Bielik, PLLuM, Gemma-4, Qwen3) separate real from fabricated entities. The probes track popularity in Polish-adapted models (ρ 0.28-0.57) but at most 0.11 in Gemma-4 and Qwen3. A familiarity direction moves Gemma-4-12B refusals both ways. They conclude that representational familiarity is separate from the abstention policy.
- *What they did not show:* no crossing with model-certified knowledge of the specific fact, and no comparison with output uncertainty.
- *What we add:* the knowledge axis, and the entropy channel that follows it.
- *Note:* their weak popularity tracking in Qwen3 is consistent with our Qwen2.5 numbers, where hedging was not measurable and the entropy popularity effect is ≈ 0. Mention it in one clause.

**Concurrent, 5 Oct 2026: Che & Li, "Does the Model Use the Feature?"** [che2026steering]
- An unknown-entity SAE latent raises abstention from 33% to 79% under steering. Installing its natural values into matched prompts transfers only 12% of the natural known/unknown contrast.
- *Consequence for us:* C4 must be worded as "steering along the familiarity direction can produce hedging", not "the model uses this direction". Our no-hedge control (`\nhOverRand`) is the same kind of caveat. Cite as concurrent and supporting.
- *Scoop-risk:* none for C1/C2; partial reframing of C4.

**Concurrent: Nguyen et al. 2026 (NeurIPS), "The Commit-Abstain Circuit"** [nguyen2026commit]
- A sparse set of heads and MLP sublayers decides commit vs abstain across 10 models in 5 families. Abstention signals arrive late and are often insufficient.
- No familiarity/knowledge split and no entropy channel. It supports C9 being appendix-only, since "a sparse abstention circuit" is now owned.

## Axis summaries

**P: Phenomenon.**
- *Known:* models carry entity-recognition / familiarity representations that steer refusal (Ferrando 2025; Lindsey 2025; Brzezinka 2026). Models' self-knowledge is partly calibrated (Kadavath 2022). Fine-tuning on unfamiliar examples teaches "I don't know" to unfamiliar inputs (Kang 2025), and evaluation incentives reward guessing (Kalai 2025).
- *Gap:* nobody has crossed familiarity with certified knowledge and compared the refusal readout with the uncertainty readout.
- *Us:* C1 and C2.

**M: Methods.**
- PopQA popularity (Mallen 2023); SliCK knowledge categories (Gekhman 2024); difference-of-means steering (Arditi 2024; Panickssery 2024); the Jacobian lens (Gurnee et al. 2026, prefitted lenses).
- We use each as published, with two departures: (i) hedging is labelled by two LLM judges (κ `\hmLlamaKappa`, no human validation, disclosed); (ii) jlens's BOS handling was audited and the figures regenerated with a single BOS.

**X: Competing explanation (hedging = output uncertainty).**
- Entropy and semantic-entropy methods (Kuhn 2023) and confidence-regulation neurons (Stolfo 2024; Gurnee 2024) treat the output distribution as the uncertainty signal.
- Ji et al. 2025 already separate verbal uncertainty from semantic uncertainty as distinct linear features. That is the closest prior work for "hedging ≠ entropy", and we must credit it.
- *Our addition:* what each signal follows. Hedging follows familiarity, entropy follows knowledge, in a factorial design (C1, C2). Entropy-selected neurons are frequency/temperature machinery (C8).

**H: Human analogue.**
- People's feeling of knowing follows cue familiarity rather than answer retrievability (Reder & Ritter 1992; Schwartz & Metcalfe 1992). Familiarity and accessibility combine (Koriat & Levy-Sadot 2001). Knowing-that-you-know and knowing-that-you-don't dissociate (Liu et al. 2007).
- The cascade prediction (familiarity decoded before knowledge in depth) failed our pre-set bar (R2). Report it as a failed prediction in one sentence.

**U: UniReps conversation.**
- Convergent representations across models (Huh 2024). Source-invariant uncertainty representations (Lippl 2024, organiser). A mid-depth abstraction phase (Cheng 2025, organiser).
- *Our cross-system content:* the entropy-follows-knowledge pattern in four families (C2); the familiarity signal in base models of three families (C6); the human FOK parallel.
- *Missing for this venue:* a quantitative cross-model similarity number (axis J §5d fix 2). This is an evidence-mode option, not a literature item.

## Scoop verdict (window 2026-10-05 to 2026-10-10)

`scoop-risk: none` for C1, C2, C6. `partial (reframing)` for C4 from che2026steering (2610.07270, submitted 2026-10-05). Ten arXiv advanced-search queries over cs, submitted 2026-10-01 to 2026-10-11, plus web searches; see QUERY_LOG.md. The earlier window (15 Aug - 5 Oct) was cleared by axis I (48 queries).
