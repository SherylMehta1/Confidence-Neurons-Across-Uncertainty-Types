# Outline: UniReps 2026 extended abstract (Checkpoint 2)

Title (locked 2026-10-09): **Saying "I Don't Know" Tracks Familiarity, Answer Uncertainty Tracks Knowledge**

Budget: 4 pages of main text, NeurIPS template. Three full-width figures take about 0.85 page, leaving about 3.1 pages (about 2,000-2,200 words) of text. References and appendix are unlimited.

## Step 1: the paper in three bullets

1. **What people assume.** Confidence work treats a model's "I don't know" as its answer uncertainty made audible, so reading the output distribution (entropy, confidence neurons) should tell you when it will abstain.
2. **The crack.** We cross entity popularity with whether the model actually knows the fact, using the model's own sampled answers.
   - Llama-3.1-8B-Instruct's "I don't know" follows popularity: effect +0.39 against +0.09 for knowledge (C1).
   - Its answer uncertainty follows knowledge, and so does every other family's: 4 of 4 families, knowledge effect at least 4× the popularity effect (C2).
   - These are two different variables, and the abstention one looks like the cue-familiarity heuristic of human feeling-of-knowing.
3. **What follows.** The familiarity variable is causal and old.
   - Steering along the obscure−popular direction makes two models say "I don't know" about facts they know (C4/C5), though mostly through hedge-associated components.
   - A familiarity signal already sits mid-depth in the decoded readout of base models from three families (C6).
   - So abstention methods built on output uncertainty measure the wrong variable, and the right one is shared across models before instruction tuning.

Every main-text claim serves a bullet: C1, C2 → 2; C4, C5, C6 → 3; C3 and C8 are one sentence each under bullets 2 and 1. C7, C9 and C10 go to the appendix or are dropped.

## Step 2: contribution sentence and thesis test

**Contribution sentence.** We show that, for factual questions, the variable that decides when an instruction-tuned model says "I don't know" (entity familiarity) is different from the variable its answer uncertainty tracks (knowledge). The answer-uncertainty half holds in four model families, and the familiarity signal is present in base models of three families.

- **Delete the setting:** "does abstention track familiarity or knowledge, and is that the same as uncertainty?" A question remains.
- **Swap the setting:** the claim reads as a testable claim for humans (feeling of knowing) or for other model families. That is the UniReps hook.

**Verbs allowed by CLAIMS.md:**
- C2 is established, so "we show".
- C1, C4, C5 and C6 are supported, so "our results indicate" or "we find", with scope.
- C3 is exploratory, so "in a post hoc check".

## Claims-to-evidence table

| Claim | Evidence file (repo) | Figure / table | Section |
|---|---|---|---|
| C1 Llama hedging: popularity > knowledge | results/hedge_measure_v2/summary.txt | Fig 1a; App. T1 (4 instruments) | §3.1 |
| C2 entropy: knowledge > popularity, 4 families | results/factorial_4models/effects.csv | Fig 1b, 1c | §3.1 |
| C3 not spelling / guessability (post hoc) | results/unireps_checks/summary.txt | one sentence | §3.1 |
| C4 steering, obscure−popular direction → hedging (qualified) | results/direction_reuse_v2/*/summary.txt; direction_nohedge | Fig 2a | §3.2 |
| C5 Gemma double dissociation | direction_reuse_v2/gemma9b/summary.txt | Fig 2b | §3.2 |
| C6 familiarity signal in base models | results/jlens_cascade/outcomes.csv | Fig 3 | §3.3 |
| C8 entropy-selected neurons are frequency/temperature machinery | RESULTS.md claims 1-2 | one sentence + App. | §1 / App. |
| R2 failed cascade prediction | results/jlens_cascade/summary.txt | one sentence | §3.3 |
| R4 hedging unmeasurable in 3 families (κ < 0.6) | results/hedge_measure_v2/summary.txt | App. T2 | §2, §4 |

## Draft 0 introduction (disposable; one paragraph per move)

1. **Stakes.** A model that says "I don't know" when it lacks a fact, and answers when it has it, is the cheapest defence against hallucination. Most methods for predicting that behaviour read the model's uncertainty over answers.
2. **Gap.** That assumes abstention and answer uncertainty are one variable. People are a counterexample: their sense that they can answer follows how familiar the question's terms are, not whether the answer is stored. Mechanistic work on LLM refusal finds "known entity" features, but defines "known" by whether the model answers correctly. That confounds familiarity with knowledge.
3. **Idea.** Separate the two. Take PopQA questions about obscure and popular entities, and label each question known or unknown from ten of the model's own answers. That gives a 2 × 2 in which familiarity and knowledge vary independently.
4. **Intuition.** If "I don't know" were uncertainty made audible, it would follow the knowledge axis just as entropy does. If it is a familiarity heuristic, it will follow the popularity axis, even for facts the model gets right every time.
5. **Contributions.** (i) the dissociation in Llama, with entropy following knowledge in four families (§3.1, Fig 1); (ii) steering evidence that the obscure−popular direction produces "I don't know" about known facts, with the caveat (§3.2, Fig 2); (iii) the familiarity signal in base models of three families (§3.3, Fig 3).
6. **Preview.** Headline: popularity effect \hmLlamaPopEff vs knowledge effect \hmLlamaKnowEff on Llama's hedging; knowledge effect on entropy at least 4× popularity in every family.

## Section outline (paragraph labels = future topic sentences)

**Abstract** (about 170 words)
1. A model's "I don't know" is usually treated as its answer uncertainty made audible; we test whether the two follow the same variable.
2. Design: popularity × model-certified knowledge on PopQA, two prompt formats, four families.
3. Result: Llama hedging follows popularity (\hmLlamaPopEff vs \hmLlamaKnowEff); entropy follows knowledge in all four families.
4. Mechanism and origin: steering and base-model lens evidence.
5. Moral: abstention and uncertainty are different variables; the first resembles human cue familiarity.

**1 Introduction** (about 0.6 page)
1. Stakes and the common assumption (move 1).
2. People and prior LLM work as counterexamples; the confound in "known entity" studies (Ferrando 2025; Lindsey 2025) (move 2).
3. Our design separates the two axes (moves 3-4).
4. Contributions with pointers and the headline numbers (moves 5-6).
5. One sentence on the earlier, negative neuron-level finding (C8) as motivation: entropy-correlated neurons turned out to be frequency and temperature machinery.

**2 Setup** (about 0.6 page)
1. Items and labels: PopQA, obscure = bottom decile and popular = top decile of page views; known/unknown from greedy plus 10 samples (SliCK); n per family.
2. **Two prompt formats, and why** (user-requested):
   - the forced-answer prompt ("The answer is") measures knowledge and answer uncertainty;
   - the free prompt measures saying "I don't know";
   - why one prompt cannot do both, with the Belorechensky example in one line;
   - the caveat that the prompt change could contribute;
   - the counterpoints: first-token log-odds and single-prompt steering and lens.
3. Measuring "I don't know": two LLM judges (Qwen2.5-32B, Gemma-2-27B) on 3 × 96-token replies; κ = \hmLlamaKappa against a pre-set bar of 0.60, met only for Llama, so hedging is reported for Llama only. **No human validation (disclosed).**
4. Effects: popularity and knowledge main effects over the 2 × 2, with item bootstrap. Each analysis had a criterion file written and hashed before its run; we say "criterion fixed before the run", not "pre-registered".

**3 Results**

§3.1 *Saying "I don't know" follows familiarity; answer uncertainty follows knowledge* (Fig 1)
1. Llama: obscure → hedges at \hmLlamaObsUnk / \hmLlamaObsKnown even when it knows the fact; popular → \hmLlamaPopUnk / \hmLlamaPopKnown. Popularity \hmLlamaPopEff vs knowledge \hmLlamaKnowEff; difference \hmLlamaDiff \hmLlamaDiffCI. Same sign under four instruments (App. T1).
2. Entropy shows the reverse in Llama (knowledge \entLlamaKnow vs popularity \entLlamaPop) and in Mistral, Qwen and Gemma.
3. Post hoc: not name spelling or guessable answers (C3), in one sentence.
4. Takeaway: two variables; the abstention one matches the human cue-familiarity account (Reder & Ritter 1992; Schwartz & Metcalfe 1992).

§3.2 *Steering along the obscure−popular direction produces "I don't know" about known facts* (Fig 2)
1. Setup: difference-of-means directions at layer ℓ*, 150 held-out popular known questions, equal norm, random-direction nulls.
2. Llama \stLlamaNone → \stLlamaFam (knowledge direction \stLlamaKnow); Gemma \stGemmaNone → \stGemmaFam. In Gemma the knowledge direction raises entropy instead (\stGemmaDE): a double dissociation. In Llama only the hedging half holds.
3. Caveat: a direction fitted on never-hedged items moves hedging by only \nhOverRand. The effect runs mostly through hedge-associated components, so this shows control, not that the model uses familiarity per se (cf. concurrent Che & Li 2026).

§3.3 *The familiarity signal predates instruction tuning* (Fig 3)
1. Jacobian lens (Gurnee et al. 2026) on four base models: mid-depth familiarity effect positive in all four (\jlLlamaBaseMidF … \jlOlmoBaseMidF).
2. Failed prediction (stated as such): we predicted, from human memory research, that familiarity would appear before knowledge in depth. It did not: both appear in the same mid-stack band (R2).

**4 Related work** (about 0.35 page): three groups
1. Knowledge awareness and refusal mechanisms (Ferrando; Lindsey; Brzezinka; Nguyen; Che & Li).
2. Uncertainty readouts (Kadavath; Kuhn; Stolfo; Ji et al. 2025, which already separate verbal from semantic uncertainty). Our addition: what each follows.
3. Human feeling of knowing and convergence across systems (Reder; Schwartz; Koriat; Huh; Lippl).

**5 Limitations and conclusion** (about 0.35 page)
1. Hedging measured in one model: the judges failed κ in three families, and the other families rarely say "I don't know".
2. Judges not validated against human labels.
3. Two prompt formats; first-token entropy only; PopQA single-entity questions; 7-9B models.
4. Steering shows control, not use.
5. Conclusion and moral: confidence methods that read answer uncertainty predict a different variable from the one that decides abstention. The testable prediction for future work: two models' hedging should agree on names even where only one knows the fact.

**Appendix**
- A: four-instrument table (T1).
- B: per-family judge κ and hedge rates (T2).
- C: C7 timing figure.
- D: the C9 circuit (equivocal), in brief.
- E: practical notes (BOS bug, judge redo).
- F: worked example.

## Terminology lock (from the glossary's last column)

| Concept | Paper term |
|---|---|
| hedge | "says 'I don't know'" / "abstains" (verb), "abstention rate" |
| familiarity / popularity | "popularity" for the measured variable (PopQA page views); "familiarity" for the construct |
| SliCK HighlyKnown / Unknown | "known / unknown facts" (defined once) |
| first-token entropy | "answer uncertainty (first-token entropy)" |
| factorial | "popularity × knowledge design" |
| steering directions | "obscure−popular direction", "unknown−known direction" |
| Jacobian lens | named once with its citation, then "the lens" |

Coined names in the main text: none beyond these.

## Register (locked here, pending your approval)

Short-workshop register: three claims at most in the main text, one figure each, rest to the appendix.

UniReps' extended-abstract call asks for "insightful negative findings". So the failed cascade prediction, the three families where hedging could not be measured, and the steering caveat are stated as findings in one or two sentences each, not as a separate audit section.
