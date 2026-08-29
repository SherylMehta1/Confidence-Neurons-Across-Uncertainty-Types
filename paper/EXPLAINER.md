# The Whole Project, Explained From Zero

*A plain-language walkthrough of everything on `emotional-atyachar`: what we asked, every claim, every method, every experiment design choice — and then how to turn it into a paper, using Neel Nanda's and Nicholas Carlini's writing advice. Written for someone joining the project today.*

---

## 1. The question, in one paragraph

When you ask a language model something it doesn't know and it says "I'm not sure, but…", **what happened inside?** The popular answer in the literature: there are special "**confidence neurons**" — individual units whose activation tracks how *spread out* (high-entropy) the model's next-token prediction is — and they regulate confidence. Our project asked: is that actually true, causally? The answer we found: **no** — those neurons are downstream plumbing. The real thing is a **decision**: a small circuit of attention heads and neurons computes an "I don't know this" verdict mid-network and routes it to where the answer begins. The decision and the spread are *different variables in different channels*. That's the paper.

## 2. Glossary (read this first)

| Term | Plain meaning |
|---|---|
| **Entropy** | How spread out the model's next-token probability distribution is. High = unsure between many tokens; low = confident in one. Measured in nats. |
| **Hedging** | The *behavior* of saying "I'm not sure / unknown / I don't know" in free generation. |
| **Hedge log-odds** | A fast proxy for hedging: at the position where the answer would start, log-probability of hedge-starting tokens minus answer-starting tokens. Positive = about to hedge. |
| **Twin pair** | Two prompts identical except for ONE change (e.g., an obscure vs a famous person in the same template). The perfect counterfactual: any internal difference is caused by that one change. |
| **The gate** | Our filter that only keeps twin pairs where the model *proves* the labels: control answered correctly 9/10 samples, uncertain 0/10, entropy gap ≥ 0.5 nats, (behavioral arm) hedging gap ≥ 0.3. |
| **Arms** | Familiarity (obscure vs famous entity — the model hedges here), contested (two disagreeing sources vs one), aleatoric ("name a prime" vs "name the smallest prime" — many valid answers). |
| **Ablation** | Deleting a neuron's contribution (setting its activation to its mean) and measuring what changes. |
| **Activation patching** | Copying internal activations from one prompt's forward pass into another's. Patch the known twin's state into the unknown prompt and see if it stops hedging. |
| **c→u / u→c** | Direction labels, always **source→target**. c→u = control (known) state patched INTO the uncertain prompt = *removing* unknownness = hedge OFF. u→c = *injecting* unknownness = hedge ON. |
| **Recovery** | (patched − target)/(source − target). 1.0 = the patch fully moved the target prompt's readout to the source twin's value. |
| **Direction** | A single vector in activation space: mean(uncertain residuals) − mean(control residuals). "Rank-1" = intervening only along this one dimension. |
| **Faithfulness** | The test of whether a proposed circuit is *sufficient*: patch only those components on held-out pairs and measure recovery, against same-size random component sets. |
| **Jacobian lens** | A prefitted tool (Anthropic) that decodes any layer's internal state into "what tokens is this state disposed to make the model say" — observational, no interventions. |
| **BH-FDR** | Statistical correction for testing many neurons at once, so a few lucky p-values don't count as discoveries. |
| **Held-out** | 30% of pairs never used for finding components — only for final testing. Prevents "discovering" patterns that only exist in the data you searched. |

## 3. The story arc (how we got here)

1. **The audit.** The original repo claimed uncertainty-specific neurons. Auditing found the results were run in 4-bit precision, without control prompts, with mis-standardized effect sizes. We rebuilt everything in bf16 with matched controls. One neuron survived — briefly.
2. **The confound.** A flat distribution amplifies the *magnitude* of any perturbation (ρ(H, |shift|) = 0.25). Adjusting for it killed every "uncertainty-specific" neuron. (Later, an external audit showed the adjustment itself is unidentifiable on gated data — so we replaced it with a better null; see claim 2.)
3. **The dataset rebuild.** We stopped trusting prompt labels and made the model prove its own uncertainty (the gate), with the grader audited by a 32B judge.
4. **The circuit.** With neurons ruled out, we looked for the mechanism with patching: position×layer maps → heads → neuron groups → joint faithfulness. Found it.
5. **The deciders.** Every criticism (sparse-vs-distributed? adjustment invalid? claim too flashy?) got its own follow-up experiment. Three adversarial audit rounds; everything either fixed or decided.
6. **The lens.** An independent observational method reproduced the causal picture and let us *watch* the verdict become the word " unknown".

## 4. The seven claims, one by one

### Claim 1 — What correlation-selected "confidence neurons" actually are
**Plain:** if you pick neurons because their activation tracks entropy, you mostly get neurons that track *word frequency*, plus one "temperature knob" neuron. Not uncertainty detectors.
**Design:** score all candidates on Stolfo-style weight criteria; then a causal test with a **temperature-matched baseline** — compare each neuron's effect against what a pure "make everything flatter" rescaling would do, so "changes entropy" can't masquerade as "frequency neuron."
**Numbers:** 9/55 frequency neurons by weights, causally confirmed (effects stable across all prompt sets, r = 0.52–0.62, 6× random neurons); L31_N2477 = temperature neuron (dose–response ρ = 0.86, ~100% mediated by RMSNorm rescaling).
**Status: established.**

### Claim 2 — No single neuron is uncertainty-specific
**Plain:** no individual neuron "knows the model doesn't know."
**Design (the interesting part):** the naive test is confounded (see §3.2). First fix: ANCOVA adjustment for baseline entropy — kills everything on the original stimuli. But on gated data the two conditions' entropies don't overlap, so ANCOVA can't be estimated there. Second fix (the one that counts): an **identified random-neuron null** — recompute the exact same statistic for 100 (familiarity) / 50 (contested) random neurons ablated on the same prompts. Random neurons *inherit* the confound, so beating them means something.
**Numbers:** four familiarity candidates beat all 100 randoms (largest z = −6.2) but at ≤ 0.013 nats ≈ **0.5% of the 2.8-nat twin gap** — real, trivial. The only large effects are contested-arm-only (L31_N11541 z = +29.4, L30_N1457 z = +14.8) and are *null on the arm where the model actually hedges* → they're answer-competition machinery, not knowledge awareness. No neuron set moves hedging behavior (77% vs 0.8% gap; clamps move it ≤ 2 points).
**Status: established** (with the identification caveat stated).

### Claim 3 — Uncertainty is a property of the (model, prompt) pair
**Plain:** you can't label a prompt "uncertain" in the abstract; models differ.
**Design:** identical candidate pools through the same gate for two models.
**Numbers:** same 1,151 PopQA candidates → **195** gated pairs for Llama, **24** for Qwen (Qwen guesses confidently instead of hedging). Contested and aleatoric arms show entropy gaps with ~zero hedging in both models — spread and behavior dissociate at the dataset level. Grader audit: 94.6–100% agreement with a 32B judge, and disagreement *lower* on hedged answers (so grader error can't fake the behavioral results).
**Status: established** — and the dataset standard is a reusable artifact.

### Claim 4 — Where the verdict is computed and how it moves
**Plain:** the model reads the entity early, decides "unknown" there, then physically moves that verdict to where the answer will start.
**Design:** patch the residual stream between twins one position-group × layer at a time (entity tokens / question tail / final position).
**Numbers:** injecting the unknown entity's activations into the known prompt (u→c) recovers the full hedge at L0–5 via entity positions alone (≈1.0), decaying by L14; the readout positions pick it up from L13 (crossing at L13–14) — routed by 15 verified heads (recovery > 0.05; biggest single head L17H20 = 0.39). Qwen contested: same picture at L17–21, with mover heads that literally attend to the conflicting passage. MLP write-windows: Llama contested L11, L13–14, L21; some layers *fight* the patch (negative recovery, largest −0.28 at L30) — the flow isn't smooth accumulation.
**Status: established in 3 of 3 runnable cells** (Qwen-familiarity has only 24 pairs — excluded by the standard's own ≥80 rule).

### Claim 5 — The circuit is sufficient and sparse (faithfulness)
**Plain:** ~120 components out of ~460,000 carry most of the behavior.
**Design:** patch the 20 heads + 100 neurons jointly on **held-out** pairs, both directions, vs **10–30 size-matched random sets** (the null that keeps you honest). Criterion fixed in code before the held-out runs: recovery > 0.7 with < 2% of components (not independently pre-registered — we say so).
**Numbers:** c→u recovery 0.80 / 0.89 / 0.86 (familiarity / Llama contested / Qwen contested); u→c 1.07 / 0.78 / 0.98; randoms 0.02–0.04. CIs: [0.70, 0.89], [0.69, 1.09], [0.77, 0.95] — one dips below 0.7, one grazes; we print that. Minimality: LOO-ranked pruning saturates (20 components → 0.71 familiarity; 40 → 0.91 contested; Qwen 10 → 0.86, plateau 1.03–1.14 = overshoot, flagged); a rank-1 direction falls far short of the set for un-hedging.
**Status: established**, minimality decided by follow-up.

### Claim 6 — Arms share a variable, not a circuit
**Plain:** familiarity-uncertainty and conflict-uncertainty use overlapping but mostly different components.
**Numbers:** heads Jaccard 0.11 (shared: L12H28, L15H4, L15H7, L29H0), neurons Jaccard 0.11, head-recovery Spearman ρ = 0.61.
**Status: supported** (one model, two arms).

### Claim 7 — The decision and the spread are different channels (directional)
**Plain:** you can remove the model's *decision to hedge* without removing its *spread*, but not vice versa — and one single direction carries the spread while switching hedging ON (not OFF).
**Design:** compare full-set patching vs rank-1 direction patching, per direction; then two independent stress tests — a **pre-stated prediction** on the aleatoric arm (spread-only → should show no decision transfer, no verbalization) and the **Jacobian lens** (observational).
**Numbers:** removing unknownness (c→u) moves the hedge with little entropy (−0.13 / +0.33); injecting it (u→c) moves both (+0.45 / +0.55); Qwen shows no asymmetry (0.83/0.73). Rank-1: can't switch the hedge off (−0.10 vs set's 0.89) but can switch it on (0.97). Lens: the hedge gap is half-formed by L16 while the entropy gap waits until L28; the rank of " unknown" at the uncertain readout falls ~1,100 → 10 across L13–17. Prediction: held cleanly in Llama (+0.30 vs +5.05), **partially failed in Qwen** (+3.67 ≈ 44% of contested) — reported as found. **Five-cell vocabulary:** each arm's direction verbalizes its own thing — the hedge lexicon (familiarity), the conflict itself (' both' → ' conflicting' → ' ambiguous'), the answer type (' example' → ' any'); replicated in Qwen partly in Chinese.
**Status: supported as a directional asymmetry** in one model; the general "decision ≠ spread" slogan is *not* claimed.

## 5. Methodology, explained

**The gate (dataset standard).** Every causal claim is only as good as its counterfactual. Twin pairs give a one-variable counterfactual; the gate makes the labels model-proven instead of assumed; keep rates are *reported, never tuned* (that sentence matters: tuning keep rates until results appear is how fields fool themselves). The grader is deterministic (alias matching on the answer head) and audited against an LLM judge — with the crucial check that its errors don't correlate with hedging.

**The neuron battery.** Mean-ablation at the readout position; explicit reference policy; matched control prompts; permutation tests; BH-FDR; entropy-magnitude confound handled twice (ANCOVA where identified, random-neuron null where not); dose–response, frozen-RMSNorm decomposition (separates "scales the distribution" from "points somewhere"), behavioral clamping, verbalized confidence. Design principle: **a null result must survive every way of not-finding-it being wrong.**

**The circuit pipeline.** Coarse→fine: positions×layers (where), then heads (attribution patching as a cheap screen, *always verified by real patching* — the screen is a linear approximation and screens can lie), then neuron groups (verified jointly vs random sets), then joint faithfulness on held-out data. Design principle: **every attribution is confirmed by intervention, and every intervention is compared to a size-matched random null.**

**The lens.** Everything above is interventional. The Jacobian lens is observational and shares zero machinery with patching — when both point at L13–19, that's *triangulation*, not repetition.

**The audits.** Two external adversarial audits plus three internal agent audits recomputed every number from the raw CSVs. The direction-semantics episode (we glossed c→u backwards in prose for a day; the code and numbers were always right) is the cautionary tale: **the code is the ground truth for what an experiment means, not anyone's memory of it.**

## 6. Experiment-design principles used (steal these)

1. **One-variable counterfactuals** (twins) beat between-group comparisons.
2. **The model certifies its own labels** — sampled correctness, not annotation.
3. **Nulls must inherit the confound**: random neurons on the same prompts, random component sets of the same size, random directions of the same norm.
4. **Held-out discipline**: selection data and evaluation data never mix.
5. **Screen cheap, verify expensive**: attribution patching proposes; activation patching disposes.
6. **State predictions before running them** — and publish the half-failure (Qwen aleatoric).
7. **Fix criteria in code before the final runs** — and admit that isn't full pre-registration.
8. **Triangulate methods**: causal (patching) + statistical (nulls) + observational (lens).
9. **Report what your own criteria couldn't decide** — then run the experiment that decides it.
10. **Audit adversarially, repeatedly** — and let the code, not prose, define semantics.

## 7. How to write the paper (Neel Nanda × Carlini, applied to *this* paper)

Both guides agree on the skeleton: **a paper is 1–3 claims, a narrative that makes readers care, and evidence rigorous enough to survive a skeptic.** Here's the merged method, with our paper as the worked example.

### Step 1 — Compress to the narrative (Neel's step 1, Carlini's "one singular idea")
Write the whole paper as three bullets before any prose. Ours:
- *"I don't know" is a computed verdict, not leaking entropy* (claims 4+5+7).
- *The neurons everyone calls confidence neurons are the spread's plumbing* (claims 1+2).
- *You can only show this on data where the model proves its own uncertainty* (claim 3).
Everything else — overlap, vocabulary, lens — is supporting evidence for these three. If a paragraph doesn't serve one of the three bullets, it goes to the appendix. (Carlini: the title you struggle to write is the symptom of having more than one idea.)

### Step 2 — Abstract by formula (Carlini) with full takeaways (Neel)
Topic → problem → results **with numbers** → why it matters. Neel adds: assume the abstract is all anyone reads — the key takeaway must be *in* it, not promised by it. Ours opens with the question ("decision, or symptom?"), gives 78–107% vs 2–4%, the rank-1 on/off asymmetry, the ~1,000→10 verbalization, and closes on the stake ("confidence methods that read the distribution are measuring the wrong channel"). No hedging words in the abstract itself.

### Step 3 — Intro as a story (both)
Carlini: meet readers where they are (what they currently believe: hedging = audible entropy). Introduce the crack (nobody verified the labels; nobody separated the channels). Turn (the twins + patching break the assumption). Best-supported evidence first (the circuit, *not* the flashy lens result — lead with what survived the most audits). Climax image (the verbalization). Stakes. Neel: the intro is an extended abstract — claims, novelty, evidence, impact all present.

### Step 4 — Figures are the paper (Neel especially)
One figure per claim, each self-contained, each caption opening with its **bolded one-sentence takeaway** ("At layers 13–17 the verdict becomes a word."). Annotate the windows (we shade L13–19). Don't defer key numbers to the text. Color-blind-safe palettes; never red/green as the only signal.

### Step 5 — Rigor section = your red-team, in print (both, loudly)
Neel: "assume you've made a mistake — what is it?" and *show the pre- vs post-hoc split*. Carlini: locally optimal — no obvious missing experiment. Our implementation is the "What our own criteria could not decide" section: CI crossings, the non-saturation scare, the unidentified ANCOVA — each *raised, then decided by a named follow-up*. This is the section reviewers at a rigor-themed venue remember. Also per Neel: distinguish pre-stated (aleatoric prediction) from post-hoc (five-cell vocabulary — found while checking, and labeled that way).

### Step 6 — Limitations without flinching (both)
List the ones a hostile reviewer would find, before they do: contested-only cross-model cells, the 24-pair Qwen arm, criterion-not-preregistered, token-proxy readout, reference-overlap. Neel: this *gains* trust with exactly the readers who matter.

### Step 7 — Conclusion states the moral (Carlini)
Not a summary in past tense. Break the fourth wall: "are you reading the decision, or only its shadow in the distribution?"

### Step 8 — The process loop (Neel's iteration)
Bullets → intro outline → full outline → draft figures → check the narrative still holds → prose → edit ruthlessly → **external feedback at every stage** (our version: adversarial audit agents + the external reviewer, five rounds). Read it aloud at the end (Carlini) — that's how "One finding survives on the full table" gets caught as a devastating double-reading.

### The don'ts that bit us (so you don't repeat them)
- Don't let prose drift from code semantics (the c→u/u→c inversion).
- Don't quote a number from memory — every number from a file, every file audited (our "five orders of magnitude" was really two).
- Don't lead with the newest result; lead with the most-audited one.
- Don't say "pre-registered" when you mean "fixed in code beforehand."
- Don't average away direction- or arm-specific results ("entity ≈1.0" was one direction only).
- Don't hide the half-failed prediction — it's the most venue-credible sentence you own.

---

*Repo: everything cited here is reproducible from `emotional-atyachar` — `RESULTS.md` (per-claim evidence pointers), `UNCERTAINTY_DEFINITION.md` (dataset standard), `paper/main.tex` (the audited reference text), `paper/DOC_MEMO.md` (the corrections memo). Sources for §7: Neel Nanda, "Highly opinionated advice on how to write ML papers" (Alignment Forum); Nicholas Carlini, "How to win a best paper award" (2026).*
