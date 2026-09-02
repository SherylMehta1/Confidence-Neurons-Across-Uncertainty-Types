# Review of dd.md (abstract + intro draft)

*Keyed to the line numbers in `paper/dd.md`. Short version: the prose is close, but it is built on the pre-audit story, and several sentences claim things our own follow-ups walked back. Every replacement number is already in `RESULTS.md` and `paper/main.tex`, so copy from there rather than re-deriving.*

---

## Title, "Uncertainty Has a Circuit" (line 1)

- Says less than we found. Our result is that uncertainty (entropy) and the circuit are different things, so the title names the wrong noun. Hedging is what has the circuit.
- The audited paper's title ("Deciding to Say 'I Don't Know'") carries the actual claim. Either adopt it or pick something that names the decision, not the uncertainty.

## Abstract, variant 1 (line 4)

- "identify what actually drives model uncertainty" is an overpromise. We identify what drives hedging. Say that.
- "no single neuron survives an entropy-adjusted causal test": the entropy adjustment is unidentifiable on gated arms. The test that counts is the random-neuron null. Rephrase around the null.
- "2 to 6% for size-matched random components" is stale. It is 2 to 4% on the 30-set null (the figure was regenerated to match).
- "roughly 2% of heads" hides that it is 2.55% in Qwen, where the sparsity clause fails. The paper discloses this now; the abstract should not paper over it.
- "replicating across two models and two distinct sources of uncertainty": three of four cells run, and the cross-model cells are contested arms where free-generation hedging is roughly zero. Add the "three of four cells" clause.
- "carries the hedging decision without carrying its output entropy, separate machinery" is the walked-back general dissociation. It is a directional asymmetry in one model, now with a test attached (p below 10^-4 and p = 0.015 in Llama, p = 0.11 in Qwen). Scope it or a reviewer will.
- "downstream readouts of that decision, not its cause" claims a mediation we never tested. The supported version: the spread's plumbing, not the decision.
- Longer than variant 2 with no added content. Cut this one.

## Abstract, variant 2 (line 6)

- The keeper: same content, less throat-clearing.
- Inherits every factual issue from variant 1 (2 to 6%, the entropy-adjusted framing, the unqualified dissociation, "readout not cause", the replication overclaim). Same fixes apply.
- Missing the venue's selling points entirely: no keep rates, no rank-1 on/off asymmetry, no verbalization moment, no pre-stated prediction. At least two of those belong in the abstract for InterpScience.

## Intro, short variant (lines 9 to 16)

- Cut. It is a compressed duplicate of the long variant, and every duplicated passage is one more place for numbers to drift.
- One thing worth saving: line 10's "readout, not a cause" framing is the cleanest statement of the alternative hypothesis in the file. Merge that sentence into the long variant (as the hypothesis being tested, not as a conclusion).
- The contribution list at 13 to 16 disagrees in wording with the list at 24 to 28. One list, one place.

## Intro, long variant (lines 18 to 28), the one to build on

- Line 19: the Stolfo mechanism description (LayerNorm rescaling, unembedding null space) is genuinely good. Keep it. It shows we understand what we are deflating.
- Line 20: the entropy-confound argument is right and well written. Keep it, it sets up the null.
- Line 21: the gate description is accurate; add the keep rates (195/1151 familiarity, 234/600 contested, Qwen 24). "Reported, never tuned" is one of our best sentences and it is absent.
- Line 22: the problem paragraph. "Critically, an ANCOVA-style adjustment" makes the unidentified estimator the centerpiece. Rewrite around the identified random-neuron null. And drop "BH-FDR at 0.01" from anywhere near the null comparison: that correction could not fire there (p-floor 1/101) and we just scrubbed it from the paper.
- Line 23: "against ten size-matched random sets" should be thirty (ten in Qwen).
- Line 25: "no single neuron in either model survives" needs the contested-arm asterisk (z = +29.4 and +14.8 exist; they pattern as answer-competition and are null on the hedging arm). Scope to the arm where the model actually hedges.
- Line 26: "the queried entity is resolved within the first eight layers" is stated for both models, but the position map is Llama familiarity. Scope it.
- Line 27: same 2 to 6% and replication fixes as the abstract.
- Line 28: same dissociation fix as the abstract. This is the third place the unqualified claim appears.
- Missing contribution: the dataset standard gets no bullet, and it is our safest novelty. See the replacement list below.

## Intro, compressed variant (lines 29 to 38)

- Delete wholesale. A worse copy of the long variant with every error intact and less of the good material (the Stolfo mechanism detail is gutted at line 30).
- Nothing here that is not in 18 to 28.

## Replacement contributions list (paste this over both lists)

**Our contributions are:**

1. **A gated-twin dataset standard.** Prompt pairs differing in one manipulation, kept only when the model itself proves the labels: control correct in at least 9/10 samples, uncertain in 0/10, entropy gap of 0.5 nats or more, and a hedging gap for the behavioral arm. Keep rates are reported, never tuned (Llama familiarity 195/1151, contested 234/600), and the grader is audited against a 32B judge, with disagreement lower on hedged answers.

2. **A causal decomposition of "confidence neurons."** Correlation-selected neurons turn out to be token-frequency machinery plus one temperature neuron. Under an identified random-neuron null (100/50 random neurons ablated on the same prompts, which inherit the entropy confound by construction), no single neuron is uncertainty-specific on the arm where the model actually hedges; the only large single-neuron effects sit on the contested arm and pattern as answer-competition, not ignorance.

3. **A verified hedging circuit.** The entity is read in the first eight layers (Llama familiarity), a verdict is routed mid-stack (layers 13 to 19 in Llama, 17 to 21 in Qwen) by verified attention heads, and mid-layer MLP neurons write it toward the readout. Jointly patching 20 heads and 100 neurons (1.95% of Llama's heads, 2.55% of Qwen's, 0.02% of neurons) transfers 78 to 107% of the held-out hedge readout, against 2 to 4% for size-matched random sets, in three of four model by arm cells; the fourth has too few gated pairs to run, itself a finding.

4. **A directional decision-vs-spread asymmetry.** In Llama, removing the unknown signal transfers the hedge decision with little of the entropy, while injecting it transfers both (paired permutation: p below 10^-4 on familiarity, p = 0.015 on contested); Qwen shows no asymmetry (p = 0.11). A rank-1 direction can switch the hedge on (0.97) but not off (-0.10); un-hedging needs the component set. We state this as a one-model result, not a general dissociation.

5. **An account of what our own criteria could not decide**, including a pre-stated prediction on the aleatoric arm that held in Llama and half-failed in Qwen, reported as found.

### Why the prediction half-fails in Qwen: DECIDED (30 Aug, sandbox run)

- The three deciders ran (test a locally from the committed lens trajectories; tests b and c on the molab GPU, with the rerun reproducing the committed L25/L26 gaps to 3 decimals before anything else was trusted). Verdict: **Reading 1, leakage**. The +3.67 gap is not a hedge decision.
- The decomposition: at L25 to 26 hedge tokens hold about 1e-6 to 1e-5 of probability in BOTH twins; there is no hedge plan to read out. The gap comes from the generic answer channel collapsing (' the' 4.6e-3 to 1.6e-5, ' that' 1.8e-4 to 3.7e-6 at L26, roughly 100x) as probability spreads over the concrete valid answers; the top gained tokens are ' February', ' Mars', ' Africa', zero hedge vocabulary.
- Behaviorally: free-generation hedge rate is 0/390 in both arms. A thin "one of the X is Y" enumerative phrasing appears on 20.5% of uncertain vs 0.5% of control completions, concentrated in 12/39 prompts, still commits to a specific answer, and is uncorrelated with the pair-level lens gap. Not a half-verdict.
- What this buys the paper: the half-failure is the same fact as Qwen's missing directional asymmetry (channels not kept apart), so Qwen becomes the natural control for the claim that the dissociation tracks the model. The paper's undecided-criteria item (b) is updated from "three tests would decide it" to the decided version.
- Evidence archived: results/circuit_aleatoric_qwen/decider_*.csv plus decider_provenance.json; the local correlation test is scripts/aleatoric_leakage_corr.py.

If five bullets are too many for the format, merge 4 and 5 and keep the first three untouched. In LaTeX, use math mode for the stats ($p < 10^{-4}$, $-0.10$).

## Everywhere at once

- Every replacement number is already in `RESULTS.md` and `paper/main.tex`. Copy, do not re-derive.
- Nothing in the file mentions the rank-1 direction, the lens, the aleatoric prediction, or the undecided-criteria section. If this intro is for the rigor workshop, those are the reasons the mock reviewers scored it accept.
- Two abstracts and three intro variants in one file means five copies of every number. Kill the duplicates before editing anything else, then fix numbers once.
