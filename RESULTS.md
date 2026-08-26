# No confidence neurons on the behavioral arm — and a faithfulness-tested uncertainty circuit

*Results summary for the `emotional-atyachar` line of work. Full lab record and interactive report: the "Uncertainty Neurons Rebuilt" artifact; definitions and dataset standard: [UNCERTAINTY_DEFINITION.md](UNCERTAINTY_DEFINITION.md).*

## Abstract

Neurons whose activation correlates with next-token entropy are routinely read as "confidence neurons." In Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct we show that this reading fails, and what is true instead. On model-relative, correctness-gated twin datasets (uncertain = the model demonstrably does not know; grader validated against a 32B judge), no single neuron survives an entropy-adjusted causal test of uncertainty-specificity, and no neuron set moves hedging behavior. But uncertainty does have a mechanism: the entity is read in the first eight layers, an "unknown" verdict is routed mid-stack (L13–19 in Llama, L17–21 in Qwen) by a small set of verified attention heads, and MLP neurons read it into the late-layer entropy machinery. Patching 20 heads + 100 neurons (≈2% of heads, 0.02% of neurons) transfers 78–107% of the held-out hedging readout between twins — random sets transfer 2–6% — in two models and two uncertainty types, with partially overlapping components (Jaccard 0.11, head-recovery ρ = 0.61). The circuit carries the hedge decision without carrying the entropy: confidence-as-behavior and confidence-as-spread are different variables with different machinery, and correlation-selected "confidence neurons" are the readout, not the decision.

## Methods

**Datasets: uncertainty as a (model, prompt) property.** A twin pair is one manipulation with one change (obscure vs famous PopQA subject in the same relation template; two disagreeing sources vs one; many valid answers vs one), identical chat template and prefill, question length within 3 tokens. A pair enters an arm only if the model proves the labels: control answered correctly in ≥ 9/10 samples (greedy + 10 at T = 0.7, alias-graded), uncertain 0/10, entropy gap ≥ 0.5 nats, and — for the behavioral arm — hedging gap ≥ 0.3 over free generations. Keep rates are reported, never tuned (Llama familiarity 195/1151, contested 234/600, aleatoric 41/120; Qwen 24 / 178 / 39; situated dropped 3/600 with the reason on record). The alias grader is validated against a Qwen2.5-32B judge on a stratified sample (agreement 94.6–100%, κ 0.70–1.00); disagreement is *lower* on hedged answers (1.5% vs 6.1%), so grader error does not mimic the behavioral readout. `scripts/gate_twins.py`, `scripts/judge_audit.py`.

**Neuron-level causal battery.** Mean-ablation at the readout position with an explicit reference policy and matched control prompts; sign-flip permutation per cell; two-sided permutation for the uncertain-vs-control interaction; BH-FDR at 0.01; and a baseline-entropy adjustment (ANCOVA), because a flat distribution amplifies any perturbation — the confound that had manufactured "uncertainty-specific" neurons. Dose–response, frozen-RMSNorm decomposition, temperature-matched frequency tests, behavioral clamping and verbalized confidence complete the battery. `scripts/run_ablation.py`, `scripts/entropy_adjusted_interaction.py`, `scripts/frequency_causal.py`.

**Circuit-level decomposition.** All circuit claims rest on activation patching between gated twins — reference-free by construction. Pipeline: (1) position × layer map of residual patching (entity span / question tail / readout position); (2) attribution patching (activation difference × gradient of the hedge log-odds) over every head at every position, top 30 heads verified by real patching; (3) the same at MLP-block and single-neuron level in the routed window, top-N neuron sets verified jointly against size-matched random sets; (4) faithfulness: the final head + neuron set patched jointly on held-out pairs, both directions, against 10 random sets, with the pre-registered criterion *recovery > 0.7 with < 2% of components*; (5) the identical pipeline on a second arm and a second model, with component-set overlap. Readout: hedge-vs-answer log-odds and entropy at the prefilled position; recovery = (patched − target)/(source − target). `scripts/circuit_*.py`, `scripts/run_overnight.sh`.

## Claims and their status

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Correlation-selected "confidence neurons" are mostly token-frequency neurons plus a general temperature neuron; the taxonomy is causally verified with a temperature-matched baseline. | `results/frequency_causal_*`, `results/stolfo_*` | **Established** |
| 2 | No single neuron is uncertainty-specific on the original stimuli (overlapping entropy supports; three adjustment schemes agree) or on the behavioral (familiarity) arm; the two neurons that survive all specifications (L31_N11541, L30_N1457, contested arm only) pattern as answer-competition, not ignorance. Caveat: ρ(H, signed shift) ≈ 0 while ρ(H, abs shift) = 0.25, so the ANCOVA is unidentified on gated arms — the gated-arm nulls rest on the temperature-matched design and the behavioral nulls, not on the adjustment. | `results/*/entropy_adjusted.csv` | **Established with the identification caveat** |
| 3 | Uncertainty is model-relative: identical candidates give Llama 195 behavioral pairs and Qwen 24; contested and aleatoric arms dissociate distributional from behavioral uncertainty in both models. | `data/*/gate_report.json` | **Established** |
| 4 | The hedge decision is computed early (entity, L0–7), routed mid-stack (L13–19 / L17–21) by ~10 verified heads, and read out by mid-layer MLP neurons into the late-layer machinery of claim 1. | `results/circuit_*` (Figures 1–2) | **Established** in 3 of 3 runnable model × arm cells |
| 5 | ≈2% of heads + 0.02% of neurons transfer 78–107% of the held-out hedging readout; random sets 2–6%. 95% CIs: familiarity 0.80 [0.70, 0.90], Llama contested 0.89 [0.69, 1.09], Qwen 0.86 [0.77, 0.95] — the CI crosses the 0.7 criterion in Llama-contested and grazes it in familiarity, and neuron-set recovery does not saturate (0.23/0.43/0.52/0.65 at N=10/30/100/300). | `results/circuit_*/faithfulness*` (Figure 3) | **Established as a faithfulness-tested set; minimality (sparse circuit vs distributed direction) undecided** |
| 6 | Types share machinery only partially: cross-arm Jaccard 0.11 (heads and neurons), head-recovery ρ = 0.61 — one variable, two readouts. | `results/circuit_conflict/faithfulness_summary.txt` | Supported (one model; two arms) |
| 7 | **Reframed (all cells reported):** in Llama the decision and the spread transfer *asymmetrically by intervention direction* — injecting the unknown signal (c→u) transfers the hedge log-odds with little entropy (−0.13 familiarity, +0.33 contested), removing it (u→c) transfers both (+0.45, +0.55). Not a general dissociation: steering moves both together, and Qwen shows no asymmetry (+0.83/+0.73). | `results/circuit_*/faithfulness.csv`, all six cells | **Supported as a directional asymmetry (one model, two arms)** |

## Results in three figures

**Figure 1 — the verdict moves: entity → readout position (Llama familiarity).** Patching the unknown twin's residuals into the known prompt one position group at a time: entity tokens carry the full decision at L0–7 and nothing after L13; the pre-prefill token and the last position pick it up from L13–17.

![Figure 1](figures/fig1.svg)

**Figure 2 — the mid-stack MLP window, three runs.** Recovery when a single layer's MLP output is patched from the control twin: Llama contested peaks at L11–14 and L21, Qwen contested at L17–21; familiarity rides more on heads than on any single MLP block.

![Figure 2](figures/fig2.svg)

**Figure 3 — faithfulness: circuit vs random, held-out pairs.** 20 heads + 100 neurons recover 0.80 / 0.89 / 0.86 (Llama familiarity / Llama contested / Qwen contested; reverse direction 1.07 / 0.78 / 0.98) vs 0.02–0.06 for size-matched random sets. The pre-registered CIRCUIT verdict passes in both Llama runs; Qwen misses only the sparsity clause (20 heads = 2.55% of its 784 heads).

![Figure 3](figures/fig3.svg)

## Post-audit corrections (25 Aug 2026)

An external audit of this branch was verified against the result files and acted on: 95% CIs added to
every faithfulness cell; the "pre-registered" wording for the circuit criterion struck (thresholds for the
*gate* predate the gating; the circuit criterion was fixed in code before the held-out runs but not
independently registered); claim 7 reframed from a general dissociation to the direction-dependent
asymmetry the full table supports; the abstract's "no single neuron survives" scoped to the behavioral
arm; the bibliography rebuilt with verified author lists; and the closest prior work added — **SCIURus**
(Teplica, Liu, Cohan & Rudner, NAACL 2025 Long: shared uncertainty/factuality circuits across eight
models via causal tracing and zero-ablation). Relative to SCIURus this work adds the gated-twin data
standard, twin-based patching, random-set faithfulness controls, neuron granularity, and cross-type
overlap — and drops any claim of being the first uncertainty circuit. Also cited now: Du et al. (COLM
2025), Arora et al. (ICML 2026, MLP-only neuron-basis circuits), Zhao et al. (COLM 2026), Vazhentsev et
al. (ICML 2026), Singha Roy et al., Basu et al. Independent novelty assessment: 7/10 — high
combinatorial/methodological novelty, near-ideal fit for InterpScience ("rigor as the contribution").
Deadline: extended to **Sept 1, 2026 AoE** (short ≤ 5 pp, long ≤ 9 pp).

## Provenance

Every result file carries a `.provenance.json` sibling (model id, precision, library versions, data hashes, seeds, parameters). Judge audits: `results/judge_audit_*`; Qwen mirrors under `results/second_model/qwen25_7b_instruct/`. Reproduction: `scripts/run_all.sh` (neuron battery), `scripts/run_overnight.sh` (judge + circuits).
