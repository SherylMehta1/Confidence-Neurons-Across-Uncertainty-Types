# Uncertainty Has a Circuit, Not a Neuron

Mechanistic study of uncertainty in `Llama-3.1-8B-Instruct` and `Qwen2.5-7B-Instruct`. The project
began as *Confidence Neurons Across Uncertainty Types* (do entropy-regulating neurons exist here, and
are they shared across uncertainty types?); after two audit-and-rebuild rounds it ends somewhere
stronger: **no single neuron is "for uncertainty" — but a sparse, patching-verified circuit of
attention heads and MLP neurons is, in two models and two uncertainty types.**

**Start here:**

| Document | What it holds |
|---|---|
| [RESULTS.md](RESULTS.md) | The paper skeleton: abstract, methods, seven claims with status, three figures |
| [UNCERTAINTY_DEFINITION.md](UNCERTAINTY_DEFINITION.md) | The dataset standard: model-relative uncertainty, gated twins, keep rates, grader audit |
| [paper/main.tex](paper/main.tex) | Workshop paper draft (NeurIPS 2026 Interpretability-as-a-Science) |
| `figures/` | Result figures (SVG; PDF versions in `paper/figures/`) |

## Headline results

1. **No confidence neurons.** Activation–entropy correlation selects token-frequency neurons and a
   general temperature neuron (causally verified with a temperature-matched baseline). After adjusting
   for baseline entropy — a flat distribution amplifies any perturbation — **zero** neurons survive an
   uncertainty-specificity test on any stimulus set, original or gated, in either model; no neuron set
   moves hedging (77% vs 0.8% clean gap; clamps shift it ≤ 2 points) or verbalized confidence.
2. **Uncertainty is a (model, prompt) property.** The same 1,151 PopQA candidates yield 195
   correctness-gated behavioral twin pairs in Llama and 24 in Qwen; contested-source and aleatoric
   arms dissociate distributional from behavioral uncertainty in both models. The alias grader is
   validated against a Qwen2.5-32B judge (agreement 94.6–100%, κ 0.70–1.00, disagreement *lower* on
   hedged answers).
3. **The circuit.** The entity is read at L0–7; an "unknown" verdict is routed mid-stack (Llama
   L13–19, Qwen L17–21) by ~10 verified heads; mid-layer MLP neurons read it into the late-layer
   entropy machinery. Patching **20 heads + 100 neurons** (≈2% of heads, 0.02% of neurons) transfers
   **78–107%** of the held-out hedging readout between twins — random sets transfer 2–6% — in three of
   three runnable model × arm cells. Cross-type overlap: Jaccard 0.11, head-recovery ρ = 0.61.
4. **Decision ≠ spread.** The circuit carries the hedge decision (0.80 recovery) without the entropy
   (≈ 0) — behavioral and distributional confidence are different variables with different machinery.
   (Newest claim; replicate before headlining.)

Every claim's evidence pointer and status: [RESULTS.md](RESULTS.md). Every result file has a
`.provenance.json` sidecar (model id, precision, versions, data hashes, seeds).

## Reproduction

One ≥ 24 GB GPU; no HF token needed with `CN_MODEL_ID=unsloth/Meta-Llama-3.1-8B-Instruct`.

```bash
python -m pytest tests -q                                   # 25 tiny-model tests, CPU, ~1 min
CN_MODEL_ID=... bash scripts/run_all.sh                     # neuron battery (detection -> ablation -> stats)
python scripts/build_familiarity_twins.py                   # gate the familiarity arm (PopQA)
python scripts/make_candidates.py conflict && \
  python scripts/gate_twins.py --candidates data/conflict/candidates.jsonl --category conflict --no-require-hedge --overwrite
MODEL=llama bash scripts/run_overnight.sh                   # judge audit + circuit pipeline (per arm)
```

Key scripts: `scripts/gate_twins.py` (model-relative gate), `scripts/judge_audit.py` (32B judge vs
alias grader), `scripts/circuit_{position_map,heads,mlp,faithfulness}.py` (position → head → neuron →
faithfulness), `scripts/entropy_adjusted_interaction.py` (the confound adjustment),
`scripts/frequency_causal.py` (temperature-matched frequency-neuron test).

## Repo structure

```
shared/                        model/tokenizer utils, detection, ablation, logit lens, provenance
scripts/                       one entry point per stage (see Reproduction); circuit_* = the circuit pipeline
data/<arm>/                    gated twin sets + gate_report.json + provenance (familiarity, conflict,
                               aleatoric, situated, and the original three categories)
results/                       all Llama outputs (ablation_*, circuit_*, judge_audit_*, frequency_*, ...)
results/second_model/          Qwen2.5-7B-Instruct mirrors (data + results + logs); llama31_base
paper/                         main.tex, references.bib, figures/ (PDF)
figures/                       SVG figures used by RESULTS.md
analysis/                      weight-based candidate criteria (Stolfo; token-frequency neurons)
person_{A,B,C}_*/              original per-category data builders (kept as the live builders)
archive/                       superseded material, never run at HEAD
tests/                         tiny-model smoke tests for every pipeline component
```

## Original framing and the audit trail (kept for the record)

The project's original hypotheses (H1 shared mechanism / H2 specialized / H3 partial overlap / H4 no
robust mechanism) resolved to **H4 at the single-neuron level** — and the circuit results above show
what H4 leaves standing: a distributed mechanism no per-neuron test could see.

Two earlier analysis rounds are preserved in full: the v3 NF4 analysis (superseded; measurement bugs
documented) and the first bf16 rerun with matched controls, whose per-neuron findings
(L31_N11541's raw interaction and its dissolution under the entropy adjustment; L31_N2477 as a
general temperature neuron; 0/55 passing Stolfo weight criteria under a norm-matched null; the
frequency-neuron taxonomy) are summarised in the tables of `results/rerun_analysis/SUMMARY.md` and in
the interactive report. Data sources, schema and setup: `DATA_SOURCES.md`, `RESULTS_SCHEMA.md`,
`SETUP.md`, `REFERENCES.md`; run order for the legacy battery: `PHASE3_GUIDE.md`.

## References

Core: Stolfo et al. 2024 (entropy neurons); Ferrando et al. 2025 (entity recognition); Ji et al. 2025
(hedging direction); Kadavath et al. 2022 & Gekhman et al. 2024 (sampled correctness);
arXiv:2604.01457 (confidence-mover circuits — closest prior work, see paper §Related). Full list:
`REFERENCES.md` and `paper/references.bib`.
