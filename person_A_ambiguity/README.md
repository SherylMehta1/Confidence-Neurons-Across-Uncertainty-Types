# Person A — Ambiguity

Dataset: AmbigQA / AmbigNQ (see ../DATA_SOURCES.md)

## Workflow
1. Preprocess data -> `../data/ambiguity/prompts.jsonl` (common schema, see DATA_SOURCES.md)
2. Load `../candidate_neurons.json` (frozen, shared — do not edit)
3. Phase 3 (mechanism check): use `shared/logit_lens.py` on each candidate neuron, this category's prompts
4. Phase 4 (causal test): use `shared/ablation.py` on each candidate neuron, this category's prompts
5. Save results to `results/results.csv` using the exact schema in `../RESULTS_SCHEMA.md`

## Notes
Verify entropy spikes at the TOKEN level, not just across full answers — AmbigQA was built for
QA evaluation, not next-token entropy analysis, so some adaptation is needed (see project chat notes).
