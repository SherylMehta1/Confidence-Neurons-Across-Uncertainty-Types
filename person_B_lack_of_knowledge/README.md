# Person B — Lack of Knowledge

Dataset: UnknownBench (NEC subset) (see ../DATA_SOURCES.md)

## Workflow
1. Preprocess data -> `../data/lack_of_knowledge/prompts.jsonl` (common schema, see DATA_SOURCES.md)
2. Load `../candidate_neurons.json` (frozen, shared — do not edit)
3. Phase 3 (mechanism check): use `shared/logit_lens.py` on each candidate neuron, this category's prompts
4. Phase 4 (causal test): use `shared/ablation.py` on each candidate neuron, this category's prompts
5. Save results to `results/results.csv` using the exact schema in `../RESULTS_SCHEMA.md`

## Notes
This category should be the fastest to validate — cleanest ground truth. Do this one first.
QA evaluation, not next-token entropy analysis, so some adaptation is needed (see project chat notes).
