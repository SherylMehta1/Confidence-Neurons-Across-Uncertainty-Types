# Person C — Contradictory Context

Dataset: Tighidet et al. knowledge-probing framework (see ../DATA_SOURCES.md)

## Workflow
1. Preprocess data -> `../data/contradictory_context/prompts.jsonl` (common schema, see DATA_SOURCES.md)
2. Load `../candidate_neurons.json` (frozen, shared — do not edit)
3. Phase 3 (mechanism check): use `shared/logit_lens.py` on each candidate neuron, this category's prompts
4. Phase 4 (causal test): use `shared/ablation.py` on each candidate neuron, this category's prompts
5. Save results to `results/results.csv` using the exact schema in `../RESULTS_SCHEMA.md`

## Notes
Watch for instruct-model meta-commentary contaminating entropy measurements (the model explicitly noting the contradiction rather than showing it via token distribution). Do this category last, after the pipeline is proven on the other two.
QA evaluation, not next-token entropy analysis, so some adaptation is needed (see project chat notes).
