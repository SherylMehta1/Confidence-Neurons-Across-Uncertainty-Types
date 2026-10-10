# Cross-model agreement of the abstention readout vs answer entropy (plan written 2026-10-10, before this run)

**Status: exploratory.** Point estimates of the item-level Spearman correlations were already printed, descriptively, in repo `results/factorial_4models/summary.txt` (CROSS-MODEL block, 7 Oct). This run adds intervals, a null and one contrast. It is labelled post hoc in the paper. Requested by review round 1 (chair item I23; framing referee F7), [user] "run 3", 2026-10-10.

**Question.** The paper's closing prediction: if abstention follows familiarity (exposure shared across model families) while answer entropy follows each model's own knowledge, then two models' abstention readouts should agree on the same questions even where the models disagree about knowing the fact. Their entropies should not agree there.

**Data (read-only).** Repo `results/factorial_4models/cross_model_items.csv` (Llama vs Qwen), `cross_model_items_gemma.csv`, `cross_model_items_mistral.csv`. These contain the questions labelled known/unknown by both models, with each model's first-token hedge-vs-answer log-odds and first-token entropy at the forced-answer prompt, and each model's free-generation regex hedge rate.

**Readouts.**
- Primary abstention readout: `hedge_logodds`. It is measured identically in all four families and does not depend on the regex, whose recall is low outside Llama.
- Secondary abstention readout: `hedge_rate` (regex, free generation). Reported, with its known low recall.
- Contrast: `entropy`.

**Subsets.** Questions where the pair agrees on knowledge, and questions where it differs (one known, one unknown).

**Statistics.**
- Spearman ρ between the two models' values, per pair, readout and subset; 95% percentile CI from 10,000 item bootstraps (seed 0).
- Main contrast on knowledge-differs questions: Δ = ρ(hedge_logodds) − ρ(entropy), with a paired bootstrap CI.
- Null for each ρ: 2,000 permutations of model B's values within relation (seed 0).
- Secondary, descriptive: ρ within each popularity arm. This shows whether agreement goes beyond the shared obscure/popular split.

**Reading, fixed now.**
- **The prediction holds for a pair if, on knowledge-differs questions, ρ(hedge_logodds) > 0 with CI excluding 0 and Δ > 0 with CI excluding 0.**
- Otherwise it is "not supported" for that pair.
- Results are reported for all three pairs whatever they show. The Gemma pair has only 60 knowledge-differs questions, so its interval will be wide.

**Caveat to state.** At the forced-answer position, Llama's hedge readout responds to knowledge as much as to popularity. So this readout is a weaker abstention proxy for Llama than its free-generation abstention.
