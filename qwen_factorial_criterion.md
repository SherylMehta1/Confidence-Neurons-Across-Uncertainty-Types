# Qwen2.5-7B-Instruct popularity x knowledge factorial: read-off criterion, stated before the run

Written 6 Oct 2026, BEFORE any Qwen item of this run was scored. Like `l5_behavioral_criterion.md`, this is a validity check with a written bar, not a full pre-registration (no power analysis). It exists so that the Qwen result cannot be read in whichever way suits the UniReps headline.

## What is run

- `scripts/build_familiarity_twins.py --model-id Qwen/Qwen2.5-7B-Instruct --n-per-relation 1000 --seed 42 --out-dir data/familiarity_v3_qwen`, with every other argument at the values recorded in `data/familiarity_v3/prompts.provenance.json` for the Llama run (lo/hi quantile 0.1/0.9, 10 samples, 20 free generations, bf16). Same PopQA dedup, same prompts, Qwen's own chat template.
- `scripts/build_frequency_knowledge_factorial.py --source data/familiarity_v3_qwen/all_measured.jsonl --prefix factorial_v3_qwen`.
- A forward-only pass adding the first-token hedge-vs-answer log-odds (`direction_bridge.HEDGE_TOKENS` vs `ANSWER_TOKENS`, at the "The answer is" prefill position) to every scored item. This is the readout all circuit numbers already use.

## Primary analysis: the 2 x 2 of cell means

Over the scored pool (HighlyKnown and Unknown only; Maybe excluded, as in the Llama factorial), cells are popularity arm (obscure = bottom decile, popular = top decile) x knowledge (HighlyKnown vs Unknown). For a readout Y:

- popularity effect on Y = mean over knowledge levels of [Y(obscure) - Y(popular)]
- knowledge effect on Y = mean over popularity levels of [Y(Unknown) - Y(HighlyKnown)]

95% CIs by item-level bootstrap within cells (10,000 resamples, seed 0).

**The dissociation replicates in Qwen if BOTH hold:**
1. for hedging, popularity effect minus knowledge effect > 0, CI excluding 0; and
2. for first-token entropy, knowledge effect minus popularity effect > 0, CI excluding 0.

Llama reference values (from `results/unireps_checks/check1_items.csv`): hedging popularity ~0.63 vs knowledge ~0.19; entropy knowledge ~2.14 vs popularity ~0.51.

## Which hedging readout governs

- **H-free:** free-generation hedge rate, as in Llama. Governs **unless** Qwen's overall hedge rate across the scored pool is below 0.05. In that case H-free is declared uninformative (a floor effect, not a null result), and:
- **H-logodds:** first-token hedge-vs-answer log-odds governs instead.

Both are reported either way. The governing readout is chosen by the pool-level hedge rate, which does not depend on the comparison being tested.

## Outcomes and what each licenses

- **REPLICATES** (1 and 2 both hold on the governing readout): the paper may say "two language models" for the familiarity claim.
- **PARTIAL** (only one holds, or the CIs straddle 0): the paper says the dissociation holds in Llama and is partial in Qwen, and reports the numbers.
- **FAILS** (the hedging difference is <= 0 with CI excluding 0, i.e. Qwen hedges on knowledge more than on popularity): reported as a model difference. The title stays Llama-scoped.

## Secondary, descriptive only

- The four pairwise factorial contrasts (gate_report.json per contrast), as for Llama.
- The name-surface checks from `scripts/unireps_confound_checks.py`, run on Qwen's pool with Qwen's tokenizer.
- Item-level agreement between Llama and Qwen on shared items (same subject and relation): Spearman correlation of hedge rates, and of entropies, overall and split by whether the two models agree on knowledge.

## Not to be done after seeing the result

Changing the 0.05 floor, swapping the governing readout, dropping relations, changing the gate or sampling settings, or rerunning with a different seed to chase a different answer.
