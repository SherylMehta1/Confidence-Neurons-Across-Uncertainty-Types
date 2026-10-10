# Saying "I don't know": popularity versus knowledge in language models

When a language model answers a factual question with "I don't know", is it reporting that it lacks the answer? This repository holds the data, code and results behind a UniReps 2026 extended abstract that tests this by crossing **entity popularity** (Wikipedia page views in PopQA) with **knowledge** (whether the model answers the fact correctly when made to answer).

> **Status (10 Oct 2026).** The paper draft is in [`paper_2.0/`](paper_2.0/); read [`paper_2.0/main.pdf`](paper_2.0/main.pdf) and see [`paper_2.0/REVIEWING.md`](paper_2.0/REVIEWING.md) for how to comment. What changed in response to review, the new experiments and how each claim moved: [`paper_2.0/WHAT_CHANGED.md`](paper_2.0/WHAT_CHANGED.md); claim status: [`paper_2.0/CLAIMS.md`](paper_2.0/CLAIMS.md); full record: [`paper_2.0/LAB_LOG.md`](paper_2.0/LAB_LOG.md). Submission: UniReps workshop at NeurIPS 2026, OpenReview deadline 11 Oct 2026, 12:00 UTC.

## Findings

Numbers are point estimates with 95% bootstrap intervals; every one is generated from a file under `results/` (see `paper_2.0/numbers.tex`).

1. **In Llama-3.1-8B-Instruct, abstention tracks popularity more than knowledge.** For facts it answers correctly at least nine times in ten, Llama says "I don't know" in 0.43 [0.35, 0.51] of replies about obscure entities and 0.00 about popular ones, at similar answer entropy (0.95 vs 0.86 nats). Popularity minus knowledge on abstention: +0.30 [+0.20, +0.40].
2. **The result survives the checks fixed in advance:** knowledge relabelled from a second prompt (+0.32), unknown facts regraded by meaning with Llama-3.3-70B (+0.23), a third judge (+0.44), counting only replies that name no candidate answer (+0.43), and an entity-recognition covariate (the effect shrinks by 5%). Post hoc, a namesake-count covariate leaves it at +0.34; ambiguity runs the wrong way, since popular names have more namesakes.
3. **Answer uncertainty follows knowledge instead.** First-token entropy under a forced-answer prompt rises 2.14 nats when Llama lacks the fact and 0.51 when the entity is obscure. Knowledge dominates in all four families (Llama, Qwen, Gemma, Mistral), partly by construction, and in Llama it holds under independent labels.
4. **Gemma shows the same preference when it is allowed to abstain.** Under the plain prompt, Qwen, Gemma and Mistral rarely say "I don't know" and the judges disagree, so their preference cannot be estimated. With "If you do not know the answer, say that you do not know" appended, popularity minus knowledge is +0.47 for Llama and +0.30 for Gemma.
5. **Mechanistic tests do not isolate a familiarity cause.** In Llama, steering along an obscure-minus-popular direction adds abstention only through its overlap with an abstention direction, even when that direction is fitted on obscure questions alone (+0.01 [+0.00, +0.03] left after removal). In Jacobian-lens readouts at the forced-answer position, the hedge-token term responds to knowledge at least as much as to popularity in four base models, and the instruct partners lean further towards knowledge.

Negative and superseded results are reported as such: the human-memory depth prediction was not supported, name surprisal does not stand in for popularity, and the earlier circuit's injection test missed its behavioural bar (see "Earlier work" below).

## Where things are

| Path | What |
|---|---|
| [`paper_2.0/`](paper_2.0/) | the UniReps paper: `main.tex`, `main.pdf`, generated `numbers.tex`, figure scripts, pre-registration notes (`PREREG/`), lab log, simulated review rounds |
| [`results/round3/README.md`](results/round3/README.md) | index of the pre-registered experiments E1-E10: criterion, script, outputs, verdict |
| [`results/OVERVIEW_2026-10-07.md`](results/OVERVIEW_2026-10-07.md) | overview of the 5-7 Oct experiments (four-family factorial, lens, steering, judges) |
| `results/hedge_measure_v2/` | free replies and three-judge labels for the four families (plus exploratory OLMo-2) |
| `results/factorial_4models/` | popularity x knowledge effects on entropy and hedge log-odds, four families |
| `results/jlens_cascade/`, `results/round3/e3/` | Jacobian-lens trajectories and window effects, base and instruct models |
| `data/familiarity_v3*/` | knowledge-labelled PopQA question pools, one per model |
| `*_criterion.md` (repo root) | analysis plans written before the four-family, judge, lens and steering runs; their SHA-256 prefixes are cited in the paper, so they are stored byte-exact |
| `scripts/` | one entry point per stage (below); `shared/` holds model loading, prompts and provenance |

Every result file has a `.provenance.json` sidecar (model id, precision, library versions, data hashes, seeds) unless it is a plain-text summary.

## Reproduce

One GPU with at least 24 GB runs the 7-9B models; the Llama-3.3-70B judge (4-bit) and the 27-32B judges need about 45-65 GB. No Hugging Face token is needed with the `unsloth/` model mirrors.

```bash
pip install -r requirements.txt

# 1. knowledge-labelled question pool for a model (greedy + 10 samples under "The answer is")
python scripts/build_familiarity_twins.py --model-id unsloth/Meta-Llama-3.1-8B-Instruct \
    --n-per-relation 1000 --seed 42 --out-dir data/familiarity_v3
# 2. free replies and judges (up to 100 questions per popularity x knowledge cell)
python scripts/hedge_measure_v2.py gen   --model-id unsloth/Meta-Llama-3.1-8B-Instruct --tag llama
python scripts/hedge_measure_v2.py judge --judge-id Qwen/Qwen2.5-32B-Instruct --judge-tag qwen32b --tags llama
python scripts/hedge_measure_v2_analysis.py          # two-judge read-off -> results/hedge_measure_v2/summary.txt
# 3. round-3 experiments (each script's docstring gives its exact commands)
python scripts/e7_audit.py gen && python scripts/e7_audit.py judge
python scripts/round3_analysis.py e7                 # read-off for any of e1 ... e10
# 4. the paper
cd paper_2.0 && python make_numbers.py && latexmk -pdf main.tex
```

For a new model, `scripts/explore_make_items.py --pool data/familiarity_v3_<tag> --tag <tag>` draws its question set. The lens runs use `scripts/jlens_cascade.py` and `scripts/e3_lens_split.py` with the prefitted lenses from `neuronpedia/jacobian-lens` (`pip install git+https://github.com/anthropics/jacobian-lens`). Tests: `python -m pytest tests -q` (tiny models, CPU).

## Pre-registration

Each confirmatory analysis has a criterion note, written before its run, that fixes the measurement, the test and the wording each outcome licenses. Notes for the round-3 experiments are in `paper_2.0/PREREG/` (hashes in `HASHES_round3.txt`); older ones are the `*_criterion.md` files at the root. `.gitattributes` marks them `-text` so line-ending conversion cannot change their bytes. Analyses decided after seeing data are labelled post hoc in the paper and in `paper_2.0/LAB_LOG.md`.

## Earlier work in this repository

The project began as *Confidence Neurons Across Uncertainty Types* and then became a circuit study of hedging in Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. That work stays here and is cited in the paper's appendix:

- **No confidence neurons.** Neurons selected by their correlation with entropy behave as token-frequency and temperature neurons; none passes an uncertainty-specificity test after adjusting for baseline entropy.
- **A sparse circuit.** 20 attention heads and 100 MLP neurons found by activation patching reproduce 0.83 [0.78, 0.87] of the first-token abstention log-odds difference between matched obscure and popular questions (1.03 in the other direction). In generation, removing its state removes abstention in 0.98 of cases but injecting it produces abstention in only 0.40, short of the bar fixed in advance. The earlier "decision, not spread" headline was retired on that result.

The earlier InterpScience workshop draft is in [`paper/`](paper/), and [`RESULTS.md`](RESULTS.md) is its claims summary; both describe the circuit framing, not the current paper. Data sources, schema and setup for the original neuron battery: `DATA_SOURCES.md`, `RESULTS_SCHEMA.md`, `SETUP.md`, `PHASE3_GUIDE.md`; superseded material is in `archive/`.

## License

See [LICENSE](LICENSE).
