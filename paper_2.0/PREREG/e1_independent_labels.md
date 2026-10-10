# E1: knowledge labels from an independent prompt, and free-prompt readouts (Llama): criterion, written before the run

Written 2026-10-10, before any E1 run, in response to review rounds 1-2:
- red-team P1/P2 (circularity; "known" under forcing is not "stored");
- methods F1;
- chair J4 and items 9(a)/(b).

Box A. Model: unsloth/Meta-Llama-3.1-8B-Instruct, bf16.

## Items

All Llama pool items in `data/familiarity_v3/all_measured.jsonl` with a subject, deduplicated by (prop, subj), both arms. There are 2,673 items. The 387 two-judge items (`results/hedge_measure_v2/item_rates.csv`, model llama) are a subset.

## New measurements

1. **Paraphrase labels.** Each relation gets one fixed paraphrased question template, listed in `scripts/e1_relabel.py` (PARA). Every template keeps the subject and changes the wording of the original PopQA question.
   - Prompt: chat template with the paraphrased question, plus the assistant prefix "Answer:". Both the wording and the prefix differ from the original "The answer is".
   - Sampling: a greedy answer and 10 samples (T = 0.7, 12 new tokens), graded with the repo's alias grader (`is_correct`), with fresh samples.
   - Label: HighlyKnown if the greedy answer and at least 9 of 10 samples are correct; Unknown if 0 of 10 are correct; otherwise Maybe.
   - Also recorded: first-token entropy at this prompt (`entropy_para`).
2. **Free-prompt first-token readouts.** At the free prompt (chat template, no prefix), record:
   - the next-token entropy;
   - the log-odds that the first token is "I" (Llama's abstentions open with "I") against all other tokens.

   These are descriptive only.

## Tests and what each outcome licenses

- **T1 (main).** Rebuild the two-judge 2x2 on the 387 items, using paraphrase labels for knowledge (items labelled Maybe under the paraphrase are dropped).
  - Effects and CIs as in the paper: 10,000 within-cell item bootstraps, seed 0.
  - **SURVIVES** if popularity − knowledge > 0 with the CI excluding 0 AND the obscure-known cell abstains in ≥ 0.25 of replies. Licenses: "the abstention result holds with knowledge labels from an independent prompt".
  - **WEAKENED** if the difference is > 0 but the CI includes 0. **FAILS** if the difference is ≤ 0. Either is reported as such.
  - If a cell has fewer than 20 items, the test is reported as underpowered and is descriptive only.
- **T2 (circularity).** Entropy measured on the ORIGINAL forced-answer prompt, crossed with knowledge from PARAPHRASE labels, on all items labelled under both.
  - **NOT CIRCULAR** if knowledge effect − popularity effect > 0 with the CI excluding 0. Licenses: "entropy follows knowledge also when knowledge is labelled from a different prompt and different samples".
  - Otherwise the entropy half is reported as tied to its own labels.
- **T3 (descriptive).** Agreement between the original and paraphrase labels, overall and per cell (a confusion table).
- **T4 (descriptive).** Free-prompt first-token entropy and "I" log-odds by cell, with popularity and knowledge effects.
  - If the free-prompt entropy follows popularity more than knowledge (difference CI excluding 0), we say so: the dissociation is then partly a prompt effect (clarity reviewer Q1).

## Not allowed after seeing data

Changing templates, prefixes, the grader, the label rule, the 0.25 bar, the item set or the bootstrap; dropping T2 or T4.
