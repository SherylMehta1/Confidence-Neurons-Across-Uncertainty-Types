# Human validation of the hedge judges, v1 (Llama-3.1-8B-Instruct replies)

Branch arr-v2, 28 Nov 2026. This folder sets up the blind human check described in
`results/hedge_measure_v2/HUMAN_VALIDATION_README.md`, now with the corrected Gemma-2-27B labels.
Nothing in `results/hedge_measure_v2/` was changed. No one has scored anything yet.

## Labellers: read this first

- **Do not open either `KEY_*.csv` file until you have finished labelling.**
- Labeller 1 uses `sheet_labeller1.csv`. Labeller 2 uses `sheet_labeller2.csv`. Work independently and don't compare answers.
- Both sheets hold the same 120 replies (30 for each popularity x knowledge cell). Sheet 1 is in the original order (shuffled with seed 0). Sheet 2 is the same rows shuffled again with seed 1, so the two labellers see them in different orders.
- In the column `label_DONT_KNOW_or_CLARIFY_or_ANSWER`, write exactly one of DONT_KNOW, CLARIFY or ANSWER, following the definitions in `HUMAN_VALIDATION_README.md`. The `note` column is optional free text and the scorer ignores it.

## Files

| File | What it is |
|---|---|
| `sheet_labeller1.csv` | `human_validation_sheet.csv` unchanged, plus an empty `note` column |
| `sheet_labeller2.csv` | the same 120 rows in a new order (`random.Random(1).shuffle`), plus a `note` column |
| `KEY_original.csv` | byte-identical copy of `results/hedge_measure_v2/human_validation_KEY_do_not_open_before_labelling.csv` (stored judge labels) |
| `KEY_corrected.csv` | same ids, row order and columns. `gemma27b` replaced by the re-judged Gemma-2-27B label. `qwen32b` and every other column unchanged |

## How the corrected key was built

The stored Gemma-2-27B labels were produced with the faulty `sdpa` attention (see `results/rejudge_gemma/README.md`).
The corrected label for each row comes from `results/rejudge_gemma/rejudge_gemma27b.jsonl` (written by `scripts/rejudge_run.py`, vLLM 0.34.0),
matched on `file == results/hedge_measure_v2/judge_gemma27b_llama.jsonl` and `id == [key, s]`.

Checks run:
- The stored `gemma27b` and `qwen32b` columns of the original key match `judge_gemma27b_llama.jsonl` and `judge_qwen32b_llama.jsonl` on 120 of 120 rows. They do not match the `llama_s1` or `llama_adj` files, so the sample comes from the plain-prompt Llama files.
- All 120 rows have a re-judged label. The question and reply text the re-judge scored are identical to the sheet's on 120 of 120.
- 25 of the 120 `gemma27b` labels change.

## Judge agreement

After the fix, the two judges (Qwen2.5-32B and Gemma-2-27B) agree with **kappa 0.91** on Llama's replies, up from the old **0.62**.
That figure is for the full Llama judge file (1,161 replies, `results/rejudge_gemma/labels_by_file.csv`).
On the 120 rows here the same comparison gives 0.60 before the fix and 0.86 after, because a sample stratified by cell is not the same as the full file.

## Scoring (not yet run)

`score_human.py` is a copy of `scripts/hedge_measure_v2_human_score.py` (left unchanged) with the same metric definitions. It takes the key as an argument and needs both labellers' files:

    .venv/bin/python results/human_validation_v1/score_human.py --key results/human_validation_v1/KEY_corrected.csv <labeller1.csv> <labeller2.csv>

Run it once with `KEY_corrected.csv` and once with `KEY_original.csv`, and report both whatever they show.
It joins on id, refuses to score unless both files have exactly the key's 120 ids, no duplicates, and a valid label on every row (it reports only the number of bad rows), and prints aggregates only.
Output: per-labeller kappa (95% bootstrap CI, 10,000 draws, seed 0), agreement, precision and recall for the both-judges rule, qwen32b and gemma27b. It also prints inter-human kappa and agreement, the both-judges kappa on replies where the labellers agree (descriptive), disagreements by cell, CLARIFY and note counts, and per-cell DONT_KNOW rates (descriptive).
All kappas on the 120 replies share one set of bootstrap resamples, so the intervals don't depend on file order.
A duplicated id counts as one bad row for each extra copy.

## What was tested (28 Nov 2026, synthetic data only)

The tests are in `test_score_human.py` (synthetic data only). Re-run them from the repo root with `python results/human_validation_v1/test_score_human.py`; this needs scikit-learn and writes its fake files to a system temp folder.
No real key or label file was opened or scored. The tests ran in a temporary folder outside the repo, with scikit-learn 1.9.1, pandas 3.0.5 and numpy 2.5.1.
The fake key had 120 ids in the four cells, with random judge verdicts. The two fake label files were the both-judges rule with exactly 12 and exactly 20 replies flipped, rows shuffled, and labels in mixed case with stray spaces. 26 of 26 checks passed:

- Every kappa (3 judges x 2 labellers, inter-human, the agreed-replies set) equals `sklearn.metrics.cohen_kappa_score` and a hand calculation from the 2x2 table to 1e-12. The printed values equal the sklearn values rounded to 2 decimals.
- Two runs with the same seed give identical intervals. The resamples are 10,000 x 120, reproducible with seed 0, and different under another seed.
- A blank label, the typo `DONT KNOW`, a missing id and a duplicate id each stop the scorer with exit code 1 and `ERROR: labeller <n>: 1 bad row(s)`, in either file position, with nothing scored and no id printed.
- Swapping the two label files swaps both columns exactly (kappa, CI, agreement, precision, recall, human DONT_KNOW rate) and leaves inter-human kappa and its interval unchanged.
- No id appears anywhere in the output.

## Checksums

`CHECKSUMS.txt` (SHA-256) covers both sheets, both keys, `score_human.py`, `test_score_human.py`, `plan.md` and this README. `plan.md` also lists the hashes fixed before scoring. Verify with `shasum -a 256 -c CHECKSUMS.txt` from inside this folder.
