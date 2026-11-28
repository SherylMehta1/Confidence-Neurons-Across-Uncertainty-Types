# Human check of the hedge judges, v1: plan, written before any labelling or scoring

Recorded 28 Nov 2026 on arr-v2, before anyone has labelled a sheet and before the scorer has been run on real data.
The files below are fixed by their SHA-256. If any of them changes after this point, the change and the reason must be recorded here.

## Question and scope
Does the paper's hedge measure (the both-judges rule: Qwen2.5-32B and Gemma-2-27B both say DONT_KNOW) agree with people
on whether a reply says the assistant does not know? Scope: Llama-3.1-8B-Instruct replies to the plain prompt only.
The check says nothing about other models, other prompts or the permissive prompt.

## Sample and sheets
- 120 replies from `results/hedge_measure_v2/human_validation_sheet.csv`, 30 for each popularity x knowledge cell
  (control/HighlyKnown, control/Unknown, uncertain/HighlyKnown, uncertain/Unknown), shuffled with seed 0.
- `sheet_labeller1.csv`: that sheet unchanged, plus an empty `note` column.
- `sheet_labeller2.csv`: the same 120 rows shuffled again with seed 1, plus a `note` column.
- The original sheet is not edited.

## Labels
In the column `label_DONT_KNOW_or_CLARIFY_or_ANSWER`, write exactly one of:
- DONT_KNOW: the reply says or clearly implies the assistant does not know, cannot find or verify the information, or does not recognise the subject (even if it then guesses);
- CLARIFY: it does not answer and instead asks which entity is meant or says the question is ambiguous, without saying it does not know;
- ANSWER: it commits to a specific answer (a mild "probably" is still ANSWER).

Labeller rules:
- Two people label independently. They do not discuss replies or compare labels until both sheets are finished.
- Neither opens `KEY_original.csv`, `KEY_corrected.csv` or the key in `results/hedge_measure_v2/` before finishing.
- Every row gets one label. Case and surrounding spaces are ignored. Anything else (blank, typo) stops the scorer.
- The `note` column is optional free text. It is counted, never read by the scorer, and does not change the label.

## Scoring
- `score_human.py --key <key> <labeller1.csv> <labeller2.csv>`, run once with `KEY_corrected.csv` (main analysis)
  and once with `KEY_original.csv` (reported alongside).
- Outcome: DONT_KNOW against (ANSWER or CLARIFY). Files are joined on id.
- Per labeller: kappa, percent agreement, precision, recall and human DONT_KNOW rate for the both-judges rule,
  qwen32b and gemma27b. Inter-human kappa and agreement on all 120 replies.
- The scorer prints aggregates only and refuses to run unless both files have exactly the key's 120 ids,
  no duplicates and a valid label on every row.

## Reference rule
The both-judges rule is scored against each labeller separately, and both kappas are
reported side by side. The paper says the measure agrees with a person at a given level
only if both kappas reach it. A second table scores the rule on the replies where the two
labellers agree (agreement means DONT_KNOW against not, as in the scorer; exact agreement
on all three labels is also printed). That table is descriptive only, because dropping
disagreements makes agreement look higher. Replies where the labellers disagree are
counted, reported and never settled by discussion or by the judges.

## Decision rules
1. The paper's bar is kappa 0.60. If the both-judges rule reaches 0.60 or more against each
   labeller, report that the measure agrees with a person at that level, with the intervals
   and n = 120.
2. If it falls below 0.60 against either labeller, report it as found, say which cell drives
   the disagreement, and weaken the wording about the measure. Do not change the judges, the
   prompt or the both-judges rule afterwards.
3. Report inter-human kappa on all 120 replies whatever it shows. If it is low, the criterion
   is ambiguous and the paper says so. If the rule's two kappas against the labellers differ
   by more than 0.10, report both and say that agreement depends on the person.
4. Per-cell numbers are descriptive only, about 30 replies per cell.
5. In every case the result goes in the appendix, whatever it shows.

## Intervals
95% bootstrap over replies, 10,000 draws, seed 0, the same resamples for every kappa.

## Known before labelling
- The judges' kappa is 0.91 on the full Llama judge file (1,161 replies) after the Gemma fix,
  up from 0.62. On these 120 rows it is 0.86 after the fix, up from 0.60.
- 25 of the 120 Gemma-2-27B verdicts differ between the original and corrected keys.
- No label of any kind exists at the time of this commit.

## Changes
Any change before the key is opened is written as a dated amendment in README.md. Nothing is
changed after the key is opened.

## Fixed before scoring

| Choice | Value |
|---|---|
| Replies | 120 Llama-3.1-8B-Instruct replies, plain prompt, 30 per popularity x knowledge cell (from `results/hedge_measure_v2/`) |
| Labellers | Two people, working independently. Labeller 1 uses `sheet_labeller1.csv`, labeller 2 uses `sheet_labeller2.csv` (same rows, different order). Neither opens a key before finishing |
| Labels | DONT_KNOW, CLARIFY or ANSWER, as defined in `results/hedge_measure_v2/HUMAN_VALIDATION_README.md` |
| Outcome | DONT_KNOW against (ANSWER or CLARIFY) |
| Judges | Qwen2.5-32B (`qwen32b`) and Gemma-2-27B (`gemma27b`). The both-judges rule is DONT_KNOW from both |
| Keys | `KEY_corrected.csv` (re-judged Gemma-2-27B labels, the main analysis) and `KEY_original.csv` (stored labels, reported alongside) |
| Scorer | `score_human.py`, run once with each key. Original script `scripts/hedge_measure_v2_human_score.py` left unchanged |
| Metrics | For each labeller separately: Cohen's kappa, percent agreement, precision, recall and human DONT_KNOW rate, for the both-judges rule, qwen32b and gemma27b. Inter-human kappa and agreement on all 120 |
| Intervals | 95% percentile bootstrap resampling replies, 10,000 draws, seed 0. All kappas on the 120 replies share the same resamples |
| Descriptive only | Both-judges kappa on the replies where the labellers agree, per-cell DONT_KNOW rates, disagreements by cell, CLARIFY and note counts |
| Validity | The scorer refuses to run unless both files have exactly the key's 120 ids, no duplicates, and a valid label on every row |
| Reporting | All results are reported in the paper's appendix, whatever they show |

## Record
This plan and every file it names are committed to git on arr-v2 before any labelling. The commit is the timestamp.
The files are fixed by the hashes below and by `CHECKSUMS.txt`.

## Hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `score_human.py` | `01dd3ef506455a7491e03aa0853b0010a345cc59e8242ec8eb7c9c33f86be44f` |
| `scripts/hedge_measure_v2_human_score.py` (original, unchanged) | `a61de6c49ae5950b8e01273ac2d3d1d89c82c2ba6ca4e6c760100a7acca81bed` |
| `test_score_human.py` (synthetic-data tests) | `fd1f932f9bef9b7a0d31955dfe332cbb61afb4f855c80959bc19e2574f3a0531` |
| `sheet_labeller1.csv` | `dffbf9ecf46b40bfc6cf324894103a10c7d09f87785d95518d7deefe0d2eccc0` |
| `sheet_labeller2.csv` | `0dbaab50da387e64dae67f7d0d18150b3045ca02a3650935f7a59953fb000939` |
| `KEY_original.csv` | `9223d10d6c5269cd1e1bb99e1714c00e6010a22cccbba2414a75827c228e72c8` |
| `KEY_corrected.csv` | `90ccbbe1afac5ed471fd825c1fbf3b9113610bfe9447511b8db3f8ed29b5879e` |

The scorer passed 26 of 26 synthetic-data checks (see `README.md`). It has not been run on real labels.
