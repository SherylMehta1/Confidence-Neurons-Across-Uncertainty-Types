# E11: name-cue control for the "known" label (Llama): criterion, written before the run

Written before any E11 run, in response to review round 4 (red-team attack; adjudication P1 and P2 "unresolved"; methods F2; chair I11). The worry: an obscure fact can be labelled known because Llama recovers the answer from the form of the name (its language, its words) or from the relation's prior, without knowing the entity. Box 7.

## Design

- **Items:** every known fact (HighlyKnown) in the 387-question two-judge sample (`results/hedge_measure_v2/items_llama.jsonl`), in both arms: 87 obscure and 100 popular.
- **Pseudo-names:** for each item, three pseudo-names made from the subject name with a fixed seed (the item key and the variant index). In every alphabetic word of at least four letters, two adjacent interior letters are swapped and one interior vowel is replaced by a different vowel. Three-letter words get only the vowel change; shorter words and non-alphabetic tokens are kept. The variants must differ from the original and from each other. An item whose question does not contain the subject string, or whose name cannot be changed, is reported as untestable and kept, unchanged.
- **Probe:** the same forced-answer prompt and functions as the original labels (`build_completion_prompt(..., cloze_suffix="The answer is", ensure_question_mark=True)`, `greedy_answer`, `sample_answers` with 10 samples at T = 0.7, `is_correct` against the gold aliases), with the subject replaced by the pseudo-name.
- **Cue-recoverable (primary, deliberately loose):** the greedy answer is correct for at least 2 of the 3 pseudo-names.
- **Cue-known (secondary, strict):** the original labelling rule (greedy correct and at least 9 of 10 samples correct) calls the fact known for at least 2 of the 3 pseudo-names.

## Test T11 (two-judge abstention, as in the paper)

Remove the cue-recoverable facts from both known cells. Recompute the popularity preference (obscure-known minus popular-unknown abstention) with a 95% within-cell item bootstrap (10,000, seed 0). Also report a relation-clustered bootstrap (5,000, seed 0) and the relation mix of the remaining obscure-known facts.

- **SURVIVES NAME-CUE CONTROL** if the preference is > 0, the within-cell interval excludes 0, and at least 20 obscure-known facts remain.
- **DOES NOT SURVIVE** if at least 20 remain and the interval includes 0, or the estimate is <= 0.
- **UNDERPOWERED** if fewer than 20 remain.

Also report: the share of cue-recoverable facts by arm and by relation; the abstention of cue-recoverable vs other obscure-known facts; and the same test with the strict rule.

## Licensing

- SURVIVES: the paper may keep "more than knowledge" in the title, adding that it survives a name-cue control.
- DOES NOT SURVIVE or UNDERPOWERED: the headline must lead with the popularity gap among unknown facts, and "more than knowledge" is stated as dependent on forced-answer labels.

## Not allowed after seeing data

Changing the perturbation rule, the number of variants, the 2-of-3 thresholds, the items, the bar of 20, or the verdict rules.
