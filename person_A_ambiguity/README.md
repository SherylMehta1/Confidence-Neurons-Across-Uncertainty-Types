# Person A -- Ambiguity

**Live entry point:** `python person_A_ambiguity/preprocess_ambiguity.py` (tokenizer only; no GPU).
Options: `--overwrite` (replace review logs), `--seed 42`, `--n 120`, `--reject-next-event`, `--pilot`.

## What the category measures
Questions that AmbigQA annotators disambiguated into >= 2 sub-questions with non-overlapping
answers -- e.g. two works sharing a title, or a word with two defensible senses. The model is asked
the question in the user turn and `The answer is` is prefilled into the assistant turn; the entropy
of the next token is the uncertainty measurement.

## Filters (every raw record gets a logged reason in `data/ambiguity/review_log.jsonl`)
1. `multipleQAs` annotation with >= 2 answer groups.
2. Factoid wh-question (not how/why/explain).
2t. Resolvable time references are rejected: `last week`, `currently`, `the latest`, ...,
   `\b(last|latest|most recent|first|current|currently|this (year|season))\b`, "when did X last/first",
   and present-tense questions whose sub-questions all split on a before/after date. AmbigQA's
   multiple answers there are annotation artifacts (every historically-true instance), not readings.
   "next <event>" future-scheduling questions are flagged in the log but kept unless `--reject-next-event`.
2b. Run-on questions (a second wh-word starting a clause) or > 20 words.
3. Full normalized alias sets of the answer groups must be pairwise disjoint.

## Controls (`data/ambiguity/controls.jsonl`, `is_control: true`)
AmbigQA `singleAnswer` records (annotators agreed on one answer), same filters 2/2t/2b, same
prefill. Records that also carry a `multipleQAs` annotation with >= 2 distinct groups are excluded.

## Extra record fields
`source_id` (AmbigQA id), `answer_groups` (list of alias lists; one list for controls).

## Known limitations
- Filters are regex heuristics; `first`/`last` also reject some genuinely ambiguous ordinal questions.
- AmbigQA questions are lower-cased NQ queries; some are grammatically rough.
- The committed data predate the time-reference filters and must be regenerated (see LIMITATIONS
  in the repo README / PHASE3_GUIDE.md).
