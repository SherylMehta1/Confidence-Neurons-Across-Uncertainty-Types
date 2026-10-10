# E7: semantic relabelling of unknown facts, and a strict "declines" measure (Llama): criterion, written before the run

Written before any E7 run, in response to review round 3 (methods F1, F3; chair items 4 and 9a-c). Box 1. All judging is done by unsloth/Llama-3.3-70B-Instruct-bnb-4bit, greedy, with the fixed prompts in scripts/e7_audit.py.

## E7a: are "unknown" facts really unknown?

**Items:** all Llama pool items labelled Unknown (both popularity arms).

**Forced answers:** under the original forced-answer prompt (chat template + "The answer is"), with a fresh greedy answer and 10 samples (T = 0.7, 12 tokens).

**Grading:** the judge sees the question, the gold aliases and one answer, and says CORRECT if the answer names the same entity as any alias (allowing spelling variants, fuller or shorter names, and equivalent descriptions), else WRONG.

**Semantic label:**
- known-semantic if the greedy answer is CORRECT and at least 9 of 10 samples are CORRECT;
- unknown-semantic if 0 of 10 are CORRECT;
- otherwise intermediate.

**Test T7a:** recompute the two-judge abstention 2x2 on the 387 items, replacing each Unknown item's label with its semantic label. Items that become intermediate are dropped; items that become known-semantic move to the known cell of their arm. HighlyKnown items are unchanged.
- **SURVIVES SEMANTIC RELABELLING** if popularity − knowledge > 0 with the 95% CI (10,000 within-cell item bootstraps, seed 0) excluding 0.
- Report the movement counts per arm.

## E7c: strict declines

**Grading:** the judge labels every 96-token Llama reply (3 per item, 387 items) as one of:
- **DECLINE:** says or implies it does not know or cannot verify, and names NO candidate answer;
- **HEDGED_ANSWER:** expresses uncertainty but names a candidate answer;
- **ANSWER:** commits to an answer;
- **CLARIFY:** asks which entity is meant.

**Test T7c:** the 2x2 effects on the DECLINE rate.
- **STRICT-SURVIVES** if popularity − knowledge > 0 with the CI excluding 0.
- Also report, per cell, the share of both-judges-DONT_KNOW replies that this judge calls HEDGED_ANSWER.

## Not allowed after seeing data

Changing the prompts, the judge, the label rules, the item sets or the bars.
