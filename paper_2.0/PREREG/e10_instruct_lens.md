# E10: instruct-model lens split, for a matched base-versus-instruct comparison: criterion, written before the run

Written before any E10 run, in response to review round 3 (framing F2, methods F8; chair items 2 and 9e). Box 4. Identical to E3 (e3_lens_split.md, sha b2b916ce) in every respect except the models.

## Models

Instruct partners with official prefitted lenses, using their chat-template forced-answer prompt as in the lens study:
- Llama-3.1-8B-Instruct
- Gemma-2-9b-it
- Gemma-2-2b-it

## Measures

Window effects (relative depth [1/3, 2/3]), popularity and knowledge, on the full readout and on the hedge term. 95% CIs from 2,000 within-cell item bootstraps (seed 0).

## Reported comparison

Descriptive, no bar: for each base/instruct pair, popularity − knowledge on the full readout and on the hedge term, with CIs.

The paper may say that instruction tuning increases the popularity preference in the lens readout only if, for a pair, the instruct difference exceeds the base difference and their CIs do not overlap. Otherwise it says the lens comparison does not show a change.

## Not allowed after seeing data

Changing the models, the readouts, the window or the rule above.
