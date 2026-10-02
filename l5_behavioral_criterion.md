# L5 / step 9 behavioral validation: read-off criterion, stated before the run

Written and committed BEFORE `scripts/circuit_behavioral.py` was run on any pairs.
`l5_behavioral_validation.md` §5 flags the choice between "quick exploratory check" and
"write a short lock-the-criterion note first" and explicitly declines to decide it, leaning
toward the lighter option. This is the lighter option, made written rather than undocumented:
it costs nothing, and the result feeds the C3 headline framing decision (decision-vs-spread),
which is exactly the situation where this project has locked criteria in advance before
(`c034557` exists because the U/U confabulator definition had already been adjusted post hoc
once).

This is a VALIDITY CHECK, not a falsification test. It is deliberately lighter than
`c034557`: no power analysis, no pre-committed fallback procedure, no significance test.
If it should not have been locked at all, delete this file -- but delete it before reading
the results, not after.

## Primary metric

Per direction, restricted to DISCORDANT pairs (pairs whose two twins differ in clean
generated hedging -- the only pairs where "did the behaviour flip" is a defined question):

    flip-to-source-behaviour rate
      = fraction of discordant pairs where the patched continuation's hedge status
        equals the SOURCE twin's clean hedge status

compared against the same quantity for the `--n-random` size-matched random component sets.

## Bar

The behavioral claim counts as SURVIVING only if, in BOTH directions:

  1. circuit flip rate >= 0.50, and
  2. circuit flip rate exceeds the random-set mean flip rate by >= 0.20 absolute.

Both directions means `control_to_uncertain` (hedging should go DOWN) and
`uncertain_to_control` (hedging should go UP), scored against their own opposite expected
signs, not pooled.

## Secondary, descriptive only (not part of the bar)

- Hedge-rate delta (patched minus clean) per direction, with its expected sign.
- The circuit's delta minus the random-set delta.
- Number of discordant pairs, reported so a small denominator cannot hide behind a rate.

## Consequences, per l5_behavioral_validation.md §7

- SURVIVES -> decision-vs-spread stays the headline framing, with the existing `" not"`
  token-selection caveat (B5) stated alongside it.
- DOES NOT SURVIVE -> the fallback: report the gated twin-pair methodology as the
  contribution, with the decision/spread dissociation reported as a finding that did not
  survive its own behavioral audit.
- A result that clears one condition but not the other, or clears one direction only, is
  reported as EQUIVOCAL and does not license the headline framing.

## Not to be done after seeing the result

Loosening either threshold, pooling the directions, dropping the discordant-pair
restriction, switching the primary metric to the hedge-rate delta, re-running on the
working split, or changing `--max-new-tokens` to chase a different answer.

## Known limitation, acknowledged in advance

The patch is applied to prompt positions during the prefill pass only, then decoding
continues unpatched (see the script's docstring for why: there is no principled source
position for a generated token). A null result therefore means "flipping the circuit at the
decision point does not change what the model goes on to say," NOT "the circuit is
behaviorally inert under any intervention." A sustained-clamp test is a different
experiment and is not what this bar governs.
