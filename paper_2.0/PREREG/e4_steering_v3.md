# E4: steering with held-out dose selection, an abstention-orthogonal direction, and judge scoring: criterion, written before the run

Written 2026-10-10, before any E4 run, in response to:
- methods referee F5-F7 (both rounds);
- chair I9, I16-I18 and J10-J14.

Box D. Models: unsloth/Meta-Llama-3.1-8B-Instruct and unsloth/gemma-2-9b-it, bf16. The layer is ℓ* = round(0.55 × (L − 1)), as before.

## Splits (seed 0)

Items are labelled from `labels_dr/<model>_inst.jsonl`.
- **FIT:** half of all items, used to fit directions.
- **The other half** gives:
  - **SELECT:** 150 popular HighlyKnown items;
  - **EVAL:** a disjoint set of 150 popular HighlyKnown items.

## Directions (fitted on FIT, at the last position of the free prompt)

- **fam** = obscure − popular, averaged over knowledge (as before).
- **know** = unknown − known, averaged over popularity.
- **abst** = mean(residual | the item's regex free-generation abstention rate ≥ 0.5) − mean(residual | rate = 0). It is fitted on FIT items of both arms, using the stored regex hedge rates (`all_measured.jsonl` hedge_rate).
- **fam_perp** = fam with its component along abst projected out, rescaled to the norm of fam.
- **5 random** directions, drawn per model with seed 0.

## Doses and generation

- Dose grid k ∈ {0.5, 1, 2, 4, 8} × ‖fam‖, applied to every direction (each is unit-normalised, then scaled).
- Generation: 3 samples per item, T = 0.7, 96 tokens, at the free prompt. The vector is added at the last prompt position and at every generated position.
- Entropy: first-token entropy at the forced-answer prompt, with the vector added at its last position.

## Dose selection (on SELECT only)

k* = the smallest k at which fam's regex abstention rate exceeds the random mean by ≥ 0.10.

## Scoring (EVAL, at k*, every direction, plus "none")

- Each reply is scored by the regex and by the two judges of the paper (Qwen2.5-32B-Instruct, Gemma-2-27B-it, same prompt).
- The primary scorer is "both judges DONT_KNOW".
- A reply is degenerate if it repeats one token more than 8 times within 24 tokens.

## Tests at k* on EVAL (judge-scored; 95% item-bootstrap CIs, seed 0)

- **D1.** fam − random mean. **CONTROLS ABSTENTION** if ≥ 0.10 with the CI excluding 0.
- **D2 (main).** fam_perp − random mean. **BEYOND ABSTENTION** if ≥ 0.10 with the CI excluding 0. Licenses: "the obscure-minus-popular direction raises abstention beyond its component along the abstention direction".
  - Otherwise: "the effect is carried by the abstention component", reported as a negative result.
- **D3.** fam − know. Report it with its CI, and say that the ranking is expected from the cell abstention rates.
- **D4.** The entropy contrast know − fam at k*, with its CI, and degenerate-output rates per condition.
- **Dose-response curves** for all directions on SELECT are reported in the appendix.

## Not allowed after seeing data

Changing the splits, the grid, the k* rule, the abst definition, the scorers or the bars.
