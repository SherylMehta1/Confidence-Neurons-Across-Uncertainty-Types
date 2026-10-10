# E8: steering with an abstention direction fitted within one popularity level: criterion, written before the run

Written before any E8 run, in response to review round 3 (methods F5; chair item 3 and 9f). In E4, the abstain-minus-answer direction was fitted on mostly obscure vs mostly popular items, so projecting it out could remove popularity by construction. Box 2.

## Design

Same as E4 (e4_steering_v3.md, sha 18c74883):
- the same FIT/SELECT/EVAL splits (seed 0), the same layer, the same free prompt;
- doses k in {0.5, 1, 2, 4, 8} x the natural norm of the obscure-minus-popular direction;
- k* = the smallest k at which fam beats the random mean by >= 0.10 in regex abstention on SELECT;
- 150 EVAL questions, 3 replies each (96 tokens).

The only change: the abstention direction is fitted **within the obscure arm**. abst_within = mean(residual | obscure FIT item with stored regex abstention >= 0.5) − mean(residual | obscure FIT item with abstention = 0). fam_perp_w = fam with its component along abst_within removed, rescaled to the norm of fam. Report cos(fam, abst_within).

Models:
- Llama-3.1-8B-Instruct (primary);
- Gemma-2-9b-it (secondary, if time allows).

**Conditions at k* on EVAL:** none, random0-4, fam, abst_within, fam_perp_w.

**Scoring:** three judges (Qwen2.5-32B, Gemma-2-27B, Llama-3.3-70B 4-bit). The primary scorer is the majority of three, because E5 showed the Gemma-27B judge is the low-agreement one. Both-of-two is reported as secondary.

## Test (Llama)

D2w = fam_perp_w − random mean, with a 95% bootstrap CI over questions.
- **BEYOND WITHIN-LEVEL ABSTENTION** if D2w >= 0.10 and the CI excludes 0. Licenses: "the obscure-minus-popular direction raises abstention beyond its component along an abstention direction fitted without popularity variation".
- Otherwise **CARRIED BY ABSTENTION**.

## Not allowed after seeing data

Changing the splits, the dose rule, the within-arm definition, the scorers or the bar.
