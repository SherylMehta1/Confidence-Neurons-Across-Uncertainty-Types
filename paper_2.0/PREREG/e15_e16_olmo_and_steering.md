# E15 and E16: the permissive prompt for OLMo-2, and steering with a popularity-free abstention direction: criteria, written before the runs

Written before any E15 or E16 run, in response to review round 4.

## E15: permissive prompt for a fifth family (box 9)

Identical to E9 and E12 (`e9_permissive_prompt.md`, sha f6459163; `e12_permissive_qwen_mistral.md`, sha 56831ebc) except the model.

- **Model:** OLMo-2-7B-Instruct (`allenai/OLMo-2-1124-7B-Instruct`).
- **Items:** its existing two-judge question set (`results/hedge_measure_v2/items_olmo2_7b.jsonl`, from the exploratory pool; obscure known n = 48).
- **Prompt, replies, judges, majority rule, κ bar (0.60), tests, the 0.25 obscure-unknown threshold and the licensing:** as in E12. Because the obscure-known cell has only 48 questions, the result is reported as a fifth-family extension, not as a test of the main claim.

## E16: steering with an abstention direction fitted within the popular arm (box 10)

Methods F10: neither E4 nor E8 can tell a generic abstention signal from a familiarity signal, because both abstention directions were fitted on questions that vary in, or are confined to, obscurity. E16 is identical to E8 (`e8_steering_within_level.md`, sha 05474954) except:

- **Abstention direction:** abst_pop = mean(residual | popular FIT question with stored regex abstention >= 0.5) − mean(residual | popular FIT question with rate 0). It is fitted within the popular arm, so it carries no obscure-vs-popular contrast and no obscure entities. fam_perp_p = fam with its component along abst_pop removed, rescaled to the norm of fam.
- **Random directions:** 20 (seeds 0-19), not 5.
- **Model and dose:** Llama-3.1-8B-Instruct only, at E4's k* = 1.
- **Conditions on the 150 EVAL questions (3 replies each):** none, random0-19, fam, abst_pop, fam_perp_p.
- **Scoring:** majority of the three judges (primary), as in E8.

**Tests:**
- **G1:** abst_pop − random mean > 0 with a 95% bootstrap CI over questions excluding 0. This checks that the popularity-free abstention direction moves abstention at all.
- **G2:** D2p = fam_perp_p − random mean, with a 95% CI.
  - **GENERIC ABSTENTION** if G1 holds and D2p < 0.10 or its CI includes 0. Licenses: "the obscure-minus-popular direction adds no abstention beyond its overlap with an abstention direction fitted without any obscure entities".
  - **BEYOND GENERIC ABSTENTION** if D2p >= 0.10 with the CI excluding 0.
  - **NOT SEPARABLE** if G1 fails. The paper then says the directions "are not separable here".
- Report cos(fam, abst_pop), and the 2.5th-97.5th percentile range of the 20 random-direction rates.

## Not allowed after seeing data

For E15: as in E12. For E16: changing the fitting arm, the regex thresholds, the splits, the dose, the number of random directions, the scorer, the bar of 0.10, or the verdict rules.
