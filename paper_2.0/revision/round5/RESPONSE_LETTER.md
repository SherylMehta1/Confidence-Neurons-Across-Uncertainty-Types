# Response to the round-4 review (2026-10-10)

Reviews: [`../../review/round4/`](../../review/round4/) (chair decision, three referees, red team, adjudication, numbers audit). Draft before this round: [`../round4/main_before_r5.tex`](../round4/main_before_r5.tex).

The chair accepted the paper on condition that 17 must-fix items were addressed. Two of those were central: run a name-only guessing baseline or narrow the headline (I11/I23), and reconcile the gold-alias figures in the obscure-known cell (I4/I15). We ran the baseline. We also ran four more pre-specified experiments that the referees asked for (E12–E16). All plans were hashed before each run (App. G).

Actions: **fix** (changed as asked), **partial** (part done, with the reason), **not done** (with the reason; carried to the open list).

## New experiments (all pre-specified)

| exp | asked by | result | where |
|---|---|---|---|
| E11 name-cue control | I11, I23 | Swapping two interior letters and one vowel in each subject's name makes 34 of the 86 obscure-known facts recoverable from the name (25 of 46 country). Those 34 abstain at 0.33, the rest at 0.50. With them removed, the preference is +0.36 [+0.25, +0.48]; relation-clustered [−0.03, +0.48]. **Verdict: survives name-cue control.** | abstract, Contrib. 1, §3.1, Table 2 |
| E12 permissive prompt, Qwen and Mistral | I3 | Judges disagree (Fleiss κ 0.40, 0.50, below the 0.60 bar). Estimates +0.15, +0.16. **Unmeasured.** | §3.2, App. B |
| E13 fresh replies, same questions | Me | +0.29 [+0.20, +0.38]. **Replicates.** | §3.1, Table 2 |
| E14 new questions, 10th–30th vs 70th–90th percentile | I19 | +0.10 [+0.01, +0.20]: smaller, as the narrower contrast predicts. Relation-clustered [−0.12, +0.27]. 85 of 100 obscure-known facts again fall in capital, country or sport. **Replicates in new questions.** | §3.1, Table 2 |
| E15 permissive prompt, OLMo-2-7B | I3 | κ 0.46. Estimate −0.09. **Unmeasured.** | §3.2, App. B |
| E16 abstention direction fitted on popular questions only | I10 | Cosine with obscure-minus-popular 0.89. Removing it leaves +0.007 [−0.003, +0.019] over 20 random directions. **Generic abstention.** | §3.3, App. C |
| E17 NLA readout of Qwen (exploratory) | — | Outputs lost at box expiry (sync misconfigured). Nothing about it is in the paper. | LAB_LOG |

## Must-fix items

| id | issue (short) | action | change and location |
|---|---|---|---|
| I23 | Headline rests on the obscure-known cell | fix | E11 was run. The abstract and Contrib. 1 lead with the preference, its survival under name-cue control (+0.36), and the gap among unknown facts (+0.34 [+0.25, +0.43]), which does not use the obscure-known cell. Limitations says "known" is specific to the forced-answer prompt. The title is unchanged; see I19. |
| I11 | No name-cue control | fix | E11 (above). §3.1 "Replications & Name-Cue Control"; Table 2 row; footnote b gives the relation-clustered interval. |
| I12 | Guessability filter: n and relation mix not given | partial | App. A now gives the count after filtering: 30 of 87 obscure-known facts are guessable (abstain 0.42); the other 57 abstain at 0.44. The relation mix is in Relation Concentration. Not done: a graded per-question prior across bins. E11 is the stronger test of the same alternative. |
| I13 | No analysis separates "knows but declines" from "mislabelled guess" | partial | App. A Free replies: with name echoes excluded, 9 of 87 obscure-known questions have one free reply that gives the answer and another that abstains. E11 removes cue-recoverable facts, and the preference holds. Not done: the held-out split (select on one reply set, score on another). |
| I4 | Gold-alias figures do not reconcile | fix (factual error) | The alias match counted echoes of the subject's name. Echo-corrected (analysis/echo_check.py): 0.25 of obscure-known free replies contain the answer (was 0.53); 0.44 of non-abstaining replies (was 0.72); 0.00 of abstaining replies; 9 of 87 questions have both (was 24). App. A says the earlier count was overstated. |
| I15 | Strict-decline 0.66 vs alias 0.53 overlap | fix | Resolved by the I4 correction: echo-corrected, no abstaining reply contains the answer. The two rates no longer overlap. |
| I1 | Abstract hides how much label audits shrink the preference | fix | The abstract gives the joint estimate, +0.14 [−0.01, +0.28], "when all label corrections are applied at once". §3.1 gives the meaning regrade, the alias drop and the joint estimate. The heading "The Preference Survives" is replaced by "Label & Measurement Checks". |
| I14 | Corrections never applied jointly | fix | +0.14 [−0.01, +0.28] with per-cell n (81/58/39/100); Table 2 row "All label corrections jointly" with footnote b; analysis/round4_checks.py (3). |
| I6 | Range across measures; primary adopted post hoc | fix | The abstract gives "+0.13 to +0.44". §3.1 says the primary measure was adopted after the first pre-specified one returned the smallest value. |
| I5 | "Known" is specific to the forced-answer prompt | partial | §3.1 states that 27 of 87 obscure-known facts stay known under a paraphrase and abstain at 0.46. Limitations says "known" is specific to the forced-answer prompt. Not done: knowledge pooled over both prompts, and the 3×3 table. |
| I8 | Relation clustering | fix | §3.1 Relation Concentration: 80 of 87 obscure-known facts lie in three relations; relation-clustered interval [−0.10, +0.43]; leave-one-relation-out range. Footnote b adds clustered intervals for the name-cue control, the new questions and the unknown-facts gap ([+0.23, +0.46]). The analysis is post hoc and labelled so. |
| I7 | Framing promises a familiarity account | fix | The abstract opens on "whether the model lacks the fact, or how popular the entity is". The intro presents familiarity as the motivating hypothesis, with page views as a stand-in. §3.1 Recognition, Ambiguity & Familiarity says no model-side measure shows obscure names to be unfamiliar, so the mechanism is not established. |
| I19 | Title and abstract not scoped to the extreme deciles | partial | The abstract says "top vs. bottom decile". Limitations says the design contrasts extreme deciles. E14 shows a smaller preference in adjacent bands. The title is unchanged, as an author decision (see open items). |
| I2 | "The Split Remains" overstates | fix | Retitled "Within One Prompt: Mixed". The pre-specified null comes first; the free-prompt reading is labelled as specified after seeing it, with units (nats, log-odds) and 95% intervals. |
| I18 | Matched pairs only for known facts | not done | The text still gives the known-facts matched-pair result (+0.76 → +0.43) beside the regression (App. A Name spelling). Matching unknown facts, and Wikidata P31 strata, need a new analysis that we could not do before the deadline. |
| I16 | No human labels | partial (the chair's stated minimum) | Limitations: "Abstention is judged by LLMs not validated against human labels, whose errors could differ by cell." No hand labelling. |
| I17 | Regex vs corrected judge differ threefold | not done | App. A still says the disagreement is not established. The by-cell cross-tabulation is a local analysis on existing outputs and is on the open list. |

## Should-fix and minor items

| id | action | change |
|---|---|---|
| I20 | partial | Contrib. 2 and §3.2 are conditional on the prompt ("Gemma Shares the Preference When Permitted"). Not done: all four plain-prompt Gemma cell rates (only the maximum, 0.11, is printed) and the odds-ratio form. |
| I3 | fix | E12 and E15 (above): reported whatever the outcome. |
| I9 | partial | The intro restates what can be asked: only Llama and Gemma are measurable. No per-question agreement analysis. |
| I10 | fix | E16 tests the reviewers' alternative directly. §3.3 now rests "generic abstention" on a direction fitted without any obscure entities. |
| I21 | partial | §3.3 cites Ferrando et al. and Brzezinka as prior steering work, and Che and Li; with E16 it shows that the effect is carried by an abstention direction. It does not state in words that those studies used random-direction controls only. |
| I22 | partial | App. C gives Gemma's three-judge differences and the within-level direction. The main text gives Llama only, for space. |
| I24, I28, I35 | fix | Table 2 is restructured: measures, replications and name-cue control, label variants, regression controls. n is filled in (macros `\nqN`, `\ekN` = 384, as the audit asked). κ is given "on the 253 questions labelled under both". |
| I25 | fix | The abstract says "43% of replies" and "greedy answer and at least 9 of 10 forced samples". |
| I26, I36 | fix | Gemma's plain prompt: "abstains in at most 0.11 of replies per cell, so its preference cannot be estimated"; no plain-prompt estimate is presented as a result. |
| I30 | fix | "knowledge raises it comparably or more in three of four". |
| I32 | not done | No conclusion section; the 4-page limit is binding. |
| I33 | fix | The cross-model sentence now names its readout ("on the forced-answer hedge readout"). |
| Audit critical 1 | fix | App. B describes the four-family test correctly: regex for Llama, Qwen and Mistral; log-odds for Gemma. |
| Audit critical 2 | fix | Same as I4. |
| Audit minors | fix | Onset caption "two of eight here"; steering caption no longer promises three-judge rates; Table 2 n = 384; OLMo-2 is reported (App. B, E15). |

## Checks after the edits

- The main text ends on page 4; the PDF has 13 pages; no overfull boxes and no undefined references.
- `check_numbers`: 15 of 15 specific literals found in result files.
- `verify_rewrite` (before vs after): every dropped macro is a deliberate replacement (stale free-reply values `rc*`/`eoFree*` → echo-corrected `ec*`). Three facts dropped by accident were restored (27 of 87 stay known; recognition shares; entropy control).
- Every new number is a macro in `numbers.tex`, generated by `make_numbers.py` from `results/round3/*_summary.txt` and `analysis/*.txt`.

**Later change (2026-10-11):** Appendix G was removed at the author's request. References to "App. G" above refer to the version before that change; the plan files and hashes are in `PREREG/`.
