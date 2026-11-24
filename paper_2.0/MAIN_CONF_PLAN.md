# Main-conference plan: from workshop result to a spotlight attempt

Drafted 2026-10-31. For the authors. Nothing here is run or pre-registered yet.

## The honest starting point

The current papers cannot go to a main conference as they stand, and adding polish will not fix that.

| What we have | Status |
|---|---|
| Popularity affects abstention at both knowledge levels | Holds within relations on the regex measure: known +0.28 [+0.21, +0.37], unknown +0.44 [+0.29, +0.59] |
| Knowledge affects abstention too | Also holds within relations: +0.48 [+0.38, +0.58] obscure, +0.32 [+0.19, +0.48] popular |
| "Popularity more than knowledge" | **Does not survive within relations:** pooled +0.44, within relations -0.04 [-0.20, +0.11] |
| Familiarity as the cause | Not shown; steering cannot separate it from abstention |

Source: `analysis/within_relation_check.txt` (post hoc; regex on 24-token replies; equal weight over the only 3 relations with at least 3 questions per cell). Effects differ widely by relation: country +0.69, composer -0.60, sport -0.21, and the composer and sport cells are very small (4 and 7 obscure known). Treat this as a warning, not an estimate.

The reason is structural. Natural data barely contains obscure known facts outside capital, country and sport: of the 16 relations, 12 have none or one in Llama's pool. More sampling from PopQA cannot fix this. A main-conference paper needs a design where popularity and knowledge do not come bundled with question type.

## The paper to aim for

**Claim to test:** exposure to an entity changes whether a model says "I don't know", separately from whether it has the fact.

**Why this could stand out:** existing work reads familiarity from representations or steers it. As far as we have found, no one has randomized familiarity and knowledge independently and measured abstention. The closest paper we have checked (Brzezinka, arXiv 2607.13568; abstract only so far) argues for separating familiarity from abstention policy and does not cross it with knowledge in behavior. Kang et al. 2025 is the nearest method and must be read in full first.

## The experiments

**A. Randomized fictional entities (the centerpiece).** Fine-tune a model on synthetic documents about invented entities, randomly assigning each one:
- *Exposure:* 0, few or many documents that mention the entity without the target fact (familiarity).
- *Fact:* taught or not taught (knowledge).

Names and relations are randomized across cells, which removes the relation, name-form and label problems at once. Outcomes: abstention on the free prompt, and forced answers as a check that "taught" worked.

Predictions, written down before running:
| Outcome | Reading |
|---|---|
| Exposure lowers abstention when the fact is absent | Misfire shown causally: the main result |
| Fact taught, little exposure, abstention stays high | The converse failure shown causally |
| Exposure has no effect | Popularity in natural data is a proxy for something else; publishable as a correction to the familiarity story |

**B. Causal mechanism, not just steering.** Fit a familiarity direction from the exposure contrast, which is not confounded with abstention arms. Then use Che & Li's tests with natural feature values: install, remove and rescue on matched prompts. This fixes the referee's objection that our direction carries abstention by construction.

**C. Where it comes from.** Measure the effect along OLMo's public post-training checkpoints, reusing the checkpoint pipeline from the geo_mama project, plus base against instruct models. Check whether abstention targets in open training data follow entity popularity.

**D. Generality.** At least 6 families, with judges validated by people on each family. Report every family, measured or not.

**E. Why it matters.** Compare the model's own "I don't know" against an entropy or probe gate on risk-coverage curves, within popularity strata. Measure who loses coverage on long-tail entities. This gives evaluators a concrete protocol and is the part most likely to lift the paper beyond interpretability readers.

## Gates: stop early if the science says stop

| Gate | Test | By | If it fails |
|---|---|---|---|
| G1 | Human labels: judges' precision and recall per cell (the 120-reply sheet exists and is unlabeled) | Nov 8 | Fix the judges before anything else |
| G2 | Literature: no one has run A; read Kang et al. and Brzezinka in full | Nov 8 | Change A or differentiate |
| G3 | Pilot A (about 200 entities): the fine-tune teaches facts, and abstention is measurable | Nov 22 | Change model or document design |
| G4 | Full A shows a clear result in either direction | Dec 13 | Move to the NeurIPS date; do not rush |

## Venue and timeline

**Recommendation: ICML 2027 if G3 passes by Nov 22; otherwise NeurIPS 2027.** All dates below are unofficial, so confirm them when the calls are published.

| Venue | Date | Status of date | Fit |
|---|---|---|---|
| ACL via ARR | Jan 4, 2027 | Third-party report; official site says TBA | Too soon for A plus B |
| **ICML 2027** | Abstract Jan 16, paper Jan 22 | Aggregator projection; no official call found | Primary, about 12 weeks |
| COLM 2027 | About Mar 4 | Marked as an estimate | Good fit if the paper is more NLP than ML |
| **NeurIPS 2027** | Mid-May | Third-party estimate | Fallback, with room for C, D and E in full |

ICLR 2027 has passed (paper deadline Sep 25, 2026).

| Weeks | Dates | Work |
|---|---|---|
| 1 | Nov 2-8 | G1, G2; external pre-registration (OSF or similar, so plans are timestamped, unlike our self-recorded hashes); power analysis |
| 2-3 | Nov 9-22 | Dataset and pilot A; G3 |
| 4-6 | Nov 23-Dec 13 | Full A, then B; G4 |
| 7-8 | Dec 14-27 | C, D, E as far as they go |
| 9-11 | Dec 28-Jan 16 | Write one story; blind audit and review rounds; abstract |
| 12 | Jan 17-22 | Fix and submit |

## What raises the chance of a spotlight

Nobody can plan a spotlight; committees decide it, and a good paper can miss it. These are the things in our control:
1. One claim a reader can repeat, shown by one randomized figure.
2. A causal design that answers the obvious objection before a referee raises it.
3. Pre-registration with outside timestamps, and human-validated measures.
4. A released benchmark and audit protocol that others can use.
5. The audit routine from this project, run before submission, not after.

## Risks

- **A may show nothing.** The plan treats that as a result, but it is a smaller paper.
- **Fine-tuning may distort abstention by itself.** Needs a control model and a pilot.
- **Compute.** Four-hour free boxes suit a pilot; the full A and B may need paid GPUs.
- **Being scooped.** This area is moving quickly; recheck the literature monthly.
- **Fixing what exists first.** UniReps and MERe carry claims this plan contradicts. Correct them before they are cited as ours.
- **Double-blind rules.** The workshop papers are non-archival, but check each venue's dual-submission and anonymity rules, and make the GitHub repo private.
