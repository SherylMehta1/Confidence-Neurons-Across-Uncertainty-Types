# Cold read of the MERe draft (blind referee, 2026-10-31)

Written for: the authors. Read the version in `mere/main_submitted_MERe.pdf`. The referee finished after the author had decided to submit, so none of this is in that PDF. Recommendation: major revision; the referee calls it worth accepting as non-archival work once framing is fixed. The referee reports the headings alone do tell the story; breaks are listed below.

## Critical

- **C1. Title vs conclusion.** The title states "tracks popularity more than knowledge" as fact; Section 8 calls that comparison "suggestive rather than settled". Referee's fix: retitle around the firm result, e.g. "In Llama, Saying 'I Don't Know' Depends on Entity Popularity, Even Among Facts It Does Not Know".
- **C2. "Facts Llama knows" does not transfer across prompts.** Appendix A: non-abstaining free replies in the obscure known cell contain the answer 0.44 of the time, against 0.98 for popular known. An "I don't know" there may be accurate about what the free prompt can retrieve. The main text never says so. Fix: say "answers correctly when forced" and move these numbers into Section 4.
- **C3. Entropy is not excluded for the firmest result.** Table 1: among unknown facts, entropy differs by arm (3.51 vs 2.58 nats) in the same direction as abstention. So the known-fact comparison excludes entropy but has contested labels; the unknown-fact gap survives label correction but does not exclude entropy. No single result does both. Fix: state this at the close of Section 4 and in the abstract.

## Major

- **M1.** The primary measure was adopted after +0.13 was seen, yet Table 2 labels it "pre". Fix: "pre, after +0.13 seen"; add to the abstract and Limitations.
- **M2.** kappa 0.79 covers only the 253 questions labeled under both prompts; most known facts lose their label (27 of 87 obscure, 35 of 100 popular stay known).
- **M3.** "Replicates" is too strong: fresh replies on the same questions check sampling noise; new questions have a different population and a clustered interval of [-0.12, +0.27].
- **M4.** The steering test cannot separate the hypotheses: a direction fitted across arms that differ in abstention carries abstention by construction, and the misfire account also predicts overlap with abstention. "No familiarity mechanism" can read as evidence of absence.
- **M5.** "Generic abstention" is stated without naming a model, but in Gemma, on the scorer Appendix C endorses, +0.05 [+0.03, +0.07] survives removal (cosine 0.63).
- **M6.** The abstract says lens readouts point to a generic signal; Section 6 says only that the readout does not show where the preference arises.
- **M7.** The recognition comparison (0.83 vs 0.75) covers known facts only, a selected group; unknown cells are not reported.
- **M8.** "Same preference" in Gemma: different prompt and scorer, no label corrections or relation resampling; Llama is +0.47 on that prompt. Fix: "a preference of the same sign".
- **M9.** The preference is a cross-cell difference with no within-relation version; the abstract's "because 80 of 87" is an untested causal claim about the interval.
- **M10.** Section 4 says the claim "holds in every variant by point estimate", but the pre-specified forced-answer readout in the same section is -0.12. True only of Table 2 rows.
- **M11.** The abstract never says the judges are LLMs, or unvalidated against humans.
- **M12.** Preference, gap, contrast and comparison are used interchangeably, and "gap" names two different quantities.
- **M13.** Abstract density: +0.30 is not explained as 0.43 minus 0.13.
- **M14.** "Name Cues Do Not Explain It" collides with "Name form explains part of the gap" (+0.76 to +0.43, post hoc), which the abstract omits.
- **M15.** The Recommendation leans on the two cells with the weakest labels and omits the most useful lesson for evaluators: the estimate moves from +0.13 to +0.44 with the measure.

## Minor (short)

- m1: "eight" models for onsets, while Appendix D names seven. m2: 86 vs 87, and "answered" under pseudo-names has no stated rule. m3: the 0.91 obscure-only direction is not described in Appendix C (its paragraph was cut for space). m4: steering scorers not labeled in the main text. m5: the entropy control is largely built in by selection. m6: "corrected" has three descriptions. m7: "all label corrections" does not include the paraphrase relabel or the name-cue control. m8: "5%" looks like 3% at printed precision. m9: "0.00 [0.00, 0.00]". m10: "misfire" is defined for one cell only. m11: instruction tuning called open though the one test found no support. m12: OLMo-2 has no citation and first appears in Section 5. m13: the four-family sentence is hard to parse. m14: how rare the headline cell is goes unstated. m15: "set independently" should be "measured separately". m16: unexplained sign convention on -0.11.

## Strengths the referee would keep

Candid weakening in the abstract; headings that form an argument; the 2x2 design with its identity; plan status marked per analysis; restraint on mechanism.
