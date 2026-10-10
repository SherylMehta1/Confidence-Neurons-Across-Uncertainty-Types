# Referee report: clarity and fresh-reader lens

**Manuscript:** "In Llama, Saying 'I Don't Know' Tracks Entity Popularity More Than Knowledge" (UniReps 2026, extended abstract)

## What a skim leaves a reader believing, and whether the body agrees

Suppose a reader skims only the abstract, the three figures and the conclusion. They come away believing the following:

1. **Llama abstains about obscure entities whether or not it knows the fact.** The body supports this (Fig 1a, Table 1).
2. **Answer entropy tracking knowledge is an independent finding that holds in four families.** The body only partly supports this. Lines 49-51 say entropy is *expected* to follow knowledge because the labels come from the same sampling distribution. The independent-label check was run on Llama only. Neither point is visible in a skim. (F4)
3. **The obscure-minus-popular steering direction raises abstention about as much as a dedicated abstention direction.** In Fig 2a the orange bar is at least as tall as the blue one, which reads as support for the familiarity "misfire" account. The body concludes the opposite. That conclusion rests on the hatched bar, and the caption never explains it. (F5)
4. **Base models already prefer popularity over knowledge.** In Fig 3a, base Llama shows popularity +0.67 against knowledge +0.36, with intervals that do not overlap. The body says the preference "is not seen in base models". (F6)
5. **Gemma has a different abstention policy that follows knowledge.** The body shows Gemma hardly abstains at all: at most 0.11 in its highest cell. A null preference near that floor says little. (F7)
6. **The two signals diverge "in every family".** Abstention was measurable in only two of the four families. (F18)

Points 1 and 2 match the body, with the caveat in F4. Points 3 to 6 do not, and each can be fixed by rewording, with no added length.

## Findings

**F1 (major). Abstract, l.7-15.**
- Evidence: "with a third judge, without guessable answers, … base models' decoded hedge readout responds to popularity"
- Problem: The abstract uses about six terms that a reader cannot decode at that point:
  - "third judge" (no first judge has been mentioned)
  - "guessable answers"
  - "made to answer"
  - "obscure-minus-popular direction"
  - "abstention component"
  - "decoded hedge readout"
- Fix: Replace the list of checks with "under four robustness checks". Replace the mechanistic sentence with "steering and logit-lens readouts do not isolate a familiarity mechanism". The abstract gets shorter.

**F2 (major). l.35, "misfire".**
- Evidence: "Our contribution is this factorial test of the misfire idea"
- Problem: This is a coined name used before it is defined. Lines 27-28 describe the idea but never name it, so the reader has to guess that "misfire" refers back to them. The name returns at l.123 as "misfire hypothesis".
- Fix: At l.28 write "…can trigger them without the fact (a *misfire*)". This costs two words.

**F3 (major). Fig 1 caption (p.2, placed above Setup).**
- Evidence: "share of free replies both judges label 'I don't know' … under the forced-answer prompt"
- Problem: The headline figure appears before Section 2. Its caption relies on four terms defined only later (l.42-56): "free replies", "both judges", "forced-answer prompt" and "item-bootstrap". A reader who goes to the figure first, as most do, cannot read panel (a) or panel (b).
- Fix: Use self-explaining glosses in the caption. For example, "free (unforced) replies, abstaining when both LLM judges say so" and "entropy of the first answer token after 'The answer is'".

**F4 (major). Abstract l.9-11, Fig 1 title, conclusion l.134.**
- Evidence: "Because the knowledge labels are sampled from the same distribution, entropy is expected to follow knowledge" (l.49-51)
- Problem: The body admits that half of the dissociation is partly built in. The abstract, Fig 1 title and conclusion present it as an empirical finding "in four model families and under the independent labels". The independent-label check (l.74-77) covers Llama only, and the abstract's wording blurs that.
- Fix: Change the abstract to "First-token entropy, which shares the labels' source, follows knowledge in four families and, in Llama, under independent labels". The length is about the same.

**F5 (major). Fig 2 title versus the bars.**
- Evidence: "Steering along the obscure-minus-popular direction adds little abstention beyond a generic abstention direction"
- Problem: In Llama the obscure-minus-popular bar (0.10) is at least as tall as abstain-minus-answer (0.09). Read naively, the figure says the familiarity direction is as potent as the abstention direction. The actual claim depends on the hatched bar, where the overlap is projected out. Neither the title nor the caption says what the hatched bar is for.
- Fix: Retitle as "The obscure-minus-popular direction raises abstention only through its overlap with the abstention direction". In (a), add "hatched: that overlap projected out".

**F6 (major). §3.3 Base models, l.116-118, against Fig 3a.**
- Evidence: "while Llama-Instruct's preference for popularity over knowledge is not seen in base models"
- Problem: Fig 3a shows base Llama with popularity +0.67 [0.59, 0.75] against knowledge +0.36 [0.28, 0.44]. That looks like exactly such a preference. The sentence also compares two different things: a behavioural measure in the instruct model and a lens readout in the base models. The comparison that would support the sentence is in Appendix B: Llama-Instruct's own forced-answer hedge readout shows no preference (-0.12). The main text never says this.
- Fix: Write "On the hedge-token term (Fig 3b) no base model prefers popularity over knowledge, and the same readout in Llama-Instruct shows none either (App. B), so this readout does not carry the abstention preference."

**F7 (major). Abstract l.16-17; §3.2 l.92-95; conclusion l.131.**
- Evidence: "no preference for popularity over knowledge (difference +0.01 [-0.03, +0.06]; at most 0.11 even for obscure, unknown facts)"
- Problem: The abstract and conclusion ("Gemma lacks it") present Gemma as a contrasting case. The body shows Gemma almost never abstains, so a null difference may just reflect the floor. A skim reader will believe Gemma has a knowledge-driven policy.
- Fix: In the abstract write "Gemma-2-9B, which rarely abstains (≤0.11 in any cell), shows no popularity preference". In the conclusion write "Gemma, which rarely abstains, shows none".

**F8 (major). §3.1 entity-knowledge check, l.79-82.**
- Evidence: "for 0.83 of obscure known facts (popular: 0.75); this entity-level knowledge tracks page views (ρ =+0.52)"
- Problem: I could not follow this paragraph on its own, for three reasons:
  - Obscure is higher than popular among known facts, yet the measure correlates positively with page views. That reads as a contradiction; presumably ρ is computed over all questions.
  - The purpose of the check is never stated. It is meant to rule out "the model knows the fact but not the entity".
  - "Entity knowledge" collides with the paper's main "knowledge" factor, which is about facts.
- Fix: Write "*Entity recognition* (does abstention track not knowing who the entity is?): … across all questions it tracks page views (ρ=+0.52), yet as a covariate…". Renaming the term costs nothing.

**F9 (major). §5 conclusion, l.134-137.**
- Absent: I expected one sentence giving the verdict on the misfire hypothesis, which l.35 names as the contribution. I looked in §3.1 (l.86-88), §3.3 and §5.
- Problem: The pieces are scattered:
  - The behaviour matches cue familiarity.
  - Page views do not correlate with the model's own name surprisal (l.88).
  - Neither steering nor the lens isolates a cause.
  The reader has to assemble the answer to the question the paper set out to test.
- Fix: Replace the last sentence with "Llama's behaviour fits the misfire account, but page views are not model-side familiarity and neither steering nor the lens isolates the mechanism."

**F10 (minor). §3.3 Base models, l.110-111.**
- Evidence: "we read a fixed set of hedge tokens against the answer token"
- Problem: "Hedge token" is never defined or exemplified, here or in Appendix D. "The answer token" is ambiguous: it could be the gold answer's first token or the greedy answer. The whole lens claim rests on both.
- Fix: Add two or three example hedge tokens in parentheses, and write "the gold answer's first token" (if that is what is meant).

**F11 (minor). Fig 1c versus Fig 3.**
- Evidence: Fig 1c labels "Llama-3.1-8B", "Gemma-2-9B" (instruct models); Fig 3 uses identical labels for base models.
- Problem: The same label means different models in two figures.
- Fix: Use "-Instruct"/"-it" in Fig 1c, or add "(instruct models)" to the Fig 1 caption.

**F12 (minor). §3.1 l.69-70.**
- Evidence: "keeps its sign under four other measurements, which range from +0.13 to +0.44 (Appendix A)"
- Problem: Table 1 has eight non-primary rows, so the reader cannot tell which four are meant. The regex detector, the source of three of them, is never introduced in the main text. The low value, +0.13, is explained only in the appendix (24-token truncation).
- Fix: Write "…under four other detectors (regex, a single corrected judge, DONT_KNOW-or-CLARIFY; Table 1)".

**F13 (minor). §3.3 Steering, l.101-103.**
- Evidence: "short of the 0.10 bar fixed in advance … Gemma's from 0.00 to 0.04 at 4× its natural norm"
- Problem: "Dose", "natural norm" and the purpose of "the 0.10 bar" are all undefined in the main text. Appendix C shows the bar was the regex-based criterion for choosing the dose. As a result, "short of the bar" reads like a failed pre-registered test, and the reader cannot tell which test.
- Fix: Write "(pre-set bar: +0.10 over random)" and "at 4× the direction's own norm".

**F14 (minor). §3.3 l.107-108.**
- Evidence: "unknown-minus-known raises it +0.66 [+0.58, +0.74] nats more (Figure 2b)"
- Problem: The sentence says "more" but not more than what.
- Fix: Add "than obscure-minus-popular".

**F15 (minor). §3.1 "One prompt", l.83-85.**
- Evidence: "follow popularity (+8.4 against +3.1) while its entropy follows knowledge (+0.16 against +0.05)"
- Problem: The two parenthetical pairs are in opposite orders (popularity first, then knowledge first). The log-odds are not stated as log-odds of "I" against what. The reader cannot judge whether +3.1 for knowledge is small.
- Fix: Label both pairs explicitly, e.g. "(popularity +8.4, knowledge +3.1)" and "(popularity +0.05, knowledge +0.16)".

**F16 (minor). §3.3 l.118-120.**
- Evidence: "(1 supporting, 1 contradicting, 6 undecided of eight models; Appendix D)"
- Problem: The paragraph has only mentioned four base models; "eight" also counts instruct models. "Onset" is undefined. The sentence is peripheral to the paper's claim.
- Fix: Move the sentence to Appendix D. That frees about 2.5 lines for the definitions in F3, F10 and F13.

**F17 (minor). §3.3 heading and the Base-models paragraph.**
- Evidence: "3.3 Mechanistic tests narrow the reading"
- Problem: The heading does not say which reading is meant. The paragraph's question (does the preference predate instruction tuning?) arrives only in its last two sentences, so the reader waits through all the numbers without knowing what they should show.
- Fix: Retitle as "Steering and lens tests do not isolate a familiarity cause". Open the paragraph with "Does a popularity-sensitive hedge signal exist before instruction tuning?"

**F18 (minor). Conclusion l.136-137.**
- Evidence: "That only the first holds in every family suggests the two are shaped separately."
- Problem: "The first" makes the reader look back to work out what it refers to. "Every family" overstates the evidence, since abstention was measurable in two of the four families.
- Fix: Write "Entropy follows knowledge in all four families; the abstention preference appears in Llama but not in Gemma, the only other measurable family."

**F19 (minor). §4 Related work, l.122-128.**
- Evidence: "Lindsey et al. [2025] describe the misfire hypothesis we test."
- Problem: The section repeats the intro's citations (Lindsey, Ferrando, Kumaran, Ji) almost verbatim. Only Brzezinka and Ni are new.
- Fix: Fold those two into the intro and cut §4. This saves about 5 lines, which can pay for F3 and F10.

**F20 (minor). Fig 1a/1b axis labels.**
- Evidence: The rendered y-labels read `P(says "I don't know"` and `answer entropy (nats`, with the closing characters clipped.
- Fix: Shorten them to "P(abstain)" and "entropy (nats)".

**F21 (minor). Fig 1 colour coding.**
- Evidence: Orange means "unknown" in panels (a) and (b), and a near-identical orange means "popularity" in panel (c).
- Problem: A reader carries the colour meaning across panels and misreads (c).
- Fix: Use a hue in (c) that is clearly distinct from the "unknown" orange.

**F22 (minor). Appendix B, Table 2.**
- Evidence: "Hedge log-odds: popularity − knowledge … Gemma … +2.11"
- Problem: Gemma's internal hedge readout prefers popularity more strongly than Llama's (-0.12), even though Gemma's abstention shows no preference. Neither the appendix nor the main text comments on this. A careful reader will find it, and it bears on F6 and F7.
- Fix: Add one appendix sentence interpreting it.

**F23 (minor). Appendices E and G.**
- Absent: I expected a main-text pointer to Appendices E and G; I looked in §3 and §5.
- Problem: Appendix G reports a circuit (0.83 reproduction; injection 0.40, short of a pre-set bar) that the main text never mentions. These are orphan results.
- Fix: Add "(further circuit and cross-model analyses: App. E, G)" at the end of §3.3, or drop the two appendices.

**F24 (minor). §2 l.60-61.**
- Evidence: "Confirmatory analyses followed criterion notes hashed before each run (Appendix H)"
- Problem: "Criterion note" is undefined. A reader cannot tell what the hashes buy unless the notes, or a timestamped record of the hashes, are available somewhere.
- Fix: Say where the notes and hashes are deposited, or write "pre-written analysis plans (hashes in App. H)".

**F25 (minor). §3.1 l.76.**
- Evidence: "obscure known 0.46, n = 27"
- Problem: n drops from 87 to 27 without explanation in the body. The reason (relabelling moves facts into the excluded middle band) is in Appendix A only.
- Fix: Write "n = 27 after relabelling".

## Strengths

1. The core design is easy to grasp. It is a 2×2 crossing, and the deciding cell is named early (l.33-34, the obscure-but-known facts). Fig 1a shows that cell directly.
2. The title states the finding together with its scope ("In Llama").
3. The popularity and knowledge effects are defined once, with sign conventions (l.58-60), and used consistently in the text and in Table 1.
4. The paper is honest about scope. Post hoc and pre-set analyses are labelled; the Gemma counterexample, the unvalidated judges and the failed human-memory prediction are all reported.
5. Table 1 puts every measurement variant of the headline contrast in one place, so a sceptical reader can check robustness in seconds.

## Questions (decision-changing)

1. **Gemma floor effect.** What popularity-minus-knowledge difference would Gemma show if it had Llama's preference scaled to its own base rate? If the answer is about 0.05, the +0.01 [-0.03, +0.06] interval does not distinguish the two cases. In that case "Gemma lacks it" has to come out of the abstract and conclusion.
2. **Base models.** Which panel supports "not seen in base models", given Fig 3a for base Llama? If the intended evidence is Fig 3b together with Appendix B, will the authors say so in the text (F6)?

## Recommendation

**Minor revision.** The findings are clear enough to follow and the design is sound. However, four of the six points a skimmer takes away (F4-F7) say more than, or the opposite of, what the body states, and the paper's central "misfire" framing is used before it is named (F2) and never given a verdict (F9). All of these can be fixed by rewording. Moving the human-memory sentence (F16) and merging Related work into the intro (F19) free the space needed for the definitions in F3, F10 and F13.
