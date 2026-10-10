# Current draft vs Nanda-style draft in your voice

**Update (2026-10-10, later):** `main_nanda.tex` is now the Nanda structure **written in your voice**. Your voice was measured from three of your papers (`AUTHOR_PROFILE.md`), with your first-author TAIS 2026 paper as the primary sample. The two fixes recommended below are applied:
- the merged Results paragraph is split again ("Recognition & Ambiguity: Neither Explains It." and "Familiarity: Not Shown Directly.");
- the Setup's "Two Readouts" paragraph is trimmed.

The pre-voice Nanda draft is kept at `revision/round4/main_nanda.v1.tex`.

## Rhythm against your own writing (`prose_gate.py`, by section)

| | Current `main.tex` | Nanda v1 | **Nanda + your voice** | Your TAIS paper |
|---|---|---|---|---|
| Abstract: mean words per sentence | 20.3 | 21.4 | **18.3** | (17.3 overall) |
| Introduction: mean (CV) | 23.4 (0.46) | 18.1 (0.57) | **15.0 (0.44)** | 17.3 (0.66) overall |
| Results: mean (CV) | 22.3 (0.53) | 27.7 (0.37) | **17.9 (0.41)** | 17.3 (0.66) overall |
| Results: passive share | 13% | 21% | **9%** | about 5% |

The voiced draft now matches your sentence length. It is still more even than your writing (CV 0.41-0.47 against your 0.66): you mix very short sentences with long ones more than this draft does.

## What your voice changed

- **Abstract:** opens "We investigate what a language model's 'I don't know' tracks: ...", as your TAIS abstract opens "We investigate how ...". "We find that ..." states the result.
- **Introduction:** shorter sentences. A bold **"Main Contributions."** run-in, then a numbered list with Title-Case labels and colons ("1. Popularity Over Knowledge in Llama: ..."), followed by one "Together, these findings mean that ..." sentence. This follows your TAIS pattern. ("Suggest" would have weakened a claim the draft states directly, so it is "mean".)
- **Headings and labels:** Title-Case colon headings ("Llama: Abstention Tracks Popularity More Than Knowledge", "Mechanistic Tests: No Familiarity Cause Isolated"), and Title-Case run-in labels ending in a period ("Core Result.", "Label & Measurement Checks: The Preference Survives.", "Plain Prompt: Too Few Abstentions to Measure.").
- **Linking:** colons and parentheses carry the detail; no em dashes.
- **Spelling:** American throughout (18 changes, e.g. relabeled, behavior, toward), appendices included.
- **Section 4:** "Limitations" as its own section, as in your papers. The implication moved into the introduction's closing sentence.
- **Table 1:** dropped its entropy question-count row to fit four pages; those counts are in Appendix A.

## Checks

- Four pages of main text; no overfull boxes; 16 of 16 specific numbers found in result files.
- `verify_rewrite.py main_nanda.v1.tex main_nanda.tex`: the many "ADDED proper noun" lines are the Title-Case words in the new labels. The two "dropped may" are the permission sense in the old heading ("when it may say 'I don't know'"), not hedges. The added "no" negations are labels restating "do not isolate". No number, citation or claim changed.

---

## Earlier comparison (current draft vs the first Nanda-style draft)

- **Current:** `main.tex` / `main.pdf` (after review round 4).
- **Nanda-style:** `main_nanda.tex` / `main_nanda.pdf`, rewritten to the style target in `AUTHOR_PROFILE.md`.

Both drafts carry the same facts, numbers, claims and appendices. `verify_rewrite.py main.tex main_nanda.tex` drops and adds nothing except one combined section reference that was split into two, and its warnings are benign (see the end of this file). Both fit four pages of main text.

**About the voice.** The Nanda style comes from his published essay "Highly Opinionated Advice on How to Write ML Papers" and from the structure of three papers he co-authored (Ferrando et al. 2025, Arditi et al. 2024, Stolfo et al. 2024). It is a style target, not his prose. No text from those sources was copied (the tool used to read them returns summaries, not verbatim text). It is also not your own voice: no sample of your writing was available. Send two or three pieces you wrote yourself and the profile can be calibrated to them.

## What changed, section by section

| Part | Current draft | Nanda-style draft | Rule (essay) |
|---|---|---|---|
| Abstract, opening | A question: "When a language model answers ... is it reporting that it lacks the answer?" | Context, then the open problem: "Language models answer factual questions they cannot answer correctly ... It is unclear, though, what a model's 'I don't know' tracks" | Sentence 1 uncontroversial context, sentence 2 the problem |
| Abstract, contribution | Implicit in the result sentence | Stated outright: "We show that in Llama-3.1-8B-Instruct, saying 'I don't know' tracks entity popularity more than knowledge." | Sentence 3 is the key contribution |
| Abstract, close | "Steering and logit-lens readouts do not isolate a familiarity mechanism." | Adds the standard of evidence: "... so the evidence says what abstention tracks, not why." | End with why it matters and the standard of evidence |
| Introduction | Three paragraphs; contribution in one sentence | Context, background, a "what is new" paragraph, our case, then a **bulleted contributions list** with a section pointer and the key number for each | Introduction as a self-contained summary; contributions as claims with pointers to evidence; state novelty plainly |
| Section headings | 3.1 "In Llama, abstention tracks popularity; answer uncertainty follows knowledge" | 3.1 "Llama's abstention tracks popularity more than knowledge"; 3.2 "Gemma tracks popularity too when it may say 'I don't know'"; 4 "Limitations and implications" | Headings state findings |
| Results paragraphs | Italic question headers ("Are the knowledge labels right?") | **Bold run-in findings** ("The knowledge labels hold up.", "The 'I don't know' replies are refusals.", "Steering moves a generic abstention signal.") | Paragraphs open with what they show (as in Arditi and Stolfo) |
| Closing section | "Conclusion and limitations": conclusion first | "Limitations and implications": limitations first, then implications, no separate conclusion | Discussion covers limitations and implications; conclusion optional |

## Measurements (`prose_gate.py --academic`, whole file)

| | Current | Nanda-style |
|---|---|---|
| Words / sentences | 3,394 / 159 | 3,421 / 156 |
| Mean sentence length (CV) | 21.3 (0.58) | 21.9 (0.56) |
| Abstract: words, sentences, mean | 146, 7, 20.3 | 174, 8, 21.4 |
| Introduction: words, sentences, mean (CV) | 317, 13, 23.4 (0.46) | 384, 20, 18.1 (0.57) |
| Results: sentences, mean (CV) | 38, 22.3 (0.53) | 29, 27.7 (0.37) |
| Passive share in Results | 13% | 21% |

The Nanda draft's introduction is shorter-sentenced and more varied, which helps a cold reader. Its **Results paragraphs are worse**: longer, more uniform sentences and more passive voice. To pay for the contributions list within four pages, I merged three short Results paragraphs (recognition, ambiguity, familiarity) into one. That is the main cost of the switch.

## Assessment

The Nanda-style draft is easier on a first pass for the reader the review described ("not readable on a first pass"):
- The abstract says what we found before what we did.
- The contributions list gives the reader the paper in three bullets with numbers.
- Every Results paragraph starts with its finding in bold, so skimming the bold text reproduces the argument.

What it gives up:
- The current draft's question headers map one-to-one onto the reviewers' objections, so a reviewer checking "did they handle X?" finds it faster there.
- The four-page budget is tighter: the contributions list took about six lines, paid for by merging three Results paragraphs.
- The abstract grew by about 28 words.

**Recommendation:** use the Nanda-style structure (abstract order, contributions list, finding-first headings and run-ins). Before submitting, split the merged Results paragraph back into "Recognition and ambiguity do not explain it." and "Familiarity itself is not shown.", and find the two to three lines it needs by trimming the Setup's "Two readouts" paragraph. Which draft goes forward is an author decision.

## Verification details

`verify_rewrite.py main.tex main_nanda.tex`: DROPPED `\cref{sec:main,sec:families}` and ADDED `\cref{sec:main}`, one combined reference split into two (both sections are still referenced). Warnings:
- "cause" is repeated: the contributions bullet restates the existing negated claim "do not isolate a familiarity cause".
- Negations rose 51 to 54, from "not why" and the split references.
- One added modal.

No number, citation or claim changed.
