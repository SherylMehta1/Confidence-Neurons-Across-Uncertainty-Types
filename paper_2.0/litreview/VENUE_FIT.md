# Venue fit: UniReps 2026 extended abstract (litreview, 2026-10-10)

Source: `C:/Users/starb/Desktop/Sher/litreview/J_venue_positioning.md` (5 Oct). The venue pages and award posts were opened there; the URLs are in J §7. This file keeps what the `write` mode needs.

## The track
- Extended abstract, non-archival, **at most 4 pages of main text**, anonymised. References and appendix do not count. Camera-ready limit in past years: 5 pages. NeurIPS template.
- The call asks for "early-stage results, insightful negative findings, and opinion pieces". Our failed cascade prediction (R2) and the equivocal injection result (C9) both fit "insightful negative findings".
- OpenReview closes **11 Oct 2026 12:00 UTC**. The form needs a reciprocal reviewer from the authors, OpenReview profiles for all authors, and the NeurIPS 2026 LLM-use policy checkbox.

## What winners look like (8 award papers, 2024-25)
1. They answer the workshop question: the *same* representation in *different* systems (models, brains, tasks), or how to measure sameness. None is about a single model alone.
2. **The title is a claim**, e.g. "Evidence from fMRI Supports a Two-Phase Abstraction Process in Language Models". Our title (b) is a claim: "Saying 'I Don't Know' Tracks Familiarity, Answer Uncertainty Tracks Knowledge".
3. Extended-abstract winners are clean cross-system empirical comparisons with **one quantitative link**.
4. Only one award involved LLM mechanistic interpretability (the 2025 emergent-misalignment shared-subspace paper), framed as "different tasks find the same directions".

## Consequences for our draft
- **Lead with the cross-system pattern**: the same dissociation in the entropy channel across 4 families (C2); the familiarity signal in base models of 3 families (C6); the human feeling-of-knowing parallel. The steering result (C4/C5) supports the claim; it is not the headline.
- **Scope honestly**: hedging is measured reliably only in Llama (R4). Say so in the abstract. A UniReps reviewer will check whether "tracks familiarity" is claimed for models where it wasn't measured.
- **Vocabulary the room uses**: "representation", "convergence across models", "shared vs model-specific", "source invariance" (Lippl, organiser), "mid-depth abstraction phase" (Cheng, organiser). Avoid circuit-paper jargon in the main text: "recovery", "twin prompts", "gated".
- **Structure for 4 pages** (J §5b, adapted to our claims):
  1. Question and claim (~½ page) with the human FOK hook.
  2. Setup: the popularity × knowledge factorial, the two prompt formats and why they differ (Methods note in PROJECT_CONTEXT), judges and κ (~½ page).
  3. Result 1: the 2 × 2 hedging vs entropy figure (Fig. 1), Llama hedging + 4-family entropy.
  4. Result 2: steering, familiarity vs knowledge directions with random nulls; the Che & Li-style caveat (Fig. 2).
  5. Result 3: the familiarity signal in base models via the Jacobian lens (Fig. 3).
  6. Related work (~⅓ page), limitations (~½ page), including no human validation of the judges.
- **References**: past extended abstracts carry roughly 20-35. Budget about 25.
- **Limitations**: the winners state limits plainly. Give one compact paragraph, not hedges sprinkled through the text.

## Gaps a UniReps reviewer will name (from J §5d, re-ranked for the current evidence)
1. "Hedging only in one model." This cannot be fixed before the deadline (judges failed κ for the other three). Make it the first limitation and keep the title's two halves scoped as measured.
2. "No quantitative cross-model similarity number." Optional in `evidence` mode, if time allows: the item-level correlation of popularity effects or entropy across families on shared PopQA items, with a permutation null.
3. "Steering ≠ mechanism." Cite che2026steering; report the no-hedge control.
4. "Numbers don't match." Every number comes from `numbers.tex`.
