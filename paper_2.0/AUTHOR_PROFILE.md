# Author profile

Two layers, in this order of authority:
1. **The author's own voice**, measured from three of their papers. It decides sentence rhythm, labels, punctuation and spelling.
2. **A structure target from Neel Nanda's writing advice.** It decides what goes where: abstract order, introduction as summary, contributions list, finding-first labels.

The two agree on most points; where they differ, the author's voice wins.

## Voice samples (measured 2026-10-10 by `voice_stats.py` on the extracted prose)

| Sample | Authorship | Weight | Sentences | Mean words (median) | CV | Share <= 12 words | Notes |
|---|---|---|---|---|---|---|---|
| Dubey & Hoelscher-Obermaier, "Loss Landscape Response to Adversarial Perturbation Is Architecture-Dependent: Comparing ViT and ResNet", TAIS 2026 camera-ready | first author of two | **primary** | 109 | 17.3 (15) | 0.66 | 39% | "We" opens 15% of sentences; "We find" x5; passive about 5% |
| Bouadi et al., "Shaping the Prior" (arXiv 2605.18971) | 4th of 6 | secondary | 339 | 23.5 (22) | 0.63 | 17% | more "The/This" openers, more passive; likely mostly co-authors' prose |
| Mukhopadhyay et al., "AMBEDKAR" (arXiv 2509.02133) | 3rd of 8 | secondary | 184 | 24.0 (20) | 0.72 | 17% | as above |

Only the first-author paper sets the rhythm. The two multi-author papers count only where all three agree.

## The author's voice (consistent across all three unless noted)

- **Short, direct sentences** (primary sample): about 17 words on average, many under 12, with longer sentences reserved for detail.
- **"We" as the actor:** "We investigate", "We find", "We extend". Sentences open on the subject itself ("ViT shows...", "ResNet-18 exhibits...").
- **Details in parentheses:** counts, settings, intervals and examples ("(Hutchinson's trace estimator, N = 500 ImageNet samples, seven attack algorithms)"). About 85 parentheses per paper.
- **Colons and semicolons do the linking:** "We find strong architectural differences: ..."; frequent "X: Y" constructions. No em dashes.
- **Bold run-in labels in Title Case, ending with a period:** "Models & Data.", "Attack Suite.", "ViT Instability." "&" inside labels is fine.
- **A "Main Contributions." run-in followed by a numbered list**, each item a Title-Case label, a colon, then the finding ("1. Uniform ViT Response: ...").
- **Colon headings:** "ViT Response: Correlated Curvature".
- **Contrast markers:** "In contrast", "however", "on the other hand".
- **Hedges with "suggest"** and "more investigation is needed"; explicit "Interpretation:" and "Recommendation:" labels in discussion.
- **Limitations in their own section**, with sample sizes stated.
- **American spelling** (-ize, -yze, -or, "toward").

## Structure target (Nanda essay; sources below)

1. One to three concrete claims.
2. Abstract: what we investigate; what we find, with a number; the key evidence; the standard of evidence. No citations.
3. Introduction: a self-contained summary (context, background, what is new, our case), ending with the contributions list.
4. Headings and run-in labels state findings; captions state findings.
5. Limitations stated fully; post hoc results labelled; claim strength matches evidence; plain language; jargon defined.

Sources: N. Nanda, "Highly Opinionated Advice on How to Write ML Papers" (https://www.greaterwrong.com/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers), and the structure of Ferrando et al. (arXiv 2411.14257), Arditi et al. (arXiv 2406.11717) and Stolfo et al. (arXiv 2406.16254). No text from them is copied.

## Venues and audience

- Target venue: UniReps 2026 (NeurIPS workshop), extended abstract, 4 pages.
- Reader: an ML or interpretability researcher arriving cold.

## Policies

- **Dashes:** none (author never uses em dashes); en-dash ranges fine.
- **Hedging:** keep hedges that carry evidence strength. Never add "suggest" to a claim the evidence states directly; rewrites must not change claim strength.
- **First person:** "we".
- **Spelling:** American.
- **Terms (one per concept):** popularity preference; abstention / says "I don't know"; known / unknown; obscure / popular; pre-specified / post hoc; misfire.

## Things the author never wants changed

- Every number comes from a macro in `numbers.tex`.
- The Llama scope of the title, unless the authors decide otherwise.
- The "pre-specified unless marked post hoc" rule.

## AI-assistance policy

- Disclose AI assistance per the UniReps / NeurIPS policy at submission.
