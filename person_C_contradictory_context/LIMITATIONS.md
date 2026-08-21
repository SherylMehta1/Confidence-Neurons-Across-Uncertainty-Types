# Contradictory Context — Known Limitation

## Contradictory context does not show a clean uncertainty-vs-control gap

The category's construction ("Redefine: [false/true fact]. [base_prompt]", with the
assistant turn prefilled to genuinely continue the sentence rather than open a fresh
chat turn) was verified to resolve the turn-boundary artifact present in an earlier
version -- generation inspection confirmed the model completes the sentence directly
(e.g., "Adobe Flash is created by" -> "Adobe Systems, not Apple") rather than producing
disclaiming preamble ("I couldn't find any information...").

However, once that artifact was fixed, a second and more fundamental issue emerged:
the model resolves the injected contradiction with roughly the same confidence it shows
when simply restating the true fact with no conflict present at all. On a 25-item sample,
working (contradiction) prompts averaged top1=0.850 versus 0.804 for matched true-object
controls -- no meaningful separation, and directionally the wrong way (contradiction more
confident than control).

We hypothesized this might be driven by the parametric-knowledge filter selecting only
very well-known facts (major tech companies, common geography), where the model's prior
is strong enough that a single contradicting sentence produces little genuine conflict.
To test this, we piloted a bounded variant of the knowledge filter requiring moderate
rather than maximal base-fact confidence (top1 between 0.3 and 0.85). This did not
produce a gap either (n=15: working mean=0.784, control mean=0.804) -- matched pairs such
as "Shablykinsky District... Belarus" (0.959) vs. its true-fact control (0.996) show the
same pattern at smaller scale.

We treat this as a genuine finding rather than a bug to engineer around: for this model
and this contradiction construction, factual correction appears to happen via confident
override rather than sustained distributional uncertainty, at least for CounterFact-style
single-fact conflicts. This is consistent with the category's original "hardest to get
clean causal results from" ranking and with Context Copying Modulation (2025)'s finding
that Llama-3-8B's context-conflict behavior doesn't transfer cleanly from GPT-2-style
entropy neuron signatures. Any downstream causal results (Phase 3/4) for this category
should be interpreted with this limitation in mind -- a null or weak result for
contradictory-context neurons may reflect genuinely weak induced uncertainty in the
dataset rather than absence of a shared mechanism.

## Reproducibility

The committed `data/contradictory_context/prompts.jsonl` and `controls.jsonl` were generated
with the NF4-quantized model, and the knows-fact filter that selected their CounterFact rows
measured the base prompt in an **unprefilled** position (the bare `base_prompt` as the user
turn, fresh assistant turn) rather than the prefilled assistant-turn position the data itself
is measured at. Both are fixed in the current `preprocess_contradictory_context.py` (bf16
required, knows-fact check at the prefilled position, pinned dataset revisions, logged in
`knows_fact_log.jsonl` / `provenance.json`). The committed files must be regenerated with
that script in bf16 before any result built on them is reported:

    python person_C_contradictory_context/preprocess_contradictory_context.py
