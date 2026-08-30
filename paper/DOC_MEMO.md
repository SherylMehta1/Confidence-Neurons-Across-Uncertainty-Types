# Memo: corrections + missing material for the draft doc

*For the team's "Draft Paper" doc · deadline Sept 1, 2026 AoE (long format ≤ 9 pp) · everything below is verified against the committed CSVs/JSONs on `uncertainty-circuit-experiments` (through `c070731`); the audited reference text is `paper/main.tex`.*

---

## 0. The one glossary everyone must use (direction semantics)

The patching direction label is **source → target** in every script. Verified from code (`twin_patching.py`, `circuit_faithfulness.py`, `circuit_direction_patch.py`, `circuit_position_map.py`) and the clean baselines (uncertain +2.62 / control −3.01 hedge log-odds).

- **c→u** (`control_to_uncertain`) = the **known** state patched **into the uncertain prompt** = *removing* the unknown signal → **switches the hedge OFF**. The headline faithfulness numbers (0.80 / 0.89 / 0.86) are this direction.
- **u→c** (`uncertain_to_control`) = the **unknown** state patched **into the known prompt** = *injecting* it → **switches the hedge ON** (1.07 / 0.78 / 0.98).

Three safe, quote-ready one-liners:

1. **Faithfulness:** "Jointly patching 20 heads + 100 neurons from the known twin into the uncertain prompt (c→u) switches the hedge off, recovering 0.80 / 0.89 / 0.86 of the twins' hedge log-odds gap on held-out pairs (Llama familiarity / Llama contested / Qwen contested); the reverse patch (u→c) switches it on at 1.07 / 0.78 / 0.98; size-matched random sets: 0.02–0.04."
2. **Directional asymmetry (claim 7):** "In Llama, removing the unknown state (c→u) transfers the hedge decision while transferring little of the entropy (entropy recovery −0.13 familiarity, +0.33 contested), whereas injecting it (u→c) transfers both (+0.45, +0.55); Qwen shows no such asymmetry (+0.83/+0.73)."
3. **Rank-1:** "Swapping only the rank-1 direction projection cannot switch the hedge off (c→u −0.10 contested vs +0.89 for the component set) even though it moves entropy (+0.57) — but it can switch the hedge on (u→c +0.97): the direction is the spread sub-channel, and un-hedging requires the component circuit."

---

## 1. Corrections to existing sections

### Claims 1–2 (Sheryl)

- **Substantive fix:** the gated-arm null does **not** rest on "the temperature-matched design of the twin pairs" (temperature-matching is the *frequency-taxonomy* control). It rests on the **random-neuron null** — every statistic recomputed for 100 (familiarity) / 50 (contested) random neurons ablated on the same prompts — plus the behavioral null. Result to quote: four familiarity candidates exceed all 100 randoms (largest L29_N5866, z = −6.2) but at ≤ 0.013 nats ≈ **0.5% of the 2.8-nat twin gap**; the contested pair reaches **z = +29.4** (L31_N11541, dz = 0.78) and **z = +14.8** (L30_N1457).
- Fill the two `xxxx` placeholders with: (a) the survivor characterization above; (b) the taxonomy stats — 9/55 frequency neurons by weights, causal under the temperature-matched baseline (stable across prompt sets r = 0.52–0.62; 6× random); one temperature neuron L31_N2477 (dose–response ρ = 0.86, ~100% RMSNorm-mediated).
- "Two neurons survive a stricter version of the test" → "survive **every** specification, including the random-neuron null" (it's not stricter, it's all of them).

### Claim 4 (Shagun)

- **"Entity ≈1.0 at L0–7" is direction-specific — must be qualified.** u→c: 0.98–1.05 at L0–5 (0.83–0.86 at L6–7), ≈0 by L14. c→u: at most 0.46–0.66 (L0–6), ≈0 by **L9**.
- **Dead zone L8–12 holds for c→u only** (u→c still carries 0.40–0.59 through L11; its minimum is L12).
- "last-token climbs to ~1.0" → "**0.76–0.86 by L29–30** (1.0 only at the trivial final-layer patch, L31)".
- "~10 verified heads per cell" → actual counts at rec_c→u > 0.05: **15 (familiarity), 6 (Llama contested), 4 (Qwen contested)** of 30 tested. State the threshold; no threshold gives ~10 everywhere.
- Interference layers: **L17–19, L25, L29–31** (largest −0.28 at L30, −0.23 at L17); L9 is −0.15 at 'all' positions. "L17–18 & L29–30" alone cherry-picks.
- MLP windows: Llama contested "**L11 and L13–14** (+0.25 to +0.35; L12 is slightly negative, −0.07) **and L21** (+0.55)"; Qwen "**L17–19** (peak +0.32 at L18) **and L21** (+0.26); L20 dips to +0.03".
- "We proved" → "we found no evidence under our tests"; Llama has exactly 32 layers.

### Claim 5 (Kanika)

- **Direction sentence is inverted** — use one-liner #1 above. Headline = c→u = known-into-uncertain.
- Familiarity CI upper bound is **0.89**, not 0.90. Llama-contested random baseline is **0.04** (0.06 was the old 10-set null; the shipped null is 30 sets: 0.018 / 0.040 / 0.044).
- Prune pairings: familiarity reaches 0.71 **at** 20 components but plateaus 0.85–0.89 only by **~60–100**; Qwen reaches 0.86 at 10 and plateaus 1.03–1.14 from **~20**. Plateau values **>1.0 are overshoot** (the patched circuit exceeds the clean-run effect) — flag, don't celebrate.
- CI method (the open item): **1.96·sd/√n over held-out pairs** (normal approximation, not bootstrap).
- Add after the rank-1 sentence: "though the direction does suffice in the other direction (u→c 0.97) — the 'why' is claim 7's."

### Abstract / Intro

- Merge the three stacked intro drafts (keep v3); one contributions list.
- The blanket "this circuit carries the hedging decision without carrying its output entropy" must become the **directional** claim (one-liner #2) — entropy transfers in 5 of 6 cells.
- Add "(overshooting only on the reverse direction)" after 78–107%.
- Title: the doc says "Uncertainty Has a Circuit"; the repo paper is "Deciding to Say 'I Don't Know': A Verdict Circuit, Not an Entropy Symptom." Pick one deliberately.

---

## 2. Missing material (ready to adapt)

### Claim 3 — the dataset standard (not yet drafted)

A twin pair is one manipulation with one change (obscure vs famous PopQA subject; two disagreeing sources vs one; many valid answers vs one), identical chat template and prefill, question length within 3 tokens. A pair enters an arm only if the model proves the labels: control correct in ≥ 9/10 samples, uncertain 0/10, entropy gap ≥ 0.5 nats, and (behavioral arm) hedging gap ≥ 0.3. Keep rates reported, never tuned: Llama 195/1151, 234/600, 41/120; Qwen 24, 178, 39; the situated arm dropped at 3/600 with the reason on record. Gate thresholds fixed before gating ran. Grader audited against a Qwen2.5-32B judge (agreement 94.6–100%, κ 0.70–1.00), with disagreement *lower* on hedged answers (1.5% vs 6.1%) — grader error cannot mimic the behavioral effect. **The cross-model keep-rate asymmetry (195 vs 24 from identical candidates) is itself a finding: "uncertain" is a property of the (model, prompt) pair.**

### Claim 6 — cross-arm overlap (not yet drafted)

Across Llama's two arms the component sets overlap partially: heads Jaccard 0.11 (shared: L12H28, L15H4, L15H7, L29H0), neurons Jaccard 0.11 (20 of 100), and recoveries of common verified heads correlate at Spearman ρ = 0.61. The types share a variable, not a full circuit — "one variable, two readouts," quantified.

### Claim 7 — asymmetry, verbalization, and the five-cell vocabulary (not yet drafted)

Use one-liners #2 and #3, then the lens material: the Jacobian-lens log-odds gap ignites exactly at the causal windows (familiarity flat to L12, 50% by L16; Qwen contested by L21) while the familiarity **entropy** gap reaches 50% only at L28 — in depth, the decision is readable ~12 layers before the spread. At the uncertain readout, the lens rank of ' unknown' falls ~1,100 (L13) → 10 (L17), single digits from L21, while the control readout never falls below ~300 and both entity spans stay in the tens of thousands (L13–30). Steering the direction at L15 on *known* twins reproduces the fall as a monotonic dose–response (margins narrowing above L26). **Five-cell vocabulary result** (`jlens_direction_tokens.txt`): each arm's decoded direction speaks its own vocabulary — familiarity a full hedge lexicon from L17 (' unknown', ' cannot', ' unable', ' unsure'); contested the conflict itself (' both'/' either' → ' conflicting'/' contradictory' → ' ambiguous'/' inconsistent', L11–25; ' unknown'/' I' never appear); aleatoric the answer type (' example' → ' any'). Both replicate in Qwen at L20–24, partly in Chinese (任意, 不同, 分歧); Qwen's mid-stack decodings are noise. Pre-stated prediction: aleatoric should show spread without decision — held cleanly in Llama (+0.30 vs +5.05/+4.75 log-odds; entropy +1.0), partially in Qwen (+3.67, ≈44% of contested, emerging only at L25–26) — report the half-failure as found.

### Related work (missing entirely)

Lead with **SCIURus** (Teplica et al., NAACL 2025: shared uncertainty/factuality circuits, 8 models, causal tracing + zero-ablation) — relative to it we add the gated-twin standard, twin patching, random-set faithfulness controls, neuron granularity, cross-type overlap, and the decision-vs-spread dissociation. Then Zhao et al. (COLM 2026, confidence-mover components, no random-set baselines), Singha Roy et al., Vazhentsev et al. (ICML 2026), Basu et al., Arora et al. (ICML 2026, MLP-only neuron-basis circuits), Du et al. (COLM 2025 — supports claim 1), Ferrando/Ji/Anthropic (directions), Stolfo/Gurnee (neurons), Kadavath/Gekhman (ground truth). Full BibTeX with verified author lists: `paper/references.bib`.

### Limitations (missing)

Both cross-model circuit cells are contested arms with near-zero free-generation hedging; Qwen's behavioral arm too small to test (itself a finding); the faithfulness criterion was fixed in code before the held-out runs but **not independently registered** (never say "pre-registered"); the hedge readout is a token log-odds proxy validated against free generation on familiarity only; the ablation reference for gated arms overlaps the evaluation distribution; the verbalization-rank analysis is Llama-familiarity (Qwen verbalizes via ' not' / its own vocabulary, via steering); two mid-size instruct models.

### Conclusion (missing — the doc ends without one)

A model's "I don't know" is not entropy leaking into words. It is a verdict — computed from the entity within eight layers, routed by a couple dozen components, made speakable mid-stack — that exists before, and moves independently of, the distributional spread most confidence methods measure. Three practices made this visible and are as much the contribution as the circuit: let the model certify its own uncertain prompts; test every component set against size-matched random ones; report which of your own criteria the evidence failed. Anyone probing an internal "feeling" in a model should start with: are you reading the decision, or only its shadow in the distribution?

### Figure inventory (takeaway-first captions; the doc has images 1–4 only — figs 4–5 are missing from it)

1. **Fig 1 (position map):** "The verdict leaves the entity by layer 8 and reaches the readout by layer 17" — entity moves the hedge only at L0–7 (u→c), readout positions only after L13.
2. **Fig 2 (MLP window):** "A single mid-stack window does the routing" — one-layer MLP patches recover the hedge only at L11–21, both models.
3. **Fig 3 (faithfulness):** "Under 2% of components carry 78–107% of the held-out hedge readout; random sets 2–4%."
4. **Fig 4 (lens gaps):** "An independent lens sees the decision before the doubt" — hedge gap half-size by L16, entropy gap not until L28. *(`paper/figures/fig4.pdf`)*
5. **Fig 5 (verbalization):** "At layers 13–17 the verdict becomes a word" — rank of ' unknown' falls ~1,100 → 10 at the uncertain readout only; steering reproduces the fall on known twins. *(`paper/figures/fig5.pdf`)*

---

## 3. Logistics

- **Deadline: Sept 1, 2026 AoE** (extended from Aug 28). Long format ≤ 9 pp (short ≤ 5 pp won't fit this paper).
- Compile: `paper/main.tex` builds standalone (venue `.sty` drops in next to it); `latexmk -pdf main.tex`.
- Interactive demo to link from the paper: the "Verbalization Moment" artifact.
- Everything quantitative above is reproducible from the repo (`RESULTS.md` has per-claim evidence pointers); three independent audit rounds have recomputed all of it from the raw CSVs.
