# ji2025hedging: Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations
Status: verified      Checked: 2026-10-10      URL: https://arxiv.org/abs/2503.14477
Authors / year / venue: Ji et al. / 2025 / venue: see VERIFICATION.md
WHY it was written: LLMs state false claims assertively.
HOW: Find a linear verbal-uncertainty feature; compare with semantic uncertainty; intervene at inference.
WHAT it found: Verbal uncertainty is a single linear feature with only moderate correlation to semantic uncertainty; the mismatch predicts hallucination; intervention cuts confident hallucinations ~30%.
Relation to us: supports
  Which claim id it touches: C1, C2
  They show verbal != semantic uncertainty; we show what each follows (familiarity vs knowledge).
Anchor: "has only moderate correlation with the actual 'semantic uncertainty' of the model" (abstract)
