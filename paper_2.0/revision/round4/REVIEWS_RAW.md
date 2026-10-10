# Round 4: external readability review (pasted by the user, 2026-10-10)

Source: pasted into the session by the user; reviewer not named. Kept verbatim.

---

Verdict: not readable on a first pass. Lexical slop is low, but the paper reads like unedited output from an automated analysis pipeline. No fabricated references.

Citations
The 2026 references resolve to real papers with matching titles: Brzezinka's Bielik paper is arXiv 2607.07670, Che and Li is arXiv 2610.07270, submitted 5 Oct 2026, Liu et al. is arXiv 2608.15772, Nguyen et al. is the Commit-Abstain Circuit paper, Kumaran et al. is arXiv 2603.22161, and Gurnee et al. is arXiv 2607.15495. For Brzezinka 2026b, the title appears on the author's page, but I did not check the ID. Separately, the bib prints all ~560 Llama 3 authors and eats about 2.5 pages. Truncate it.
arxiv.org +6

Slop (pipeline leakage)

ALL-CAPS verdict strings (PARTIAL, REPLICATES) sit in §3.2 prose and are never defined.
About a dozen provenance tags in four pages: "(fixed in advance)" ×7, "pre-set", "pre-registered", "(post hoc)". Three names for one idea. "Pre-registered" overclaims: the hashes are self-recorded and one note admits "a dating error in the note itself". Say "pre-specified" and move provenance into a Table 1 column.
Lab-notebook narration: the swapped-dose run discarded for degenerate text, the earlier regex version that "overstated the effect". Cut both.
Appendix H is a list of 8-hex hash prefixes. Nobody verifies these.
Appendix G (20 heads, 100 neurons, patching fractions) has no methods or figure and is never mentioned in the main text. Cut it or promote it.

Readability

The key abstract sentence is unparseable. Try: "On obscure entities, Llama-3.1-8B-Instruct says 'I don't know' to 43% of questions it answers correctly in at least 9 of 10 forced samples. On popular entities the rate is 0%."
"Preference" is the paper's core technical term (popularity effect minus knowledge effect) and is never defined. "Misfire" and "default not released" are also undefined. "Hedge tokens" is used in §3.1 but only defined in the Fig. 3 caption and Appendix D.
The headline metric is really a two-cell contrast (obscure-known n=87 minus popular-unknown n=100) dressed as a difference of marginal effects. Put a 2×2 table of rates with n in the main text.
The §3.1 "Alternatives" paragraph packs six checks into one block. Split it and move the numbers to Table 1. "Ambiguity runs the wrong way" and the matched-pairs name-form sentence are opaque.
The §3.2 title and the "One prompt" paragraph label say nothing.
§3.3 cites Figure 3, which is in Appendix D.
§4 has no conclusion, only limitations.
The abstract lists the by-construction entropy result as a finding. Move it to a control.
The intro's platonic-representation hook promises a cross-family comparison. Abstention is only estimable in Llama, and in Gemma under the permissive prompt.
The intro says feeling-of-knowing follows cue familiarity "not whether the answer is stored". Your own Koriat and Levy-Sadot cite says both contribute.

Internal inconsistencies to fix

The Fig. 3 title says the hedge readout responds "at least as much to knowledge". Table 4's full readout has popularity above knowledge for Llama (+0.67 vs +0.36), Gemma-2B (+0.40 vs +0.11) and OLMo (+0.29 vs −0.30). Only panel (b) fits the title. For OLMo's hedge term the point estimate is also reversed (+0.61 vs +0.53).
Table 1's entity-recognition and namesake covariate rows are both +0.34 [+0.27, +0.41]. Check for a copy-paste. Those rows are regression coefficients sitting in a column headed "Popularity − knowledge".
The text says the sign holds "under five other measurements", but Table 1 lists about ten.
Table 6 shows zero-width bootstrap intervals on onset layers ([0, 0], [−5, −5]). A bootstrap on an integer layer index is not informative there.
