# Response to the round-4 readability review (2026-10-10)

Raw review: [`REVIEWS_RAW.md`](REVIEWS_RAW.md). Draft before this round: [`main_before.tex`](main_before.tex). We thank the reviewer: the review found one factual error (the OLMo hedge term) and correctly diagnosed the main readability problem, which was pipeline provenance leaking into the prose.

Actions: **fix** (changed), **push back** (with the evidence pointed to), **partial** (fixed in part, with the reason).

| id | comment (short) | action | change and location |
|---|---|---|---|
| R4.1 | Llama 3 bib prints ~560 authors | fix | Every author list over ten names is truncated to "first author et al." (8 entries). The PDF dropped from 15 to 12 pages |
| R4.2 | Brzezinka 2026b arXiv ID unchecked | push back | Checked: arXiv 2607.07670 resolves to "Does Bielik Know What It Doesn't Know? ...", Brzezinka, 8 Jul 2026 (verify_citations: title match 1.00) |
| R4.3 | ALL-CAPS verdicts (PARTIAL, REPLICATES) undefined in 3.2 | fix | 3.2 states the test's two conditions in words ("found both in Gemma and one in Qwen and Mistral"); the App. B table row reads "conditions met: both hold / one holds" and the caption defines the conditions |
| R4.4 | About a dozen provenance tags, three names for one idea; "pre-registered" overclaims | fix | One rule in Setup: unless marked post hoc, every analysis followed a plan written and hashed before it ran (*pre-specified*). "(fixed in advance)", "pre-set" and "pre-registered" are gone from the paper; only "(post hoc)" marks remain in the main text. Table 2 has a Plan column. App. G says the hashes are self-recorded, not externally timestamped |
| R4.5 | Lab-notebook narration (discarded swapped-dose run; earlier regex version "overstated") | fix | Both sentences cut from App. C. The record stays in LAB_LOG.md and WHAT_CHANGED.md |
| R4.6 | App. H is a list of hash prefixes nobody verifies | partial | Kept, because the plans go in the supplementary material and the prefixes are how a reader checks them. Now a short table (analysis, plan file, prefix), with an explicit statement that the prefixes are self-recorded |
| R4.7 | App. G (circuit) has no methods and is never mentioned | fix | Cut. It was earlier work outside this paper's claims; its six citations were removed with it |
| R4.8 | Key abstract sentence unparseable | fix | Uses the reviewer's wording: "says 'I don't know' to 43% of questions it answers correctly in at least 9 of 10 forced samples; on popular entities the rate is 0%" (percent macros generated from the result file) |
| R4.9 | "Preference", "misfire", "default not released", "hedge tokens" undefined | fix | Setup defines the *popularity preference* (popularity effect minus knowledge effect, equal to obscure-known minus popular-unknown abstention). The intro defines *misfire* and its converse. Hedge tokens are defined at first use in 3.1 |
| R4.10 | Headline is a two-cell contrast; put a 2x2 table with n in the main text | fix | New Table 1: abstention and entropy for all four cells, with intervals and n. The Setup states the two-cell identity |
| R4.11 | "Alternatives" packs six checks; opaque sentences | fix | Split into question-headed paragraphs (labels right? refusals? recognition? ambiguity? familiarity? two prompts?). Numbers moved to Table 2. The name-form detail moved to App. A with units ("regex abstention gap between matched pairs") |
| R4.12 | 3.2 title and the "One prompt" label say nothing | fix | 3.2 is now "Other families: Gemma tracks popularity too when it may say 'I don't know'"; the paragraph is "Does the split need two prompts?" |
| R4.13 | 3.3 cites a figure in the appendix | fix | Figure references now name the appendix ("Appendix D, Figure 3"). The steering figure also moved to App. C to keep four pages |
| R4.14 | No conclusion | fix | Section 4 is "Conclusion and limitations" and opens with the conclusion |
| R4.15 | Abstract lists the by-construction entropy result as a finding | fix | The abstract presents it as a control ("As a control, answer entropy follows knowledge, as the labelling implies"), and so does 3.1 |
| R4.16 | Platonic hook promises a cross-family comparison | fix | The intro now says the question can be asked only where abstention is measurable, here Llama and, under a permissive prompt, Gemma. The conclusion answers it in those terms |
| R4.17 | Feeling-of-knowing misstated; Koriat says both contribute | fix | "depends on how familiar its terms are as well as on whether the answer can be retrieved", citing Koriat and Levy-Sadot alongside Reder and Schwartz |
| R4.18 | Fig. 3 title contradicts panel (a); OLMo hedge term reversed | fix (factual error) | Correct: OLMo's hedge-term popularity +0.61 exceeds knowledge +0.53 (intervals overlap). The title now covers panel (b) only ("about as much to knowledge"), the caption describes both panels and names the OLMo exception, and 3.3 no longer says "at least as much in each" |
| R4.19 | Table rows for entity recognition and namesakes both +0.34 [+0.27, +0.41]: copy-paste? Coefficients in a difference column | push back + fix | Not a copy-paste: different models, +0.337 [+0.266, +0.411] and +0.340 [+0.267, +0.412] (results/round3/e6_summary.txt; analysis/ambiguity_check.txt), equal at two decimals. Table 2 now puts the coefficients in their own block ("coefficient of the obscure arm"), below the no-covariate baseline (+0.35) |
| R4.20 | "Five other measurements" but the table lists about ten | fix | Table 2 is split into measures of abstention (primary plus five others: the "five"), other knowledge labels, and regression controls; the text says "five other measures of abstention" |
| R4.21 | Zero-width bootstrap intervals on integer onsets | partial | Kept, because they are the inputs to the pre-specified onset rule. The caption now says that integer onsets make intervals collapse to a point and that they are not a precision estimate |

## Checks after the edits

- Main text ends on page 4; the PDF has 12 pages; no overfull boxes; no undefined references.
- check_numbers: 16 of 16 specific literals found in result files, 0 missing.
- verify_citations --online: 33 verified. Two mismatches predate this round and are benign: the ICML title prefix "Position:" for Huh et al., and Rimsky published as Panickssery (the DOI matches). One web page (Lindsey et al.) is unresolved, as before.
- verify_rewrite main_before.tex vs main.tex: every macro it flags as dropped either still appears (moved into tables) or belongs to the deliberate cuts (App. G, the superseded steering run). Two numbers that fell out of the prose by accident were restored in App. A: the independent-label entropy check (+1.40) and the 73 unique-name facts.
