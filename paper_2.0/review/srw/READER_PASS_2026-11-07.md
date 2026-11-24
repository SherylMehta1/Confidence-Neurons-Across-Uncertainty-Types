# Reader pass on the EACL SRW mentorship draft (2026-11-07)

Three blind readers (outsider student, skeptical referee, fact-checker), then fixes in srw/main.tex.

Fixed:
- Sign error: the conclusion said knowledge "raises" abstention; it is missing knowledge that raises it. Lens sentence reworded the same way.
- "Natural Questions" read as the NQ dataset; now "real-world questions" / "questions about real entities".
- Abstract named one set of three relations for two different sets; both now named.
- +0.18 cited to a table that did not show it; now in the Appendix A caption.
- "Holds across other judges and labels" -> "stays positive ... though it shrinks under some".
- Primary measure now marked as adopted post hoc (Design and Table 3).
- Entropy "follows knowledge, not popularity" -> "much more than popularity" (unknown cells differ by about 0.9 nats).
- kappa: above the bar at the point estimate, not across the interval.
- Section 5 heading no longer says familiarity is ruled out.
- Lens claim now names four base models and points to a table (Appendix B).
- Jargon simplified: arms -> groups, preference -> popularity margin, rho -> rank correlation, steering described as adding fixed vectors, cosine -> cosine similarity, post hoc defined once.
- Kalai et al. no longer credited with "worst" hallucinations; Gekhman et al. labels described as adapted.

Checked and clean: all 13 citations exist (verify_citations; Lindsey et al. is a web page); cited claims supported (Reder and Ritter, Koriat and Levy-Sadot: abstracts could not be opened, titles match the claim); method statements match the scripts; every number is a macro traced to a result file.

Not fixed (honest limits, stated in the paper): country's +0.69 has no interval; the within-relation comparison is underpowered; regex 20-reply count confirmed only via the script default.
