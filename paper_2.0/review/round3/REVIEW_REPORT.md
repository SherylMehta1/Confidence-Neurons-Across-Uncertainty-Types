# Review report, round 3 (2026-10-10): main.pdf at commit f723070

**Final decision (area chair): ACCEPT**, conditional on camera-ready items 1-8 in chair_decision.md. These are rewording and tabulation of existing data. "A new claim not covered by this list is grounds for withdrawing acceptance."

**Referees:**
- methods: minor revision;
- framing: minor revision;
- clarity: minor revision;
- no reject.

**Red team:** the attack is referential ambiguity, plus "page views are not familiarity". The adjudication finds 0 critical, 5 partial and 1 unresolved (ambiguity).

**Numbers audit:** 370 items; no wrong number; all 13 hashes match. Two critical wording issues, which the chair lowered to major:
- K1: Gemma's pre-registered forced-readout verdict REPLICATES is unreported;
- K2: "not seen in base models" holds only for the hedge term.

**Compliance with round 2:**
- rank 1 addressed;
- ranks 2-8 partly addressed (rank 4 has one element made worse);
- rank 9 largely addressed (6 of 8 encouraged analyses run).

**Post-review analyses already done on existing data** (post hoc; analysis/round3_label_checks.*), answering chair items 4 and 9(a)-(c):
- popular-unknown audit: contrast +0.189 [+0.069, +0.307] after dropping questions with a gold alias in any free reply;
- original vs paraphrase labels: κ 0.79; contrast on agreeing items +0.314 [+0.147, +0.480];
- 0% of both-judges-DONT_KNOW replies in the obscure cells contain the gold alias, which refutes the "about a quarter" estimate.

**Item 9(d):** the homonymy covariate is running (analysis/ambiguity_check.py; Wikidata rate-limited).

**Files:** reviewer_{methods,framing,clarity}.md, attack_memo.md, adjudication.md, numbers_audit.md, chair_decision.md.
Independence note: all agents are the same model family as the author agent.
