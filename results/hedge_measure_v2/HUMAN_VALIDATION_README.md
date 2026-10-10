# Human check of the two hedge judges (Llama-3.1-8B-Instruct)

Why: Llama's headline hedging result rests on two LLM judges whose agreement (kappa 0.62) only just clears the 0.60 bar. A person labelling a blind sample grounds it.

1. Open `human_validation_sheet.csv` (120 replies, 30 per popularity x knowledge cell, shuffled, seed 0). Do NOT open the KEY file first.
2. In the last column write exactly one of:
   - DONT_KNOW: the reply says or clearly implies the assistant does not know, cannot find or verify the information, or does not recognise the subject (even if it then guesses);
   - CLARIFY: it does not answer and instead asks which entity is meant or says the question is ambiguous, without saying it does not know;
   - ANSWER: it commits to a specific answer (a mild "probably" is still ANSWER).
3. Ideally two people label independently.
4. Then run, from the repo root:
   `python scripts/hedge_measure_v2_human_score.py results/hedge_measure_v2/human_validation_sheet.csv`
   This reports agreement (kappa) of each judge and of the "both judges" rule with the human labels.

Report the result in the paper's appendix whatever it shows.
