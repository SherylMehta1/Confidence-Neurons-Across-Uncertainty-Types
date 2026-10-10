# What changed, and why: from the InterpScience draft to the UniReps camera-ready

For collaborators catching up. This summarises the work from 5 to 10 Oct 2026: the reframing, three rounds of simulated blind review, the experiments run in response, and how the claims moved. Each item names the file that holds the detail. The full dated record is [`LAB_LOG.md`](LAB_LOG.md), and the claim-by-claim status is [`CLAIMS.md`](CLAIMS.md).

## 1. In one paragraph

The paper is no longer about an uncertainty circuit. Its claim is that **in Llama-3.1-8B-Instruct, saying "I don't know" tracks entity popularity more than knowledge, while answer entropy tracks knowledge**. That claim survived three review rounds and ten pre-registered experiments, and every check made it more robust, not less. What changed is everything around it. Secondary claims that went beyond the evidence were withdrawn: the strong steering effect, the instruct-versus-base origin, and "Gemma shows no preference". Two new results were added: Gemma shows the same preference when the prompt permits "I don't know", and steering acts through a generic abstention signal even when that signal is fitted without any popularity contrast.

## 2. Timeline

| Date | What happened | Where |
|---|---|---|
| before 5 Oct | InterpScience workshop rejected the circuit paper ("Deciding to Say 'I Don't Know'"); the decision-vs-spread headline was retired on its own behavioural test | `../paper/` (old draft), CLAIMS.md R1 |
| 5-7 Oct | Four-family factorial, judge audits, 8-model lens study, steering; the literature review pointed to the popularity-vs-knowledge dissociation | `../results/OVERVIEW_2026-10-07.md` |
| 9 Oct | Paper rewritten for UniReps around the dissociation; figures and numbers generated from result files | OUTLINE.md, FIGURE_PLAN.md, numbers.tex |
| 10 Oct, round 1 | Blind review: methods major, framing major, clarity minor | review/round1/, revision/round1/RESPONSE_LETTER.md |
| 10 Oct, round 2 | ACCEPT with required changes | review/round2/chair_decision.md |
| 10 Oct, round 3 | E1-E6 run; review: three minor revisions, chair ACCEPT with camera-ready items 1-8 and encouraged analyses 9a-h | review/round3/chair_decision.md |
| 10 Oct, lock-in | E7-E10 run on six GPU boxes; camera-ready written, humanized and verified | results/round3/README.md, this file |

## 3. What the reviewers asked, and what we did

### Round 1 (major revision)
Main concerns: the title and abstract generalised a one-model result; obscure "known" facts might be guessable; the knowledge labels and the entropy readout come from the same sampling; the steering and lens claims were oversold; prior work (Lindsey, Ferrando, Kumaran) was missing.
**Response:** retitled to "In Llama, ..."; added a guessability check (post hoc); the shared distribution is stated openly and later tested (E1); steering and lens claims softened; citations added. Comment-by-comment table: [`revision/round1/RESPONSE_LETTER.md`](revision/round1/RESPONSE_LETTER.md).

### Round 2 (accept with changes)
Asked for the deciding observation up front (0.43 vs 0.00), honest measurement reporting, and a rewrite of the steering and lens claims. It also encouraged eight analyses (free-reply accuracy, free-prompt entropy, the lens split, an item-level model, entity knowledge, projection steering, among others).
**Response:** six of the eight analyses were run, several pre-registered as E1-E6.

### Round 3 (accept; camera-ready items binding)
The chair found four new overstatements and three disclosure gaps. Each is listed below with what changed. The full list is in [`review/round3/chair_decision.md`](review/round3/chair_decision.md).

| Chair item | Asked | Done | Paper |
|---|---|---|---|
| 1 | Withdraw "Gemma shows no popularity preference"; report the pre-registered four-family verdicts | Withdrawn. Verdicts reported (Qwen PARTIAL, Gemma REPLICATES, Mistral PARTIAL, on a readout that shows no preference in Llama either). Then E9: Gemma does follow popularity when the prompt permits "I don't know" | Abstract, 3.2, App. B |
| 2 | Fix "not seen in base models"; add instruct effects on the same readout or drop the inference | Both lens panels described. E10 run: instruct models lean further towards knowledge, so the base-vs-instruct inference is dropped | 3.3, App. D |
| 3 | Fix "only through its abstention component" (built in by construction) | Rewritten. E8 refits the abstention direction within obscure questions only: still carried by abstention | 3.3, App. C |
| 4 | Report the headline cells honestly | Stated that the headline difference equals obscure-known minus popular-unknown abstention. Intervals added for 0.43 and both entropies. Label cross-tab with kappa. E7 regrades unknown facts by meaning (40 of 100 popular-unknown were not unknown; result +0.23). Under 3% of "I don't know" replies contain the answer; strict declines +0.43 | 3.1, App. A |
| 5 | Reframe: a dissociation the misfire idea predicts, not a test it passed; name ambiguity | Reframed. Stated that neither model-side measure shows the names to be unfamiliar. Namesake check added (post hoc): ambiguity runs the wrong way | 1, 3.1, 4 |
| 6 | Disclose that the one-prompt check was chosen after the forced-prompt result | Disclosed, with that result beside it | 3.1 |
| 7 | Abstract scope; "five other measurements"; readable abstract | Done | Abstract, 3.1 |
| 8 | Housekeeping: UniReps link, citations (Brzezinka 2026b, Kang), figures, footer, hashes, counts, kappa interval | All done, including corrected hedge-token and answer-token definitions and a corrected criterion hash | throughout |
| 9a-h | Encouraged analyses | (a) E7, (b) consensus labels, (c) strict declines, (d) namesakes, (e) E10, (f) E8, (h) E9 run. (g) human audit not run, by decision | App. A-D |

## 4. New experiments and checks

Pre-registered: each had a criterion note hashed before the run (`PREREG/`). Index with files: [`../results/round3/README.md`](../results/round3/README.md).

| Exp | Question | Verdict |
|---|---|---|
| E1 | Do independent knowledge labels keep the result? | Yes: +0.32 [+0.16, +0.49]; entropy follows the new labels |
| E2 | Is name surprisal a familiarity measure? | No (uncorrelated with page views) |
| E3 | Is the base-model lens popularity effect in the hedge term? | Yes in 4/4, but knowledge moves it at least as much |
| E4 | Steering with a held-out dose | Below the bar (+0.09); carried by the abstention component |
| E5 | Third judge | Llama +0.44 (kappa 0.74); Gemma near floor; Qwen, Mistral unmeasured |
| E6 | Entity recognition as a covariate | Effect shrinks only 5% |
| E7 | Regrade unknown facts by meaning; strict declines | +0.23 [+0.12, +0.35]; strict +0.43 [+0.31, +0.54] |
| E8 | Steering with a within-obscure abstention direction | Carried by abstention (+0.01 left) |
| E9 | Prompt that permits "I don't know" | Llama +0.47, Gemma +0.30: both follow popularity |
| E10 | Instruct lens vs base | No increase; instruct leans towards knowledge |

Post hoc (labelled as such in the paper): popular-unknown alias audit (+0.19), consensus labels (+0.31), namesake covariate (+0.34), known-arm intervals and bootstrapped kappa (`analysis/lockin_checks.txt`), surprisal check (`analysis/surprisal_check.txt`).

## 5. How the claims moved

- **Strengthened:** the Llama abstention result (C1) and the entropy result in Llama (C2).
- **Added:** Gemma follows popularity when permitted (C12); steering acts through abstention even when it is fitted within one popularity level (C14).
- **Narrowed:** the base-model lens signal exists but is not a preference (C6); Gemma steering is equivocal (C5).
- **Retired:** the strong steering claim (R6), instruct = 3-4x base (R7), "Gemma shows no preference" (R8), "not seen in base models" (R9), "only through its abstention component" as worded (R10).

## 6. Mistakes we made and fixed

Recorded so nobody repeats them:
- **E8 launched with the two models' doses swapped.** Llama produced garbled text that the judges scored as abstention. The run was discarded, kept in `../results/round3/e8_INVALID_swapped_kstar/` and rerun.
- **E3 window bug** in the read-off: the wrong layer denominator. Fixed, and the values now match the original.
- **The lens "answer token" was misdescribed.** It is six answer-opening tokens, not the gold answer. Corrected in App. D.
- **One criterion hash in the paper was computed from a Windows (CRLF) copy.** Corrected to the committed note's hash. Criterion notes are now stored byte-exact.
- **The E4 amendment note carries a wrong internal date**, a day after its recorded hash time. This is disclosed in App. H.
- GPU boxes expired mid-run twice. Outputs are now synced every 15 minutes.

## 7. Still open (needs the authors)

- Author list, reciprocal reviewer, OpenReview profiles, LLM-use disclosure.
- Title scope: keep "In Llama" or widen it, given E9.
- Upload the criterion notes as an anonymised supplement (the paper says they are there).
- Human audit of the judges (not done).
- Make the repository private after submission (it is public now).
