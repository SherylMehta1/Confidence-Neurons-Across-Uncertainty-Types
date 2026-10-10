# NeurIPS 2026 "Interpretability as a Science" workshop: decision and reviews for Submission 195

Saved 2026-10-06 from the team's own paste of the OpenReview author page. This file is the verbatim text
the user pasted into the Claude Code session
`C:/Users/starb/.claude/projects/c--Users-starb-Desktop-shshk/877bb149-8657-41db-b137-cd9d45abc3d3.jsonl`
(transcript line 868, timestamp 2026-09-30T03:38:29Z = 09:08 IST on 30 Sep 2026; a repeat of the second review's
last paragraphs at line 881, 2026-09-30T03:46:05Z). Nothing below was edited, except that OpenReview's
run-together header lines are split for readability. Typos ("trhe", "abstrac") are the reviewers' own.

Paper as submitted: "Doubt Is a Decision: A Sparse Circuit Commits Language Models to 'I Don't Know'",
17-page PDF (9 pages main text + 2 pages appendix + NeurIPS checklist), built 2026-09-02 08:46 UTC.
Byte-identical copies: `C:/Users/starb/Downloads/Doubt_Is_a_Decision.pdf` and
`C:/Users/starb/Desktop/papers/195_Doubt_Is_a_Decision_A_Spar.pdf` (md5 026e65e2cfd176f886f3907f9f98030b).
Source: `C:/Users/starb/Desktop/shshk/overleaf_submission/main.tex` (301 lines, outside git).

Timeline (OpenReview timestamps as displayed in the author's browser; the paste does not show a timezone,
most likely IST):

- Submission deadline: 1 Sep 2026 (CFP, https://interpscience.github.io/cfp).
- Reviewer 14N6 review: 13 Sept 2026, 22:38. Reviewer gbEf review: 19 Sept 2026, 13:02.
- Area Chair e8tZ meta review: 29 Sept 2026, 08:29 (modified 30 Sept 2026, 07:35). Reviews made visible
  to authors ("modified") 30 Sept 2026, 06:38. CFP notification date: 29 Sep 2026.
- The team had read the decision by 09:08 IST on 30 Sep 2026 (the paste). `shshk/EXPERIMENT_PLAN.md:3` says
  "rejected on 1 Oct 2026"; that is the date the plan was written, not the decision date.

---

## Verbatim paste (OpenReview author view)

```
Decision: Reject
Meta Review of Submission195 by Area Chair e8tZ
Meta Review by Area Chair e8tZ  29 Sept 2026, 08:29 (modified: 30 Sept 2026, 07:35)  Program Chairs, Area Chairs, Authors, Reviewers Submitted  Revisions
Recommendation: Reject

Review of doubt is a decision
Official Review by Reviewer gbEf  19 Sept 2026, 13:02 (modified: 30 Sept 2026, 06:38)  Program Chairs, Area Chairs, Reviewers Submitted, Authors  Revisions
Review:
The paper essentially looks for circuits and neurons that changes based on uncertainty-related-words using activation patching. The idea is that uncertain prompts are annotated rather than verified and single-component accounts conflate the decision to hedge with the spread of the output distribution. Using their experimental setup they are able to identify uncertainty directions and steer using activation patching. The paper is very difficult to read in my opinion, and they use a lot of unnecessary jargon and overly-specific language throughout. This dense style is a weakness of trhe paper in my opinion. In terms of the contribution, it seems to be in the comparison and identification of these circuits as opposed to a new methodology or a general phenomena identified. I would say the strengths are the testing and comparisons seem to be fair and the experiment is decently well set up. I will rate this as accept but I don't really follow all the parts of the paper despite spending some time on this. In terms of the novelty, I could not find anything that was exactly covering this question, but I am not sure if this result is particularly significant due to the lack of generality and breadth of models considered.

Rating: 3: Accept
Confidence: 2: The reviewer is willing to defend the evaluation, but it is quite likely that the reviewer did not understand central parts of the paper

substantially AI generated, abstract is extremely difficult to understand due to unsubstantiated jargon

Official Review by Reviewer 14N6  13 Sept 2026, 22:38 (modified: 30 Sept 2026, 06:38)  Program Chairs, Area Chairs, Reviewers Submitted, Authors  Revisions
Review:
the abstract is very hard to understand. pretty much every sentence introduces terminology that is not standard in the field and not defined. specifically: "flat output" is not defined but I assume means high entropy next token distribution. what does it mean to annotate or verify an uncertain prompt? what is a single component account? "we separate the two" -> which two specifically, do you mean the overall research Q of decision to say IDK vs high entropy distribution? what is meant by "gated", and how does the model certify a prompt using those 3 things you mention? what do you mean by token frequency and temperature machinery in the context of "resolve into anything at all"? what is the entity you're talking about? what is the familiarity arm? what are the "cells" (3 that can run and fourth with too few certified pairs)?

I also don't see how a circuit can "commit a language model to IDK", a circuit is a post-hoc description, not itself the cause of "IDK".

the paper title and abstrac overclaim, selling the work as general IDK but in fact they only evaluate on 3 very specific types of prompts where deliberate IDK computation is expected:

unknown entities like fake names
knowledge conflicts
multiple correct answers the abstract does not mention this limitation at all.
rather than e.g. getting stuck on math and not knowing whether or where it's stuck, which is arguably the interesting part of this space.

Rating: 1: Clear rejection
Confidence: 3: The reviewer is confident but not absolutely certain that the evaluation is correct
```

## Notes on the paste

1. The meta review carries only "Recommendation: Reject"; no AC text was pasted. Either the AC wrote nothing
   beyond the recommendation or the paste cut it. [unverified; check OpenReview]
2. The line "substantially AI generated, abstract is extremely difficult to understand due to unsubstantiated
   jargon" sits between Reviewer gbEf's confidence line and Reviewer 14N6's header. Its field is not shown.
   It reads like a short free-text field (for example a "summary" or "flag" box) attached to one of the reviews
   or to the meta review. Treat the "substantially AI generated" judgement as a third, separate complaint,
   whoever wrote it. [field attribution unverified]
3. Reviewer 14N6's paragraph about the three prompt types came out of OpenReview as a run-together list:
   "unknown entities like fake names / knowledge conflicts / multiple correct answers" are the three types, and
   "the abstract does not mention this limitation at all" is the next sentence.

## What the two reviewers actually asked for (from the text above, not the 5 Oct paraphrase)

- gbEf (Accept 3, confidence 2): readability ("unnecessary jargon and overly-specific language throughout",
  "dense style"); contribution read as "comparison and identification" rather than "a new methodology or a
  general phenomena"; experiments "fair" and "decently well set up"; novelty: "could not find anything that was
  exactly covering this question"; significance doubted for "lack of generality and breadth of models".
- 14N6 (Clear rejection 1, confidence 3): nine undefined terms in the abstract, listed one by one ("flat output",
  annotate vs verify, "single component account", "we separate the two", "gated", model "certify", "token
  frequency and temperature machinery", "entity", "familiarity arm", "cells"); the causal objection ("a circuit is
  a post-hoc description, not itself the cause"); title/abstract overclaim ("selling the work as general IDK");
  the three prompt types are ones "where deliberate IDK computation is expected"; the abstract "does not mention
  this limitation at all"; the missing case is reasoning ("getting stuck on math and not knowing whether or where
  it's stuck").
- Unattributed: "substantially AI generated".

Related but different: `C:/Users/starb/Desktop/shshk/reviews/` (five files, 2 Sep 2026 03:57-04:11 IST) is a
simulated pre-submission "cold panel" (R1 statistics, R2 technical, R3 construct validity, mentor, AC; all 6/10,
"accept (poster)"), described in commit `e883412`. Those are not the workshop's reviews.
