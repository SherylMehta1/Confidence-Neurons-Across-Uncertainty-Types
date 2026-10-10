# Hedging measured properly (96-token answers, two judges, three-way labels): criterion, stated before the run

Written 7 Oct 2026, after the judge audit (`results/hedge_judge/FINDINGS.md`) showed two things. The regex misses 86-90% of Qwen, Gemma and Mistral hedges. The 24-token free generations are often cut off before the answer appears, which makes single-judge corrections unreliable. Nothing below has been run.

## Decision fixed in advance

If the two judges agree adequately (Cohen's κ ≥ 0.60 on the primary label), **this measurement supersedes the regex** for the factorial hedging comparison in every model, including Llama. The paper reports it as the hedging result, and the earlier locked regex verdicts as a superseded first measurement. If κ < 0.60, the hedging comparison is reported as unmeasured for the models where the judges disagree, and the regex verdicts are reported with their known recall.

## Generation

- **Models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3.
- **Items:** from each model's own factorial pool (HighlyKnown and Unknown items), a stratified random sample of 100 items per popularity × knowledge cell (all items if the cell has fewer), seed 0.
- **Prompt:** the free-generation prompt, `format_chat_prompt(q)`, identical to the original hedge-rate prompt.
- **Decoding:** 3 samples per item, T = 0.7, **96 new tokens** (originally 24), and the **full generated text** is kept. The original `sample_answers` keeps only the first line of each reply, which also cuts list-style answers ("…include:" followed by a list); that truncation is removed too. *(This clarification was added before anything ran.)*

## Labelling

Two judges, each greedy, with the same fixed prompt: **Qwen/Qwen2.5-32B-Instruct** and **unsloth/gemma-2-27b-it**.

> You are checking replies from an AI assistant to a factual question. Classify the reply into exactly one category.
> DONT_KNOW: the reply says or clearly implies that the assistant does not know the answer, cannot find or verify the information, or does not recognise the subject (even if it then offers a guess).
> CLARIFY: the reply does not answer and instead asks which entity is meant, or says the question is ambiguous or needs more context, without saying it does not know.
> ANSWER: the reply commits to a specific answer (possibly with a mild qualifier such as "probably").
> Question: {question}
> Reply: {reply}
> Respond with exactly one word: DONT_KNOW, CLARIFY or ANSWER.

- **Primary hedge measure:** a sample counts as hedged if **both** judges say DONT_KNOW. The secondary measure is "both judges say DONT_KNOW or CLARIFY".
- **Agreement:** κ between the two judges on DONT_KNOW vs not, per model and overall.
- **Item hedge rate** = the fraction of its 3 samples that are hedged.

## Analysis (per model, primary measure)

- Popularity and knowledge effects and their difference on item hedge rates, over the four cells, with the 2 × 2 definitions of `qwen_factorial_criterion.md` and a 10,000-sample within-cell item bootstrap.
- **Condition 1:** popularity effect − knowledge effect > 0, CI excluding 0. Report it, and also the sign when the CI excludes 0 on the negative side ("hedging follows knowledge more than familiarity").
- **Regex recall on the new generations** (P(regex | primary hedged)), to show whether the 24-token truncation or the detector itself drove the earlier gap.

## What may be claimed

- **"In model X, saying 'I don't know' follows how familiar the entity is more than whether the model knows the fact":** only if condition 1 holds for X on the primary measure and κ ≥ 0.60 for X.
- **"…follows knowledge more than familiarity":** only if the difference's CI lies entirely below 0 (with κ ≥ 0.60).
- **Cross-model wording:** the count of models in each category, with all four reported. No title claim beyond what the count supports.

## Not to be done after seeing results

Changing the prompt, the judges, the token budget, the sample sizes, the "both judges" rule or the κ bar; switching the primary measure.
