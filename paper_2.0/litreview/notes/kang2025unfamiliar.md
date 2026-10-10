# kang2025unfamiliar: Unfamiliar Finetuning Examples Control How Language Models Hallucinate
Status: verified      Checked: 2026-10-10      URL: https://arxiv.org/abs/2403.05612
Authors / year / venue: Kang et al. / 2024 (arXiv) / venue: see VERIFICATION.md
WHY it was written: How do models hallucinate on unfamiliar queries?
HOW: Controlled SFT/RL/reward-model fine-tuning on TriviaQA and MMLU, varying the supervision of unfamiliar examples.
WHAT it found: Hallucinated predictions mirror the responses given to unfamiliar fine-tuning examples; supervising them with 'I don't know' changes behaviour.
Relation to us: supports
  Which claim id it touches: C1 (mechanism hypothesis)
  Training-side account; predicts hedging keys on unfamiliarity rather than retrieval failure.
Anchor: "an LLM's hallucinated predictions tend to mirror the responses associated with its unfamiliar finetuning examples" (abstract)
