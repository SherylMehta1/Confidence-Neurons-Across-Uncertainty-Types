# brzezinka2026familiarity: Graded Entity-Familiarity Readouts in Language Models: Polish Adaptation, Cross-Language Robustness, and Refusal Steering
Status: verified      Checked: 2026-10-10      URL: https://arxiv.org/abs/2607.13568
Authors / year / venue: Brzezinka et al. / 2026 (15 Jul) / arXiv preprint
WHY it was written: Can a model estimate its familiarity with an entity before answering?
HOW: Last-prompt-token probes in 12 instruction-tuned models (Bielik, PLLuM, Gemma-4, Qwen3), 1,440 Polish entities over 10 pageview deciles plus fabricated controls; familiarity steering in Gemma-4-12B.
WHAT it found: Probes separate real from fabricated entities in every family; popularity tracking rho 0.28-0.57 in Polish-adapted models vs <= 0.11 in Gemma-4 and Qwen3; steering moves refusal both ways.
Relation to us: supports
  Which claim id it touches: C1, C2, C4
  No crossing with certified knowledge of the specific fact and no entropy channel.
Anchor: "a separation between representational familiarity and the policy that converts it into abstention" (abstract)
