# ferrando2025iknow: Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models
Status: verified      Checked: 2026-10-10      URL: https://arxiv.org/abs/2411.14257 (+ PDF pp. 3-4)
Authors / year / venue: Ferrando et al. / 2025 / ICLR 2025 (PDF header)
WHY it was written: Hallucination mechanisms are poorly understood; does the model represent whether it can recall facts about an entity?
HOW: SAE latents at the entity's final token in Gemma 2 2B/9B; entities labelled known (>= 2 attributes correct) or unknown (all wrong); latents steered in the chat model.
WHAT it found: Known/unknown entity latents are causal: steering makes the chat model refuse known entities or hallucinate unknown ones; base-model SAEs transfer to chat.
Relation to us: reframes
  Which claim id it touches: C1, C4
  Their 'known' confounds recognition with fact knowledge; we cross popularity with per-question certified knowledge.
Anchor: "Known entities are those where the model gets at least two attributes correct, while unknown are where it gets them all wrong" (Sec. 3)
