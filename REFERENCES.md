# References

## Mechanism
- Gurnee, W. et al. (2024). Universal Neurons in GPT-2 Models.
- Stolfo, A., Belinkov, Y., & Sachan, M. (2024). Confidence Regulation Neurons in Language Models. NeurIPS 2024.
- Menta, T. et al. (2025). Transformers Don't Need LayerNorm at Inference Time. arXiv:2507.02559.
- Context Copying Modulation: The Role of Entropy Neurons in Managing Parametric and Contextual
  Knowledge Conflicts (2025). arXiv:2509.10663.
- How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and
  Confidence (2025). arXiv:2504.02904.
- nostalgebraist (2020). interpreting GPT: the logit lens. (blog post, origin of the logit lens technique)
- Wang, K. et al. (2023). Interpretability in the Wild: IOI Circuit in GPT-2 small. ICLR 2023.

## Datasets
- Liu, G. et al. (2024). Examining LLMs' Uncertainty Expression Towards Questions Outside
  Parametric Knowledge (UnknownBench). arXiv:2311.09731.
- Min, S. et al. (2020). AmbigQA: Answering Ambiguous Open-domain Questions. arXiv:2004.10645.
- Tighidet, Z. et al. (2024). Probing Language Models on Their Knowledge Source. arXiv:2410.05817.
- Elazar, Y. et al. (2021). Measuring and Improving Consistency in Pretrained Language Models (ParaRel).

## Reading order (do this before writing any code)
1. Gurnee et al. (2024) — most accessible, sets up vocabulary
2. Stolfo et al. (2024) — central mechanism paper, everything else builds on this
3. Menta et al. (2025) — gives you the exact ablation protocol
4. Post-training study (arXiv:2504.02904) — motivates the whole project, read last
