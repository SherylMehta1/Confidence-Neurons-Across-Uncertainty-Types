# E4 amendment (written 2026-10-10 ~22:10 UTC, before any E4 reply was judged)

The EVAL replies are also scored by the E5 third judge (unsloth/Llama-3.3-70B-Instruct-bnb-4bit, same prompt). The majority-of-three rule (≥ 2 of 3 DONT_KNOW) is reported as a SECONDARY scorer for D1-D3.

The primary scorer stays "both of the two original judges say DONT_KNOW", as fixed in e4_steering_v3.md (sha 18c74883). Nothing else changes.

Reason: E5 showed that the two original judges disagree on non-Llama families; Gemma-27B is the low-agreement judge. A majority rule is more robust for Gemma's steered replies.
