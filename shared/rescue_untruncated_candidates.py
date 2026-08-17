"""
rescue_untruncated_candidates.py -- recovers the full split-half-stable
neuron set WITHOUT re-running detection.

detect_candidate_neurons_split_half() already found 17 neurons that landed
in the top-60 (by |correlation|) of BOTH independently-computed halves --
that agreement IS the validation criterion. save_candidate_neurons_v2()
then truncated to top_k_final=15 before writing candidate_neurons.json,
silently dropping 2 neurons that had already passed the real bar. 15 was
just matching the old Phase 2 candidate count for comparison, not a
methodologically meaningful cutoff -- once split-half agreement is met,
there's no principled reason to drop the last 2 by rank alone.

full_correlation_distribution.json already contains both halves' complete
correlation dictionaries, so this just re-derives top_a/top_b/stable from
that file (same top_k_per_half=60 used originally) and re-saves
candidate_neurons.json with the FULL stable set, untruncated. No model
needed, no GPU needed -- this is pure post-processing on what you already
computed.

Run with:
    exec(open("shared/rescue_untruncated_candidates.py").read())
    rescue()
"""

import json


def rescue(
    distribution_path="full_correlation_distribution.json",
    candidates_path="candidate_neurons.json",
    top_k_per_half=60,
):
    with open(distribution_path) as f:
        dist = json.load(f)

    def parse_key(k):
        # "L{layer}_N{neuron}" -> (layer, neuron)
        l_part, n_part = k.split("_")
        return int(l_part[1:]), int(n_part[1:])

    corr_a = {parse_key(k): v for k, v in dist["half_a"].items()}
    corr_b = {parse_key(k): v for k, v in dist["half_b"].items()}

    top_a = set(sorted(corr_a, key=lambda k: -abs(corr_a[k]))[:top_k_per_half])
    top_b = set(sorted(corr_b, key=lambda k: -abs(corr_b[k]))[:top_k_per_half])
    stable = top_a & top_b
    print(f"{len(top_a)} in top-{top_k_per_half} of half A, {len(top_b)} of half B, "
          f"{len(stable)} in BOTH (split-half-stable) -- keeping ALL of these, no truncation.")

    ranked = sorted(stable, key=lambda k: -min(abs(corr_a[k]), abs(corr_b[k])))

    candidates = [
        {
            "neuron_id": f"L{l}_N{n}", "layer": l, "neuron_idx": n,
            "detection_correlation_half_a": corr_a[(l, n)],
            "detection_correlation_half_b": corr_b[(l, n)],
            "detection_correlation_min_abs": min(abs(corr_a[(l, n)]), abs(corr_b[(l, n)])),
        }
        for (l, n) in ranked
    ]

    with open(candidates_path) as f:
        existing = json.load(f)
    provenance = existing.get("provenance", {})
    provenance["method"] = "split_half_validated"
    provenance["note"] = (
        f"Corrected via rescue script: original save truncated "
        f"to top_k_final=15, dropping {len(candidates) - 15 if len(candidates) >= 15 else 0} "
        f"neuron(s) that had already passed split-half validation. This file keeps the "
        f"full validated set ({len(candidates)} neurons)."
    )

    payload = {"provenance": provenance, "candidates": candidates}
    with open(candidates_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Re-saved {len(candidates)} split-half-validated candidates to {candidates_path} "
          f"(was 15, truncated).")
    for c in candidates:
        print(f"  {c['neuron_id']}: min|r|={c['detection_correlation_min_abs']:.4f}")

    return candidates


if __name__ == "__main__":
    rescue()
