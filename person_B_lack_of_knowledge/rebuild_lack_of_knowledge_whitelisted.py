"""
rebuild_lack_of_knowledge_whitelisted.py -- Person B, final Phase 3 data build

WHY THIS EXISTS: screen_templates.py's screen_all_templates() (n=6) found 15
templates with gap >= 0.15. rescreen_candidates() (n=15, or full pool where
smaller) re-tested those 15 and found 3 of them were sampling noise that
collapsed below threshold at the larger sample:
    - "What equipment and playing environment are required for {}?"  0.193 -> 0.093
    - "How accessible and affordable is {} for casual players?"      0.152 -> 0.136
    - "How long does a regular game or match tend to last in {}?"    0.173 -> 0.030
3 more shrank but stayed above 0.15 (borderline -- smaller/shakier pools):
    - "Can {} lead to dependence or addiction with long-term use?"          0.260 -> 0.177
    - "What are the basic rules and objectives of {}?"                     0.174 -> 0.169
    - "Who are some famous historical and contemporary figures from {}?"   (pool capped ~10) -> 0.152
9 held up as genuinely stable and are what this script uses:

    gap(n=15)  template
    +0.716     What is the capital city of {}?
    +0.627     Are there any interesting behaviors or habits associated with {}?
    +0.546     Does {} interact with any foods, drinks, or other medications?
    +0.477     Is there significance to when or how often {} is served?
    +0.466     Does {} play any significant role in its ecosystem?
    +0.375     What are the official languages spoken in {}?
    +0.295     What are some major historical events that happened in {}?
    +0.271     What cooking methods and techniques are used to prepare {}?
    +0.246     What form of government does {} have?

Combined pool (min(unanswerable, answerable) per template, from the original
full screening run): ~187 per side -- comfortably above the 120/side target,
so this deliberately drops the 3 borderline templates rather than stretching
to include them. A smaller, template-validated set beats a padded one that
still contains noisy phrasings -- same principle the project already used to
justify a shrunk split-half candidate list in Step 2.

CAVEAT WORTH ONE LINE IN THE WRITEUP: not all 9 templates fail the same way
for a fabricated entity. "capital city" / "official languages" / "form of
government" fail via genuine fact-retrieval collapse -- there's no capital
to retrieve. Several of the others (interesting behaviors, interacts with
foods/medications, significant role in ecosystem) plausibly fail via a
different route: the model recognizing the entity name doesn't pattern-match
anything real, which is closer to an entity-recognition confidence signal
than pure fact-retrieval. Both are legitimate "the model doesn't know this"
signals for this project's purposes, but they're not identical mechanisms --
worth a sentence in limitations, not a reason to drop templates.

Run in the same Kaggle/RunPod session, tokenizer loaded (model not required
here, only for the optional verify step at the end):

    exec(open("person_B_lack_of_knowledge/rebuild_lack_of_knowledge_whitelisted.py").read())
    records, control_records = rebuild()

This OVERWRITES data/lack_of_knowledge/prompts.jsonl and controls.jsonl --
the ones from preprocess_lack_of_knowledge_v2.py used the broad
is_factoid_question filter across all templates; this replaces that filter
with the empirically-validated 9-template whitelist.
"""

import sys
sys.path.append(".")
import json
import random
from pathlib import Path

from person_B_lack_of_knowledge.screen_templates import load_and_classify
from shared.schema_utils import load_tokenizer
from shared.prompt_format import build_records_with_formatter, build_completion_prompt

OUTPUT_PATH = "data/lack_of_knowledge/prompts.jsonl"
CONTROLS_OUTPUT_PATH = "data/lack_of_knowledge/controls.jsonl"

WHITELIST = [
    "What is the capital city of {}?",
    "Are there any interesting behaviors or habits associated with {}?",
    "Does {} interact with any foods, drinks, or other medications?",
    "Is there significance to when or how often {} is served?",
    "Does {} play any significant role in its ecosystem?",
    "What are the official languages spoken in {}?",
    "What are some major historical events that happened in {}?",
    "What cooking methods and techniques are used to prepare {}?",
    "What form of government does {} have?",
]

TARGET_PER_SIDE = 120


def stratified_sample(pools: dict, target: int, seed: int) -> list[str]:
    """
    pools: {template: [prompt, ...]}. Allocates a per-template quota
    proportional to each template's pool size (largest-remainder method),
    capped at that template's own pool size, so no single template can
    dominate the final set just because it happens to have the biggest pool.
    Falls back to taking everything if the combined pool is short of target.
    """
    rng = random.Random(seed)
    total_available = sum(len(p) for p in pools.values())
    if total_available <= target:
        out = []
        for t, p in pools.items():
            out.extend(p)
        rng.shuffle(out)
        return out

    # largest-remainder proportional allocation
    raw_quotas = {t: len(p) / total_available * target for t, p in pools.items()}
    quotas = {t: int(raw_quotas[t]) for t in pools}
    remainders = sorted(pools.keys(), key=lambda t: raw_quotas[t] - quotas[t], reverse=True)
    shortfall = target - sum(quotas.values())
    for t in remainders[:shortfall]:
        quotas[t] += 1
    # cap at pool size, redistribute any overflow to templates with spare capacity
    overflow = 0
    for t in pools:
        if quotas[t] > len(pools[t]):
            overflow += quotas[t] - len(pools[t])
            quotas[t] = len(pools[t])
    if overflow > 0:
        spare = [t for t in pools if quotas[t] < len(pools[t])]
        spare.sort(key=lambda t: len(pools[t]) - quotas[t], reverse=True)
        i = 0
        while overflow > 0 and spare:
            t = spare[i % len(spare)]
            if quotas[t] < len(pools[t]):
                quotas[t] += 1
                overflow -= 1
            i += 1
            if i > 10000:
                break  # safety valve, shouldn't trigger given the ~187 pool math

    out = []
    for t, p in pools.items():
        out.extend(rng.sample(p, min(quotas[t], len(p))))
    rng.shuffle(out)
    return out


def rebuild(seed: int = 42):
    unans, ans = load_and_classify()

    u_pools, a_pools = {}, {}
    for r in unans:
        if r["template"] in WHITELIST:
            u_pools.setdefault(r["template"], []).append(r["prompt"])
    for r in ans:
        if r["template"] in WHITELIST:
            a_pools.setdefault(r["template"], []).append(r["prompt"])

    print("Whitelisted-template pool sizes (unanswerable / answerable):")
    for t in WHITELIST:
        print(f"  {len(u_pools.get(t, [])):>4} / {len(a_pools.get(t, [])):>4}   {t}")

    unanswerable_qs = stratified_sample(u_pools, TARGET_PER_SIDE, seed=seed)
    answerable_qs = stratified_sample(a_pools, TARGET_PER_SIDE, seed=seed + 1)
    print(f"\nSelected {len(unanswerable_qs)} unanswerable, {len(answerable_qs)} answerable "
          f"(target was {TARGET_PER_SIDE} each).")

    tokenizer = load_tokenizer()

    records = build_records_with_formatter(
        raw_prompts=unanswerable_qs,
        category="lack_of_knowledge",
        source_dataset="UnknownBench-NEC (unanswerable, template-whitelisted)",
        prefix="lok",
        tokenizer=tokenizer,
        formatter=build_completion_prompt,
        n_working=len(unanswerable_qs),
        split_ratio=0.7,
    )
    from shared.schema_utils import save_records
    save_records(records, OUTPUT_PATH)

    control_records = build_records_with_formatter(
        raw_prompts=answerable_qs,
        category="lack_of_knowledge",
        source_dataset="UnknownBench-NEC (answerable, template-whitelisted, matched control)",
        prefix="lok_ctrl",
        tokenizer=tokenizer,
        formatter=build_completion_prompt,
        n_working=len(answerable_qs),
        split_ratio=0.7,
        is_control=True,
    )
    Path(CONTROLS_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CONTROLS_OUTPUT_PATH, "w") as f:
        for r in control_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(records)} -> {OUTPUT_PATH}")
    print(f"Saved {len(control_records)} -> {CONTROLS_OUTPUT_PATH}")
    print(
        "\nNEXT: with model+tokenizer loaded, re-run the induction-quality check "
        "on this rebuilt set as a final sanity gate before moving to Step 2 "
        "(detection):\n"
        "    from shared.prompt_format import verify_induction_quality\n"
        "    verify_induction_quality(model, tokenizer, records[:25])\n"
        "    verify_induction_quality(model, tokenizer, control_records[:25])\n"
        "This time also print the two MEANS yourself and diff them directly -- "
        "don't rely on verify_induction_quality's own PASS/FAIL line, it only "
        "checks individual items against an absolute ceiling and won't catch "
        "two similar group means (that's exactly how the first, unwhitelisted "
        "version of this data slipped through)."
    )

    return records, control_records


if __name__ == "__main__":
    print("Load model/tokenizer first, then call rebuild() directly (see module docstring).")