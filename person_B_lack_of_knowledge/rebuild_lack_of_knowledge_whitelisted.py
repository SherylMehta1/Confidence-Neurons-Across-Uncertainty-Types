"""
rebuild_lack_of_knowledge_whitelisted.py -- Person B, lack-of-knowledge.
Step 2 of the live pipeline (run screen_templates.py first).

    python person_B_lack_of_knowledge/rebuild_lack_of_knowledge_whitelisted.py
        [--n 120] [--seed 42] [--min-gap 0.15] [--use-fallback-whitelist]

Builds
    data/lack_of_knowledge/prompts.jsonl    fabricated-entity (unanswerable) prompts
    data/lack_of_knowledge/controls.jsonl   real-entity (answerable) matched controls
    person_B_lack_of_knowledge/template_quota_report.json

from UnknownBench-NEC (pinned commit, see nec_templates.UNKNOWNBENCH_COMMIT),
restricted to the templates that empirically separate known from unknown
under the cloze prefill -- the whitelist is READ from
person_B_lack_of_knowledge/template_screen_results.json (gap >= --min-gap,
unanswerable mean top1 <= 0.5). FALLBACK_WHITELIST (the 9 templates from the
original bf16 n=15 rescreen) is used only with --use-fallback-whitelist or
when the screen results file is absent, and the choice is recorded in
template_quota_report.json.

Sampling: the SAME per-template quota is used on both sides. Quotas are
proportional to min(unanswerable_pool, answerable_pool) per template
(largest-remainder rounding, capped at the smaller pool), drawn with
random.Random(seed) per side, so the two files have identical template
composition and differ only in whether the entity is real.

Each record carries the 7 shared fields plus `template` (the NEC template
string) and `nec_category` (animals/food/countries/medicines/sports/generic).

Needs only the tokenizer (no model, no torch import).
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from person_B_lack_of_knowledge.nec_templates import (  # noqa: E402
    SOURCE_DATASET_BASE, TEMPLATE_TO_CATEGORY, UNKNOWNBENCH_COMMIT, load_and_classify,
)

DATA_DIR = REPO_ROOT / "data" / "lack_of_knowledge"
OUTPUT_PATH = DATA_DIR / "prompts.jsonl"
CONTROLS_OUTPUT_PATH = DATA_DIR / "controls.jsonl"
SCREEN_RESULTS_PATH = REPO_ROOT / "person_B_lack_of_knowledge" / "template_screen_results.json"
QUOTA_REPORT_PATH = REPO_ROOT / "person_B_lack_of_knowledge" / "template_quota_report.json"

# Fallback only. These 9 templates held gap >= 0.15 in the original bf16
# rescreen at n=15 (capital city +0.716 ... form of government +0.246); 3
# more templates that passed the n=6 screen collapsed at n=15 and were
# dropped. Prefer the derivable whitelist from template_screen_results.json.
FALLBACK_WHITELIST = [
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


def load_whitelist(min_gap: float, use_fallback: bool, max_unanswerable_mean: float = 0.5):
    """Returns (whitelist, provenance dict)."""
    if not use_fallback and SCREEN_RESULTS_PATH.exists():
        with open(SCREEN_RESULTS_PATH, encoding="utf-8") as f:
            screen = json.load(f)
        wl = [r["template"] for r in screen["results"]
              if "gap" in r and r["gap"] >= min_gap
              and r["unanswerable_mean_top1"] <= max_unanswerable_mean]
        prov = {"source": str(SCREEN_RESULTS_PATH.relative_to(REPO_ROOT)),
                "screen_precision": screen.get("precision"),
                "screen_n_per_side": screen.get("n_per_side"),
                "min_gap": min_gap, "max_unanswerable_mean_top1": max_unanswerable_mean}
        if screen.get("precision") != "bf16":
            print(f"WARNING: template_screen_results.json was produced at precision "
                  f"{screen.get('precision')!r}, not bf16. Re-run screen_templates.py in bf16 "
                  f"before committing data built from it.")
        return wl, prov
    if not use_fallback:
        print(f"{SCREEN_RESULTS_PATH.name} not found -- using FALLBACK_WHITELIST "
              f"(run screen_templates.py to derive it properly).")
    return list(FALLBACK_WHITELIST), {"source": "FALLBACK_WHITELIST (hard-coded)"}


def allocate_quotas(u_pools: dict, a_pools: dict, target: int) -> dict:
    """Per-template quota proportional to min(len(u), len(a)), largest-
    remainder rounding, capped at that min. Same quota applies to both sides."""
    caps = {t: min(len(u_pools.get(t, [])), len(a_pools.get(t, []))) for t in set(u_pools) | set(a_pools)}
    caps = {t: c for t, c in caps.items() if c > 0}
    total = sum(caps.values())
    if total <= target:
        return dict(caps)
    raw = {t: c / total * target for t, c in caps.items()}
    quotas = {t: int(raw[t]) for t in caps}
    shortfall = target - sum(quotas.values())
    for t in sorted(caps, key=lambda t: raw[t] - quotas[t], reverse=True)[:shortfall]:
        quotas[t] += 1
    # cap and redistribute overflow
    overflow = 0
    for t in caps:
        if quotas[t] > caps[t]:
            overflow += quotas[t] - caps[t]
            quotas[t] = caps[t]
    guard = 0
    while overflow > 0:
        spare = [t for t in caps if quotas[t] < caps[t]]
        if not spare or guard > 10000:
            break
        for t in sorted(spare, key=lambda t: caps[t] - quotas[t], reverse=True):
            if overflow == 0:
                break
            quotas[t] += 1
            overflow -= 1
        guard += 1
    return quotas


def sample_side(pools: dict, quotas: dict, seed: int) -> list:
    """Draw quotas[t] prompts from each template pool with random.Random(seed)."""
    rng = random.Random(seed)
    out = []
    for t in sorted(quotas):
        out.extend(rng.sample(sorted(pools[t]), quotas[t]))
    rng.shuffle(out)
    return out


def _attach(records, prompt_to_template):
    for r in records:
        t = prompt_to_template.get(r["raw_prompt"])
        r["template"] = t
        r["nec_category"] = TEMPLATE_TO_CATEGORY.get(t)
    return records


def _save(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_w = sum(1 for r in records if r["split"] == "working")
    print(f"Saved {len(records)} -> {path} ({n_w} working / {len(records) - n_w} held-out)")


def rebuild(n: int = 120, seed: int = 42, min_gap: float = 0.15, use_fallback: bool = False):
    from shared.schema_utils import load_tokenizer
    from shared.prompt_format import build_records_with_formatter, build_completion_prompt

    whitelist, wl_prov = load_whitelist(min_gap, use_fallback)
    if not whitelist:
        raise SystemExit("Empty whitelist -- nothing passes the gap threshold.")
    unans, ans, unmatched = load_and_classify()

    u_pools, a_pools = {}, {}
    for r in unans:
        if r["template"] in whitelist:
            u_pools.setdefault(r["template"], []).append(r["prompt"])
    for r in ans:
        if r["template"] in whitelist:
            a_pools.setdefault(r["template"], []).append(r["prompt"])

    quotas = allocate_quotas(u_pools, a_pools, n)
    print("Whitelisted templates: quota  (unanswerable pool / answerable pool)")
    for t in whitelist:
        print(f"  {quotas.get(t, 0):>4}   ({len(u_pools.get(t, [])):>4} / {len(a_pools.get(t, [])):>4})   {t}")
    n_sel = sum(quotas.values())
    if n_sel < n:
        print(f"WARNING: only {n_sel} matched pairs available (target {n}); building the smaller set.")

    unanswerable_qs = sample_side(u_pools, quotas, seed)
    answerable_qs = sample_side(a_pools, quotas, seed)
    print(f"\nSelected {len(unanswerable_qs)} unanswerable, {len(answerable_qs)} answerable.")

    prompt_to_template = {r["prompt"]: r["template"] for r in unans + ans}
    # build_completion_prompt may append '?' -- key both forms
    for p, t in list(prompt_to_template.items()):
        if not p.endswith("?"):
            prompt_to_template[p + "?"] = t

    tokenizer = load_tokenizer()
    records = build_records_with_formatter(
        raw_prompts=unanswerable_qs, category="lack_of_knowledge",
        source_dataset=f"{SOURCE_DATASET_BASE} (unanswerable, template-whitelisted)",
        prefix="lok", tokenizer=tokenizer, formatter=build_completion_prompt,
        n_working=len(unanswerable_qs), split_ratio=0.7, seed=seed,
    )
    _attach(records, prompt_to_template)
    _save(records, OUTPUT_PATH)

    control_records = build_records_with_formatter(
        raw_prompts=answerable_qs, category="lack_of_knowledge",
        source_dataset=f"{SOURCE_DATASET_BASE} (answerable, template-whitelisted, matched control)",
        prefix="lok_ctrl", tokenizer=tokenizer, formatter=build_completion_prompt,
        n_working=len(answerable_qs), split_ratio=0.7, seed=seed, is_control=True,
    )
    _attach(control_records, prompt_to_template)
    _save(control_records, CONTROLS_OUTPUT_PATH)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "unknownbench_commit": UNKNOWNBENCH_COMMIT,
        "seed": seed, "target_per_side": n,
        "whitelist_provenance": wl_prov,
        "whitelist": whitelist,
        "n_unmatched_nec_prompts": len(unmatched),
        "per_template": [
            {"template": t, "nec_category": TEMPLATE_TO_CATEGORY.get(t), "quota": quotas.get(t, 0),
             "unanswerable_pool": len(u_pools.get(t, [])), "answerable_pool": len(a_pools.get(t, []))}
            for t in whitelist
        ],
        "n_records": len(records), "n_control_records": len(control_records),
    }
    with open(QUOTA_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {QUOTA_REPORT_PATH}")
    print("\nNEXT: with the bf16 model loaded, run verify_induction_quality on records[:25] and "
          "control_records[:25] and compare the two MEANS directly.")
    return records, control_records


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=120, help="records per side")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-gap", type=float, default=0.15)
    p.add_argument("--use-fallback-whitelist", action="store_true",
                   help="ignore template_screen_results.json and use the hard-coded 9-template list")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return rebuild(n=args.n, seed=args.seed, min_gap=args.min_gap, use_fallback=args.use_fallback_whitelist)


if __name__ == "__main__":
    main()
