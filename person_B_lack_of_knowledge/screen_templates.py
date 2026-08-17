"""
screen_templates.py -- Person B, follow-up to the induction-quality check

WHY THIS EXISTS: verify_induction_quality on the first 25 lack-of-knowledge /
control items showed almost no separation overall (mean top1 0.377 vs 0.410),
but the per-template breakdown showed the signal is real for SOME of NEC's
question templates (capital city: +0.744 gap) and reversed for others
(typical diet: -0.187, dosage/schedule: -0.280). NEC's 78 templates
(prompts/NEC_question_templates.py in the UnknownBench repo) were built for
full free-text response evaluation, not single-next-token cloze completion --
most of them don't have one canonical short answer even when the entity is
real, so top1 there measures phrasing convergence, not knowledge.

This script tests EVERY template (not just the 8 that happened to appear in
one 25-item sample) on a small matched batch of real + fabricated entities,
and reports which ones show a genuine, consistently-signed gap. Use the
output to build a template WHITELIST, then rebuild
data/lack_of_knowledge/prompts.jsonl and controls.jsonl from only those
templates -- that replaces the current is_factoid_question phrase-matching
filter (which screens by sentence-starter, not by whether the template
actually has a single-fact answer) with something that screens by what
actually matters: does this template empirically separate known from
unknown.

Needs model + tokenizer loaded. Run with:
    exec(open("person_B_lack_of_knowledge/screen_templates.py").read())
    template_results = screen_all_templates(model, tokenizer)
"""

import re
import json
from pathlib import Path
from collections import defaultdict

from shared.model_utils import get_next_token_probs, compute_top1_prob
from shared.prompt_format import build_completion_prompt

# Pulled directly from UnknownBench's prompts/NEC_question_templates.py --
# keep this in sync if that file ever changes upstream.
NEC_TEMPLATES = {
    "animals": [
        "What is the habitat of {}?", "What is the typical diet of {}?",
        "How does {} reproduce?", "What are the physical characteristics of {}?",
        "Is {} considered to be endangered or threatened?",
        "What are the main predators of {}?",
        "How does {} communicate with others of its species?",
        "Are there any interesting behaviors or habits associated with {}?",
        "How long is the typical lifespan of {} in the wild?",
        "Does {} play any significant role in its ecosystem?",
    ],
    "food": [
        "What are the main ingredients in {}?",
        "What cuisine or culture does {} originate from?",
        "What cooking methods and techniques are used to prepare {}?",
        "How long does it take to prepare and cook {}?",
        "What are the different flavors and seasonings used to flavor {}?",
        "Is there significance to when or how often {} is served?",
        "How is {} typically presented or plated?",
        "What sides or accompaniments complement {}?",
        "Are there any variations or regional differences for {}?",
        "What is the nutrition breakdown and calorie count per serving for {}?",
        "Are there certain ingredients that could be substituted or modified in {}?",
        "What is the proper way to eat and enjoy {}?",
    ],
    "countries": [
        "What is the capital city of {}?", "What form of government does {} have?",
        "What are the official languages spoken in {}?",
        "What are some major geographic features located in {}?",
        "What religions are predominantly practiced in {}?",
        "What are some of {}'s major exports and industries?",
        "What type of climate exist in different regions of {}?",
        "What are some major historical events that happened in {}?",
        "Who are some famous historical and contemporary figures from {}?",
        "What are some examples of art, music, and cuisine native to {}?",
        "What ethnic and cultural groups live in or originate from {}?",
        "How does the education system work in {}?",
        "What are some national holidays and traditions celebrated in {}?",
        "What are some societal issues or challenges currently facing {}?",
    ],
    "medicines": [
        "What medical condition(s) is {} used to treat?",
        "What is the generic name and chemical makeup of {}?",
        "How is {} administered (oral, injection, topical, etc.)?",
        "What is the typical dosage and schedule for taking {}?",
        "What are the common side effects and risks associated with taking {}?",
        "Does {} interact with any foods, drinks, or other medications?",
        "How long does it take for the medicine {} to take effect?",
        "How long will the effects of {} last after taking it?",
        "Can {} lead to dependence or addiction with long-term use?",
        "How should {} be stored properly at home?",
        "How expensive is {} and is it covered by insurance plans?",
        "When did {} first become available and who manufactures it?",
        "What should you do if you overdose on {}?",
    ],
    "sports": [
        "What are the basic rules and objectives of {}?",
        "What equipment and playing environment are required for {}?",
        "How many players are on each team for {}?",
        "What are the different positions or roles that players take on in {}?",
        "How long does a regular game or match tend to last in {}?",
        "How does scoring work in {} and how do you win?",
        "What major leagues, competitions, or tournaments exist for {}?",
        "What skills, strengths, and abilities are required to excel at {}?",
        "Who are some all-time great professional players in {}?",
        "What are some common injuries or health risks associated with {}?",
        "Has {} seen any major rule changes, innovations, or controversies over time?",
        "What equipment and safety gear are required or recommended for {}?",
        "Is {} predominantly played by males, females, or both equally?",
        "At what age do most athletes begin competing in {}?",
        "How accessible and affordable is {} for casual players?",
    ],
    "generic": [
        "Can you provide examples to help illustrate {}?", "Explain {} to me.",
        "What is {}?", "What is the meaning of {}?",
        "What do people usually use {} for?", "Explain {} to a 5-year-old child.",
        "What is the definition of {}?",
        "The concept of {} is not very widely known. Explain it in layman's terms.",
        "Explain this concept: {}.",
        "What are some related concepts that are often confused with {}?",
        "Describe some real-world applications of {}?",
        "What are some historical events or figures associated with {}?",
        "What are some interesting facts or trivia related to {}?",
        "How do people use {} in everyday life?",
    ],
}


def _template_to_regex(template: str) -> re.Pattern:
    """Turn a "...{}..." template into a regex that captures the entity."""
    escaped = re.escape(template).replace(re.escape("{}"), "(.+)")
    return re.compile("^" + escaped + "$")


def load_and_classify(nec_dir="UnknownBench/data/NEC"):
    """
    Loads NEC_answerable.json / NEC_unanswerable.json and tags each record
    with which exact template generated it, by regex-matching against
    NEC_TEMPLATES for that record's category. Records that don't match any
    template for their category (rare -- covers upstream data quirks) are
    dropped with a warning count.
    """
    regexes = {
        cat: [(t, _template_to_regex(t)) for t in templates]
        for cat, templates in NEC_TEMPLATES.items()
    }

    def load_one(path, is_unanswerable):
        with open(path) as f:
            records = [json.loads(l) for l in f]
        classified, unmatched = [], 0
        for r in records:
            cat = r.get("category")
            prompt = r.get("prompt", "")
            if cat not in regexes:
                unmatched += 1
                continue
            matched_template = None
            for template, rx in regexes[cat]:
                if rx.match(prompt):
                    matched_template = template
                    break
            if matched_template is None:
                unmatched += 1
                continue
            classified.append({
                "prompt": prompt, "category": cat, "template": matched_template,
                "is_unanswerable": is_unanswerable,
            })
        print(f"{path}: {len(classified)} classified, {unmatched} unmatched (dropped)")
        return classified

    nec_dir = Path(nec_dir)
    unans = load_one(nec_dir / "NEC_unanswerable.json", True)
    ans = load_one(nec_dir / "NEC_answerable.json", False)
    return unans, ans


def screen_all_templates(model, tokenizer, n_per_side: int = 6, seed: int = 42):
    """
    For every (category, template) pair, samples n_per_side unanswerable +
    n_per_side answerable items, measures top1 under the position-fixed
    cloze prompt, and reports the gap. Returns a list of dicts sorted by
    gap descending -- use this to decide your whitelist.
    """
    import random
    random.seed(seed)

    unans, ans = load_and_classify()
    by_template_u = defaultdict(list)
    by_template_a = defaultdict(list)
    for r in unans:
        by_template_u[r["template"]].append(r["prompt"])
    for r in ans:
        by_template_a[r["template"]].append(r["prompt"])

    results = []
    all_templates = sorted(set(by_template_u) | set(by_template_a))
    for i, template in enumerate(all_templates):
        u_pool = by_template_u.get(template, [])
        a_pool = by_template_a.get(template, [])
        if len(u_pool) < 2 or len(a_pool) < 2:
            continue  # too few instances of this template to say anything

        u_sample = random.sample(u_pool, min(n_per_side, len(u_pool)))
        a_sample = random.sample(a_pool, min(n_per_side, len(a_pool)))

        u_top1s, a_top1s = [], []
        for prompt in u_sample:
            formatted = build_completion_prompt(tokenizer, prompt)
            probs = get_next_token_probs(model, tokenizer, formatted["chat_formatted_prompt"])
            u_top1s.append(compute_top1_prob(probs))
        for prompt in a_sample:
            formatted = build_completion_prompt(tokenizer, prompt)
            probs = get_next_token_probs(model, tokenizer, formatted["chat_formatted_prompt"])
            a_top1s.append(compute_top1_prob(probs))

        u_mean = sum(u_top1s) / len(u_top1s)
        a_mean = sum(a_top1s) / len(a_top1s)
        results.append({
            "template": template,
            "n_unanswerable_pool": len(u_pool), "n_answerable_pool": len(a_pool),
            "unanswerable_mean_top1": u_mean, "control_mean_top1": a_mean,
            "gap": a_mean - u_mean,
        })
        print(f"[{i+1}/{len(all_templates)}] gap={a_mean-u_mean:+.3f}  "
              f"(unans={u_mean:.3f}, ctrl={a_mean:.3f})  {template}")

    results.sort(key=lambda r: -r["gap"])
    return results


def print_whitelist_recommendation(results, min_gap=0.15, max_unanswerable_mean=0.5):
    """
    A template is worth keeping if fabricated entities are clearly LESS
    confident than real ones (gap >= min_gap) AND the fabricated-entity
    mean isn't already so peaked that the position fix isn't doing its job
    for this phrasing (unanswerable_mean <= max_unanswerable_mean). Adjust
    both thresholds after eyeballing the full table -- these are starting
    points, not fixed rules.
    """
    good = [r for r in results if r["gap"] >= min_gap and r["unanswerable_mean_top1"] <= max_unanswerable_mean]
    total_pool = sum(min(r["n_unanswerable_pool"], r["n_answerable_pool"]) for r in good)
    print(f"\n{len(good)} / {len(results)} templates pass (gap >= {min_gap}, "
          f"unanswerable_mean <= {max_unanswerable_mean}):")
    for r in good:
        print(f"  gap={r['gap']:+.3f}  pool(unans/ctrl)={r['n_unanswerable_pool']}/{r['n_answerable_pool']}  {r['template']}")
    print(f"\nTotal available pool from passing templates (min of unans/ctrl per template, summed): ~{total_pool}")
    print("Compare this to the 120 you need per side -- if it's short, either "
          "lower min_gap and re-inspect, or accept a smaller working set (a "
          "smaller-but-clean dataset is better than a padded-but-noisy one, "
          "per the project's own scope/limitations language).")
    return good
