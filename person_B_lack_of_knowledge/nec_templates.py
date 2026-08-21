"""
nec_templates.py -- Person B, lack-of-knowledge category.

Dependency-free (stdlib only; no torch, no transformers) module shared by
screen_templates.py and rebuild_lack_of_knowledge_whitelisted.py:

  * NEC_TEMPLATES         the 78 question templates from UnknownBench's
                          prompts/NEC_question_templates.py, by NEC category
  * fetch_unknownbench()  git-clone UnknownBench and check out the pinned commit
  * load_and_classify()   load NEC_unanswerable.json / NEC_answerable.json and
                          tag each prompt with the exact template that generated
                          it (anchored, non-greedy regex; longest template first)
  * classify_prompt()     the same matcher for a single prompt string

Unmatched prompts are NOT rare (~15% of the NEC files do not match any
template for their category -- upstream data quirks such as extra words or
hand-edited phrasings). load_and_classify returns them so the caller can
inspect them, and prints the count and percentage.
"""

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

UNKNOWNBENCH_REPO_URL = "https://github.com/genglinliu/UnknownBench.git"
# Pinned: the commit the committed data was built from (also the gitlink
# recorded for UnknownBench/ in this repo's tree).
UNKNOWNBENCH_COMMIT = "7283e4218b9146275d3069306927c3289fad576a"
UNKNOWNBENCH_DIR = REPO_ROOT / "UnknownBench"
NEC_DIR = UNKNOWNBENCH_DIR / "data" / "NEC"
SOURCE_DATASET_BASE = f"UnknownBench-NEC@{UNKNOWNBENCH_COMMIT[:7]}"

# Pulled directly from UnknownBench's prompts/NEC_question_templates.py at
# UNKNOWNBENCH_COMMIT -- keep in sync if the pin ever changes.
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

TEMPLATE_TO_CATEGORY = {t: cat for cat, ts in NEC_TEMPLATES.items() for t in ts}


def _template_to_regex(template: str) -> re.Pattern:
    """'...{}...' template -> anchored regex with a non-greedy entity capture."""
    escaped = re.escape(template).replace(re.escape("{}"), "(.+?)")
    return re.compile("^" + escaped + "$")


def _compile_regexes():
    """Per category, (template, regex) pairs sorted longest-template-first so
    a more specific template wins over a shorter one it contains (e.g.
    'What is the meaning of {}?' before 'What is {}?')."""
    out = {}
    for cat, templates in NEC_TEMPLATES.items():
        ordered = sorted(templates, key=len, reverse=True)
        out[cat] = [(t, _template_to_regex(t)) for t in ordered]
    return out


_REGEXES = _compile_regexes()
_ALL_REGEXES = sorted(
    ((t, _template_to_regex(t)) for ts in NEC_TEMPLATES.values() for t in ts),
    key=lambda tr: len(tr[0]), reverse=True,
)


def classify_prompt(prompt: str, category: str = None):
    """Return (template, entity) for prompt, searching only `category`'s
    templates if given (else all). Returns (None, None) if nothing matches."""
    prompt = prompt.strip()
    candidates = _REGEXES.get(category) if category else _ALL_REGEXES
    if candidates is None:
        return None, None
    for template, rx in candidates:
        m = rx.match(prompt)
        if m:
            return template, m.group(1)
    return None, None


def fetch_unknownbench(dest: Path = UNKNOWNBENCH_DIR, commit: str = UNKNOWNBENCH_COMMIT) -> Path:
    """Clone UnknownBench (if needed) and check out the pinned commit."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        print(f"Cloning {UNKNOWNBENCH_REPO_URL} -> {dest} ...")
        subprocess.run(["git", "clone", UNKNOWNBENCH_REPO_URL, str(dest)], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != commit:
        print(f"Checking out UnknownBench@{commit[:7]} (was {head[:7]})")
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "origin"], check=False)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
    return dest


def _load_json_or_jsonl(path: Path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else list(data.values())
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


def load_and_classify(nec_dir=NEC_DIR, fetch: bool = True, verbose: bool = True):
    """
    Returns (unanswerable, answerable, unmatched):
      unanswerable / answerable: list of
          {"prompt", "category", "template", "entity", "is_unanswerable"}
      unmatched: list of {"prompt", "category", "is_unanswerable", "file"}
          for prompts that matched no template of their category.
    """
    nec_dir = Path(nec_dir)
    if fetch and not nec_dir.exists():
        fetch_unknownbench()
    if not nec_dir.exists():
        raise FileNotFoundError(f"{nec_dir} not found -- run fetch_unknownbench() first")

    unmatched = []

    def load_one(path, is_unanswerable):
        records = _load_json_or_jsonl(path)
        classified = []
        for r in records:
            cat = r.get("category")
            prompt = (r.get("prompt") or r.get("question") or "").strip()
            template, entity = classify_prompt(prompt, cat)
            if template is None:
                unmatched.append({"prompt": prompt, "category": cat,
                                  "is_unanswerable": is_unanswerable, "file": path.name})
                continue
            classified.append({"prompt": prompt, "category": cat, "template": template,
                               "entity": entity, "is_unanswerable": is_unanswerable})
        n_total = len(records)
        n_un = n_total - len(classified)
        if verbose:
            print(f"{path.name}: {len(classified)} classified, {n_un} unmatched "
                  f"({100.0 * n_un / max(n_total, 1):.1f}%)")
        return classified

    unans = load_one(nec_dir / "NEC_unanswerable.json", True)
    ans = load_one(nec_dir / "NEC_answerable.json", False)
    if verbose:
        total = len(unans) + len(ans) + len(unmatched)
        print(f"Unmatched overall: {len(unmatched)} / {total} "
              f"({100.0 * len(unmatched) / max(total, 1):.1f}%) -- see returned list")
    return unans, ans, unmatched


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Classify NEC prompts by template (no model needed).")
    p.add_argument("--fetch", action="store_true", help="clone/checkout UnknownBench at the pinned commit")
    p.add_argument("--dump-unmatched", type=str, default=None, help="write unmatched prompts to this jsonl")
    args = p.parse_args()
    if args.fetch:
        fetch_unknownbench()
    u, a, um = load_and_classify()
    if args.dump_unmatched:
        with open(args.dump_unmatched, "w", encoding="utf-8") as f:
            for r in um:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(um)} unmatched prompts to {args.dump_unmatched}")
