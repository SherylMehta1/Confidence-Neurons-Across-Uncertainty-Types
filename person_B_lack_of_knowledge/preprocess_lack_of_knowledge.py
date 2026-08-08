"""
preprocess_lack_of_knowledge.py -- Person B
Category: lack_of_knowledge
Source: UnknownBench (github.com/genglinliu/UnknownBench), NEC
(Non-Existent Concepts) subset -- 2,078 fabricated-entity questions with
matched answerable controls.

IMPORTANT -- READ BEFORE RUNNING:
UnknownBench is not on the HF Hub, so we clone the repo and load its raw
files directly. The exact field names inside the NEC json/jsonl files are
NOT verified here -- run the `inspect_structure()` step first (it just
prints the first raw record) and adjust `extract_prompt_and_label()` to
match what you actually see. This script is written to fail loudly rather
than silently guess wrong field names.
"""

import sys
sys.path.append(".")  # run this script from the repo root
import json
import subprocess
from pathlib import Path

from shared.schema_utils import load_tokenizer, build_records, save_records

REPO_URL = "https://github.com/genglinliu/UnknownBench.git"
CLONE_DIR = Path("UnknownBench")
OUTPUT_PATH = "data/lack_of_knowledge/prompts.jsonl"


def clone_repo():
    if CLONE_DIR.exists():
        print(f"{CLONE_DIR} already exists, skipping clone.")
        return
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
                    check=True)


def find_nec_files():
    """
    The repo's `data/NEC/` folder holds the actual dataset (2 clean files:
    NEC_answerable.json, NEC_unanswerable.json). Everything under `outputs/`
    is experiment results (model responses, confidence scores, eval logs
    from the paper) -- not the base dataset, and some of those are
    JSON-Lines rather than plain JSON, which breaks a naive json.load().
    Only search data/NEC/ to avoid pulling those in.
    """
    data_dir = CLONE_DIR / "data" / "NEC"
    candidates = list(data_dir.glob("*.json"))
    return candidates


def _load_json_or_jsonl(filepath):
    """
    Some files in this repo are named .json but are actually JSON-Lines
    (one JSON object per line) rather than a single JSON array/object.
    Try standard json.load() first; if that fails with "Extra data"
    (the telltale sign of JSONL), fall back to line-by-line parsing.
    """
    with open(filepath) as fh:
        content = fh.read()
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else list(data.values())
    except json.JSONDecodeError:
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def inspect_structure(files):
    """
    Run this FIRST, before trusting extract_prompt_and_label(). Prints the
    first record of each file so you can confirm the question/prompt field name.
    """
    for f in files:
        print(f"\n--- {f} ---")
        items = _load_json_or_jsonl(f)
        print(items[0])


def convert_to_cloze_prompt(question: str, suffix: str = " The answer is") -> str:
    """
    Converts an open-ended question into a completion-style prompt.

    WHY: raw questions like "What is the proper way to eat X?" put the
    meaningful uncertainty several tokens into the model's response, not at
    the very next token -- the first generated token is mostly determined
    by English grammar / chat-assistant habits ("The", "There", "I"), not
    by whether the model actually knows the entity. Appending a fixed
    completion trigger pulls the uncertainty signal to the next token,
    matching the cloze-style measurement already used by the ambiguity and
    contradictory-context categories, so entropy means roughly the same
    thing across all three categories.
    """
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    return q + suffix


def extract_prompt_and_label(record: dict) -> tuple[str, bool] | None:
    """
    is_non_existent comes from load_nec_records() tagging based on which
    file the record was loaded from (reliable). Only the question/prompt
    TEXT field name still needs verifying via inspect_structure() --
    adjust the fallbacks below if none of them match what you see printed.
    """
    question = (
        record.get("question")
        or record.get("query")
        or record.get("prompt")
        or record.get("text")
    )
    if question is None:
        return None

    is_non_existent = record.get("_is_non_existent", False)
    return question.strip(), bool(is_non_existent)


def load_nec_records(files) -> list[dict]:
    """
    Tags each record with is_non_existent based on which FILE it came from
    (NEC_unanswerable.json vs NEC_answerable.json), since the filename is a
    more reliable signal than guessing an internal field name.
    """
    all_records = []
    for f in files:
        is_unanswerable_file = "unanswerable" in f.stem.lower()
        items = _load_json_or_jsonl(f)
        for item in items:
            item = dict(item) if isinstance(item, dict) else {"raw": item}
            item["_is_non_existent"] = is_unanswerable_file
            all_records.append(item)
    return all_records


def main():
    clone_repo()
    files = find_nec_files()
    if not files:
        raise FileNotFoundError(
            "No files matching '*nec*' found in the cloned repo. "
            "List UnknownBench/ manually and update find_nec_files()."
        )

    print(f"Found candidate NEC files: {[str(f) for f in files]}")
    print("\n>>> Run inspect_structure(files) interactively first if you "
          "haven't verified the schema yet. <<<\n")
    inspect_structure(files)

    raw_records = load_nec_records(files)
    print(f"Loaded {len(raw_records)} raw records.")

    raw_prompts = []
    for r in raw_records:
        parsed = extract_prompt_and_label(r)
        if parsed is None:
            continue
        question, is_non_existent = parsed
        if is_non_existent:  # keep only the NEC (unanswerable) items
            raw_prompts.append(convert_to_cloze_prompt(question))

    print(f"Extracted {len(raw_prompts)} non-existent-concept questions "
          f"(need >= 120 to subsample).")

    tokenizer = load_tokenizer()
    records = build_records(
        raw_prompts=raw_prompts,
        category="lack_of_knowledge",
        source_dataset="UnknownBench-NEC",
        prefix="lok",
        tokenizer=tokenizer,
        n_working=120,
        split_ratio=0.7,
    )
    save_records(records, OUTPUT_PATH)


if __name__ == "__main__":
    main()
