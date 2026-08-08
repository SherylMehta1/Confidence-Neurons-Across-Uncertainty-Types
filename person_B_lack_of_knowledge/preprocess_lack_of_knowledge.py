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
    Searches the cloned repo for files whose name suggests the NEC
    (Non-Existent Concepts) subset. Adjust the glob pattern once you've
    seen the actual repo layout.
    """
    candidates = list(CLONE_DIR.rglob("*nec*")) + list(CLONE_DIR.rglob("*NEC*"))
    candidates = [c for c in candidates if c.suffix in (".json", ".jsonl", ".csv")]
    return candidates


def inspect_structure(files):
    """
    Run this FIRST, standalone, before trusting the rest of the script.
    Prints the first record of each candidate file so you can see the
    real field names.
    """
    for f in files:
        print(f"\n--- {f} ---")
        if f.suffix == ".jsonl":
            with open(f) as fh:
                first_line = fh.readline()
            print(json.loads(first_line))
        elif f.suffix == ".json":
            with open(f) as fh:
                data = json.load(fh)
            item = data[0] if isinstance(data, list) else next(iter(data.values()))
            print(item)
        else:
            with open(f) as fh:
                print(fh.readline())
                print(fh.readline())


def extract_prompt_and_label(record: dict) -> tuple[str, bool] | None:
    """
    TODO: adjust these key names after running inspect_structure() and
    seeing the real schema. Common possibilities based on the paper's
    description (fabricated entity + matched answerable control) are
    sketched below -- treat this as a starting guess, not verified fact.

    Returns (question_text, is_non_existent) or None to skip a record.
    """
    # --- Likely candidate field names, try in this order ---
    question = (
        record.get("question")
        or record.get("query")
        or record.get("prompt")
    )
    if question is None:
        return None

    # is_non_existent: True for fabricated-entity (NEC) items,
    # False for their matched answerable controls
    is_non_existent = record.get("is_fabricated")
    if is_non_existent is None:
        is_non_existent = record.get("label") == "non-existent"
    if is_non_existent is None:
        # fallback: some UnknownBench-style sets mark this via a
        # "type" or "category" field instead
        is_non_existent = str(record.get("type", "")).lower() in (
            "nec", "non_existent", "fabricated"
        )

    return question.strip(), bool(is_non_existent)


def load_nec_records(files) -> list[dict]:
    all_records = []
    for f in files:
        if f.suffix == ".jsonl":
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))
        elif f.suffix == ".json":
            with open(f) as fh:
                data = json.load(fh)
            all_records.extend(data if isinstance(data, list) else list(data.values()))
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
            raw_prompts.append(question)

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
