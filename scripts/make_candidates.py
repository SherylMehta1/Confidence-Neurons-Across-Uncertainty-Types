"""
Candidate twin generators (no model needed) for scripts/gate_twins.py.

  situated   : SituatedQA (siyue/SituatedQA, temp + geo) -- uncertain = bare `question`,
               control = `edited_question` (same question + resolving clause). gold = answers.
  aleatoric  : hand-built families -- uncertain = many valid answers (gold=null), control = unique answer.
  conflict   : ConflictQA (osunlp/ConflictQA popQA split) -- control = question + Wikipedia evidence for the
               true (PopQA) answer; uncertain = the same + ChatGPT's fabricated passage for a different answer
               (two disagreeing sources). gold: control = PopQA ground-truth aliases; uncertain = null (no single
               right answer) -- the gate then relies on entropy + hedging.

Writes data/<category>/candidates.jsonl.
Usage: python scripts/make_candidates.py situated [--n 600] ; python scripts/make_candidates.py aleatoric ; ...
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import ast
import json
import random

from _common import REPO_ROOT
from build_familiarity_twins import is_correct

ALEATORIC_FAMILIES = [
    ("Name a prime number below 100.", "Name the smallest prime number.", ["2", "two"]),
    ("Name a colour of the rainbow.", "Name the first colour of the rainbow.", ["red"]),
    ("Name a US state.", "Name the US state whose capital is Boise.", ["Idaho"]),
    ("Name a chemical element.", "Name the chemical element with atomic number 1.", ["hydrogen"]),
    ("Name a Shakespeare play.", "Name the Shakespeare play featuring Prince Hamlet.", ["Hamlet"]),
    ("Name a planet in the solar system.", "Name the largest planet in the solar system.", ["Jupiter"]),
    ("Name a month of the year.", "Name the first month of the year.", ["January"]),
    ("Name a day of the week.", "Name the day of the week that comes after Monday.", ["Tuesday"]),
    ("Name a continent.", "Name the continent that contains Brazil.", ["South America"]),
    ("Name an ocean.", "Name the largest ocean on Earth.", ["Pacific", "Pacific Ocean"]),
    ("Name a programming language.", "Name the programming language created by Guido van Rossum.", ["Python"]),
    ("Name a European capital city.", "Name the capital city of France.", ["Paris"]),
    ("Name a Beatles song.", "Name the Beatles song that begins 'Yesterday, all my troubles seemed so far away'.", ["Yesterday"]),
    ("Name a noble gas.", "Name the noble gas with atomic number 2.", ["helium"]),
    ("Name a bone in the human body.", "Name the longest bone in the human body.", ["femur"]),
    ("Name a Greek letter.", "Name the first letter of the Greek alphabet.", ["alpha"]),
    ("Name a playing-card suit.", "Name the playing-card suit that is red and shaped like a heart.", ["hearts"]),
    ("Name a primary colour.", "Name the colour of a ripe banana.", ["yellow"]),
    ("Name a number between 1 and 10.", "Name the number that comes after 6.", ["7", "seven"]),
    ("Name a mammal.", "Name the largest mammal on Earth.", ["blue whale", "whale"]),
]
ALEATORIC_FILLERS = ["", "In one word: ", "Please answer briefly. ", "Quick question: ", "Give one example. ", "I'm curious: "]


def pair(twin_id, category, uq, cq, ugold, cgold, source, umeta=None, cmeta=None, context=None):
    d = dict(twin_id=twin_id, category=category, source=source, uncertain=dict(question=uq, gold=ugold, meta=umeta or {}),
             control=dict(question=cq, gold=cgold, meta=cmeta or {}))
    if context:
        d["context"] = context
    return d


def situated(n, seed):
    import urllib.request
    rng = random.Random(seed); out = []
    for cfg in ("temp", "geo"):
        # the HF repo only carries a loading script (unsupported by datasets>=4); read the raw files it points to
        url = f"https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/{cfg}.test.jsonl"
        rows = [json.loads(l) for l in urllib.request.urlopen(url, timeout=60).read().decode("utf-8").splitlines() if l.strip()]
        rng.shuffle(rows)
        for r in rows[: n // 2]:
            q, eq = r["question"].strip(), r["edited_question"].strip()
            if not q.endswith("?"): q += "?"
            if not eq.endswith("?"): eq += "?"
            gold = r.get("answer") or []
            gold = gold if isinstance(gold, list) else [gold]
            out.append(pair(f"sit_{cfg}_{r['id']}", "situated", q, eq, None, gold, f"siyue/SituatedQA:{cfg}",
                            dict(date_type=r.get("date_type"), any_answer=r.get("any_answer")), dict(date=r.get("date"))))
    return out


def aleatoric(seed):
    rng = random.Random(seed); out = []
    for i, (u, c, gold) in enumerate(ALEATORIC_FAMILIES):
        for j, f in enumerate(ALEATORIC_FILLERS):
            out.append(pair(f"ale_{i:02d}_{j}", "aleatoric", f + u, f + c, None, gold, "handbuilt", dict(family=i, filler=j), dict(family=i, filler=j)))
    rng.shuffle(out)
    return out


def conflict(n, seed, config="ConflictQA-popQA-chatgpt"):
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    fname = {"ConflictQA-popQA-chatgpt": "conflictQA-popQA-chatgpt.json", "ConflictQA-popQA-gpt4": "conflictQA-popQA-gpt4.json"}.get(config, config)
    path = hf_hub_download("osunlp/ConflictQA", fname, repo_type="dataset")  # raw json; the repo's loading script is unsupported by datasets>=4
    txt = open(path, encoding="utf-8").read().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]  # file is JSONL despite .json
    rows = [r for r in rows if all(k in r for k in ("question", "parametric_memory", "counter_memory_aligned_evidence", "ground_truth"))]
    rng.shuffle(rows); out = []
    for r in rows:
        if len(out) >= n:
            break
        i = len(out)
        q = r["question"].strip(); q = q if q.endswith("?") else q + "?"
        gt = r["ground_truth"]
        try:
            gold = list(ast.literal_eval(gt)) if isinstance(gt, str) and gt.strip().startswith("[") else ([gt] if isinstance(gt, str) else list(gt))
        except Exception:
            gold = [gt]
        truth = r["counter_memory_aligned_evidence"].strip()   # Wikipedia evidence supporting the real (PopQA) answer
        fake = r["parametric_memory"].strip()                  # ChatGPT-generated passage supporting its wrong belief
        if not any(str(g).strip().lower() in truth.lower() for g in gold if str(g).strip()):
            continue  # the evidence passage must actually state the answer (62.8% of ConflictQA rows do)
        if is_correct(str(r.get("memory_answer", "")), [str(g) for g in gold]):
            continue  # ChatGPT's parametric answer is already right here, so the "counter" evidence would support a wrong answer
        two = [truth, fake]; rng.shuffle(two)
        ctx_u = "Source 1: " + two[0] + "\n\nSource 2: " + two[1] + "\n\nBased on the sources above, answer briefly."
        ctx_c = "Source 1: " + truth + "\n\nBased on the source above, answer briefly."
        out.append(pair(f"cq_{i:05d}", "conflict", q, q, None, gold, f"osunlp/ConflictQA:{config}",
                        dict(truth_answer=gold[0], conflicting_answer=r.get("memory_answer"), popularity=r.get("popularity")), dict(truth_answer=gold[0]),
                        context=dict(uncertain=ctx_u, control=ctx_c)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("category", choices=["situated", "aleatoric", "conflict"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = {"situated": lambda: situated(args.n, args.seed), "aleatoric": lambda: aleatoric(args.seed), "conflict": lambda: conflict(args.n, args.seed)}[args.category]()
    d = REPO_ROOT / "data" / args.category; d.mkdir(parents=True, exist_ok=True)
    with open(d / "candidates.jsonl", "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} candidate pairs to {d / 'candidates.jsonl'}")


if __name__ == "__main__":
    main()
