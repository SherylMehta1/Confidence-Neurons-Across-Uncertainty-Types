"""E2 (paper/PREREG/e2_name_surprisal.md, sha 2b69e06c): mean per-token surprisal of each subject name under the
base model, in the fixed context "Here is a name:" + " " + subject.

  python scripts/e2_surprisal.py --out results/round3/e2_surprisal.jsonl"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import torch

from _common import REPO_ROOT
from shared.model_utils import load_model

CONTEXT = "Here is a name:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--items", default="data/familiarity_v3/all_measured.jsonl")
    ap.add_argument("--out", default="results/round3/e2_surprisal.jsonl")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    subjects, seen = [], set()
    for l in open(REPO_ROOT / a.items, encoding="utf-8"):
        r = json.loads(l)
        if r.get("subj") and r["subj"] not in seen:
            seen.add(r["subj"]); subjects.append(r["subj"])
    model, tok = load_model(a.model_id)
    model.eval()
    pre = tok(CONTEXT, add_special_tokens=True)["input_ids"]
    bad = 0
    with open(out, "w", encoding="utf-8") as f, torch.no_grad():
        for s in subjects:
            ids = tok(CONTEXT + " " + s, add_special_tokens=True)["input_ids"]
            if ids[:len(pre)] != pre:
                bad += 1
                continue
            x = torch.tensor([ids], device=model.device)
            lp = torch.log_softmax(model(x).logits[0].float(), -1)
            pos = range(len(pre), len(ids))
            surp = [-float(lp[p - 1, ids[p]]) for p in pos]
            f.write(json.dumps(dict(subj=s, name_tokens=len(surp), name_surprisal=sum(surp) / len(surp),
                                    name_surprisal_total=sum(surp))) + "\n")
    print(f"subjects {len(subjects)}, skipped (tokenizer boundary) {bad}", flush=True)
    print("E2_DONE", flush=True)


if __name__ == "__main__":
    main()
