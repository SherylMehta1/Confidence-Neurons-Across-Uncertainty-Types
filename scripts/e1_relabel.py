"""E1 (paper/PREREG/e1_independent_labels.md, sha af50f53f): knowledge labels from a paraphrased forced-answer prompt,
plus free-prompt first-token readouts, for Llama-3.1-8B-Instruct on the familiarity pool.

  python scripts/e1_relabel.py --out results/round3/e1_llama.jsonl
Resumable: items already in --out are skipped."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import torch

from _common import REPO_ROOT
from build_familiarity_twins import greedy_answer, is_correct, sample_answers, slick_class
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt

PARA = {  # one fixed paraphrase per relation; keeps the subject, changes the wording of the PopQA question
    "author": "Which writer wrote {s}?",
    "capital": "Name the capital city of {s}.",
    "capital of": "{s} is the capital of which place?",
    "color": "Which colour is {s}?",
    "composer": "Which composer wrote the music for {s}?",
    "country": "Which country is {s} located in?",
    "director": "Which filmmaker directed {s}?",
    "father": "Name the father of {s}.",
    "genre": "Which genre does {s} belong to?",
    "mother": "Name the mother of {s}.",
    "occupation": "What does {s} do for a living?",
    "place of birth": "Which city is {s}'s birthplace?",
    "producer": "Who produced {s}?",
    "religion": "Which religion does {s} follow?",
    "screenwriter": "Who wrote the screenplay for {s}?",
    "sport": "Which sport is {s} known for playing?",
}
PREFIX = "Answer:"


@torch.no_grad()
def last_logits(model, tok, prompt):
    enc = tok([prompt], return_tensors="pt", add_special_tokens=False).to(model.device)
    return model(**enc).logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--items", default="data/familiarity_v3/all_measured.jsonl")
    ap.add_argument("--out", default="results/round3/e1_llama.jsonl")
    a = ap.parse_args()
    torch.manual_seed(0)
    out = REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    items, seen = [], set()
    for l in open(REPO_ROOT / a.items, encoding="utf-8"):
        r = json.loads(l)
        k = f"{r['prop']}::{r['subj']}" if r.get("subj") else None
        if k and k not in seen:
            seen.add(k); items.append(r)
    model, tok = load_model(a.model_id)
    model.eval()
    i_id = tok.encode("I", add_special_tokens=False)[0]
    print(f"items {len(items)}, done {len(done)}, token 'I' = {i_id}", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for n, r in enumerate(items):
            k = f"{r['prop']}::{r['subj']}"
            if k in done:
                continue
            aliases = r["aliases"] if isinstance(r["aliases"], list) else json.loads(r["aliases"])
            para_q = PARA[r["prop"]].format(s=r["subj"])
            p_para = format_chat_prompt(tok, para_q) + PREFIX
            g = greedy_answer(model, tok, p_para)
            s = sample_answers(model, tok, p_para, 10)
            frac = sum(is_correct(x, aliases) for x in s) / 10
            lp = torch.log_softmax(last_logits(model, tok, p_para), -1)
            ent_para = float(-(lp.exp() * lp).sum())
            lf = torch.log_softmax(last_logits(model, tok, format_chat_prompt(tok, r["question"])), -1)
            ent_free = float(-(lf.exp() * lf).sum())
            p_i = float(lf[i_id].exp())
            rec = dict(key=k, prop=r["prop"], arm=r["arm"], slick=r["slick"], s_pop=r.get("s_pop"), para_q=para_q,
                       para_greedy=g, para_frac=frac, slick_para=slick_class(is_correct(g, aliases), frac),
                       entropy_para=ent_para, entropy_free=ent_free,
                       logodds_I_free=float(torch.log(torch.tensor(p_i)) - torch.log1p(torch.tensor(-p_i))))
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (n + 1) % 200 == 0:
                print(f"  {n + 1}/{len(items)}", flush=True)
    print("E1_DONE", flush=True)


if __name__ == "__main__":
    main()
