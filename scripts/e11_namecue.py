"""E11 (paper/PREREG/e11_name_cue_control.md, sha f716d396): can Llama recover the answers of facts it "knows" from
the form of the name alone? Asks the forced-answer question with three pseudo-names (subject with two interior letters
swapped and one interior vowel changed) for every known fact in the 387-question sample, using the original labelling
functions. Output: results/round3/e11_namecue.jsonl (one line per item and variant).

  python scripts/e11_namecue.py
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import json
import random

import torch

from _common import REPO_ROOT
from build_familiarity_twins import PREFILL, greedy_answer, is_correct, sample_answers, slick_class
from shared.model_utils import load_model
from shared.prompt_format import build_completion_prompt

OUT = REPO_ROOT / "results/round3/e11_namecue.jsonl"
VOWELS = "aeiou"


def perturb_word(w, rng):
    if not w.isalpha() or len(w) < 3:
        return w
    c = list(w)
    if len(c) >= 4:  # swap two adjacent interior letters (positions 1 .. len-2)
        j = rng.randrange(1, len(c) - 2)
        c[j], c[j + 1] = c[j + 1], c[j]
    idx = [i for i in range(1, len(c) - 1) if c[i].lower() in VOWELS]
    if idx:  # replace one interior vowel with a different vowel, keeping case
        i = rng.choice(idx)
        new = rng.choice([v for v in VOWELS if v != c[i].lower()])
        c[i] = new.upper() if c[i].isupper() else new
    return "".join(c)


def pseudo_names(key, subj, k=3):
    out, tries = [], 0
    while len(out) < k and tries < 50:
        rng = random.Random(f"{key}|{len(out)}|{tries}")
        p = " ".join(perturb_word(w, rng) for w in subj.split(" "))
        if p != subj and p not in out:
            out.append(p)
        tries += 1
    return out


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


@torch.no_grad()
def main():
    items = [r for r in read(REPO_ROOT / "results/hedge_measure_v2/items_llama.jsonl") if r["slick"] == "HighlyKnown"]
    pool = {}
    for r in read(REPO_ROOT / "data/familiarity_v3/all_measured.jsonl"):
        if r.get("subj"):
            pool.setdefault(f"{r['prop']}::{r['subj']}", r)
    done = {(x["key"], x["variant"]) for x in read(OUT)} if OUT.exists() else set()
    model, tok = load_model("unsloth/Meta-Llama-3.1-8B-Instruct")
    model.eval()
    torch.manual_seed(0)
    n_untestable = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for i, it in enumerate(items):
            p = pool[it["key"]]
            subj, q = p["subj"], it["q"]
            al = p.get("aliases") or json.loads(p.get("possible_answers") or "[]")
            al = al if isinstance(al, list) else json.loads(al)
            names = pseudo_names(it["key"], subj)
            if subj not in q or not names:
                n_untestable += 1
                f.write(json.dumps(dict(key=it["key"], arm=it["arm"], prop=p["prop"], variant=-1, untestable=True)) + "\n")
                continue
            for v, name in enumerate(names):
                if (it["key"], v) in done:
                    continue
                q2 = q.replace(subj, name)
                chat = build_completion_prompt(tok, q2, cloze_suffix=PREFILL, ensure_question_mark=True)["chat_formatted_prompt"]
                g = greedy_answer(model, tok, chat)
                s = sample_answers(model, tok, chat, 10)
                gok = is_correct(g, al)
                frac = sum(is_correct(x, al) for x in s) / len(s)
                f.write(json.dumps(dict(key=it["key"], arm=it["arm"], prop=p["prop"], variant=v, pseudo=name, q=q2, greedy=g,
                                        greedy_ok=gok, frac_ok=frac, slick=slick_class(gok, frac)), ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(items)} items", flush=True)
    print(f"E11_DONE untestable {n_untestable}", flush=True)


if __name__ == "__main__":
    main()
