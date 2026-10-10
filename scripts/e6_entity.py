"""E6 (paper/PREREG/e6_entity_knowledge.md, sha 937b46e3): entity-level knowledge probe and its grading.

  python scripts/e6_entity.py gen                    -> results/round3/e6_gen.jsonl  (Llama-3.1-8B-Instruct)
  python scripts/e6_entity.py judge --shard 0 --of 2 -> results/round3/e6_judge_0.jsonl (Llama-3.3-70B, 4-bit)"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import torch

from _common import REPO_ROOT
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt

R3 = REPO_ROOT / "results/round3"
JUDGE = ("You are grading whether an AI assistant correctly identified an entity.\n"
         "Entity name: {subj}\nReference description (Wikidata): {desc}\nAssistant's completion: {subj} is {text}\n"
         "CORRECT: the completion identifies the same kind of entity as the reference and any distinguishing detail it gives "
         "is consistent with the reference (if the reference gives only a type, the right type suffices).\n"
         "WRONG: it describes a different entity or the wrong kind of entity.\n"
         "VAGUE: too generic to tell.\n"
         "Respond with exactly one word: CORRECT, WRONG or VAGUE.")


@torch.no_grad()
def gen(a):
    desc = {x["key"]: x["description"] for x in map(json.loads, open(R3 / "e6_wikidata.jsonl", encoding="utf-8"))}
    items, seen = [], set()
    for l in open(REPO_ROOT / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8"):
        r = json.loads(l)
        k = f"{r['prop']}::{r['subj']}" if r.get("subj") else None
        if k and k in desc and k not in seen:
            seen.add(k); items.append(r)
    out = R3 / "e6_gen.jsonl"
    done = {json.loads(l)["key"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    model, tok = load_model("unsloth/Meta-Llama-3.1-8B-Instruct")
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    todo = [r for r in items if f"{r['prop']}::{r['subj']}" not in done]
    print(f"items {len(items)}, todo {len(todo)}", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 16):
            b = todo[i:i + 16]
            prompts = [format_chat_prompt(tok, f"What is {r['subj']}? Answer in a few words.") + f"{r['subj']} is" for r in b]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            enc.pop("token_type_ids", None)
            g = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
            s = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=0.7, num_return_sequences=5, pad_token_id=tok.pad_token_id)
            L = enc["input_ids"].shape[1]
            gt = tok.batch_decode(g[:, L:], skip_special_tokens=True)
            st = tok.batch_decode(s[:, L:], skip_special_tokens=True)
            for j, r in enumerate(b):
                k = f"{r['prop']}::{r['subj']}"
                f.write(json.dumps(dict(key=k, subj=r["subj"], description=desc[k],
                                        completions=[gt[j].strip()] + [x.strip() for x in st[j * 5:(j + 1) * 5]]), ensure_ascii=False) + "\n")
            f.flush()
            if (i // 16) % 20 == 0:
                print(f"  {i + len(b)}/{len(todo)}", flush=True)
    print("E6_GEN_DONE", flush=True)


@torch.no_grad()
def judge(a):
    rows = [json.loads(l) for l in open(R3 / "e6_gen.jsonl", encoding="utf-8")]
    rows = [r for i, r in enumerate(rows) if i % a.of == a.shard]
    out = R3 / f"e6_judge_{a.shard}.jsonl"
    done = {(x["key"], x["c"]) for x in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else set()
    todo = [(r, c, t) for r in rows for c, t in enumerate(r["completions"]) if (r["key"], c) not in done]
    print(f"shard {a.shard}/{a.of}: {len(todo)} completions to grade", flush=True)
    model, tok = load_model("unsloth/Llama-3.3-70B-Instruct-bnb-4bit")
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 16):
            b = todo[i:i + 16]
            texts = [tok.apply_chat_template([{"role": "user", "content": JUDGE.format(subj=r["subj"], desc=r["description"], text=t)}],
                                             tokenize=False, add_generation_prompt=True) for r, c, t in b]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            o = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
            for (r, c, t), raw in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                raw = raw.strip().upper()
                lab = next((x for x in ("CORRECT", "WRONG", "VAGUE") if raw.startswith(x)), "UNPARSED")
                f.write(json.dumps(dict(key=r["key"], c=c, label=lab, raw=raw)) + "\n")
            f.flush()
            if (i // 16) % 50 == 0:
                print(f"  {i + len(b)}/{len(todo)}", flush=True)
    print(f"E6_JUDGE_DONE {a.shard}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen")
    j = sub.add_parser("judge"); j.add_argument("--shard", type=int, default=0); j.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else judge(a)
