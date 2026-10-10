"""E7 (paper/PREREG/e7_label_and_decline_audit.md, sha 1498be91).
  python scripts/e7_audit.py gen     # Llama forced answers (greedy + 10 samples) for every Unknown pool item
  python scripts/e7_audit.py judge   # Llama-3.3-70B: semantic grading of those answers, and strict-decline labels of
                                     # the 96-token two-judge replies
Outputs: results/round3/e7_answers.jsonl, e7_grades.jsonl, e7_declines.jsonl"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import torch

from _common import REPO_ROOT
from shared.model_utils import load_model
from shared.prompt_format import build_completion_prompt

R3 = REPO_ROOT / "results/round3"
GRADE = ("You are grading an answer to a factual question.\nQuestion: {q}\nAccepted answers (any one is correct): {aliases}\n"
         "Answer given: {a}\nIs the answer given correct, i.e. does it name the same entity as one of the accepted answers "
         "(allowing spelling variants, fuller or shorter names, or an equivalent description)?\n"
         "Respond with exactly one word: CORRECT or WRONG.")
DECLINE = ("You are checking a reply from an AI assistant to a factual question. Classify the reply into exactly one category.\n"
           "DECLINE: the reply says or implies that the assistant does not know or cannot verify the answer, and it names NO candidate answer.\n"
           "HEDGED_ANSWER: the reply expresses uncertainty but names a candidate answer.\n"
           "ANSWER: the reply commits to a specific answer.\n"
           "CLARIFY: the reply asks which entity is meant, without answering.\n"
           "Question: {q}\nReply: {r}\nRespond with exactly one word: DECLINE, HEDGED_ANSWER, ANSWER or CLARIFY.")


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


@torch.no_grad()
def gen(a):
    items, seen = [], set()
    for r in read(REPO_ROOT / "data/familiarity_v3/all_measured.jsonl"):
        k = f"{r['prop']}::{r['subj']}" if r.get("subj") else None
        if k and k not in seen and r.get("slick") == "Unknown":
            seen.add(k); items.append(r)
    out = R3 / "e7_answers.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {x["key"] for x in read(out)}
    model, tok = load_model("unsloth/Meta-Llama-3.1-8B-Instruct"); model.eval()
    tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    torch.manual_seed(0)
    todo = [r for r in items if f"{r['prop']}::{r['subj']}" not in done]
    print(f"Unknown items {len(items)}, todo {len(todo)}", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 16):
            b = todo[i:i + 16]
            prompts = [build_completion_prompt(tok, r["q"])["chat_formatted_prompt"] for r in b]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device); enc.pop("token_type_ids", None)
            L = enc["input_ids"].shape[1]
            g = tok.batch_decode(model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)[:, L:], skip_special_tokens=True)
            s = tok.batch_decode(model.generate(**enc, max_new_tokens=12, do_sample=True, temperature=0.7, num_return_sequences=10, pad_token_id=tok.pad_token_id)[:, L:], skip_special_tokens=True)
            for j, r in enumerate(b):
                al = r["aliases"] if isinstance(r["aliases"], list) else json.loads(r["aliases"])
                f.write(json.dumps(dict(key=f"{r['prop']}::{r['subj']}", arm=r["arm"], q=r["q"], aliases=al,
                                        answers=[g[j].split("\n")[0].strip()] + [x.split("\n")[0].strip() for x in s[j * 10:(j + 1) * 10]]), ensure_ascii=False) + "\n")
            f.flush()
            if (i // 16) % 20 == 0:
                print(f"  {i + len(b)}/{len(todo)}", flush=True)
    print("E7_GEN_DONE", flush=True)


@torch.no_grad()
def judge(a):
    model, tok = load_model("unsloth/Llama-3.3-70B-Instruct-bnb-4bit"); model.eval()
    tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token

    def run(tasks, outp, labels, maxtok, bs=32):
        done = {(x["key"], x["i"]) for x in read(outp)}
        tasks = [t for t in tasks if (t[0], t[1]) not in done]
        print(f"{outp.name}: {len(tasks)} to label", flush=True)
        with open(outp, "a", encoding="utf-8") as f:
            for i in range(0, len(tasks), bs):
                b = tasks[i:i + bs]
                texts = [tok.apply_chat_template([{"role": "user", "content": t[2]}], tokenize=False, add_generation_prompt=True) for t in b]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                o = tok.batch_decode(model.generate(**enc, max_new_tokens=maxtok, do_sample=False, pad_token_id=tok.pad_token_id)[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for t, raw in zip(b, o):
                    raw = raw.strip().upper().replace(" ", "_")
                    lab = next((x for x in labels if raw.startswith(x)), "UNPARSED")
                    f.write(json.dumps(dict(key=t[0], i=t[1], label=lab, raw=raw[:40])) + "\n")
                f.flush()
                if (i // bs) % 50 == 0:
                    print(f"  {i + len(b)}/{len(tasks)}", flush=True)

    ans = read(R3 / "e7_answers.jsonl")
    # each distinct answer string is graded once; "i" holds the answer text
    run([(r["key"], x, GRADE.format(q=r["q"], aliases="; ".join(r["aliases"]), a=x)) for r in ans for x in dict.fromkeys(r["answers"])],
        R3 / "e7_grades.jsonl", ("CORRECT", "WRONG"), 3)
    gens = read(REPO_ROOT / "results/hedge_measure_v2/gens_llama.jsonl")
    run([(g["key"], s, DECLINE.format(q=g["q"], r=t)) for g in gens for s, t in enumerate(g["samples"])],
        R3 / "e7_declines.jsonl", ("DECLINE", "HEDGED_ANSWER", "ANSWER", "CLARIFY"), 6)
    print("E7_JUDGE_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen"); sub.add_parser("judge")
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else judge(a)
