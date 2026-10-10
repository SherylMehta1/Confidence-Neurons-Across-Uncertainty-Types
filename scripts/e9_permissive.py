"""E9 (paper/PREREG/e9_permissive_prompt.md, sha f6459163): free replies under a prompt that permits "I don't know",
scored by the three judges of E5.

  python scripts/e9_permissive.py gen --model llama        # then --model gemma
  python scripts/e9_permissive.py judge --judges qwen32b,gemma27b,llama70b
Outputs in results/round3/e9/."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import gc
import json

import torch

from _common import REPO_ROOT
from hedge_measure_v2 import PROMPT
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt

JUDGES = {"qwen32b": "Qwen/Qwen2.5-32B-Instruct", "gemma27b": "unsloth/gemma-2-27b-it",
          "llama70b": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"}  # as in E5
MODELS = {"llama": "unsloth/Meta-Llama-3.1-8B-Instruct", "gemma": "unsloth/gemma-2-9b-it",
          "qwen": "Qwen/Qwen2.5-7B-Instruct", "mistral": "unsloth/mistral-7b-instruct-v0.3",
          "olmo2_7b": "allenai/OLMo-2-1124-7B-Instruct"}  # qwen, mistral: E12; olmo2_7b: E15
SUFFIX = " If you do not know the answer, say that you do not know."
OUT = REPO_ROOT / "results/round3/e9"


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


@torch.no_grad()
def gen(a):
    items = read(REPO_ROOT / f"results/hedge_measure_v2/items_{a.model}.jsonl")
    outp = OUT / f"gens_{a.model}.jsonl"
    done = {r["key"] for r in read(outp)}
    model, tok = load_model(MODELS[a.model]); model.eval()
    tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    torch.manual_seed(0)
    todo = [r for r in items if r["key"] not in done]
    print(f"[{a.model}] items {len(items)}, todo {len(todo)}", flush=True)
    with open(outp, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), 16):
            b = todo[i:i + 16]
            qs = [(r["q"].strip() if r["q"].strip().endswith("?") else r["q"].strip() + "?") + SUFFIX for r in b]
            enc = tok([format_chat_prompt(tok, q) for q in qs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            enc.pop("token_type_ids", None)
            o = model.generate(**enc, max_new_tokens=96, do_sample=True, temperature=0.7, num_return_sequences=3, pad_token_id=tok.pad_token_id)
            txt = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, r in enumerate(b):
                f.write(json.dumps(dict(r, prompt_q=qs[j], samples=[t.strip() for t in txt[j * 3:(j + 1) * 3]]), ensure_ascii=False) + "\n")
            f.flush()
    print(f"E9_GEN_DONE {a.model}", flush=True)


@torch.no_grad()
def judge(a):
    gens = {m: read(OUT / f"gens_{m}.jsonl") for m in MODELS}
    for jt in a.judges.split(","):
        model, tok = load_model(JUDGES[jt]); model.eval()
        tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
        for m, gs in gens.items():
            outp = OUT / f"judge_{jt}_{m}.jsonl"
            done = {(x["key"], x["s"]) for x in read(outp)}
            # the judge sees the original question, as in E5
            todo = [(g["key"], s, g["q"], t) for g in gs for s, t in enumerate(g["samples"]) if (g["key"], s) not in done]
            print(f"[judge {jt}] {m}: {len(todo)} replies", flush=True)
            with open(outp, "a", encoding="utf-8") as f:
                for i in range(0, len(todo), 16):
                    b = todo[i:i + 16]
                    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(question=q, reply=t)}],
                                                     tokenize=False, add_generation_prompt=True) for _, _, q, t in b]
                    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                    o = model.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id)
                    for (k, s, _, _), raw in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                        raw = raw.strip().upper()
                        lab = next((c for c in ("DONT_KNOW", "CLARIFY", "ANSWER") if raw.startswith(c) or raw.startswith(c.replace("_", " "))), "UNPARSED")
                        f.write(json.dumps(dict(key=k, s=s, raw=raw, label=lab)) + "\n")
                    f.flush()
        del model; gc.collect(); torch.cuda.empty_cache()
        print(f"JUDGED {jt}", flush=True)
    print("E9_JUDGE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--model", choices=list(MODELS), required=True)
    j = sub.add_parser("judge"); j.add_argument("--judges", default="qwen32b,gemma27b,llama70b")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gen(a) if a.cmd == "gen" else judge(a)


if __name__ == "__main__":
    main()
