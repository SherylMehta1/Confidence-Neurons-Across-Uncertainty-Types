"""
hedge_measure_v2_criterion.md: regenerate free answers (3 samples, T=0.7, 96 tokens, full text), then label each with
two judges (DONT_KNOW / CLARIFY / ANSWER). Two subcommands; both resumable.

  python scripts/hedge_measure_v2.py gen   --model-id unsloth/Meta-Llama-3.1-8B-Instruct --tag llama
  python scripts/hedge_measure_v2.py judge --judge-id Qwen/Qwen2.5-32B-Instruct --judge-tag qwen32b --tags llama,qwen,gemma,mistral
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import torch

from _common import REPO_ROOT
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt
from shared.provenance import build_provenance, write_provenance

D = REPO_ROOT / "results" / "hedge_measure_v2"
PROMPT = ("You are checking replies from an AI assistant to a factual question. Classify the reply into exactly one category.\n"
          "DONT_KNOW: the reply says or clearly implies that the assistant does not know the answer, cannot find or verify the "
          "information, or does not recognise the subject (even if it then offers a guess).\n"
          "CLARIFY: the reply does not answer and instead asks which entity is meant, or says the question is ambiguous or needs "
          "more context, without saying it does not know.\n"
          "ANSWER: the reply commits to a specific answer (possibly with a mild qualifier such as \"probably\").\n"
          "Question: {question}\nReply: {reply}\nRespond with exactly one word: DONT_KNOW, CLARIFY or ANSWER.")


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


@torch.no_grad()
def gen(args):
    items = read(D / f"items_{args.tag}.jsonl")
    outp = D / f"gens_{args.tag}.jsonl"
    done = {r["key"] for r in read(outp)}
    model, tok = load_model(args.model_id)
    model.eval()
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(0)
    with open(outp, "a", encoding="utf-8") as f:
        for i, r in enumerate(x for x in items if x["key"] not in done):
            q = r["q"].strip(); q = q if q.endswith("?") else q + "?"
            enc = tok([format_chat_prompt(tok, q)], return_tensors="pt", add_special_tokens=False).to(model.device)
            enc.pop("token_type_ids", None)
            out = model.generate(**enc, max_new_tokens=96, do_sample=True, temperature=0.7, num_return_sequences=3, pad_token_id=pad)
            texts = [tok.decode(t[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for t in out]
            f.write(json.dumps(dict(r, samples=texts), ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 50 == 0:
                print(f"  [{args.tag}] {i + 1} items", flush=True)
    write_provenance(outp, build_provenance(model, script="scripts/hedge_measure_v2.py gen", model_id=args.model_id, n=len(items)))
    print("GEN_DONE", flush=True)


@torch.no_grad()
def judge(args):
    model, tok = load_model(args.judge_id)
    model.eval()
    for tag in args.tags.split(","):
        gens = read(D / f"gens_{tag}.jsonl")
        outp = D / f"judge_{args.judge_tag}_{tag}.jsonl"
        done = {(r["key"], r["s"]) for r in read(outp)}
        with open(outp, "a", encoding="utf-8") as f:
            for g in gens:
                for s, reply in enumerate(g["samples"]):
                    if (g["key"], s) in done:
                        continue
                    msgs = [{"role": "user", "content": PROMPT.format(question=g["q"], reply=reply)}]
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                    o = model.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
                    raw = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
                    lab = next((c for c in ("DONT_KNOW", "CLARIFY", "ANSWER") if raw.startswith(c) or raw.startswith(c.replace("_", " "))), "UNPARSED")
                    f.write(json.dumps(dict(key=g["key"], s=s, raw=raw, label=lab)) + "\n"); f.flush()
        print(f"  [{args.judge_tag}] judged {tag}", flush=True)
    print("JUDGE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--model-id", required=True); g.add_argument("--tag", required=True)
    j = sub.add_parser("judge"); j.add_argument("--judge-id", required=True); j.add_argument("--judge-tag", required=True)
    j.add_argument("--tags", default="llama,qwen,gemma,mistral")
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else judge(a)


if __name__ == "__main__":
    main()
