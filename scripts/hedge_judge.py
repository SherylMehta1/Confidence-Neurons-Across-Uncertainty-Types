"""
Judge pass for hedge_judge_criterion.md: label each sampled free generation HEDGE or ANSWER with a fixed prompt.

Reads  results/hedge_judge/sample.jsonl (drawn on CPU before judging; seed 0)
Writes results/hedge_judge/judgments.jsonl (sid, judge_raw, judge) -- resumable

  python scripts/hedge_judge.py --judge-id Qwen/Qwen2.5-32B-Instruct
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import torch

from _common import REPO_ROOT
from shared.model_utils import load_model
from shared.provenance import build_provenance, write_provenance

PROMPT = ("You are checking replies from an AI assistant to factual questions. Decide whether the reply says or clearly "
          "implies that the assistant does not know the answer, cannot identify the subject, or lacks the information "
          "(HEDGE), or whether it simply commits to an answer (ANSWER). A reply that says it does not know and then offers "
          "a guess is still HEDGE. A reply that gives an answer with a mild qualifier (\"probably\", \"I believe\") is ANSWER.\n"
          "Question: {question}\nReply: {reply}\nRespond with exactly one word: HEDGE or ANSWER.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-id", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--sample", default="results/hedge_judge/sample.jsonl")
    ap.add_argument("--out", default="results/hedge_judge/judgments.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(REPO_ROOT / args.sample, encoding="utf-8") if l.strip()]
    outp = REPO_ROOT / args.out
    done = {json.loads(l)["sid"] for l in open(outp, encoding="utf-8")} if outp.exists() else set()
    model, tok = load_model(args.judge_id)
    model.eval()
    with open(outp, "a", encoding="utf-8") as f:
        for i, r in enumerate(x for x in rows if x["sid"] not in done):
            msgs = [{"role": "user", "content": PROMPT.format(question=r["q"], reply=r["reply"])}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            j = "HEDGE" if raw.upper().startswith("HEDGE") else ("ANSWER" if raw.upper().startswith("ANSWER") else "UNPARSED")
            f.write(json.dumps(dict(sid=r["sid"], judge_raw=raw, judge=j)) + "\n"); f.flush()
            if (i + 1) % 200 == 0:
                print(f"  judged {i + 1}", flush=True)
    write_provenance(outp, build_provenance(model, script="scripts/hedge_judge.py", judge_id=args.judge_id, n=len(rows)))
    print("HEDGE_JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
