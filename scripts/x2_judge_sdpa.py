"""X2 amendment: label the full-run replies with the paper's own Gemma path (transformers 5.14.1, default sdpa attention, one reply at a
time, as scripts/hedge_measure_v2.py). Faulty by construction (no attention soft-capping); kept only for comparison. Resumable.
  python x2_judge_sdpa.py --out DIR"""
import argparse
import ast
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
MID = "unsloth/gemma-2-27b-it"
env = {}
tree = ast.parse((REPO / "scripts/hedge_measure_v2.py").read_text(encoding="utf-8"))
exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PROMPT" for t in n.targets)], []), "p", "exec"), env)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out) / "judge_gemma27b_sdpa_full.jsonl"
done = {(r["key"], r["s"]) for r in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else set()
assert transformers.__version__ == "5.14.1"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="auto").eval()
assert model.config._attn_implementation == "sdpa"
t0 = time.time()
with open(out, "a", encoding="utf-8") as f, torch.no_grad():
    for g in map(json.loads, open(Path(a.out) / "gens_full.jsonl", encoding="utf-8")):
        for s, reply in enumerate(g["long"]):
            if (g["key"], s) in done:
                continue
            enc = tok(tok.apply_chat_template([{"role": "user", "content": env["PROMPT"].format(question=g["q"], reply=reply)}], tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to(model.device)
            o = model.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
            raw = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            lab = next((c for c in ("DONT_KNOW", "CLARIFY", "ANSWER") if raw.startswith(c) or raw.startswith(c.replace("_", " "))), "UNPARSED")
            f.write(json.dumps(dict(key=g["key"], s=s, raw=raw, label=lab)) + "\n"); f.flush()
print("STAGE_DONE judge_gemma27b_sdpa_full", round(time.time() - t0), flush=True)
