"""Is Gemma-2-9B-it, as a tested model, affected by transformers' sdpa attention (no soft-capping)?
  gen   (vLLM, correct attention): for the paper's 400 Gemma questions, forced greedy + 10 samples, and 3 free replies under the plain and
        the permissive prompt. Same prompts and sampling values as the paper's runs (T = 0.7, top_k 50; Gemma's generation_config sets no top_p).
  diag --attn sdpa|eager (transformers 5.14.1, one prompt at a time): forced-answer greedy text, first-token entropy and top-1 probability on 300
        pool questions (seed 0), to compare with the stored values.
Outputs: DIR/gemma9b_replies.jsonl, DIR/gemma9b_diag_{attn}.jsonl"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
MID = "unsloth/gemma-2-9b-it"
SUFFIX = " If you do not know the answer, say that you do not know."
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--attn"); ap.add_argument("--out", required=True); a = ap.parse_args()
from shared.prompt_format import build_completion_prompt, format_chat_prompt  # noqa: E402

line = lambda t: t.split("\n")[0].strip()
if a.cmd == "gen":
    from vllm import SamplingParams as SP
    from x2_run import engine, ids
    items = read(REPO / "results/rejudge_gemma/gemma9b_items.jsonl")
    llm = engine(MID, 1024); tok = llm.get_tokenizer()
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
    q = [x["q"].strip() if x["q"].strip().endswith("?") else x["q"].strip() + "?" for x in items]
    forced = ids(tok, [build_completion_prompt(tok, t, cloze_suffix="The answer is", ensure_question_mark=True)["chat_formatted_prompt"] for t in q])
    sp = lambda n, m, o: [SP(n=n, temperature=0.7, top_k=50, max_tokens=m, stop_token_ids=stop, seed=o + 100 * i) for i in range(len(items))]
    g = llm.generate(forced, SP(temperature=0.0, max_tokens=12, stop_token_ids=stop)); s = llm.generate(forced, sp(10, 12, 0))
    with open(Path(a.out) / "gemma9b_replies.jsonl", "w", encoding="utf-8") as f:
        for x, gg, ss in zip(items, g, s):
            f.write(json.dumps(dict(key=x["key"], q=x["q"], prompt="forced", greedy=line(gg.outputs[0].text), samples=[line(o.text) for o in ss.outputs]), ensure_ascii=False) + "\n")
        for name, qs, o in (("plain", q, 20), ("permissive", [t + SUFFIX for t in q], 40)):
            for x, r in zip(items, llm.generate(ids(tok, [format_chat_prompt(tok, t) for t in qs]), sp(3, 96, o))):
                f.write(json.dumps(dict(key=x["key"], q=x["q"], prompt=name, long=[k.text.strip() for k in r.outputs]), ensure_ascii=False) + "\n")
    print("STAGE_DONE gemma9b gen", flush=True)
else:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert transformers.__version__ == "5.14.1"
    pool = random.Random(0).sample(read(REPO / "results/rejudge_gemma/gemma9b_pool.jsonl"), 300)
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="auto", attn_implementation=a.attn).eval()
    t0 = time.time()
    with open(Path(a.out) / f"gemma9b_diag_{a.attn}.jsonl", "w", encoding="utf-8") as f, torch.no_grad():
        for r in pool:
            enc = tok([r["chat"]], return_tensors="pt", add_special_tokens=False).to(model.device)
            lg = model(**enc, use_cache=False).logits[0, -1].float(); p = lg.softmax(-1)
            o = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
            f.write(json.dumps(dict(key=r["key"], entropy=float(-torch.special.xlogy(p, p).sum()), top1=float(p.max()), greedy=line(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))), ensure_ascii=False) + "\n")
    print("STAGE_DONE gemma9b diag", a.attn, model.config._attn_implementation, round(time.time() - t0), flush=True)
