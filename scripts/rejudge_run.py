"""Re-judge the stored replies in results/rejudge_gemma/todo.jsonl.gz with vLLM. Judge prompt, label parsing, engine and settings are those of
scripts/x2_run.py (Gemma equal to transformers eager attention on the paper's 1,161 replies). --judge gemma27b: the correction.
--judge qwen32b: a check that prompts are rebuilt exactly (Qwen's stored labels must come back)."""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
from x2_run import JUDGE_PROMPT, JUDGES, REPO, engine, ids, meta  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True); ap.add_argument("--out", required=True); ap.add_argument("--per-file", type=int, default=0); a = ap.parse_args()
    from vllm import SamplingParams as SP
    mid, rp = JUDGES[a.judge]
    todo = [json.loads(l) for l in gzip.open(REPO / "results/rejudge_gemma/todo.jsonl.gz", "rt", encoding="utf-8")]
    if a.per_file:  # a seeded random subset per stored file
        import random
        by = {}
        for x in todo:
            by.setdefault(x["file"], []).append(x)
        todo = [x for f in sorted(by) for x in random.Random(0).sample(by[f], min(a.per_file, len(by[f])))]
    g9 = Path(a.out) / "gemma9b_replies.jsonl"  # fresh Gemma-2-9B replies from scripts/gemma9b_check.py gen, if present
    if g9.exists():
        todo += [dict(file="gemma9b_replies.jsonl:" + r["prompt"], id=[r["key"], s], q=r["q"], reply=t) for r in map(json.loads, open(g9, encoding="utf-8")) if "long" in r for s, t in enumerate(r["long"])]
    llm = engine(mid, 2048); tok = llm.get_tokenizer()
    pr = ids(tok, [tok.apply_chat_template([{"role": "user", "content": JUDGE_PROMPT.format(question=x["q"], reply=x["reply"])}], tokenize=False, add_generation_prompt=True) for x in todo])
    t0 = time.time()
    out = llm.generate(pr, SP(temperature=0.0, max_tokens=5, repetition_penalty=rp))
    with open(Path(a.out) / f"rejudge_{a.judge}.jsonl", "w", encoding="utf-8") as f:
        for x, o in zip(todo, out):
            raw = o.outputs[0].text.strip().upper()
            lab = next((c for c in ("DONT_KNOW", "CLARIFY", "ANSWER") if raw.startswith(c) or raw.startswith(c.replace("_", " "))), "UNPARSED")
            f.write(json.dumps(dict(file=x["file"], id=x["id"], raw=raw, label=lab), ensure_ascii=False) + "\n")
    meta(a, f"rejudge_{a.judge}", model=mid, repetition_penalty=rp, seconds=round(time.time() - t0), n=len(todo))
