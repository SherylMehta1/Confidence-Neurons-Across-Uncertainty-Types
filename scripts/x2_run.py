"""X2 generation and judging with vLLM. Prompts, answer grader, regex, judge prompt and label parsing are taken from the existing
repo scripts by parsing their source, so nothing is retyped. Sampling settings are the ones transformers resolved in the paper's runs:
T=0.7 with top_p=0.9 (the model's generation_config) and top_k=50 (library default); judges greedy, with the repetition penalty 1.05 that
Qwen's generation_config applies.

  python x2_run.py gen   --tag pilot|full --out DIR
  python x2_run.py judge --judge qwen32b|gemma27b --tag pilot|full|stored --out DIR      (stored: the paper's 1,161 Llama replies)
"""
import argparse
import ast
import json
import re
import string
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from shared.prompt_format import build_completion_prompt, format_chat_prompt  # noqa: E402

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
JUDGES = {"qwen32b": ("Qwen/Qwen2.5-32B-Instruct", 1.05), "gemma27b": ("unsloth/gemma-2-27b-it", 1.0)}
SEED = {"pilot": 1, "full": 0}


def repo_names(path, names, env):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    keep = [n for n in tree.body if getattr(n, "name", None) in names or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    exec(compile(ast.Module(keep, []), path, "exec"), env)
    return [env[n] for n in names]


normalize, answer_head, is_correct, PREFILL = repo_names("scripts/build_familiarity_twins.py", ["normalize", "answer_head", "is_correct", "PREFILL"], {"re": re, "string": string})
(HEDGE_RE,) = repo_names("scripts/behavioral_test.py", ["HEDGE_RE"], {"re": re})
(JUDGE_PROMPT,) = repo_names("scripts/hedge_measure_v2.py", ["PROMPT"], {})


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def engine(model_id, max_len):
    from vllm import LLM
    return LLM(model=model_id, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.92, seed=0, enable_prefix_caching=True)


def ids(tok, prompts):
    return [{"prompt_token_ids": tok(p, add_special_tokens=False)["input_ids"]} for p in prompts]


def gen(a):
    from vllm import SamplingParams as SP
    items = [x for x in read(REPO / "results/x2/sample.jsonl") if a.tag == "full" or x["pilot"]]
    llm = engine(MODEL, 1024); tok = llm.get_tokenizer()
    stop = [128001, 128008, 128009]
    q = [x["question"] if x["question"].endswith("?") else x["question"] + "?" for x in items]
    forced = ids(tok, [build_completion_prompt(tok, t, cloze_suffix=PREFILL, ensure_question_mark=True)["chat_formatted_prompt"] for t in q])
    free = ids(tok, [format_chat_prompt(tok, t) for t in q])
    bos = tok.bos_token_id
    assert all(p["prompt_token_ids"][0] == bos and p["prompt_token_ids"][1] != bos for p in forced + free)
    base = SEED[a.tag] * 10_000_000
    samp = lambda n, m, off: [SP(n=n, temperature=0.7, top_p=0.9, top_k=50, max_tokens=m, stop_token_ids=stop, seed=base + off + 100 * i) for i in range(len(items))]
    line = lambda t: t.split("\n")[0].strip()
    t0 = time.time()
    g = llm.generate(forced, SP(temperature=0.0, max_tokens=12, stop_token_ids=stop))
    s = llm.generate(forced, samp(10, 12, 0))
    sh = llm.generate(free, samp(20, 24, 20))
    lo = llm.generate(free, samp(3, 96, 50))
    with open(Path(a.out) / f"gens_{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for x, gg, ss, ff, ll in zip(items, g, s, sh, lo):
            txt = lambda r: [o.text for o in r.outputs]
            r = dict(key=x["key"], q=x["question"], greedy=line(gg.outputs[0].text), samples=[line(t) for t in txt(ss)], short=[line(t) for t in txt(ff)], long=[t.strip() for t in txt(ll)])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta(a, f"gens_{a.tag}", model=MODEL, seconds=round(time.time() - t0), n=len(items), seed=SEED[a.tag])


def judge(a):
    from vllm import SamplingParams as SP
    mid, rp = JUDGES[a.judge]
    if a.tag == "stored":
        reps = [(g["key"], s, g["q"], t) for g in read(REPO / "results/hedge_measure_v2/gens_llama.jsonl") for s, t in enumerate(g["samples"])]
    else:
        reps = [(g["key"], s, g["q"], t) for g in read(Path(a.out) / f"gens_{a.tag}.jsonl") for s, t in enumerate(g["long"])]
    llm = engine(mid, 2048); tok = llm.get_tokenizer()
    pr = ids(tok, [tok.apply_chat_template([{"role": "user", "content": JUDGE_PROMPT.format(question=q, reply=t)}], tokenize=False, add_generation_prompt=True) for _, _, q, t in reps])
    t0 = time.time()
    out = llm.generate(pr, SP(temperature=0.0, max_tokens=5, repetition_penalty=rp))
    with open(Path(a.out) / f"judge_{a.judge}_{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for (k, s, _, _), o in zip(reps, out):
            raw = o.outputs[0].text.strip().upper()
            lab = next((c for c in ("DONT_KNOW", "CLARIFY", "ANSWER") if raw.startswith(c) or raw.startswith(c.replace("_", " "))), "UNPARSED")
            f.write(json.dumps(dict(key=k, s=s, raw=raw, label=lab)) + "\n")
    meta(a, f"judge_{a.judge}_{a.tag}", model=mid, repetition_penalty=rp, seconds=round(time.time() - t0), n=len(reps))


def meta(a, name, **k):
    import datetime
    import hashlib
    import torch
    import transformers
    import vllm
    from huggingface_hub import snapshot_download
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    rev = Path(snapshot_download(k["model"], local_files_only=True, allow_patterns=["config.json"])).name
    p = Path(a.out) / f"{name}.jsonl"
    (Path(a.out) / f"{name}.provenance.json").write_text(json.dumps(dict(k, revision=rev, vllm=vllm.__version__, torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0), dtype="bfloat16",
                                                             script_sha256=h(Path(__file__)), sample_sha256=h(REPO / "results/x2/sample.jsonl"), output_sha256=h(p), utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=1))
    print("STAGE_DONE", name, k, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "judge"]); ap.add_argument("--tag", required=True); ap.add_argument("--judge"); ap.add_argument("--out", required=True)
    A = ap.parse_args()
    gen(A) if A.cmd == "gen" else judge(A)
