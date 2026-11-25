"""X3 generation: Gemma-2-9B-it with vLLM (correct attention) on the X2 question list, free replies under the paper's permissive prompt.
Everything except the model, the permissive suffix and Gemma's own sampling defaults is imported from scripts/x2_run.py.
T = 0.7, top_k = 50 (library default); Gemma's generation_config sets no top_p. Stop at <eos> and <end_of_turn>.
  python x3_run.py --tag pilot|full --out DIR
Judging uses scripts/x2_run.py judge unchanged (the judges see the original question, as in E9)."""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x2_run import PREFILL, REPO, SEED, build_completion_prompt, engine, format_chat_prompt, ids, meta, read  # noqa: E402

MODEL = "unsloth/gemma-2-9b-it"
(SUFFIX,) = [ast.literal_eval(n.value) for n in ast.parse((REPO / "scripts/e9_permissive.py").read_text(encoding="utf-8")).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "SUFFIX"]


def gen(a):
    from vllm import SamplingParams as SP
    items = [x for x in read(REPO / "results/x2/sample.jsonl") if a.tag == "full" or x["pilot"]]
    llm = engine(MODEL, 1024); tok = llm.get_tokenizer()
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
    q = [x["question"] if x["question"].endswith("?") else x["question"] + "?" for x in items]
    forced = ids(tok, [build_completion_prompt(tok, t, cloze_suffix=PREFILL, ensure_question_mark=True)["chat_formatted_prompt"] for t in q])
    free = ids(tok, [format_chat_prompt(tok, t + SUFFIX) for t in q])
    assert all(p["prompt_token_ids"][0] == tok.bos_token_id and p["prompt_token_ids"][1] != tok.bos_token_id for p in forced + free)
    base = SEED[a.tag] * 10_000_000
    samp = lambda n, m, off: [SP(n=n, temperature=0.7, top_k=50, max_tokens=m, stop_token_ids=stop, seed=base + off + 100 * i) for i in range(len(items))]
    line = lambda t: t.split("\n")[0].strip()
    t0 = time.time()
    g = llm.generate(forced, SP(temperature=0.0, max_tokens=12, stop_token_ids=stop))
    s = llm.generate(forced, samp(10, 12, 0))
    sh = llm.generate(free, samp(20, 24, 20))
    lo = llm.generate(free, samp(3, 96, 50))
    with open(Path(a.out) / f"gens_{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for x, gg, ss, ff, ll in zip(items, g, s, sh, lo):
            txt = lambda r: [o.text for o in r.outputs]
            f.write(json.dumps(dict(key=x["key"], q=x["question"], greedy=line(gg.outputs[0].text), samples=[line(t) for t in txt(ss)], short=[line(t) for t in txt(ff)], long=[t.strip() for t in txt(ll)]), ensure_ascii=False) + "\n")
    meta(a, f"gens_{a.tag}", model=MODEL, seconds=round(time.time() - t0), n=len(items), seed=SEED[a.tag], suffix=SUFFIX, sampling="T=0.7 top_k=50")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    gen(ap.parse_args())
