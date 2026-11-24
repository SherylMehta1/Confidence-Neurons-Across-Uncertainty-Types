"""List every stored reply that has a stored Gemma-2-27B judge label -> results/rejudge_gemma/todo.jsonl (file, id, q, reply).
Only replies with a stored label are listed, in the stored order, so old and new labels line up one to one."""
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HM, R3 = REPO / "results/hedge_measure_v2", REPO / "results/round3"
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
out, n = [], {}


def add(stored, replies, ident):
    rel = stored.relative_to(REPO).as_posix()
    for r in read(stored):
        q, t = replies[tuple(r[k] for k in ident)]
        out.append(dict(file=rel, id=[r[k] for k in ident], q=q, reply=t))
        n[rel] = n.get(rel, 0) + 1


for base, pre in ((HM, "gens"), (R3 / "e9", "gens")):
    for p in sorted(base.glob("judge_gemma27b_*.jsonl")):
        tag = p.stem.replace("judge_gemma27b_", "")
        add(p, {(g["key"], s): (g["q"], t) for g in read(base / f"{pre}_{tag}.jsonl") for s, t in enumerate(g["samples"])}, ("key", "s"))
E = {}
for m in ("llama", "gemma9b"):
    for r in read(R3 / f"e4/replies_{m}.jsonl"):
        if r["split"] == "eval":
            E[r["model"], r["cond"], r["key"], r["s"]] = (r["q"], r["text"])
add(R3 / "e4/judge_gemma27b.jsonl", E, ("model", "cond", "key", "s"))
for d in ("e8", "e8_INVALID_swapped_kstar", "e16"):
    for p in sorted((R3 / d).glob("judge_gemma27b_*.jsonl")):
        tag = p.stem.replace("judge_gemma27b_", "")
        reps = [r for r in read(R3 / d / f"replies_{tag}.jsonl")]
        add(p, {(r["cond"], r["key"], r["s"]): (r["q"], r["text"]) for r in reps}, ("cond", "key", "s"))
with gzip.open(REPO / "results/rejudge_gemma/todo.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(len(out), n)
