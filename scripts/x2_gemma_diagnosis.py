"""Post hoc, found by the X2 pilot: the stored Gemma-2-27B judge labels came from transformers' sdpa attention, which skips Gemma-2's
attention-logit soft-capping. Compares stored, sdpa (re-run), eager (re-run) and vLLM labels on the paper's 1,161 Llama replies and
recomputes the paper's two-judge quantities with the eager labels. CPU; seed 0, 10,000 draws (5,000 for relation resampling)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "results/x0"))
import os
os.environ.setdefault("OUT", str(REPO / "results/x2/gemma_judge_diagnosis"))
from x0_common import *  # noqa: E402,F403

D = REPO / "results/x2/gemma_judge_diagnosis"
L = lambda p: {(r["key"], r["s"]): r for r in read(p)}
st = {j: L(HM / f"judge_{j}_llama.jsonl") for j in ("qwen32b", "gemma27b", "llama70b")}
V = {"stored (the paper)": st["gemma27b"], "re-run, sdpa (transformers default)": L(D / "judge_gemma27b_llama_sdpa.jsonl"), "re-run, eager (soft-capping on)": L(D / "judge_gemma27b_llama_eager.jsonl"),
     "vLLM": L(REPO / "results/x2/raw/judge_gemma27b_stored.jsonl")}
rows = []
dk = lambda x: np.array([x[k]["label"] == "DONT_KNOW" for k in st["gemma27b"]])
for n, x in V.items():
    rows.append(dict(gemma_labels=n, dont_know_rate=dk(x).mean(), unparsed=sum(r["label"] == "UNPARSED" for r in x.values()), raw_identical_to_stored=np.mean([x[k]["raw"] == st["gemma27b"][k]["raw"] for k in x]),
                     label_identical_to_eager=np.mean([x[k]["label"] == V["re-run, eager (soft-capping on)"][k]["label"] for k in x]), kappa_with_qwen=kappa(dk(x), dk(st["qwen32b"])), kappa_with_llama70=kappa(dk(x), dk(st["llama70b"]))))
T = pd.DataFrame(rows); save(T, "gemma_judge_engines.csv"); print(T.round(3).to_string(index=False))
info = {r["key"]: r for r in read(HM / "items_llama.jsonl")}
out = []
for n, x in V.items():
    R = pd.DataFrame([dict(key=k, both=st["qwen32b"][k, s]["label"] == "DONT_KNOW" and r["label"] == "DONT_KNOW", gemma=r["label"] == "DONT_KNOW") for (k, s), r in x.items()])
    Q = R.groupby("key").mean().reset_index()
    Q["cell"] = [CELL_OF[info[k]["arm"], info[k]["slick"]] for k in Q.key]; Q["prop"] = Q.key.str.split("::").str[0]
    for y in ("both", "gemma"):
        pt, bs = question_boot(Q, y); e, b = effects(pt), effects(bs); rl = effects(relation_boot(Q, y, False))
        out.append(dict(gemma_labels=n, measure="two judges" if y == "both" else "Gemma alone", **{c: pt[c] for c in CELLS}, **{f"{q}{s}": v for q in ("popularity effect", "knowledge effect", "margin", "gap among unknown") for s, v in zip(("", " lo", " hi"), (e[q], *ci(b[q])))},
                        margin_relation_lo=ci(rl["margin"])[0], margin_relation_hi=ci(rl["margin"])[1]))
O = pd.DataFrame(out); save(O, "paper_numbers_by_gemma_labels.csv")
for r in out:
    print(f"{r['gemma_labels']:<38} {r['measure']:<12} OK {r['OK']:.3f} OU {r['OU']:.3f} PK {r['PK']:.3f} PU {r['PU']:.3f} | pop {r['popularity effect']:+.3f} know {r['knowledge effect']:+.3f} margin {fmt(r['margin'], r['margin lo'], r['margin hi'])} unknown gap {fmt(r['gap among unknown'], r['gap among unknown lo'], r['gap among unknown hi'])}")
