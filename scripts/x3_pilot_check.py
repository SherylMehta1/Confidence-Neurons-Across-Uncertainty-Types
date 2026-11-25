"""X3 pilot checks, exactly the rules in results/x3/x3_plan.md. Writes results/x3/pilot_checks.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from x2_run import HEDGE_RE, is_correct, read  # noqa: E402

D = REPO / "results/x3/raw"
S = {x["key"]: x for x in read(REPO / "results/x2/sample.jsonl")}
G = read(D / "gens_pilot.jsonl")
rows = []
add = lambda name, value, rule, ok: rows.append(dict(check=name, value=value, rule=rule, passed=bool(ok)))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


J = {j: {(r["key"], r["s"]): r for r in read(D / f"judge_{j}_pilot.jsonl")} for j in ("qwen32b", "gemma27b")}
want = {(g["key"], s) for g in G for s in range(3)}
n = sum(len(g["samples"]) == 10 and len(g["short"]) == 20 and len(g["long"]) == 3 for g in G)
add("questions with all outputs", n, "= 200", n == 200)
e = sum(not t.strip() for g in G for t in g["long"]); add("empty 96-token replies", e, "= 0", e == 0)
for j in J:
    add(f"{j}: missing labels", len(want - set(J[j])), "= 0", want == set(J[j]))
    m = sum(not r["raw"] for r in J[j].values()); add(f"{j}: empty judge outputs", m, "= 0", m == 0)
    u = np.mean([r["label"] == "UNPARSED" for r in J[j].values()]); add(f"{j}: off-format share", round(float(u), 4), "<= 0.01", u <= 0.01)
P = {}
for r in read(REPO / "data/familiarity_v3_gemma/all_measured.jsonl"):
    if r.get("subj"):
        P.setdefault(f"{r['prop']}::{r['subj']}", r)
pairs, lab = [], []
for g in G:
    r = P.get(g["key"])
    if r and r["question"].strip() == g["q"]:
        al = S[g["key"]]["possible_answers"]
        pairs.append((is_correct(g["greedy"], al), is_correct(r["greedy"], al))); lab.append((np.mean([is_correct(t, al) for t in g["samples"]]), r["frac_correct"]))
pairs, lab = np.array(pairs), np.array(lab)
a = (pairs[:, 0] == pairs[:, 1]).mean(); add(f"greedy correctness equals the stored one (n={len(pairs)})", round(float(a), 4), ">= 0.95", a >= 0.95)
add("information only: correlation of knowledge score with the stored share correct", round(float(np.corrcoef(lab.T)[0, 1]), 4), "report", True)
q, m = (np.array([[J[j][g["key"], s]["label"] == "DONT_KNOW" for s in range(3)] for g in G]) for j in ("qwen32b", "gemma27b"))
ab = (q & m).mean(); add("two-judge abstention", round(float(ab), 4), "0.05 to 0.95", 0.05 <= ab <= 0.95)
k = kappa(q.ravel(), m.ravel()); i = np.random.default_rng(0).integers(0, len(q), (10_000, len(q)))
lo, hi = np.nanpercentile([kappa(q[x].ravel(), m[x].ravel()) for x in i], [2.5, 97.5])
add("two-judge kappa", f"{k:.3f} [{lo:.3f}, {hi:.3f}]", ">= 0.60", k >= 0.60)
for j, ref in (("qwen32b", REPO / "results/hedge_measure_v2/judge_qwen32b_llama.jsonl"), ("gemma27b", REPO / "results/x2/gemma_judge_diagnosis/judge_gemma27b_llama_eager.jsonl")):
    st = {(r["key"], r["s"]): r["label"] for r in read(ref)}; nw = {(r["key"], r["s"]): r["label"] for r in read(D / f"judge_{j}_stored.jsonl")}
    assert set(st) == set(nw) and len(st) == 1161
    v = np.mean([(st[x] == "DONT_KNOW") == (nw[x] == "DONT_KNOW") for x in st])
    add(f"{j}: DONT_KNOW labels equal the reference on the paper's 1,161 Llama replies" + (" (eager)" if j == "gemma27b" else " (stored)"), round(float(v), 4), ">= 0.95", v >= 0.95)
add("information only: regex rate, 20 short replies", round(float(np.mean([bool(HEDGE_RE.search(t)) for g in G for t in g["short"]])), 4), "report", True)
T = pd.DataFrame(rows); T.to_csv(REPO / "results/x3/pilot_checks.csv", index=False)
print(T.to_string(index=False)); print("PILOT CLEAN" if T.passed.all() else "PILOT NOT CLEAN")
