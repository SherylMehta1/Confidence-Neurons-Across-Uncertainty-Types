"""X2 pilot checks, exactly the rules in results/x2/x2_plan.md. Writes results/x2/pilot_checks.csv and prints PILOT CLEAN / NOT CLEAN."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from x2_run import HEDGE_RE, is_correct, read  # noqa: E402

D = REPO / "results/x2/raw"
S = {x["key"]: x for x in read(REPO / "results/x2/sample.jsonl")}
G = read(D / "gens_pilot.jsonl")
rows = []


def add(name, value, rule, ok):
    rows.append(dict(check=name, value=value, rule=rule, passed=bool(ok)))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


J = {j: {(r["key"], r["s"]): r for r in read(D / f"judge_{j}_pilot.jsonl")} for j in ("qwen32b", "gemma27b")}
want = {(g["key"], s) for g in G for s in range(3)}
add("questions with all outputs", sum(len(g["samples"]) == 10 and len(g["short"]) == 20 and len(g["long"]) == 3 for g in G), "= 200", sum(len(g["samples"]) == 10 and len(g["short"]) == 20 and len(g["long"]) == 3 for g in G) == 200)
add("empty 96-token replies", sum(not t.strip() for g in G for t in g["long"]), "= 0", not any(not t.strip() for g in G for t in g["long"]))
for j in J:
    add(f"{j}: missing labels", len(want - set(J[j])), "= 0", want == set(J[j]))
    add(f"{j}: empty judge outputs", sum(not r["raw"] for r in J[j].values()), "= 0", all(r["raw"] for r in J[j].values()))
    u = np.mean([r["label"] == "UNPARSED" for r in J[j].values()])
    add(f"{j}: unparsed share", round(u, 4), "<= 0.01", u <= 0.01)
P = {}
for n in ("familiarity_v3", "familiarity_v3_adj"):
    for r in read(REPO / f"data/{n}/all_measured.jsonl"):
        if r.get("subj"):
            P.setdefault(f"{r['prop']}::{r['subj']}", r)
pairs, lab = [], []
for g in G:
    r = P.get(g["key"])
    if r and r["question"].strip() == g["q"]:
        pairs.append((is_correct(g["greedy"], S[g["key"]]["possible_answers"]), is_correct(r["greedy"], S[g["key"]]["possible_answers"])))
        fc = np.mean([is_correct(t, S[g["key"]]["possible_answers"]) for t in g["samples"]]); lab.append((fc, r["frac_correct"]))
pairs, lab = np.array(pairs), np.array(lab)
ag = (pairs[:, 0] == pairs[:, 1]).mean()
add(f"greedy correctness equals the stored one (n={len(pairs)})", round(ag, 4), ">= 0.95", ag >= 0.95)
add("information only: correlation of knowledge score with the stored share correct", round(float(np.corrcoef(lab.T)[0, 1]), 4), "report", True)
add("information only: mean knowledge score minus stored", round(float((lab[:, 0] - lab[:, 1]).mean()), 4), "report", True)
q = np.array([[J["qwen32b"][g["key"], s]["label"] == "DONT_KNOW" for s in range(3)] for g in G])
m = np.array([[J["gemma27b"][g["key"], s]["label"] == "DONT_KNOW" for s in range(3)] for g in G])
ab = (q & m).mean()
add("two-judge abstention", round(float(ab), 4), "0.05 to 0.95", 0.05 <= ab <= 0.95)
k = kappa(q.ravel(), m.ravel())
i = np.random.default_rng(0).integers(0, len(q), (10_000, len(q)))
kb = np.array([kappa(q[x].ravel(), m[x].ravel()) for x in i]); lo, hi = np.nanpercentile(kb, [2.5, 97.5])
add("two-judge kappa", f"{k:.3f} [{lo:.3f}, {hi:.3f}]", ">= 0.50 and its interval overlaps [0.57, 0.67]", k >= 0.50 and lo <= 0.67 and hi >= 0.57)
for j in J:
    st = {(r["key"], r["s"]): r["label"] for r in read(REPO / f"results/hedge_measure_v2/judge_{j}_llama.jsonl")}
    nw = {(r["key"], r["s"]): r["label"] for r in read(D / f"judge_{j}_stored.jsonl")}
    assert set(st) == set(nw) and len(st) == 1161
    a = np.mean([(st[x] == "DONT_KNOW") == (nw[x] == "DONT_KNOW") for x in st])
    add(f"{j}: vLLM against stored DONT_KNOW labels on the paper's 1,161 replies", round(float(a), 4), ">= 0.95", a >= 0.95)
add("information only: regex rate, 20 short replies", round(float(np.mean([bool(HEDGE_RE.search(t)) for g in G for t in g["short"]])), 4), "report", True)
T = pd.DataFrame(rows); T.to_csv(REPO / "results/x2/pilot_checks.csv", index=False)
print(T.to_string(index=False)); print("PILOT CLEAN" if T.passed.all() else "PILOT NOT CLEAN")
