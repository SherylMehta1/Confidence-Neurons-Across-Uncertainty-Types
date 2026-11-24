"""Gemma-2-9B-it check, read-off. Compares stored values, sdpa re-run and eager re-run (300 pool questions) and stored against fresh correct-attention
replies (the paper's 400 questions). Outputs results/rejudge_gemma/gemma9b_{diag,replies}.csv. CPU."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from x2_run import HEDGE_RE, is_correct, read, repo_names  # noqa: E402

(slick_class,) = repo_names("scripts/build_familiarity_twins.py", ["slick_class"], {})
D = REPO / "results/rejudge_gemma"
P = {}
for r in read(REPO / "data/familiarity_v3_gemma/all_measured.jsonl"):
    if r.get("subj"):
        P.setdefault(f"{r['prop']}::{r['subj']}", r)
X = {a: {r["key"]: r for r in read(D / f"gemma9b_diag_{a}.jsonl")} for a in ("sdpa", "eager")}
keys = list(X["sdpa"]); assert set(keys) == set(X["eager"])
st = {k: dict(entropy=P[k]["entropy"], top1=P[k]["top1"], greedy=P[k]["greedy"]) for k in keys}
V = {"stored": st, **X}
ok = {v: np.array([is_correct(V[v][k]["greedy"], P[k]["aliases"]) for k in keys]) for v in V}
E = {v: np.array([V[v][k]["entropy"] for k in keys]) for v in V}
rows = []
for a, b in (("stored", "sdpa"), ("stored", "eager"), ("sdpa", "eager")):
    rows.append(dict(a=a, b=b, n=len(keys), greedy_text_identical=np.mean([V[a][k]["greedy"] == V[b][k]["greedy"] for k in keys]), greedy_correctness_agree=(ok[a] == ok[b]).mean(), greedy_correct_a=ok[a].mean(), greedy_correct_b=ok[b].mean(),
                     entropy_abs_diff_mean=np.abs(E[a] - E[b]).mean(), entropy_abs_diff_max=np.abs(E[a] - E[b]).max(), entropy_mean_a=E[a].mean(), entropy_mean_b=E[b].mean(), entropy_corr=np.corrcoef(E[a], E[b])[0, 1]))
T = pd.DataFrame(rows); T.to_csv(D / "gemma9b_diag.csv", index=False, float_format="%.5g"); print(T.round(3).T.to_string(header=False))
arm = []
for v in V:
    d = pd.DataFrame(dict(e=E[v], arm=[P[k]["arm"] for k in keys], kn=[P[k]["slick"] for k in keys])); m = d.groupby(["arm", "kn"]).e.mean().round(2).to_dict(); arm.append((v, m))
print("entropy by cell:", arm)
items = {r["key"]: r for r in read(REPO / "results/hedge_measure_v2/items_gemma.jsonl")}
G = read(D / "gemma9b_replies.jsonl")
lab = []
for r in G:
    if r["prompt"] == "forced":
        al = P[r["key"]]["aliases"]; s = [is_correct(t, al) for t in r["samples"]]
        lab.append(dict(key=r["key"], stored=items[r["key"]]["slick"], arm=items[r["key"]]["arm"], new=slick_class(is_correct(r["greedy"], al), float(np.mean(s))), frac=float(np.mean(s)), frac_stored=P[r["key"]]["frac_correct"]))
L = pd.DataFrame(lab)
print("label with correct attention (rows: stored, by arm):"); print(L.groupby(["arm", "stored", "new"]).size().unstack(fill_value=0))
L.groupby(["arm", "stored", "new"]).size().rename("n").reset_index().to_csv(D / "gemma9b_labels.csv", index=False)
rows = []
for name, path in (("plain", REPO / "results/hedge_measure_v2/gens_gemma.jsonl"), ("permissive", REPO / "results/round3/e9/gens_gemma.jsonl")):
    old = {r["key"]: r["samples"] for r in read(path)}; new = {r["key"]: r["long"] for r in G if r["prompt"] == name}
    for k, it in items.items():
        rows.append(dict(prompt=name, arm=it["arm"], know=it["slick"], stored=np.mean([bool(HEDGE_RE.search(t)) for t in old[k]]), correct=np.mean([bool(HEDGE_RE.search(t)) for t in new[k]])))
R = pd.DataFrame(rows).groupby(["prompt", "arm", "know"])[["stored", "correct"]].mean().reset_index()
R.to_csv(D / "gemma9b_replies_regex.csv", index=False, float_format="%.5g"); print("regex abstention on 96-token replies, stored against correct attention"); print(R.round(3).to_string(index=False))
