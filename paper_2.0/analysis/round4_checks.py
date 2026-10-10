"""Round-4 review (post hoc, existing data only). Answers referee questions that existing outputs can settle:
(1) relation mix of the 387-question sample by cell, and of the obscure-known facts;
(2) relation-clustered bootstrap interval for the popularity preference, and a leave-one-relation-out range;
(3) the preference with all label corrections applied jointly: E7 semantic labels for unknown facts, popular-unknown
    questions with the gold answer (name echoes excluded) in any free reply dropped, guessable facts removed;
(4) obscure-known facts left after the guessability rule, and their abstention.
Writes analysis/round4_checks.txt."""
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parent.parent, HERE.parent.parent / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
D = REPO / "results/hedge_measure_v2"
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def nrm(x):
    return " " + re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip() + " "


pool = {}
for r in read(REPO / "data/familiarity_v3/all_measured.jsonl"):
    if r.get("subj"):
        pool.setdefault(f"{r['prop']}::{r['subj']}", r)


def aliases(r):
    a = r.get("aliases") or json.loads(r.get("possible_answers") or "[]")
    a = a if isinstance(a, list) else json.loads(a)
    return [x for x in a if x]


# guessability rule exactly as in analysis/guessability.py
lab_pool = {k: r for k, r in pool.items() if r.get("slick") in ("HighlyKnown", "Unknown")}
mode = {}
for prop in {r["prop"] for r in lab_pool.values()}:
    c = collections.Counter(r["obj"] for r in lab_pool.values() if r["prop"] == prop)
    top, n = c.most_common(1)[0]; mode[prop] = (top, n / sum(c.values()))
norm_g = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def guess(r):
    s = " " + norm_g(r["subj"]) + " "
    nm = any(len(norm_g(a)) >= 3 and (" " + norm_g(a) + " ") in s for a in aliases(r))
    top, share = mode[r["prop"]]
    return nm or (r["obj"] == top and share >= 0.2)


rates = pd.read_csv(D / "item_rates.csv").query("model == 'llama'")[["key", "prim"]]
items = {r["key"]: r for r in read(D / "items_llama.jsonl")}
rates["arm"] = [items[k]["arm"] for k in rates.key]
rates["know"] = [items[k]["slick"] for k in rates.key]
rates["prop"] = [k.split("::")[0] for k in rates.key]
rates["guess"] = [guess(pool[k]) for k in rates.key]
CELLS = [("uncertain", "Unknown"), ("uncertain", "HighlyKnown"), ("control", "Unknown"), ("control", "HighlyKnown")]


def pref(df):
    m = {c: df[(df.arm == c[0]) & (df.know == c[1])].prim.mean() for c in CELLS}
    return (m["uncertain", "HighlyKnown"] - m["control", "Unknown"]) if all(np.isfinite(list(m.values()))) else np.nan


def within_boot(df, B=10_000, seed=0):
    r = np.random.default_rng(seed)
    cells = {c: df[(df.arm == c[0]) & (df.know == c[1])].prim.to_numpy() for c in CELLS}
    bs = [np.mean(r.choice(cells["uncertain", "HighlyKnown"], len(cells["uncertain", "HighlyKnown"])))
          - np.mean(r.choice(cells["control", "Unknown"], len(cells["control", "Unknown"]))) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


# (1) relation mix
say("(1) relation mix of the 387-question sample (columns: obscure unknown, obscure known, popular unknown, popular known)")
tab = rates.groupby(["prop", "arm", "know"]).size().unstack(["arm", "know"]).fillna(0).astype(int)
tab = tab[[c for c in CELLS if c in tab.columns]]
say(tab.to_string())
ok = rates[(rates.arm == "uncertain") & (rates.know == "HighlyKnown")]
top = ok.prop.value_counts()
say(f"obscure known: {len(ok)} facts over {len(top)} relations; top 3 relations hold {top.iloc[:3].sum()} ({top.iloc[:3].sum() / len(ok):.0%}): {dict(top.iloc[:5])}")

# (2) relation-clustered bootstrap and leave-one-relation-out
p0 = pref(rates)
r = np.random.default_rng(0)
props = rates.prop.unique()
bs = []
for _ in range(5000):
    pick = r.choice(props, len(props))
    df = pd.concat([rates[rates.prop == p] for p in pick])
    v = pref(df)
    if np.isfinite(v):
        bs.append(v)
lo, hi = np.percentile(bs, [2.5, 97.5])
loo = {p: pref(rates[rates.prop != p]) for p in props}
say(f"(2) preference {p0:+.3f}; within-cell item bootstrap {within_boot(rates).round(3)}; relation-clustered bootstrap (5,000) [{lo:+.3f}, {hi:+.3f}]")
say(f"    leave-one-relation-out range [{min(loo.values()):+.3f}, {max(loo.values()):+.3f}] (lowest when dropping {min(loo, key=loo.get)})")

# (3) all label corrections jointly
A = read(REPO / "results/round3/e7_answers.jsonl")
G = {(x["key"], x["i"]): x["label"] for x in read(REPO / "results/round3/e7_grades.jsonl")}
sem = {}
for a in A:
    g = [G.get((a["key"], x)) for x in a["answers"]]
    if any(v is None for v in g):
        continue
    c = [v == "CORRECT" for v in g]
    sem[a["key"]] = "HighlyKnown" if (c[0] and sum(c[1:]) >= 9) else ("Unknown" if sum(c[1:]) == 0 else "intermediate")
alias_hit = {}
for g in read(D / "gens_llama.jsonl"):
    p = pool[g["key"]]; subj = nrm(p["subj"])
    al = [nrm(x).strip() for x in aliases(p) if len(str(x).strip()) >= 3]
    alias_hit[g["key"]] = any(any(" " + a + " " in nrm(t) for a in al if (" " + a + " ") not in subj) for t in g["samples"])
J = rates.copy()
J["know"] = [sem.get(k, "UNGRADED") if kn == "Unknown" else kn for k, kn in zip(J.key, J.know)]
J = J[J.know.isin(["HighlyKnown", "Unknown"])]
J = J[~((J.arm == "control") & (J.know == "Unknown") & J.key.map(alias_hit))]
J = J[~J.guess]
n = {f"{a[:3]}/{k[:4]}": int(((J.arm == a) & (J.know == k)).sum()) for a, k in CELLS}
say(f"(3) all corrections jointly (semantic regrade, popular-unknown alias drop with name echoes excluded, guessable removed): "
    f"preference {pref(J):+.3f}, within-cell bootstrap {within_boot(J).round(3)}; n {n}")
say("    cell rates: " + "  ".join(f"{a[:3]}/{k[:4]} {J[(J.arm == a) & (J.know == k)].prim.mean():.3f}" for a, k in CELLS))

# (4) guessability in the obscure-known cell
ng = ok[~ok.guess]
say(f"(4) obscure known: {int(ok.guess.sum())} of {len(ok)} guessable; the other {len(ng)} abstain at {ng.prim.mean():.3f}; "
    f"guessable ones at {ok[ok.guess].prim.mean():.3f}")
(HERE / "round4_checks.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")


# (5) the label-robust candidate headline: popularity gap among unknown facts, before and after all corrections
def gap_unknown(df, B=10_000, seed=0):
    a = df[(df.arm == "uncertain") & (df.know == "Unknown")].prim.to_numpy()
    b = df[(df.arm == "control") & (df.know == "Unknown")].prim.to_numpy()
    r = np.random.default_rng(seed)
    bs = [r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(B)]
    return a.mean() - b.mean(), *np.percentile(bs, [2.5, 97.5])


g0, g1 = gap_unknown(rates), gap_unknown(J)
say(f"(5) popularity gap among unknown facts: all {g0[0]:+.3f} [{g0[1]:+.3f}, {g0[2]:+.3f}]; after all corrections {g1[0]:+.3f} [{g1[1]:+.3f}, {g1[2]:+.3f}]")
r = np.random.default_rng(0)
bs = []
U = rates[rates.know == "Unknown"]
props_u = U.prop.unique()
for _ in range(5000):
    df = pd.concat([U[U.prop == p] for p in r.choice(props_u, len(props_u))])
    if ((df.arm == "uncertain").any()) and ((df.arm == "control").any()):
        bs.append(df[df.arm == "uncertain"].prim.mean() - df[df.arm == "control"].prim.mean())
say(f"    relation-clustered bootstrap for the unknown-facts gap (all): [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
(HERE / "round4_checks.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
