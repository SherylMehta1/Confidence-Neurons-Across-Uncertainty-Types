"""Read-offs for the round-3 experiments, each exactly as its criterion note states (paper/PREREG/e1..e5_*.md).

  python scripts/round3_analysis.py e1|e2|e3|e4|e5
Each writes results/round3/<exp>_summary.txt (and a CSV where useful) and prints it."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import csv
import json
from collections import Counter

import numpy as np
import pandas as pd

from _common import REPO_ROOT

R3 = REPO_ROOT / "results/round3"
CELLS = [("uncertain", "Unknown"), ("uncertain", "HighlyKnown"), ("control", "Unknown"), ("control", "HighlyKnown")]
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def effects(df, y, arm="arm", know="know", B=10_000, seed=0):
    """Popularity and knowledge main effects on y over the 2x2 (obscure=uncertain), within-cell item bootstrap."""
    cells = {c: df[(df[arm] == c[0]) & (df[know] == c[1])][y].dropna().to_numpy(float) for c in CELLS}
    n = {f"{a[:3]}/{k[:4]}": len(v) for (a, k), v in cells.items()}
    if min(len(v) for v in cells.values()) == 0:
        return None, n

    def stat(c):
        m = {k: v.mean() for k, v in c.items()}
        pop = ((m["uncertain", "Unknown"] - m["control", "Unknown"]) + (m["uncertain", "HighlyKnown"] - m["control", "HighlyKnown"])) / 2
        kn = ((m["uncertain", "Unknown"] - m["uncertain", "HighlyKnown"]) + (m["control", "Unknown"] - m["control", "HighlyKnown"])) / 2
        return pop, kn, pop - kn

    pt = stat(cells)
    rng = np.random.default_rng(seed)
    bs = np.array([stat({k: rng.choice(v, len(v)) for k, v in cells.items()}) for _ in range(B)])
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    out = {nm: (pt[i], lo[i], hi[i]) for i, nm in enumerate(("pop", "know", "diff"))}
    out["means"] = {f"{a[:3]}/{k[:4]}": float(v.mean()) for (a, k), v in cells.items()}
    return out, n


def fmt(e):
    return "  ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in e.items() if k != "means")


def lpm(df, y, cols, B=2000, seed=0):
    X = pd.get_dummies(df[cols + ["prop"]], columns=["prop"], drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    Xv, yv = X.to_numpy(float), df[y].to_numpy(float)
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = np.array([np.linalg.lstsq(Xv[i], yv[i], rcond=None)[0] for i in (rng.integers(0, len(yv), len(yv)) for _ in range(B))])
    return {c: (beta[X.columns.get_loc(c)], *np.percentile(bs[:, X.columns.get_loc(c)], [2.5, 97.5])) for c in cols}


def pool_llama():
    p = {}
    for r in read_jsonl(REPO_ROOT / "data/familiarity_v3/all_measured.jsonl"):
        if r.get("subj"):
            p.setdefault(f"{r['prop']}::{r['subj']}", r)
    return p


# ------------------------------------------------------------------ E1
def e1():
    say("E1: independent knowledge labels (PREREG e1_independent_labels.md, sha af50f53f)")
    E = pd.DataFrame(read_jsonl(R3 / "e1_llama.jsonl"))
    P = pool_llama()
    E["entropy_orig"] = [P[k]["entropy"] for k in E.key]
    say(f"items relabelled: {len(E)}; paraphrase labels {dict(Counter(E.slick_para))}")
    say("T3 agreement (rows original label, cols paraphrase label), by arm:")
    for a in ("uncertain", "control"):
        t = pd.crosstab(E[E.arm == a].slick, E[E.arm == a].slick_para)
        say(f"  {('obscure' if a == 'uncertain' else 'popular')}:\n" + "\n".join("    " + l for l in t.to_string().splitlines()))
    J = pd.read_csv(REPO_ROOT / "results/hedge_measure_v2/item_rates.csv").query("model == 'llama'")[["key", "prim"]]
    J = J.merge(E[["key", "arm", "slick_para"]], on="key")
    J = J[J.slick_para.isin(["HighlyKnown", "Unknown"])].rename(columns={"slick_para": "know"})
    e, n = effects(J, "prim")
    say(f"T1 two-judge abstention with paraphrase knowledge labels, cells n {n}")
    if e is None or min(n.values()) < 20:
        say("  T1 UNDERPOWERED (a cell has < 20 items): descriptive only" + ("" if e is None else "  " + fmt(e)))
        if e is not None:
            say(f"  cell means {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}")
    else:
        say("  " + fmt(e) + f"\n  cell means {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}")
        ok = e["diff"][1] > 0 and e["means"]["unc/High"] >= 0.25
        say("  T1 verdict: " + ("SURVIVES" if ok else ("WEAKENED" if e["diff"][0] > 0 else "FAILS")))
    T2 = E[E.slick.isin(["HighlyKnown", "Unknown"]) & E.slick_para.isin(["HighlyKnown", "Unknown"])].rename(columns={"slick_para": "know"})
    e, n = effects(T2, "entropy_orig")
    kmp = (e["know"][0] - e["pop"][0])
    # CI of know - pop = -(diff)
    say(f"T2 original-prompt entropy x paraphrase knowledge, n {n}\n  {fmt(e)}")
    say(f"  know - pop {kmp:+.3f} [{-e['diff'][2]:+.3f}, {-e['diff'][1]:+.3f}]  -> " +
        ("NOT CIRCULAR" if -e["diff"][2] > 0 else "TIED TO ITS OWN LABELS"))
    T4 = E[E.slick.isin(["HighlyKnown", "Unknown"])].rename(columns={"slick": "know"})
    for y in ("entropy_free", "logodds_I_free"):
        e, n = effects(T4, y)
        flag = " -> FREE-PROMPT READOUT FOLLOWS POPULARITY MORE (prompt effect)" if (y == "entropy_free" and e["diff"][1] > 0) else ""
        say(f"T4 {y} (original labels), n {n}\n  {fmt(e)}{flag}\n  cell means {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}")
    (R3 / "e1_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E2
def e2():
    say("E2: name surprisal (PREREG e2_name_surprisal.md, sha 2b69e06c)")
    S = {r["subj"]: r for r in read_jsonl(R3 / "e2_surprisal.jsonl")}
    P = pool_llama()
    rows = []
    for k, r in P.items():
        if r["subj"] in S and r.get("s_pop"):
            rows.append(dict(key=k, prop=r["prop"], arm=r["arm"], slick=r["slick"], s_pop=r["s_pop"], regex=r["hedge_rate"],
                             surp=S[r["subj"]]["name_surprisal"], ntok=S[r["subj"]]["name_tokens"]))
    D = pd.DataFrame(rows)
    from scipy.stats import spearmanr
    rho = spearmanr(D.surp, np.log(D.s_pop.astype(float))).correlation
    say(f"S1 Spearman(surprisal, log page views) = {rho:+.3f} (n={len(D)})")
    D = D[D.slick.isin(["HighlyKnown", "Unknown"])].copy()
    D["surprisal_z"] = (D.surp - D.surp.mean()) / D.surp.std()
    D["name_tokens_z"] = (D.ntok - D.ntok.mean()) / D.ntok.std()
    D["unknown"] = (D.slick == "Unknown").astype(float)
    D["obscure"] = (D.arm == "uncertain").astype(float)
    J = pd.read_csv(REPO_ROOT / "results/hedge_measure_v2/item_rates.csv").query("model == 'llama'")[["key", "prim"]].merge(D, on="key")
    res = {}
    for name, df, y in (("(a) two judges", J, "prim"), ("(b) regex", D, "regex")):
        for tag, cols in (("S2", ["surprisal_z", "unknown", "name_tokens_z"]), ("S3", ["surprisal_z", "obscure", "unknown", "name_tokens_z"])):
            r = lpm(df, y, cols)
            res[(name, tag)] = r
            say(f"{tag} {name} n={len(df)}: " + "  ".join(f"{c} {b:+.3f} [{lo:+.3f},{hi:+.3f}]" for c, (b, lo, hi) in r.items()))
    ok = all(res[(nm, "S2")]["surprisal_z"][1] > 0 for nm in ("(a) two judges", "(b) regex"))
    say("S2 verdict: " + ("TRACKS OWN FAMILIARITY" if ok else "NOT SUPPORTED"))
    (R3 / "e2_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E3
def window_means(d, items):
    lab = {x["key"]: x["slick"] for x in read_jsonl(d / "labels.jsonl")}
    L = pd.read_csv(d / "lens.csv")
    L = L[L.layer >= 0]
    nl = L.layer.max() + 1  # lens layers 0..n_layers-2; rel depth = layer / (n_layers - 1), as in the lens criterion
    W = L[(L.layer / nl >= 1 / 3) & (L.layer / nl <= 2 / 3)]
    M = W.groupby("key")[["logodds", "hedge_term", "answer_term"]].mean().reset_index()
    M["neg_answer"] = -M.answer_term
    M["arm"] = [items[k]["arm"] for k in M.key]
    M["know"] = [lab.get(k) for k in M.key]
    return M[M.know.isin(["HighlyKnown", "Unknown"])]


def e3():
    say("E3: lens split (PREREG e3_lens_split.md, sha b2b916ce); window rel depth [1/3, 2/3]; 2,000 within-cell item bootstraps")
    items = {f"{r['prop']}::{r['subj']}": r for r in read_jsonl(REPO_ROOT / "data/familiarity_v3/all_measured.jsonl") if r.get("subj")}
    rows, l1 = [], {}
    for d in sorted((R3 / "e3").iterdir()):
        if not (d / "lens.csv").exists():
            continue
        M = window_means(d, items)
        for y in ("logodds", "hedge_term", "neg_answer"):
            e, n = effects(M, y, B=2000)
            rows.append(dict(model=d.name, term=y, pop=e["pop"][0], pop_lo=e["pop"][1], pop_hi=e["pop"][2],
                             know=e["know"][0], know_lo=e["know"][1], know_hi=e["know"][2], n=sum(n.values())))
            say(f"{d.name:16s} {y:11s} {fmt(e)}")
            if y == "hedge_term":
                l1[d.name] = e["pop"][1] > 0
    pd.DataFrame(rows).to_csv(R3 / "e3_window_effects.csv", index=False)
    base = [m for m in ("llama3.1-8b", "gemma-2-9b", "gemma-2-2b", "olmo-3-1025-7b") if m in l1]
    k = sum(l1[m] for m in base)
    say(f"L1: hedge-term popularity effect > 0 (CI excl 0) in {k} of {len(base)} base models analysed -> " +
        ("HEDGE-SIDE" if (len(base) == 4 and k >= 3) else ("INCOMPLETE" if len(base) < 4 else "CARRIED PARTLY BY THE ANSWER TERM")))
    (R3 / "e3_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E5
def fleiss(labels):  # labels: list of lists (raters per item), binary
    a = np.array(labels, float)
    n = a.shape[1]
    p1 = a.mean()
    pj = np.array([p1, 1 - p1])
    Pi = (a.sum(1) * (a.sum(1) - 1) + (n - a.sum(1)) * (n - a.sum(1) - 1)) / (n * (n - 1))
    Pe = (pj ** 2).sum()
    return (Pi.mean() - Pe) / (1 - Pe)


def cohen(x, y):
    x, y = np.array(x, float), np.array(y, float)
    po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe)


def e5():
    say("E5: third judge (PREREG e5_third_judge.md, sha f35aa4d9); majority = >= 2 of 3 DONT_KNOW; MEASURED if Fleiss kappa >= 0.60")
    D = REPO_ROOT / "results/hedge_measure_v2"
    out = []
    for fam in ("llama", "qwen", "gemma", "mistral"):
        J = {}
        for jt in ("qwen32b", "gemma27b", "llama70b"):
            for x in read_jsonl(D / f"judge_{jt}_{fam}.jsonl"):
                J.setdefault((x["key"], x["s"]), {})[jt] = x["label"] == "DONT_KNOW"
        both = {k: v for k, v in J.items() if len(v) == 3}
        lab = [[v["qwen32b"], v["gemma27b"], v["llama70b"]] for v in both.values()]
        fk = fleiss(lab)
        pair = {f"{a}-{b}": cohen([v[a] for v in both.values()], [v[b] for v in both.values()])
                for a, b in (("qwen32b", "gemma27b"), ("qwen32b", "llama70b"), ("gemma27b", "llama70b"))}
        meta = {r["key"]: r for r in read_jsonl(D / f"items_{fam}.jsonl")}
        it = {}
        for (k, s), v in both.items():
            it.setdefault(k, []).append(float(sum(v.values()) >= 2))
        df = pd.DataFrame([dict(key=k, arm=meta[k]["arm"], know=meta[k]["slick"], maj=np.mean(v)) for k, v in it.items() if k in meta])
        e, n = effects(df, "maj")
        measured = fk >= 0.60
        verdict = "UNMEASURED" if not measured else ("FOLLOWS POPULARITY" if e["diff"][1] > 0 else ("FOLLOWS KNOWLEDGE" if e["diff"][2] < 0 else "NEITHER"))
        say(f"{fam:8s} replies {len(both)}  Fleiss kappa {fk:.3f}  pairwise " + " ".join(f"{k} {v:.2f}" for k, v in pair.items()))
        say(f"         majority-rule {fmt(e)}  cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}  -> {verdict}")
        out.append(dict(family=fam, fleiss=fk, **{f"k_{k}": v for k, v in pair.items()}, pop=e["pop"][0], know=e["know"][0],
                        diff=e["diff"][0], diff_lo=e["diff"][1], diff_hi=e["diff"][2], verdict=verdict,
                        obs_unk=e["means"]["unc/Unkn"]))
    pd.DataFrame(out).to_csv(R3 / "e5_summary.csv", index=False)
    k = sum(1 for r in out if r["family"] != "llama" and r["verdict"] == "FOLLOWS POPULARITY")
    say(f"Other families MEASURED and FOLLOWS POPULARITY: {k} of 3 -> title " + ("may drop the Llama scope" if k >= 2 else "stays scoped to Llama"))
    (R3 / "e5_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E4
def e4():
    say("E4: steering v3 (PREREG e4_steering_v3.md, sha 18c74883); scorer = both judges DONT_KNOW; 10,000 item bootstraps")
    O = R3 / "e4"
    jl = {}
    for jt in ("qwen32b", "gemma27b"):
        for x in read_jsonl(O / f"judge_{jt}.jsonl"):
            jl.setdefault((x["model"], x["cond"], x["key"], x["s"]), {})[jt] = x["label"]
    for m in ("llama", "gemma9b"):
        p = O / f"replies_{m}.jsonl"
        if not p.exists():
            continue
        R = pd.DataFrame(read_jsonl(p))
        geo = json.load(open(O / f"geometry_{m}.json"))
        ks = json.load(open(O / f"kstar_{m}.json"))["kstar"]
        say(f"\n[{m}] layer {geo['layer']} natural norms fam {geo['nat_norm_fam']:.2f} know {geo['nat_norm_know']:.2f} abst {geo['nat_norm_abst']:.2f}; "
            f"abst fit on {geo['n_abst_hi']} vs {geo['n_abst_lo']} items; cos(fam,abst) {geo['cos_fam_abst']:+.3f}; k* = {ks}")
        S = R[R.split == "select"].groupby(["cond", "dose"]).regex.mean().unstack()
        say("  SELECT dose-response (regex abstention rate):\n" + "\n".join("    " + l for l in S.round(3).to_string().splitlines()))
        if ks is None:
            say("  no k*: NOT TESTABLE"); continue
        E = R[R.split == "eval"].copy()
        E["judge"] = [jl.get((m, c, k, s), {}) for c, k, s in zip(E.cond, E.key, E.s)]
        E = E[[len(j) == 2 for j in E.judge]]
        E["abstain"] = [float(all(v == "DONT_KNOW" for v in j.values())) for j in E.judge]
        it = E.groupby(["cond", "key"]).agg(abstain=("abstain", "mean"), regex=("regex", "mean"), degen=("degenerate", "mean"),
                                           entropy=("entropy", "first")).reset_index()
        piv = it.pivot(index="key", columns="cond", values="abstain")
        rand = piv[[f"random{j}" for j in range(5)]].mean(1)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(piv), (10_000, len(piv)))

        def ci(v):
            v = v.to_numpy(float)
            return v.mean(), *np.percentile(v[idx].mean(1), [2.5, 97.5])

        for c in ["none", "fam", "know", "abst", "fam_perp"] + [f"random{j}" for j in range(5)]:
            g = it[it.cond == c]
            say(f"  EVAL {c:9s} judges {g.abstain.mean():.3f}  regex {g.regex.mean():.3f}  degenerate {g.degen.mean():.3f}  entropy {g.entropy.mean():.3f}")
        d1, d2, d3 = ci(piv["fam"] - rand), ci(piv["fam_perp"] - rand), ci(piv["fam"] - piv["know"])
        ent = it.pivot(index="key", columns="cond", values="entropy")
        d4 = ci(ent["know"] - ent["fam"])
        say(f"  D1 fam - random      {d1[0]:+.3f} [{d1[1]:+.3f},{d1[2]:+.3f}] -> " + ("CONTROLS ABSTENTION" if d1[0] >= 0.10 and d1[1] > 0 else "NOT MET"))
        say(f"  D2 fam_perp - random {d2[0]:+.3f} [{d2[1]:+.3f},{d2[2]:+.3f}] -> " + ("BEYOND ABSTENTION" if d2[0] >= 0.10 and d2[1] > 0 else "CARRIED BY THE ABSTENTION COMPONENT"))
        say(f"  D3 fam - know        {d3[0]:+.3f} [{d3[1]:+.3f},{d3[2]:+.3f}]")
        say(f"  D4 entropy know - fam {d4[0]:+.3f} [{d4[1]:+.3f},{d4[2]:+.3f}]")
        p70 = O / "judge_llama70b.jsonl"
        if p70.exists():  # secondary scorer, PREREG e4_amendment_third_judge.md
            l70 = {(x["model"], x["cond"], x["key"], x["s"]): x["label"] for x in read_jsonl(p70)}
            E["maj"] = [float(sum(v == "DONT_KNOW" for v in list(j.values()) + [l70.get((m, c, k, s))]) >= 2)
                        for j, c, k, s in zip(E.judge, E.cond, E.key, E.s)]
            pm = E.groupby(["cond", "key"]).maj.mean().reset_index().pivot(index="key", columns="cond", values="maj")
            rm = pm[[f"random{j}" for j in range(5)]].mean(1)
            for nm, v in (("D1 maj fam - random", pm["fam"] - rm), ("D2 maj fam_perp - random", pm["fam_perp"] - rm), ("D3 maj fam - know", pm["fam"] - pm["know"])):
                c_ = ci(v)
                say(f"  (secondary, majority of 3) {nm} {c_[0]:+.3f} [{c_[1]:+.3f},{c_[2]:+.3f}]")
            say("  (secondary) majority-rule rates: " + "  ".join(f"{c} {pm[c].mean():.3f}" for c in pm.columns))
    (R3 / "e4_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E6
def e6():
    say("E6: entity-level knowledge covariate (PREREG e6_entity_knowledge.md, sha 937b46e3)")
    labs = {}
    for p in sorted(R3.glob("e6_judge_*.jsonl")):
        for x in read_jsonl(p):
            labs.setdefault(x["key"], {})[x["c"]] = x["label"]
    ek = {k: sum(v == "CORRECT" for v in d.values()) / len(d) for k, d in labs.items() if len(d) == 6}
    say(f"items graded on all 6 completions: {len(ek)}; label counts {dict(Counter(v for d in labs.values() for v in d.values()))}")
    P = pool_llama()
    D = pd.DataFrame([dict(key=k, prop=r["prop"], arm=r["arm"], slick=r["slick"], s_pop=r.get("s_pop"), regex=r["hedge_rate"], ek=ek[k])
                      for k, r in P.items() if k in ek])
    from scipy.stats import spearmanr
    say(f"K3 Spearman(entity_known, log page views) = {spearmanr(D.ek, np.log(D.s_pop.astype(float))).correlation:+.3f} (n={len(D)})")
    say("K3 mean entity_known by cell: " + "  ".join(
        f"{a[:3]}/{k[:4]} {D[(D.arm == a) & (D.slick == k)].ek.mean():.3f}" for a, k in CELLS))
    D = D[D.slick.isin(["HighlyKnown", "Unknown"])].copy()
    D["obscure"] = (D.arm == "uncertain").astype(float)
    D["unknown"] = (D.slick == "Unknown").astype(float)
    D["entity_known_z"] = (D.ek - D.ek.mean()) / D.ek.std()
    J = pd.read_csv(REPO_ROOT / "results/hedge_measure_v2/item_rates.csv").query("model == 'llama'")[["key", "prim"]].merge(D, on="key")
    ok = []
    for name, df, y in (("(a) two judges", J, "prim"), ("(b) regex", D, "regex")):
        base = lpm(df, y, ["obscure", "unknown"])
        full = lpm(df, y, ["obscure", "unknown", "entity_known_z"])
        shrink = 1 - full["obscure"][0] / base["obscure"][0]
        say(f"{name} n={len(df)} without: " + "  ".join(f"{c} {b:+.3f} [{lo:+.3f},{hi:+.3f}]" for c, (b, lo, hi) in base.items()))
        say(f"{name} n={len(df)} with:    " + "  ".join(f"{c} {b:+.3f} [{lo:+.3f},{hi:+.3f}]" for c, (b, lo, hi) in full.items()) +
            f"  | obscure coefficient shrinks by {shrink:.0%}")
        ok.append(full["obscure"][1] > 0)
    say("K1 verdict: " + ("SURVIVES ENTITY KNOWLEDGE" if all(ok) else ("PARTIAL" if any(ok) else "ACCOUNTED FOR BY ENTITY KNOWLEDGE")))
    (R3 / "e6_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E7
def two_judge(fam):
    D = REPO_ROOT / "results/hedge_measure_v2"
    J = {}
    for jt in ("qwen32b", "gemma27b"):
        for x in read_jsonl(D / f"judge_{jt}_{fam}.jsonl"):
            J.setdefault((x["key"], x["s"]), {})[jt] = x["label"]
    return J


def e7():
    say("E7: semantic relabelling and strict declines (PREREG e7_label_and_decline_audit.md, sha 1498be91); judge Llama-3.3-70B")
    A = read_jsonl(R3 / "e7_answers.jsonl")
    G = {(x["key"], x["i"]): x["label"] for x in read_jsonl(R3 / "e7_grades.jsonl")}
    sem, miss = {}, 0
    for r in A:
        g = [G.get((r["key"], a)) for a in r["answers"]]
        if any(v is None for v in g):
            miss += 1; continue
        c = [v == "CORRECT" for v in g]
        sem[r["key"]] = "HighlyKnown" if (c[0] and sum(c[1:]) >= 9) else ("Unknown" if sum(c[1:]) == 0 else "intermediate")
    say(f"pool Unknown items with forced answers {len(A)}, fully graded {len(sem)} (incomplete {miss}); semantic labels {dict(Counter(sem.values()))}")
    say("  by arm: " + "  ".join(f"{('obscure' if a == 'uncertain' else 'popular')} {dict(Counter(sem[r['key']] for r in A if r['key'] in sem and r['arm'] == a))}"
                                 for a in ("uncertain", "control")))
    items = {r["key"]: r for r in read_jsonl(REPO_ROOT / "results/hedge_measure_v2/items_llama.jsonl")}
    rates = pd.read_csv(REPO_ROOT / "results/hedge_measure_v2/item_rates.csv").query("model == 'llama'")[["key", "prim"]]
    rates["arm"] = [items[k]["arm"] for k in rates.key]
    rates["orig"] = [items[k]["slick"] for k in rates.key]
    rates["know"] = [sem.get(k, "UNGRADED") if o == "Unknown" else o for k, o in zip(rates.key, rates.orig)]
    mv = rates[rates.orig == "Unknown"].groupby(["arm", "know"]).size()
    say("T7a movement of the 2x2's Unknown items (arm, semantic label -> count):\n" + "\n".join("    " + l for l in mv.to_string().splitlines()))
    T = rates[rates.know.isin(["HighlyKnown", "Unknown"])]
    e, n = effects(T, "prim")
    say(f"T7a two-judge abstention with semantic labels, n {n}\n  {fmt(e)}\n  cell means {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}")
    say("  T7a verdict: " + ("SURVIVES SEMANTIC RELABELLING" if e["diff"][1] > 0 else "DOES NOT SURVIVE"))
    # E7c
    Dl = {(x["key"], x["i"]): x["label"] for x in read_jsonl(R3 / "e7_declines.jsonl")}
    say(f"T7c strict labels: {len(Dl)} replies, {dict(Counter(Dl.values()))}")
    J2 = two_judge("llama")
    rows = []
    for (k, s), lab in Dl.items():
        if k in items:
            j = J2.get((k, s), {})
            rows.append(dict(key=k, s=s, arm=items[k]["arm"], know=items[k]["slick"], decline=float(lab == "DECLINE"), strict=lab,
                             dk2=len(j) == 2 and all(v == "DONT_KNOW" for v in j.values())))
    R = pd.DataFrame(rows)
    it = R.groupby(["key", "arm", "know"]).decline.mean().reset_index()
    e, n = effects(it, "decline")
    say(f"T7c DECLINE rate, n {n}\n  {fmt(e)}\n  cell means {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}")
    say("  T7c verdict: " + ("STRICT-SURVIVES" if e["diff"][1] > 0 else "STRICT DOES NOT SURVIVE"))
    say("  share of both-judges-DONT_KNOW replies labelled HEDGED_ANSWER, by cell: " + "  ".join(
        f"{a[:3]}/{k[:4]} {(R[(R.arm == a) & (R.know == k) & R.dk2].strict == 'HEDGED_ANSWER').mean():.3f} (of {int(((R.arm == a) & (R.know == k) & R.dk2).sum())})"
        for a, k in CELLS))
    (R3 / "e7_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E8
def e8():
    say("E8: steering with a within-obscure abstention direction (PREREG e8_steering_within_level.md, sha 05474954); "
        "primary scorer majority of 3 judges; 10,000 question bootstraps")
    O = R3 / "e8"
    for m in ("llama", "gemma9b"):
        p = O / f"replies_{m}.jsonl"
        if not p.exists():
            continue
        R = pd.DataFrame(read_jsonl(p))
        geo = json.load(open(O / f"geometry_{m}.json"))
        say(f"\n[{m}] layer {geo['layer']} k* {geo['kstar']} nat norm fam {geo['nat_norm_fam']:.2f}; abst_within fit on {geo['n_abst_hi']} vs "
            f"{geo['n_abst_lo']} obscure items; cos(fam, abst_within) {geo['cos_fam_abst_within']:+.3f}")
        jl = {}
        for jt in ("qwen32b", "gemma27b", "llama70b"):
            q = O / f"judge_{jt}_{m}.jsonl"
            if q.exists():
                for x in read_jsonl(q):
                    jl.setdefault((x["cond"], x["key"], x["s"]), {})[jt] = x["label"] == "DONT_KNOW"
        R["j"] = [jl.get((c, k, s), {}) for c, k, s in zip(R.cond, R.key, R.s)]
        R = R[[len(j) == 3 for j in R.j]]
        if R.empty:
            say("  not all three judges done"); continue
        R["maj"] = [float(sum(j.values()) >= 2) for j in R.j]
        R["two"] = [float(j["qwen32b"] and j["gemma27b"]) for j in R.j]
        rng = np.random.default_rng(0)
        for sc, nm in (("maj", "primary, majority of 3"), ("two", "secondary, both of two")):
            piv = R.groupby(["cond", "key"])[sc].mean().reset_index().pivot(index="key", columns="cond", values=sc)
            rand = piv[[f"random{j}" for j in range(5)]].mean(1)
            idx = rng.integers(0, len(piv), (10_000, len(piv)))

            def ci(v):
                v = v.to_numpy(float)
                return v.mean(), *np.percentile(v[idx].mean(1), [2.5, 97.5])

            say(f"  [{nm}] rates: " + "  ".join(f"{c} {piv[c].mean():.3f}" for c in piv.columns) + f"  (n questions {len(piv)})")
            for lab, v in (("D1w fam - random", piv["fam"] - rand), ("D2w fam_perp_w - random", piv["fam_perp_w"] - rand),
                           ("abst_within - random", piv["abst_within"] - rand)):
                c_ = ci(v)
                tail = ""
                if lab.startswith("D2w") and sc == "maj" and m == "llama":
                    tail = " -> " + ("BEYOND WITHIN-LEVEL ABSTENTION" if c_[0] >= 0.10 and c_[1] > 0 else "CARRIED BY ABSTENTION")
                say(f"    {lab:24s} {c_[0]:+.3f} [{c_[1]:+.3f},{c_[2]:+.3f}]{tail}")
        say("  degenerate share by cond: " + "  ".join(f"{c} {g.degenerate.mean():.3f}" for c, g in R.groupby("cond")))
    (R3 / "e8_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E9
def e9():
    say("E9: permissive prompt (PREREG e9_permissive_prompt.md, sha f6459163); majority of 3 judges; MEASURED if Fleiss kappa >= 0.60")
    O = R3 / "e9"
    for m in ("llama", "gemma"):
        meta = {r["key"]: r for r in read_jsonl(REPO_ROOT / f"results/hedge_measure_v2/items_{m}.jsonl")}
        J = {}
        for jt in ("qwen32b", "gemma27b", "llama70b"):
            q = O / f"judge_{jt}_{m}.jsonl"
            if q.exists():
                for x in read_jsonl(q):
                    J.setdefault((x["key"], x["s"]), {})[jt] = x["label"] == "DONT_KNOW"
        full = {k: v for k, v in J.items() if len(v) == 3}
        if not full:
            say(f"{m}: no complete judgements yet"); continue
        fk = fleiss([[v["qwen32b"], v["gemma27b"], v["llama70b"]] for v in full.values()])
        it = {}
        for (k, s), v in full.items():
            it.setdefault(k, []).append(float(sum(v.values()) >= 2))
        df = pd.DataFrame([dict(key=k, arm=meta[k]["arm"], know=meta[k]["slick"], maj=np.mean(v)) for k, v in it.items() if k in meta])
        e, n = effects(df, "maj")
        measured = fk >= 0.60
        verdict = "UNMEASURED" if not measured else ("FOLLOWS POPULARITY" if e["diff"][1] > 0 else ("FOLLOWS KNOWLEDGE" if e["diff"][2] < 0 else "NEITHER"))
        say(f"{m:6s} replies {len(full)}  Fleiss kappa {fk:.3f}  n {n}\n  majority-rule {fmt(e)}\n  cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}  -> {verdict}")
        if m == "gemma":
            ou = e["means"]["unc/Unkn"]
            say(f"  Gemma licensing: obscure-unknown rate {ou:.3f} " + (">= 0.25: verdict replaces 'cannot be estimated' (permissive prompt)" if measured and ou >= 0.25
                                                                       else "< 0.25 or unmeasured: cannot be estimated even when permitted"))
    (R3 / "e9_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E10
def e10():
    say("E10: instruct lens split vs base (PREREG e10_instruct_lens.md, sha 6af179ce); popularity - knowledge, 2,000 within-cell item bootstraps")
    items = {f"{r['prop']}::{r['subj']}": r for r in read_jsonl(REPO_ROOT / "data/familiarity_v3/all_measured.jsonl") if r.get("subj")}
    rows = []
    for base, inst in (("llama3.1-8b", "llama3.1-8b-it"), ("gemma-2-9b", "gemma-2-9b-it"), ("gemma-2-2b", "gemma-2-2b-it")):
        if not all((R3 / "e3" / d / "lens.csv").exists() for d in (base, inst)):
            say(f"{base} / {inst}: lens output missing"); continue
        for y in ("logodds", "hedge_term"):
            r = {}
            for d in (base, inst):
                e, n = effects(window_means(R3 / "e3" / d, items), y, B=2000)
                r[d] = e
                say(f"  {d:16s} {y:11s} {fmt(e)}  n {sum(n.values())}")
                rows.append(dict(model=d, pair=base, term=y, **{f"{k}{s}": e[k][i] for k in ("pop", "know", "diff") for i, s in enumerate(("", "_lo", "_hi"))}))
            b, i = r[base]["diff"], r[inst]["diff"]
            up = i[0] > b[0] and i[1] > b[2]
            say(f"  {base} {y}: pop-know base {b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}] vs instruct {i[0]:+.3f} [{i[1]:+.3f},{i[2]:+.3f}] -> " +
                ("INSTRUCTION TUNING INCREASES THE POPULARITY PREFERENCE" if up else "NO CHANGE SHOWN"))
    pd.DataFrame(rows).to_csv(R3 / "e10_summary.csv", index=False)
    (R3 / "e10_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E12
def e12(models=("qwen", "mistral"), name="e12", head="E12: permissive prompt for Qwen and Mistral (PREREG e12_permissive_qwen_mistral.md, sha 56831ebc); identical to E9; "):
    say(head + "majority of 3 judges; MEASURED if Fleiss kappa >= 0.60")
    O = R3 / "e9"
    for m in models:
        meta = {r["key"]: r for r in read_jsonl(REPO_ROOT / f"results/hedge_measure_v2/items_{m}.jsonl")}
        J = {}
        for jt in ("qwen32b", "gemma27b", "llama70b"):
            q = O / f"judge_{jt}_{m}.jsonl"
            if q.exists():
                for x in read_jsonl(q):
                    J.setdefault((x["key"], x["s"]), {})[jt] = x["label"] == "DONT_KNOW"
        full = {k: v for k, v in J.items() if len(v) == 3}
        if not full:
            say(f"{m}: no complete judgements yet"); continue
        fk = fleiss([[v["qwen32b"], v["gemma27b"], v["llama70b"]] for v in full.values()])
        it = {}
        for (k, s), v in full.items():
            it.setdefault(k, []).append(float(sum(v.values()) >= 2))
        df = pd.DataFrame([dict(key=k, arm=meta[k]["arm"], know=meta[k]["slick"], maj=np.mean(v)) for k, v in it.items() if k in meta])
        e, n = effects(df, "maj")
        measured = fk >= 0.60
        verdict = "UNMEASURED" if not measured else ("FOLLOWS POPULARITY" if e["diff"][1] > 0 else ("FOLLOWS KNOWLEDGE" if e["diff"][2] < 0 else "NEITHER"))
        ou = e["means"]["unc/Unkn"]
        say(f"{m:7s} replies {len(full)}  Fleiss kappa {fk:.3f}  n {n}\n  majority-rule {fmt(e)}\n  cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}  -> {verdict}")
        say(f"  licensing: obscure-unknown rate {ou:.3f} -> " + ("reported as a permissive-prompt result" if measured and ou >= 0.25
                                                               else "cannot be estimated even when permitted"))
    (R3 / f"{name}_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E11
def e11():
    say("E11: name-cue control (PREREG e11_name_cue_control.md, sha f716d396); pseudo-names with two interior letters swapped "
        "and one vowel changed; cue-recoverable (primary) = greedy correct for >= 2 of 3 pseudo-names; cue-known (secondary) = "
        "the original labelling rule calls it known for >= 2 of 3")
    X = pd.DataFrame(read_jsonl(R3 / "e11_namecue.jsonl"))
    unt = set(X[X.variant == -1].key)
    T = X[X.variant >= 0]
    agg = T.groupby("key").agg(arm=("arm", "first"), prop=("prop", "first"), n=("variant", "nunique"),
                               g=("greedy_ok", "sum"), k=("slick", lambda v: int((v == "HighlyKnown").sum()))).reset_index()
    agg = agg[agg.n == 3]
    rec = set(agg[agg.g >= 2].key)
    kn = set(agg[agg.k >= 2].key)
    say(f"known items probed {len(agg)} (untestable {len(unt)}); cue-recoverable {len(rec)}; cue-known (strict) {len(kn)}")
    for a, nm in (("uncertain", "obscure"), ("control", "popular")):
        g = agg[agg.arm == a]
        say(f"  {nm} known: {len(g)} probed; cue-recoverable {int((g.g >= 2).sum())} ({(g.g >= 2).mean():.2f}); cue-known {int((g.k >= 2).sum())}")
    say("  cue-recoverable share by relation (obscure known): " + "  ".join(
        f"{p} {int((gg.g >= 2).sum())}/{len(gg)}" for p, gg in agg[agg.arm == "uncertain"].groupby("prop")))
    items = {r["key"]: r for r in read_jsonl(REPO_ROOT / "results/hedge_measure_v2/items_llama.jsonl")}
    rates = pd.read_csv(REPO_ROOT / "results/hedge_measure_v2/item_rates.csv").query("model == 'llama'")[["key", "prim"]]
    rates["arm"] = [items[k]["arm"] for k in rates.key]
    rates["know"] = [items[k]["slick"] for k in rates.key]
    rates["prop"] = [k.split("::")[0] for k in rates.key]
    ok = rates[(rates.arm == "uncertain") & (rates.know == "HighlyKnown")]
    say(f"  obscure known abstention: cue-recoverable {ok[ok.key.isin(rec)].prim.mean():.3f} (n {int(ok.key.isin(rec).sum())}); "
        f"other {ok[~ok.key.isin(rec)].prim.mean():.3f} (n {int((~ok.key.isin(rec)).sum())})")

    def pref(df):
        m = {c: df[(df.arm == c[0]) & (df.know == c[1])].prim.mean() for c in CELLS}
        return m["uncertain", "HighlyKnown"] - m["control", "Unknown"]

    for tag, drop in (("primary (drop cue-recoverable)", rec), ("secondary (drop cue-known)", kn)):
        D = rates[~((rates.know == "HighlyKnown") & rates.key.isin(drop))]
        e, n = effects(D, "prim")
        r = np.random.default_rng(0)
        props = D.prop.unique()
        bs = []
        for _ in range(5000):
            dd = pd.concat([D[D.prop == p] for p in r.choice(props, len(props))])
            if all(((dd.arm == c[0]) & (dd.know == c[1])).any() for c in CELLS):
                bs.append(pref(dd))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        n_ok = n["unc/High"]
        left = D[(D.arm == "uncertain") & (D.know == "HighlyKnown")].prop.value_counts().to_dict()
        say(f"  {tag}: n {n}\n    {fmt(e)}\n    cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}\n"
            f"    relation-clustered bootstrap [{lo:+.3f}, {hi:+.3f}]; remaining obscure known by relation {left}")
        if tag.startswith("primary"):
            v = ("UNDERPOWERED" if n_ok < 20 else ("SURVIVES NAME-CUE CONTROL" if e["diff"][0] > 0 and e["diff"][1] > 0 else "DOES NOT SURVIVE"))
            say(f"  T11 verdict: {v}")
    (R3 / "e11_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ E13 / E14
def _two_judge_items(tag):
    D = REPO_ROOT / "results/hedge_measure_v2"
    meta = {r["key"]: r for r in read_jsonl(D / f"items_{tag}.jsonl")}
    J = {}
    for jt in ("qwen32b", "gemma27b"):
        q = D / f"judge_{jt}_{tag}.jsonl"
        if q.exists():
            for x in read_jsonl(q):
                J.setdefault((x["key"], x["s"]), {})[jt] = x["label"]
    it = {}
    for (k, s_), v in J.items():
        if len(v) == 2:
            it.setdefault(k, []).append(float(all(l == "DONT_KNOW" for l in v.values())))
    df = pd.DataFrame([dict(key=k, arm=meta[k]["arm"], know=meta[k]["slick"], prop=k.split("::")[0], prim=np.mean(v))
                       for k, v in it.items() if k in meta])
    kap = cohen([v["qwen32b"] == "DONT_KNOW" for v in J.values() if len(v) == 2],
                [v["gemma27b"] == "DONT_KNOW" for v in J.values() if len(v) == 2])
    return df, kap


def _clustered(df, B=5000, seed=0):
    def pref(d):
        m = {c: d[(d.arm == c[0]) & (d.know == c[1])].prim.mean() for c in CELLS}
        return m["uncertain", "HighlyKnown"] - m["control", "Unknown"]
    r = np.random.default_rng(seed)
    props = df.prop.unique()
    bs = []
    for _ in range(B):
        d = pd.concat([df[df.prop == p] for p in r.choice(props, len(props))])
        if all(((d.arm == c[0]) & (d.know == c[1])).any() for c in CELLS):
            bs.append(pref(d))
    return np.percentile(bs, [2.5, 97.5])


def e13():
    say("E13: fresh replies on the same 387 questions (PREREG e13_e14_replications.md, sha fbbc8c25); seed 1; two judges, both DONT_KNOW")
    df, kap = _two_judge_items("llama_s1")
    e, n = effects(df.rename(columns={"prim": "prim"}), "prim")
    say(f"two-judge kappa {kap:.3f}; n {n}\n  {fmt(e)}\n  cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}")
    say("  E13 verdict: " + ("REPLICATES" if e["diff"][0] > 0 and e["diff"][1] > 0 else "DOES NOT REPLICATE"))
    (R3 / "e13_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


def e14():
    say("E14: new questions from the 10-30th and 70-90th page-view percentiles (PREREG e13_e14_replications.md, sha fbbc8c25); "
        "two judges, both DONT_KNOW")
    df, kap = _two_judge_items("llama_adj")
    e, n = effects(df, "prim")
    lo, hi = _clustered(df)
    say(f"two-judge kappa {kap:.3f}; n {n}\n  {fmt(e)}\n  cells {json.dumps({k: round(v, 3) for k, v in e['means'].items()})}\n"
        f"  relation-clustered bootstrap for the preference [{lo:+.3f}, {hi:+.3f}]")
    for c in CELLS:
        say(f"  relation mix {c[0][:3]}/{c[1][:4]}: " + json.dumps(df[(df.arm == c[0]) & (df.know == c[1])].prop.value_counts().to_dict()))
    nk = n["unc/High"]
    v = "UNDERPOWERED" if nk < 30 else ("REPLICATES IN NEW QUESTIONS" if e["diff"][0] > 0 and e["diff"][1] > 0 else "DOES NOT REPLICATE")
    say(f"  E14 verdict: {v} (obscure-adjacent known n = {nk})")
    (R3 / "e14_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


def e15():
    e12(models=("olmo2_7b",), name="e15", head="E15: permissive prompt for OLMo-2-7B-Instruct (PREREG e15_e16_olmo_and_steering.md, sha 2b3e665d); identical to E12; ")


def e16():
    """E16: as E8 with the abstention direction fitted within the popular arm and 20 random directions (sha 2b3e665d)."""
    say("E16: steering with a popular-arm abstention direction (PREREG e15_e16_olmo_and_steering.md, sha 2b3e665d); majority of 3 judges; "
        "20 random directions; 10,000 question bootstraps")
    O = R3 / "e16"
    R = pd.DataFrame(read_jsonl(O / "replies_llama.jsonl"))
    geo = json.load(open(O / "geometry_llama.json"))
    say(f"layer {geo['layer']} k* {geo['kstar']}; abst_pop fit on {geo['n_abst_hi']} vs {geo['n_abst_lo']} popular items; "
        f"cos(fam, abst_pop) {geo['cos_fam_abst_pop']:+.3f}")
    jl = {}
    for jt in ("qwen32b", "gemma27b", "llama70b"):
        q = O / f"judge_{jt}_llama.jsonl"
        if q.exists():
            for x in read_jsonl(q):
                jl.setdefault((x["cond"], x["key"], x["s"]), {})[jt] = x["label"] == "DONT_KNOW"
    R["j"] = [jl.get((c, k, s_), {}) for c, k, s_ in zip(R.cond, R.key, R.s)]
    R = R[[len(j) == 3 for j in R.j]]
    R["maj"] = [float(sum(j.values()) >= 2) for j in R.j]
    piv = R.groupby(["cond", "key"]).maj.mean().reset_index().pivot(index="key", columns="cond", values="maj")
    rnd_cols = [c for c in piv.columns if c.startswith("random")]
    rand = piv[rnd_cols].mean(1)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(piv), (10_000, len(piv)))

    def ci(v):
        v = v.to_numpy(float)
        return v.mean(), *np.percentile(v[idx].mean(1), [2.5, 97.5])

    rr = piv[rnd_cols].mean(0)
    say(f"rates: none {piv['none'].mean():.3f}  fam {piv['fam'].mean():.3f}  abst_pop {piv['abst_pop'].mean():.3f}  fam_perp_p {piv['fam_perp_p'].mean():.3f}  "
        f"random mean {rr.mean():.3f} (2.5-97.5 pct over {len(rnd_cols)} directions {np.percentile(rr, 2.5):.3f}-{np.percentile(rr, 97.5):.3f})")
    g1 = ci(piv["abst_pop"] - rand); d1 = ci(piv["fam"] - rand); d2 = ci(piv["fam_perp_p"] - rand)
    say(f"  G1 abst_pop - random {g1[0]:+.3f} [{g1[1]:+.3f},{g1[2]:+.3f}]")
    say(f"  fam - random {d1[0]:+.3f} [{d1[1]:+.3f},{d1[2]:+.3f}]")
    say(f"  D2p fam_perp_p - random {d2[0]:+.3f} [{d2[1]:+.3f},{d2[2]:+.3f}]")
    g1ok = g1[0] > 0 and g1[1] > 0
    v = "NOT SEPARABLE" if not g1ok else ("BEYOND GENERIC ABSTENTION" if d2[0] >= 0.10 and d2[1] > 0 else "GENERIC ABSTENTION")
    say(f"  E16 verdict: {v}; degenerate share {R.degenerate.mean():.3f}")
    (R3 / "e16_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6, "e7": e7, "e8": e8, "e9": e9, "e10": e10, "e11": e11, "e12": e12, "e13": e13, "e14": e14, "e15": e15, "e16": e16}[_sys.argv[1]]()
