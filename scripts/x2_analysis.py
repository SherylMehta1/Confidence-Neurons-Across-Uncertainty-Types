"""X2 analysis, exactly as in results/x2/x2_plan.md. CPU only. Outputs results/x2/{questions.csv, coefficients.csv, gradient.csv, summary.txt}."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from x2_run import HEDGE_RE, is_correct, read  # noqa: E402

D, OUT = REPO / "results/x2/raw", REPO / "results/x2"
SEED, B = 0, 5000
LINES = []


def say(s=""):
    print(s); LINES.append(str(s))


S = pd.DataFrame(read(OUT / "sample.jsonl")).set_index("key")
J = {j: {(r["key"], r["s"]): r["label"] for r in read(D / f"judge_{j}_full.jsonl")} for j in ("qwen32b", "gemma27b")}
rows = []
for g in read(D / "gens_full.jsonl"):
    al = S.possible_answers[g["key"]]
    ok = [is_correct(t, al) for t in g["samples"]]
    q = [J["qwen32b"][g["key"], s] == "DONT_KNOW" for s in range(3)]
    m = [J["gemma27b"][g["key"], s] == "DONT_KNOW" for s in range(3)]
    rows.append(dict(key=g["key"], know=np.mean(ok), greedy=float(is_correct(g["greedy"], al)), both=np.mean([a and b for a, b in zip(q, m)]), qwen=np.mean(q), gemma=np.mean(m),
                     regex24=np.mean([bool(HEDGE_RE.search(t)) for t in g["short"]]), regex96=np.mean([bool(HEDGE_RE.search(t)) for t in g["long"]])))
Q = pd.DataFrame(rows).set_index("key").join(S[["prop", "s_pop", "pop_rank", "decile", "n_relation", "pilot"]])
assert len(Q) == len(S) == 4376 and Q.notna().all().all()
Q["obscurity"] = 1 - Q.pop_rank  # 0 = most viewed question of its relation, 1 = least viewed
Q["band"] = pd.cut(Q.know, [-0.01, 0.05, 0.85, 1.01], labels=["0 of 10 correct", "1 to 8 of 10", "9 or 10 of 10"]).astype(str)
Q.reset_index().to_csv(OUT / "questions.csv", index=False, float_format="%.6g")


def irls(X, y, w, b=None):
    """Weighted fractional logit by Newton steps; minimum-norm step when a relation has no weight."""
    b = np.zeros(X.shape[1]) if b is None else b.copy()
    for _ in range(60):
        eta = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-eta))
        H = X.T @ (X * (w * p * (1 - p))[:, None]); g = X.T @ (w * (y - p))
        step = np.linalg.lstsq(H + 1e-9 * np.eye(len(b)), g, rcond=None)[0]
        b += np.clip(step, -5, 5)
        if np.abs(step).max() < 1e-8:
            return b, True
    return b, False


def fit(label, role, d, y, know="linear", greedy=False):
    v = d.groupby("prop")[y].agg(["min", "max"])
    const = sorted(v.index[(v["min"] == v["max"])])  # constant outcome: the fixed effect diverges and carries no slope information
    d = d[~d.prop.isin(const)]
    props = sorted(d.prop.unique())
    cols = [d.obscurity.to_numpy(float)]
    cols += [d.know.to_numpy(float)] if know == "linear" else [np.isclose(d.know.to_numpy(float), j / 10).astype(float) for j in range(1, 11)]
    if greedy:
        cols.append(d.greedy.to_numpy(float))
    k = len(cols)
    ri = d.prop.map({p: i for i, p in enumerate(props)}).to_numpy()
    X = np.column_stack(cols + [(ri == i).astype(float) for i in range(len(props))])
    Y = d[y].to_numpy(float)
    b, conv = irls(X, Y, np.ones(len(Y)))
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(props), np.full(len(props), 1 / len(props)), size=B)
    bs, bad = np.full((B, 2), np.nan), 0
    for t in range(B):
        r = np.flatnonzero(W[t]); keep = np.isin(ri, r)
        c = np.r_[np.arange(k), k + r]
        e, ok = irls(X[np.ix_(keep, c)], Y[keep], W[t][ri[keep]].astype(float), b[c])
        if ok:
            bs[t] = e[:2]
        else:
            bad += 1
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    # average predicted abstention at the 10th and 90th percentile of popularity (other covariates as observed)
    pr = lambda o: float(np.mean(1 / (1 + np.exp(-np.clip(np.column_stack([np.full(len(Y), o), X[:, 1:]]) @ b, -30, 30)))))
    r = dict(model=label, role=role, outcome=y, n=len(d), relations=len(props), relations_dropped_constant="; ".join(const), b_obscurity=b[0], obscurity_lo=lo[0], obscurity_hi=hi[0],
             b_knowledge=b[1] if know == "linear" else np.nan, knowledge_lo=lo[1] if know == "linear" else np.nan, knowledge_hi=hi[1] if know == "linear" else np.nan,
             predicted_at_popularity_90th=pr(0.1), predicted_at_popularity_10th=pr(0.9), converged=conv, draws=B, draws_not_converged=bad, seed=SEED)
    say(f"{role:<10} {label:<52} {y:<8} n {len(d):4d} rel {len(props):2d} | obscurity {b[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]" + (f" knowledge {b[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]" if know == "linear" else "")
        + f" | predicted {r['predicted_at_popularity_90th']:.3f} (popular) to {r['predicted_at_popularity_10th']:.3f} (obscure) | not converged {bad}")
    return r


say(f"X2: {len(Q)} questions, {Q.prop.nunique()} relations; mean abstention two judges {Q.both.mean():.3f}, regex short {Q.regex24.mean():.3f}; knowledge bands {Q.band.value_counts().to_dict()}")
C = [fit("two judges", "PRIMARY", Q, "both"), fit("regex, 20 short replies", "second", Q, "regex24")]
mid = Q[Q.decile.between(2, 9)]
for y in ("both", "regex24"):
    C.append(fit("deciles 2 to 9 only", "secondary", mid, y))
    C.append(fit("knowledge score as 11 levels", "secondary", Q, y, know="levels"))
    C.append(fit("plus greedy answer correct", "secondary", Q, y, greedy=True))
    C.append(fit("without country, capital and sport", "secondary", Q[~Q.prop.isin(["country", "capital", "sport"])], y))
for y in ("qwen", "gemma", "regex96"):
    C.append(fit("other abstention measure", "secondary", Q, y))
pd.DataFrame(C).to_csv(OUT / "coefficients.csv", index=False, float_format="%.6g")
p = C[0]
v = ("ABOVE ZERO: the popularity effect holds across the range (Llama, PopQA)" if p["obscurity_lo"] > 0 else "BELOW ZERO: report as found, rewrite the claim" if p["obscurity_hi"] < 0 else "INCLUDES ZERO: report as a limit of the extreme-tenths result")
say(f"DECISION (primary, obscurity coefficient {p['b_obscurity']:+.3f} [{p['obscurity_lo']:+.3f}, {p['obscurity_hi']:+.3f}]): {v}")

rng = np.random.default_rng(SEED)
rows = []
for y in ("both", "regex24", "regex96"):
    for band, g0 in [("all", Q), *Q.groupby("band")]:
        for dec, g in g0.groupby("decile"):
            v = g[y].to_numpy(float)
            bt = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
            rows.append(dict(outcome=y, band=band, decile=dec, n=len(v), mean=v.mean(), lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5), method="questions within decile and band", draws=B, seed=SEED))
pd.DataFrame(rows).to_csv(OUT / "gradient.csv", index=False, float_format="%.6g")
say(pd.DataFrame(rows).query("outcome == 'both'").pivot(index="decile", columns="band", values="mean").round(3).to_string())
say(pd.DataFrame(rows).query("outcome == 'both'").pivot(index="decile", columns="band", values="n").to_string())
(OUT / "summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
