"""
Read-off for qwen_factorial_criterion.md (written before the Qwen run), plus the Llama reference and
the cross-model item agreement. CPU only.

Inputs  data/familiarity_v3/{all_measured,hedge_logodds}.jsonl        (Llama-3.1-8B-Instruct)
        data/familiarity_v3_qwen/{all_measured,hedge_logodds}.jsonl   (Qwen2.5-7B-Instruct)
Output  results/qwen_factorial/{summary.txt, cells.csv, effects.csv, cross_model_items.csv}

Primary analysis (exactly as written in the criterion):
  cells = popularity arm (obscure = bottom decile, popular = top decile) x knowledge (HighlyKnown vs
  Unknown; Maybe excluded); popularity effect = mean over knowledge of Y(obscure) - Y(popular);
  knowledge effect = mean over popularity of Y(Unknown) - Y(HighlyKnown); 95% CI from 10,000 item
  bootstraps within cells (seed 0). REPLICATES iff hedging(pop - know) > 0 and entropy(know - pop) > 0,
  both CIs excluding 0, on the governing hedging readout: free-generation hedge rate unless the
  model's pool-level hedge rate (all scored items) is below 0.05, then first-token hedge log-odds.
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = _Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "qwen_factorial"
POOLS = {"llama": "familiarity_v3", "qwen": "familiarity_v3_qwen"}
CRITERIA = {"qwen": "qwen_factorial_criterion.md", "gemma": "gemma_mistral_factorial_criterion.md",
            "mistral": "gemma_mistral_factorial_criterion.md"}
FLOOR = 0.05
B = 10_000
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def load(pool):
    rows = [json.loads(l) for l in open(REPO / "data" / pool / "all_measured.jsonl", encoding="utf-8") if l.strip()]
    d = pd.DataFrame([dict(prop=r["prop"], subj=r["subj"], arm=r["arm"], slick=r["slick"], s_pop=r.get("s_pop"),
                           hedge_rate=r["hedge_rate"], entropy=r["entropy"], top1=r["top1"],
                           frac_correct=r["frac_correct"]) for r in rows if r.get("subj")])
    lo_path = REPO / "data" / pool / "hedge_logodds.jsonl"
    if lo_path.exists():
        lo = pd.DataFrame([json.loads(l) for l in open(lo_path, encoding="utf-8") if l.strip()])
        d = d.merge(lo[["prop", "subj", "hedge_logodds", "hedge_mass"]], on=["prop", "subj"], how="left")
    else:
        d["hedge_logodds"] = np.nan
        d["hedge_mass"] = np.nan
    return d.drop_duplicates(["prop", "subj"])


def effects(d, y, rng):
    """Popularity and knowledge main effects on y over the 2 x 2, with within-cell item bootstrap."""
    cells = {(a, k): d[(d.arm == a) & (d.slick == k)][y].dropna().to_numpy()
             for a in ("uncertain", "control") for k in ("Unknown", "HighlyKnown")}

    def stat(c):
        m = {key: v.mean() for key, v in c.items()}
        pop = ((m["uncertain", "Unknown"] - m["control", "Unknown"]) + (m["uncertain", "HighlyKnown"] - m["control", "HighlyKnown"])) / 2
        know = ((m["uncertain", "Unknown"] - m["uncertain", "HighlyKnown"]) + (m["control", "Unknown"] - m["control", "HighlyKnown"])) / 2
        return pop, know, pop - know

    point = stat(cells)
    boots = np.array([stat({k: rng.choice(v, len(v)) for k, v in cells.items()}) for _ in range(B)])
    lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
    return dict(pop=point[0], pop_lo=lo[0], pop_hi=hi[0], know=point[1], know_lo=lo[1], know_hi=hi[1],
                diff=point[2], diff_lo=lo[2], diff_hi=hi[2],
                n={f"{a}/{k}": len(v) for (a, k), v in cells.items()},
                means={f"{a}/{k}": float(v.mean()) for (a, k), v in cells.items()})


def fmt(e, name):
    return (f"{name:<14} popularity {e['pop']:+.3f} [{e['pop_lo']:+.3f}, {e['pop_hi']:+.3f}]   "
            f"knowledge {e['know']:+.3f} [{e['know_lo']:+.3f}, {e['know_hi']:+.3f}]   "
            f"pop - know {e['diff']:+.3f} [{e['diff_lo']:+.3f}, {e['diff_hi']:+.3f}]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {m: load(p) for m, p in POOLS.items()}
    eff_rows, cell_rows = [], []
    verdicts = {}
    for m, d in data.items():
        rng = np.random.default_rng(0)
        k = d[d.slick.isin(["HighlyKnown", "Unknown"])]
        pool_hedge = d.hedge_rate.mean()
        govern = "hedge_rate" if pool_hedge >= FLOOR else "hedge_logodds"
        say("=" * 96)
        say(f"{m.upper()}  ({POOLS[m]}): {len(d)} scored items, {len(k)} HighlyKnown/Unknown; "
            f"pool-level hedge rate {pool_hedge:.3f} -> governing hedging readout: {govern}")
        say("=" * 96)
        res = {}
        for y in ["hedge_rate", "hedge_logodds", "entropy"]:
            if k[y].notna().sum() == 0:
                say(f"{y:<14} (not available)")
                continue
            e = effects(k, y, rng)
            res[y] = e
            say(fmt(e, y))
            say(f"{'':<14} cell means " + "  ".join(f"{c} {v:+.3f} (n={e['n'][c]})" for c, v in e["means"].items()))
            eff_rows.append(dict(model=m, readout=y, **{x: e[x] for x in e if x not in ("n", "means")}))
            for c, v in e["means"].items():
                cell_rows.append(dict(model=m, readout=y, cell=c, mean=v, n=e["n"][c]))
        if govern in res and "entropy" in res:
            h, en = res[govern], res["entropy"]
            c1 = h["diff_lo"] > 0
            c2 = -en["diff_hi"] > 0  # knowledge - popularity > 0 with CI excluding 0  <=>  (pop - know) CI entirely < 0
            fails = h["diff_hi"] < 0
            v = "REPLICATES" if (c1 and c2) else ("FAILS" if fails else "PARTIAL")
            verdicts[m] = v
            say(f"\n  condition 1 (hedging: popularity > knowledge, CI excludes 0): {'yes' if c1 else 'no'}")
            say(f"  condition 2 (entropy: knowledge > popularity, CI excludes 0): {'yes' if c2 else 'no'}")
            say(f"  -> {v}" + ((f"   (criterion applies to {', '.join(x.capitalize() for x in data if x != 'llama')}; Llama shown as the reference)") if m == "llama" else ""))
        if d.hedge_mass.notna().any():
            say(f"  hedge-token probability mass at the prefill: median {d.hedge_mass.median():.4f}, "
                f"90th pct {d.hedge_mass.quantile(.9):.4f}")
        say("")

    pd.DataFrame(eff_rows).to_csv(OUT / "effects.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(OUT / "cells.csv", index=False)

    # ---- cross-model item agreement with Llama (descriptive) ----
    others = [m for m in data if m != "llama"]
    for o in others:
        cross_model(data["llama"], data[o], o, len(others) > 1)
    for o in others:
        say(f"\nVERDICT ({o.capitalize()}, per {CRITERIA.get(o, 'criterion')}): " + verdicts.get(o, "not computed"))
    (OUT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")


def cross_model(L, Q, name, tagged):
    j = L.merge(Q, on=["prop", "subj"], suffixes=("_l", "_q"))
    j = j[j.slick_l.isin(["HighlyKnown", "Unknown"]) & j.slick_q.isin(["HighlyKnown", "Unknown"])].copy()
    j["logpop"] = np.log10(j.s_pop_l.astype(float) + 1)
    j["agree"] = np.where(j.slick_l == j.slick_q, "same knowledge", "knowledge differs")
    j.to_csv(OUT / ("cross_model_items.csv" if name == "qwen" else f"cross_model_items_{name}.csv"), index=False)
    say("=" * 96)
    say(f"CROSS-MODEL item agreement (descriptive): {len(j)} items scored HighlyKnown/Unknown by both models"
        + (f"  [Llama vs {name.capitalize()}]" if tagged else ""))
    say("=" * 96)
    for y in ["hedge_rate", "hedge_logodds", "entropy"]:
        x = j[[f"{y}_l", f"{y}_q", "agree"]].dropna()
        if len(x) < 10:
            continue
        r, p = spearmanr(x[f"{y}_l"], x[f"{y}_q"])
        parts = [f"all rho {r:+.3f} (n={len(x)})"]
        for a, g in x.groupby("agree"):
            if len(g) >= 10:
                ra, _ = spearmanr(g[f"{y}_l"], g[f"{y}_q"])
                parts.append(f"{a} rho {ra:+.3f} (n={len(g)})")
        say(f"  {y:<14} " + "   ".join(parts))
    dis = j[j.agree == "knowledge differs"].copy()
    if len(dis) >= 20:
        import statsmodels.formula.api as smf
        dis["q_known"] = (dis.slick_q == "HighlyKnown").astype(int)
        for y in ["hedge_rate", "hedge_logodds"]:
            if dis[f"{y}_q"].notna().sum() < 20:
                continue
            z = dis.dropna(subset=[f"{y}_q", f"{y}_l"])
            m = smf.ols(f"{y}_q ~ {y}_l + logpop + q_known", z).fit(cov_type="HC3")
            say(f"  knowledge-differs items, {y}: {name.capitalize()} ~ Llama + logpop + {name.capitalize()}-knows  (n={len(z)})  " +
                "  ".join(f"{t} {m.params[t]:+.3f} (p={m.pvalues[t]:.2g})" for t in m.params.index if t != "Intercept"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="name=pool list, e.g. llama=familiarity_v3,gemma=familiarity_v3_gemma (default: llama + qwen)")
    ap.add_argument("--out", default=None, help="results subdirectory (default results/qwen_factorial)")
    a = ap.parse_args()
    if a.models:
        POOLS = dict(x.split("=") for x in a.models.split(","))
    if a.out:
        OUT = REPO / a.out
    main()
