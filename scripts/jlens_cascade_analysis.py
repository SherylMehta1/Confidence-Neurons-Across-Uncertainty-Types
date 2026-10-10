"""
Read-off for jlens_cascade_criterion.md: familiarity vs knowledge onset of the lens-decoded hedge readout, per model,
and the cross-model claims. CPU only.

Inputs  results/jlens_cascade/<model>/{labels.jsonl, lens.csv}, data/familiarity_v3/all_measured.jsonl (arms)
Outputs results/jlens_cascade/{summary.txt, curves.csv, outcomes.csv, fig_cascade.pdf/.png}
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np
import pandas as pd

REPO = _Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "jlens_cascade"
MODELS = ["llama3.1-8b-it", "llama3.1-8b", "qwen2.5-7b-it", "gemma-2-9b-it", "gemma-2-9b", "gemma-2-2b-it", "gemma-2-2b", "olmo-3-1025-7b"]
BASE = ["llama3.1-8b", "gemma-2-9b", "gemma-2-2b", "olmo-3-1025-7b"]
CELLS = [("uncertain", "Unknown"), ("uncertain", "HighlyKnown"), ("control", "Unknown"), ("control", "HighlyKnown")]
B = 2000
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def curves_from_means(m):
    """m: dict cell -> per-layer mean vector. Returns F, K vectors."""
    F = ((m["uncertain", "Unknown"] - m["control", "Unknown"]) + (m["uncertain", "HighlyKnown"] - m["control", "HighlyKnown"])) / 2
    K = ((m["uncertain", "Unknown"] - m["uncertain", "HighlyKnown"]) + (m["control", "Unknown"] - m["control", "HighlyKnown"])) / 2
    return F, K


def onset_simple(c):
    mx = np.nanmax(c)
    if not np.isfinite(mx) or mx <= 0:
        return None
    idx = np.where(c >= 0.5 * mx)[0]
    return int(idx[0]) if len(idx) else None


def onset_with_ci(c, lo):
    mx = np.nanmax(c)
    if not np.isfinite(mx) or mx <= 0:
        return None
    idx = np.where((c >= 0.5 * mx) & (lo > 0))[0]
    return int(idx[0]) if len(idx) else None


def analyse(model, arms, rng, measure="logodds"):
    d = ROOT / model
    lab = {x["key"]: x["slick"] for x in map(json.loads, open(d / "labels.jsonl", encoding="utf-8")) if x}
    lens = pd.read_csv(d / "lens.csv")
    lens = lens[lens.layer >= 0]
    Y = lens.pivot_table(index="key", columns="layer", values=measure)
    layers = np.array(sorted(Y.columns))
    Y = Y[layers]
    keys = [k for k in Y.index if lab.get(k) in ("Unknown", "HighlyKnown") and k in arms]
    Y = Y.loc[keys]
    cell_rows = {c: np.where([(arms[k], lab[k]) == c for k in keys])[0] for c in CELLS}
    A = Y.to_numpy()
    n = {c: len(ix) for c, ix in cell_rows.items()}
    if min(n.values()) < 5:
        return dict(model=model, error=f"cell too small: {n}")
    F, K = curves_from_means({c: A[ix].mean(0) for c, ix in cell_rows.items()})
    bF, bK, bD, bW = [], [], [], []
    rel = layers / (layers.max() + 1)
    win = (rel >= 1 / 3) & (rel <= 2 / 3)
    for _ in range(B):
        m = {c: A[rng.choice(ix, len(ix))].mean(0) for c, ix in cell_rows.items()}
        f, k = curves_from_means(m)
        bF.append(f); bK.append(k); bW.append(f[win].mean())
        of, ok = onset_simple(f), onset_simple(k)
        bD.append(np.nan if (of is None or ok is None) else layers[ok] - layers[of])
    bF, bK, bD, bW = map(np.array, (bF, bK, bD, bW))
    Flo, Fhi = np.percentile(bF, [2.5, 97.5], 0)
    Klo, Khi = np.percentile(bK, [2.5, 97.5], 0)
    oF, oK = onset_with_ci(F, Flo), onset_with_ci(K, Klo)
    Dd = bD[np.isfinite(bD)]
    Dlo, Dhi = (np.percentile(Dd, [2.5, 97.5]) if len(Dd) else (np.nan, np.nan))
    if oF is not None and oK is None:
        outcome = "SUPPORTS"
    elif oK is not None and oF is None:
        outcome = "CONTRADICTS"
    elif oF is None and oK is None:
        outcome = "UNDECIDED"
    elif Dlo > 0:
        outcome = "SUPPORTS"
    elif Dhi < 0:
        outcome = "CONTRADICTS"
    else:
        outcome = "UNDECIDED"
    curves = pd.DataFrame(dict(model=model, measure=measure, layer=layers, rel_depth=rel, F=F, F_lo=Flo, F_hi=Fhi, K=K, K_lo=Klo, K_hi=Khi))
    return dict(model=model, n=n, n_layers=int(layers.max() + 2), onset_F=None if oF is None else int(layers[oF]),
                onset_K=None if oK is None else int(layers[oK]), rel_F=None if oF is None else float(rel[oF]),
                rel_K=None if oK is None else float(rel[oK]), D=None if (oF is None or oK is None) else int(layers[oK] - layers[oF]),
                D_lo=float(Dlo), D_hi=float(Dhi), D_defined=float(np.isfinite(bD).mean()), outcome=outcome,
                midF=float(F[win].mean()), midF_lo=float(np.percentile(bW, 2.5)), midF_hi=float(np.percentile(bW, 97.5)), curves=curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if (ROOT / m / "lens.csv").exists()]
    items = [json.loads(l) for l in open(REPO / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8") if l.strip()]
    arms = {f"{r['prop']}::{r['subj']}": r["arm"] for r in items if r.get("subj")}
    rng = np.random.default_rng(0)
    res, allc = [], []
    say("=" * 100)
    say("J-LENS FAMILIARITY-vs-KNOWLEDGE ONSET (jlens_cascade_criterion.md); readout: lens hedge-vs-answer log-odds")
    say("=" * 100)
    for m in models:
        r = analyse(m, arms, rng)
        if "error" in r:
            say(f"{m:<16} NOT ANALYSED: {r['error']}"); continue
        bos = (ROOT / m / "bos_check.txt").read_text().strip() if (ROOT / m / "bos_check.txt").exists() else "?"
        res.append(r); allc.append(r.pop("curves"))
        say(f"{m:<16} layers {r['n_layers']:>2}  cells {r['n']}  bos-check {bos}")
        say(f"{'':<16} familiarity onset L{r['onset_F']} (rel {r['rel_F'] if r['rel_F'] is None else round(r['rel_F'], 2)})   "
            f"knowledge onset L{r['onset_K']} (rel {r['rel_K'] if r['rel_K'] is None else round(r['rel_K'], 2)})   "
            f"D = K - F = {r['D']} layers, 95% CI [{r['D_lo']:+.1f}, {r['D_hi']:+.1f}] (defined in {r['D_defined']:.0%} of replicates)")
        say(f"{'':<16} mid-depth familiarity F [1/3,2/3] = {r['midF']:+.3f} [{r['midF_lo']:+.3f}, {r['midF_hi']:+.3f}]   -> {r['outcome']}")
    out = pd.DataFrame([{k: v for k, v in r.items() if k != "n"} | {"cells": json.dumps({f'{a}/{b}': x for (a, b), x in r['n'].items()})} for r in res])
    out.to_csv(ROOT / "outcomes.csv", index=False)
    if allc:
        pd.concat(allc).to_csv(ROOT / "curves.csv", index=False)
    sup = sum(r["outcome"] == "SUPPORTS" for r in res); con = sum(r["outcome"] == "CONTRADICTS" for r in res)
    say("")
    say(f"Models analysed: {len(res)} of {len(MODELS)} planned; SUPPORTS {sup}, CONTRADICTS {con}, UNDECIDED {len(res) - sup - con}")
    if len(res) < len(MODELS):
        say("  (claims below are evaluated on the planned 8; models not run count as not supporting)")
    claim = ("familiarity decoded before knowledge across models" if (sup >= 5 and con <= 1) else
             ("order differs across models" if con >= 2 else "mixed"))
    say(f"CROSS-MODEL CLAIM: {claim}")
    base = [r for r in res if r["model"] in BASE]
    pre = len(base) == len(BASE) and all(r["midF_lo"] > 0 for r in base)
    say(f"PREDATES INSTRUCTION TUNING (all 4 base models mid-depth F > 0, CI excl. 0): {'yes' if pre else 'no'}  "
        + "; ".join(f"{r['model']} {r['midF']:+.2f} [{r['midF_lo']:+.2f},{r['midF_hi']:+.2f}]" for r in base))
    (ROOT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")
    if allc:
        plot(pd.concat(allc), res)


def plot(c, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [r["model"] for r in res]
    ncol = 4; nrow = int(np.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), sharex=True, squeeze=False)
    for ax, r in zip(axes.flat, res):
        d = c[c.model == r["model"]]
        for col, lab, colr in [("F", "familiarity (obscure - popular)", "#1f77b4"), ("K", "knowledge (unknown - known)", "#d62728")]:
            ax.plot(d.rel_depth, d[col], color=colr, lw=1.6, label=lab)
            ax.fill_between(d.rel_depth, d[col + "_lo"], d[col + "_hi"], color=colr, alpha=0.18, lw=0)
            on = r["rel_F"] if col == "F" else r["rel_K"]
            if on is not None:
                ax.axvline(on, color=colr, ls=":", lw=1)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"{r['model']}  [{r['outcome']}]", fontsize=8)
    for ax in axes.flat[len(res):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("relative depth", fontsize=8)
    for row in axes:
        row[0].set_ylabel("lens hedge log-odds gap", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_cascade.pdf"); fig.savefig(ROOT / "fig_cascade.png", dpi=160)


if __name__ == "__main__":
    main()
