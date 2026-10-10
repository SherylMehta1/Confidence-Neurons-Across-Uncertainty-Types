"""Figure 1: saying "I don't know" follows popularity; answer entropy follows knowledge.

(a) Llama-3.1-8B-Instruct hedge rate in free generation (both judges label DONT_KNOW), 2 x 2 cells.
    Source: results/hedge_measure_v2/item_rates.csv (model == llama, column prim).
(b) Llama first-token entropy after "The answer is", same 2 x 2.
    Source: data/familiarity_v3/all_measured.jsonl (entropy), cells as in scripts/qwen_factorial_analysis.py.
(c) Popularity and knowledge main effects on entropy in four families, with the published 95% CIs.
    Source: results/factorial_4models/effects.csv (readout == entropy).
Cell intervals in (a, b): 95% percentile bootstrap over items within cell (2,000 resamples, seed 0).
"""
import json

import numpy as np
import pandas as pd

from style import KNOW, KNOWN, REPO, TEXTWIDTH, UNKNOWN, panel_label, plt, save

ITEMS = REPO / "results/hedge_measure_v2/item_rates.csv"
POOL = REPO / "data/familiarity_v3/all_measured.jsonl"
EFFECTS = REPO / "results/factorial_4models/effects.csv"
ARMS = [("uncertain", "obscure"), ("control", "popular")]
KN = [("HighlyKnown", "known", KNOWN, "o", "-"), ("Unknown", "unknown", UNKNOWN, "s", "--")]
rng = np.random.default_rng(0)
POP_C = "#CC79A7"  # reddish purple: popularity in (c), distinct from the vermillion "unknown" in (a, b)


def ci(v, B=2000):
    b = rng.choice(v, (B, len(v))).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def cells(d, y):
    out = {}
    for a, _ in ARMS:
        for k, *_ in KN:
            v = d[(d.arm == a) & (d.slick == k)][y].dropna().to_numpy()
            out[a, k] = (*ci(v), len(v))
    return out


def interaction(ax, c, ylabel, ylim):
    x = np.arange(2)
    for k, lab, col, mk, ls in KN:
        m = np.array([c[a, k][0] for a, _ in ARMS])
        lo = np.array([c[a, k][1] for a, _ in ARMS]); hi = np.array([c[a, k][2] for a, _ in ARMS])
        ax.errorbar(x, m, yerr=[m - lo, hi - m], color=col, marker=mk, ls=ls, capsize=2, lw=1.4, ms=4.5, label=lab)
    ax.set_xticks(x, [n for _, n in ARMS], fontsize=8); ax.set_xlim(-0.35, 1.35); ax.set_ylim(*ylim)
    ax.set_xlabel("entity popularity"); ax.set_ylabel(ylabel)


def main():
    plotted = {}
    it = pd.read_csv(ITEMS); it = it[it.model == "llama"]
    ca = cells(it, "prim")
    rows = [json.loads(l) for l in open(POOL, encoding="utf-8") if l.strip()]
    pool = pd.DataFrame([dict(prop=r["prop"], subj=r["subj"], arm=r["arm"], slick=r["slick"], entropy=r["entropy"])
                         for r in rows if r.get("subj")]).drop_duplicates(["prop", "subj"])
    cb = cells(pool, "entropy")
    eff = pd.read_csv(EFFECTS); eff = eff[eff.readout == "entropy"]

    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.52), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    interaction(axs[0], ca, 'share "I don\'t know"', (0, 1))
    interaction(axs[1], cb, "entropy (nats)", (0, 4))
    axs[0].legend(frameon=False, loc="upper right", handlelength=1.8, borderaxespad=0.1)
    panel_label(axs[0], "a"); panel_label(axs[1], "b"); panel_label(axs[2], "c")

    ax = axs[2]
    order = ["llama", "mistral", "qwen", "gemma"]
    names = {"llama": "Llama-3.1-8B-Inst.", "mistral": "Mistral-7B-Inst.", "qwen": "Qwen2.5-7B-Inst.", "gemma": "Gemma-2-9B-it"}
    for i, m in enumerate(order):
        r = eff[eff.model == m].iloc[0]
        y = len(order) - 1 - i
        for val, lo, hi, col, mk, dy in [(r["pop"], r.pop_lo, r.pop_hi, POP_C, "D", 0.13), (r.know, r.know_lo, r.know_hi, KNOW, "^", -0.13)]:
            ax.errorbar(val, y + dy, xerr=[[val - lo], [hi - val]], color=col, marker=mk, ms=4.5, capsize=2, lw=1.4, ls="none")
        plotted[f"c/{m}"] = dict(pop=r["pop"], pop_ci=(r.pop_lo, r.pop_hi), know=r.know, know_ci=(r.know_lo, r.know_hi))
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(range(len(order)), [names[m] for m in order[::-1]]); ax.set_ylim(-0.6, len(order) + 0.35)
    ax.set_xlabel("entropy effect (nats)"); ax.set_xlim(-0.3, 2.45)
    ax.text(2.14, 3.78, "knowledge", color=KNOW, ha="center", fontsize=8)
    ax.text(0.51, 3.78, "popularity", color=POP_C, ha="center", fontsize=8)

    for tag, c in (("a", ca), ("b", cb)):
        for (a, k), (m, lo, hi, n) in c.items():
            plotted[f"{tag}/{a}/{k}"] = dict(mean=m, ci=(lo, hi), n=n)
            print(f"{tag} {a:9s} {k:11s} mean {m:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
    for m in order:
        print("c", m, {k: round(v, 3) if isinstance(v, float) else v for k, v in plotted[f"c/{m}"].items()})
    save(fig, "fig_dissociation", [ITEMS, POOL, EFFECTS], plotted)


if __name__ == "__main__":
    main()
