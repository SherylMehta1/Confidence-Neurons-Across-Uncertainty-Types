"""Figure 2 (round 3): steering with the dose chosen on held-out questions, scored by two LLM judges.

PREREG e4_steering_v3.md (sha 18c74883). 150 EVAL popular, HighlyKnown questions per model, disjoint from the
questions used to fit directions and to choose the dose; 3 free replies each (96 tokens); vector added at layer l*.
(a) abstention: share of replies both judges label DONT_KNOW, per condition at the selected dose.
(b) first-token entropy at the forced-answer prompt, same conditions.
Sources: results/round3/e4/{replies_llama,replies_gemma9b}.jsonl, judge_{qwen32b,gemma27b}.jsonl, kstar_*.json.
Intervals: 95% percentile bootstrap over questions (2,000 resamples, seed 0). Grey dots: the 5 random directions.
"""
import json

import numpy as np
import pandas as pd

from style import FAM, KNOW, NOHEDGE, NONE, RAND, REPO, TEXTWIDTH, panel_label, plt, save

E4 = REPO / "results/round3/e4"
MODELS = ["llama", "gemma9b"]
rng = np.random.default_rng(0)
ABST = "#56B4E9"  # sky blue: abstention direction


def boot(v, B=2000):
    v = np.asarray(v, float)
    b = rng.choice(v, (B, len(v))).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def load():
    jl = {}
    for jt in ("qwen32b", "gemma27b"):
        for l in open(E4 / f"judge_{jt}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            jl.setdefault((x["model"], x["cond"], x["key"], x["s"]), []).append(x["label"])
    out = {}
    for m in MODELS:
        R = pd.DataFrame([json.loads(l) for l in open(E4 / f"replies_{m}.jsonl", encoding="utf-8")])
        R = R[R.split == "eval"].copy()
        R["abstain"] = [float(jl.get((m, c, k, s), []) == ["DONT_KNOW", "DONT_KNOW"]) for c, k, s in zip(R.cond, R.key, R.s)]
        out[m] = R.groupby(["cond", "key"]).agg(abstain=("abstain", "mean"), entropy=("entropy", "first")).reset_index()
    return out


def main():
    data = load()
    ks = {m: json.load(open(E4 / f"kstar_{m}.json"))["kstar"] for m in MODELS}
    names = {"llama": f"Llama-3.1-8B-Instruct\n(dose {ks['llama']:g}×)", "gemma9b": f"Gemma-2-9B-it\n(dose {ks['gemma9b']:g}×)"}
    conds = [("none", "no steering", NONE, ""), ("random", "5 random directions", RAND, ""),
             ("know", "unknown−known", KNOW, ""), ("abst", "abstain−answer", ABST, ""),
             ("fam", "obscure−popular", FAM, ""), ("fam_perp", "obscure−popular, abstention part removed", NOHEDGE, "////")]
    plotted = {}
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.5))
    w = 0.135
    for pi, (ax, y, ylab) in enumerate([(axs[0], "abstain", 'share "I don\'t know"'), (axs[1], "entropy", "entropy (nats)")]):
        for gi, m in enumerate(MODELS):
            d = data[m]
            for ci, (c, lab, col, hatch) in enumerate(conds):
                x = gi + (ci - 2.5) * w
                if c == "random":
                    per = [d[d.cond == f"random{j}"][y].mean() for j in range(5)]
                    mean, lo, hi = boot(d[d.cond.str.startswith("random")].groupby("key")[y].mean())
                    ax.scatter([x] * 5, per, s=4, color="0.25", zorder=3, lw=0)
                else:
                    mean, lo, hi = boot(d[d.cond == c][y])
                ax.bar(x, mean, w * 0.92, color=col if hatch == "" else "white", edgecolor=col, hatch=hatch, lw=1.0,
                       label=lab if (gi == 0 and pi == 0) else None)
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color="0.2", capsize=1.5, lw=0.8)
                plotted[f"{'ab'[pi]}/{m}/{c}"] = dict(mean=mean, ci=(lo, hi))
                print(f"{'ab'[pi]} {m:8s} {c:9s} {mean:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.set_xticks(range(len(MODELS)), [names[m] for m in MODELS], fontsize=8)
        ax.set_ylabel(ylab)
        panel_label(ax, "ab"[pi])
    axs[0].set_ylim(0, 0.25)
    fig.legend(loc="outside upper center", ncol=3, frameon=False, fontsize=7.2, handlelength=1.4, columnspacing=1.0)
    save(fig, "fig_steering", [E4 / "replies_llama.jsonl", E4 / "replies_gemma9b.jsonl", E4 / "judge_qwen32b.jsonl", E4 / "judge_gemma27b.jsonl"], plotted)


if __name__ == "__main__":
    main()
