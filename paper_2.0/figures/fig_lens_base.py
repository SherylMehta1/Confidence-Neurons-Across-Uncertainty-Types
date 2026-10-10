"""Figure 3 (round 3): in base models the decoded hedge readout responds to popularity, and about as much to knowledge.

PREREG e3_lens_split.md (sha b2b916ce). Jacobian-lens readout at the end of "Question: ... Answer: The answer is",
averaged over the pre-set mid-depth window (relative depth 1/3 to 2/3), for four base models.
Rows: the full hedge-vs-answer log-odds, and its hedge-token term alone (logsumexp of hedge-token log-probabilities).
Points: popularity effect (obscure minus popular) and knowledge effect (unknown minus known), each averaged over the
other factor; bars: 95% within-cell item-bootstrap intervals (2,000 resamples, seed 0).
Source: results/round3/e3_window_effects.csv (scripts/round3_analysis.py e3 on results/round3/e3/<model>/lens.csv).
"""
import pandas as pd

from style import FAM, KNOW, REPO, TEXTWIDTH, panel_label, plt, save

SRC = REPO / "results/round3/e3_window_effects.csv"
MODELS = [("llama3.1-8b", "Llama-3.1-8B"), ("gemma-2-9b", "Gemma-2-9B"), ("gemma-2-2b", "Gemma-2-2B"), ("olmo-3-1025-7b", "OLMo-3-7B")]


def main():
    d = pd.read_csv(SRC)
    plotted = {}
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.65), sharey=True)
    for pi, (ax, term, title) in enumerate([(axs[0], "logodds", "full readout (hedge − answer)"), (axs[1], "hedge_term", "hedge-token term alone")]):
        ax.axvline(0, color="0.6", lw=0.8)
        for i, (m, name) in enumerate(MODELS):
            r = d[(d.model == m) & (d.term == term)].iloc[0]
            y = len(MODELS) - 1 - i
            for val, lo, hi, col, mk, dy in [(r["pop"], r.pop_lo, r.pop_hi, FAM, "D", 0.13), (r.know, r.know_lo, r.know_hi, KNOW, "^", -0.13)]:
                ax.errorbar(val, y + dy, xerr=[[val - lo], [hi - val]], color=col, marker=mk, ms=4, capsize=2, lw=1.2, ls="none")
            plotted[f"{'ab'[pi]}/{m}"] = dict(pop=(r["pop"], r.pop_lo, r.pop_hi), know=(r.know, r.know_lo, r.know_hi))
            print(f"{'ab'[pi]} {term:10s} {m:16s} pop {r['pop']:+.2f} [{r.pop_lo:+.2f}, {r.pop_hi:+.2f}]  know {r.know:+.2f} [{r.know_lo:+.2f}, {r.know_hi:+.2f}]")
        ax.set_yticks(range(len(MODELS)), [n for _, n in MODELS[::-1]])
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.set_xlabel(f"effect on {title}\n(log-odds, mid-depth window)", fontsize=8)
        panel_label(ax, "ab"[pi])
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=FAM, marker="D", ls="none", ms=4), Line2D([], [], color=KNOW, marker="^", ls="none", ms=4)],
               ["popularity effect (obscure − popular)", "knowledge effect (unknown − known)"],
               loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)
    save(fig, "fig_lens_base", [SRC], plotted)


if __name__ == "__main__":
    main()
