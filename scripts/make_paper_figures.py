"""All paper figures from the committed result CSVs, one uniform palette, WITH
per-pair 95% confidence bands/whiskers (camera-ready revision).

Outputs paper/figures/fig{1..8}.pdf (+ refreshed figures/fig{4,5}.svg for RESULTS.md).
fig1  position x layer relay (Llama familiarity) + CI bands
fig2  one-layer MLP patching window, three runs + CI bands
fig3  faithfulness bars + 95% CI whiskers (30-set null; Qwen 10)
fig4  Jacobian-lens gap trajectories, BOTH panels normalized by the SIGNED final value,
      with CI bands (the sign-reversing contested excursions stay visible)
fig5  verbalization moment (median + IQR band) + steering dose-response
fig6  rank-1 direction vs random vs projection-matched vs full set, with whiskers
fig7  single-neuron effects vs the identified random-neuron null
fig8  LOO-pruning curves, three cells (the previously unplotted minimality claim)

Palette: BLUE #3B4FA8 (Llama familiarity/primary), AMBER #D97706 (Llama contested),
TEAL #0E7C86 dashed (Qwen), GREY #6B7280 (nulls/refs). Run from the repo root.
"""
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "axes.titlesize": 9, "legend.frameon": False,
})
BLUE, AMBER, TEAL, GREY = "#3B4FA8", "#D97706", "#0E7C86", "#6B7280"
QWEN_STYLE = dict(color=TEAL, ls="--")

os.makedirs("paper/figures", exist_ok=True)


def save(fig, name, svg=False):
    fig.savefig(f"paper/figures/{name}.pdf")
    if svg:
        buf = io.StringIO()
        fig.savefig(buf, format="svg")
        s = buf.getvalue()
        open(f"figures/{name}.svg", "w", encoding="utf-8").write(s[s.index("<svg"):])
    plt.close(fig)
    print(name, "written")


def mean_ci(df, layercol, valcol):
    g = df.groupby(layercol)[valcol]
    m, sd, n = g.mean(), g.std(ddof=1), g.count()
    half = 1.96 * sd / np.sqrt(n)
    return m, half


def ci95(v):
    v = np.asarray(v, dtype=float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


# ---- fig1: position x layer relay (Llama familiarity, uncertain -> control) + bands
pm = pd.read_csv("results/circuit_familiarity/position_map.csv.gz")
u2c = pm[pm.direction == "uncertain_to_control"]
tp = pd.read_csv("results/twin_patching_familiarity.csv.gz")
tpu = tp[tp.direction == "uncertain_to_control"]
fig, ax = plt.subplots(figsize=(5.2, 2.7))
for grp, lab, c in [("entity", "entity tokens", BLUE),
                    ("suffix-1", "token before prefill", AMBER),
                    ("last", "last token", TEAL)]:
    d = u2c[u2c.group == grp]
    m, half = mean_ci(d, "layer", "logodds_rec")
    ax.plot(m.index, m.values, color=c, lw=1.8, label=lab)
    ax.fill_between(m.index, m - half, m + half, color=c, alpha=0.15, lw=0)
m, half = mean_ci(tpu, "layer", "logodds_recovery")
ax.plot(m.index, m.values, color=GREY, lw=1.4, ls="--", label="whole last position (195 pairs)")
ax.fill_between(m.index, m - half, m + half, color=GREY, alpha=0.15, lw=0)
ax.set_xlabel("layer"); ax.set_ylabel("hedge log-odds recovery")
ax.set_xlim(0, 31); ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.legend(fontsize=7.5, loc="center right")
fig.tight_layout(); save(fig, "fig1")

# ---- fig2: MLP window, three runs + bands
fig, ax = plt.subplots(figsize=(5.2, 2.7))
for path, lab, kw in [
    ("results/circuit_familiarity/mlp_patching.csv", "Llama familiarity", dict(color=BLUE)),
    ("results/circuit_conflict/mlp_patching.csv", "Llama contested", dict(color=AMBER)),
    ("results/second_model/qwen25_7b_instruct/results/circuit_conflict/mlp_patching.csv",
     "Qwen contested", QWEN_STYLE),
]:
    d = pd.read_csv(path)
    d = d[d.positions == "all"]
    m, half = mean_ci(d, "layer", "rec")
    ax.plot(m.index, m.values, lw=1.8, label=lab, **kw)
    ax.fill_between(m.index, m - half, m + half, color=kw["color"], alpha=0.15, lw=0)
ax.set_xlabel("layer"); ax.set_ylabel("recovery (MLP output patched)")
ax.set_xlim(0, 31); ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); save(fig, "fig2")

# ---- fig3: faithfulness bars + whiskers (30-set null; Qwen 10)
CELLS = [
    ("Llama\nfamiliarity", "results/circuit_familiarity/faithfulness.csv"),
    ("Llama\ncontested", "results/circuit_conflict/faithfulness.csv"),
    ("Qwen\ncontested", "results/second_model/qwen25_7b_instruct/results/circuit_conflict/faithfulness.csv"),
]
runs = []
for label, path in CELLS:
    d = pd.read_csv(path)
    d = d[d.direction == "control_to_uncertain"]
    circ = d[d["set"] == "circuit"].logodds_rec
    rmeans = d[d["set"].str.startswith("random")].groupby("set").logodds_rec.mean()
    runs.append((label, circ.mean(), ci95(circ), rmeans.mean(), 1.96 * rmeans.std(ddof=1)))
fig, ax = plt.subplots(figsize=(3.6, 2.6))
x = np.arange(len(runs))
ax.bar(x - 0.19, [r[1] for r in runs], width=0.36, color=BLUE,
       yerr=[r[2] for r in runs], capsize=3, error_kw=dict(lw=1),
       label="circuit (20 heads + 100 neurons), 95% CI")
ax.bar(x + 0.19, [r[3] for r in runs], width=0.36, color=GREY, alpha=0.6,
       yerr=[r[4] for r in runs], capsize=3, error_kw=dict(lw=1),
       label="random sets (mean of 30; Qwen 10)")
for i, r in enumerate(runs):
    ax.text(i - 0.19, r[1] + r[2] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=8)
    ax.text(i + 0.19, r[3] + r[4] + 0.02, f"{r[3]:.2f}", ha="center", fontsize=8)
ax.axhline(0.7, color=GREY, lw=0.9, ls=":", zorder=0)
ax.text(2.35, 0.71, "0.7 criterion", fontsize=6.5, color=GREY, ha="right")
ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in runs], fontsize=8)
ax.set_ylabel("held-out log-odds recovery"); ax.set_ylim(0, 1.12)
ax.legend(fontsize=7.0, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); save(fig, "fig3")

# ---- fig4: lens gap trajectories, SAME signed-final denominator both panels, CI bands
def gap_ci(path, col):
    d = pd.read_csv(path)
    d = d[d.layer >= 0]
    w = d.pivot_table(index=["pair", "layer"], columns="arm", values=col).reset_index()
    w["gap"] = w["uncertain"] - w["control"]
    m, half = mean_ci(w, "layer", "gap")
    fin = m.iloc[-1]
    return m / fin, half / abs(fin)

lens_runs = [("results/circuit_familiarity/jlens_trajectory.csv.gz", "Llama familiarity", dict(color=BLUE)),
             ("results/circuit_conflict/jlens_trajectory.csv.gz", "Llama contested", dict(color=AMBER)),
             ("results/circuit_conflict_qwen/jlens_trajectory.csv.gz", "Qwen contested", QWEN_STYLE)]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
for path, lab, kw in lens_runs:
    for ax, col in ((axes[0], "lens_logodds"), (axes[1], "lens_entropy")):
        m, half = gap_ci(path, col)
        ax.plot(m.index, m.values, lw=1.8, label=lab, **kw)
        ax.fill_between(m.index, m - half, m + half, color=kw["color"], alpha=0.15, lw=0)
for ax, ttl in zip(axes, ("hedge log-odds gap (fraction of final)", "entropy gap (fraction of final)")):
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axhline(0.5, color="0.85", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("layer"); ax.set_title(ttl)
axes[0].axvspan(13, 19, color=BLUE, alpha=0.08)
axes[0].legend(fontsize=7.2, loc="upper left")
fig.tight_layout(); save(fig, "fig4", svg=True)

# ---- fig5: verbalization moment (median + IQR band) + steering
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
d = pd.read_csv("results/circuit_familiarity/jlens_entity.csv")
for (arm, grp), (c, ls, lw, lab, band) in {
    ("uncertain", "readout"): (BLUE, "-", 2.2, "uncertain · readout position", True),
    ("control", "readout"): (GREY, "-", 1.6, "control · readout", True),
    ("uncertain", "entity"): (AMBER, ":", 1.4, "uncertain · entity span", False),
    ("control", "entity"): (TEAL, ":", 1.4, "control · entity span", False),
}.items():
    g = d[(d.arm == arm) & (d.group == grp)].groupby("layer").rank_unknown
    med = g.median()
    axes[0].plot(med.index, med.values, color=c, ls=ls, lw=lw, label=lab)
    if band:
        q1, q3 = g.quantile(0.25), g.quantile(0.75)
        axes[0].fill_between(med.index, q1, q3, color=c, alpha=0.12, lw=0)
axes[0].set_yscale("log"); axes[0].set_ylim(1, 2e5)
axes[0].axvspan(13, 19, color=BLUE, alpha=0.08)
axes[0].set_xlabel("layer"); axes[0].set_ylabel("lens rank of ' unknown' (median, IQR)")
axes[0].set_title("the verbalization moment (Llama familiarity)")
axes[0].legend(fontsize=7)
s = pd.read_csv("results/circuit_familiarity/jlens_steer.csv")
for a, c in ((-3.0, GREY), (0.0, AMBER), (3.0, BLUE)):
    g = s[s.alpha == a].groupby("layer")["rankunknown"].median()
    axes[1].plot(g.index, g.values, color=c, lw=1.8, label=f"α = {a:+g}σ")
axes[1].set_yscale("log"); axes[1].set_ylim(1, 2e5)
axes[1].set_xlabel("layer (downstream of the L15 push)")
axes[1].set_ylabel("lens rank of ' unknown' (median)")
axes[1].set_title("steering the direction on known twins (Llama)")
axes[1].legend(fontsize=7.5)
fig.tight_layout(); save(fig, "fig5", svg=True)

# ---- fig6: direction vs random vs projection-matched vs set, with whiskers
dp = pd.read_csv("results/circuit_conflict/direction_patch_u2c_null.csv")
hm = pd.read_csv("results/circuit_conflict/direction_patch_u2c_hedgematched.csv")
fa = pd.read_csv("results/circuit_conflict/faithfulness.csv")

def bar(df, setname, direc):
    v = df[(df["set"] == setname) & (df.direction == direc)]["logodds_rec"]
    return v.mean(), ci95(v)

def rndbar(df, direc):
    m = df[df["set"].str.startswith("random") & (df.direction == direc)].groupby("set").logodds_rec.mean()
    return m.mean(), 1.96 * m.std(ddof=1)

on = [bar(dp, "direction", "uncertain_to_control"), rndbar(dp, "uncertain_to_control"),
      rndbar(hm, "uncertain_to_control"), bar(fa, "circuit", "uncertain_to_control")]
off = [bar(dp, "direction", "control_to_uncertain"), rndbar(dp, "control_to_uncertain"),
       (np.nan, 0), bar(fa, "circuit", "control_to_uncertain")]
labels = ["rank-1 direction", "random unit directions",
          "projection-matched\nrandom directions", "full set (20h+100n)"]
colors = [AMBER, GREY, "#9A6FB0", BLUE]
fig, ax = plt.subplots(figsize=(4.2, 2.6))
w = 0.2
for j, (lab, col) in enumerate(zip(labels, colors)):
    xs, ms, es = [], [], []
    for gi, vals in enumerate((on, off)):
        m, e = vals[j]
        if not np.isnan(m):
            xs.append(gi + (j - 1.5) * w); ms.append(m); es.append(e)
    ax.bar(xs, ms, width=w, color=col, alpha=0.6 if col == GREY else 1.0,
           yerr=es, capsize=2.5, error_kw=dict(lw=0.9), label=lab)
    for xi, mi, ei in zip(xs, ms, es):
        ax.text(xi, mi + ei + 0.03 if mi >= 0 else mi - ei - 0.10, f"{mi:.2f}",
                ha="center", fontsize=6.8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["switch hedge ON\n(inject unknown)", "switch hedge OFF\n(remove unknown)"], fontsize=8)
ax.set_ylabel("hedge log-odds recovery")
ax.set_ylim(-0.35, 1.25); ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); save(fig, "fig6")

# ---- fig7: single-neuron effects vs identified null (unchanged content)
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from gated_specificity_test import paired_interaction  # noqa: E402

run_dir = "results/ablation_fam_ctrl100"
prov = json.load(open(f"{run_dir}/results_familiarity.provenance.json", encoding="utf-8"))
ctrl = set(map(tuple, prov.get("control_neurons") or [])) if isinstance(
    (prov.get("control_neurons") or [None])[0], list) else set(prov.get("control_neurons") or [])
raw = pd.read_csv(f"{run_dir}/results_familiarity.csv.gz")
stats = paired_interaction(raw)
is_ctrl = stats.index.isin(ctrl)
null_v = np.abs(stats[is_ctrl].inter.values)
cand = pd.read_csv(f"{run_dir}/specificity_random_null.csv")
cand_v = np.abs(cand.inter.values)
beat_all = cand_v > null_v.max()
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(5.2, 2.5))
ax.scatter(np.clip(null_v, 1e-5, None), rng.uniform(-0.25, 0.25, len(null_v)),
           s=12, color=GREY, alpha=0.55, label=f"{len(null_v)} random neurons (identified null)")
ax.scatter(np.clip(cand_v[~beat_all], 1e-5, None), rng.uniform(0.6, 1.1, int((~beat_all).sum())),
           s=16, color=AMBER, alpha=0.8, label="candidates")
ax.scatter(np.clip(cand_v[beat_all], 1e-5, None), rng.uniform(0.6, 1.1, int(beat_all.sum())),
           s=26, color=BLUE, label="candidates beating every null")
ax.axvline(2.8, color=GREY, lw=1.2, ls="--")
ax.text(2.8, 1.45, "twins' 2.8-nat\nentropy gap", ha="center", fontsize=7.5, color=GREY)
ax.set_xscale("log"); ax.set_xlim(1e-5, 8); ax.set_ylim(-0.5, 1.9)
ax.set_yticks([]); ax.set_xlabel("|uncertain-vs-control interaction| (nats, log scale)")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); save(fig, "fig7")

# ---- fig8: LOO-pruning curves (the previously unplotted minimality claim)
fig, ax = plt.subplots(figsize=(4.2, 2.5))
for path, lab, kw in [
    ("results/circuit_familiarity/prune_curve.csv", "Llama familiarity", dict(color=BLUE)),
    ("results/circuit_conflict/prune_curve.csv", "Llama contested", dict(color=AMBER)),
    ("results/second_model/qwen25_7b_instruct/results/circuit_conflict/prune_curve.csv",
     "Qwen contested", QWEN_STYLE),
]:
    d = pd.read_csv(path).sort_values("k")
    ax.plot(d.k, d.recovery, marker="o", ms=3.5, lw=1.6, label=lab, **kw)
ax.axhline(0.7, color=GREY, lw=0.9, ls=":", zorder=0)
ax.text(3, 0.72, "0.7 criterion", fontsize=6.5, color=GREY)
ax.set_xlabel("components kept (LOO-ranked)"); ax.set_ylabel("held-out log-odds recovery")
ax.legend(fontsize=7.2, loc="lower right")
fig.tight_layout(); save(fig, "fig8")
