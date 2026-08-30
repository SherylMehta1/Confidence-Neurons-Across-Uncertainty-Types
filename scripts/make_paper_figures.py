"""All paper figures from the committed result CSVs, one uniform palette.

Outputs paper/figures/fig{1..7}.pdf (+ refreshed figures/fig{4,5}.svg for RESULTS.md).
fig1  position x layer relay (Llama familiarity)
fig2  one-layer MLP patching window, three runs
fig3  faithfulness bars, circuit vs size-matched random sets (30-set null; Qwen 10)
fig4  Jacobian-lens gap trajectories (decision before doubt)
fig5  the verbalization moment + steering dose-response
fig6  rank-1 direction vs full component set, switch-on vs switch-off (Llama contested)
fig7  single-neuron effects vs the 100-random-neuron null (familiarity arm)

Palette (colorblind-safe; Qwen lines additionally dashed):
  BLUE  #3B4FA8  Llama familiarity / circuit / primary
  AMBER #D97706  Llama contested / secondary
  TEAL  #0E7C86  Qwen
  GREY  #6B7280  nulls, controls, references
Run from the repo root: python scripts/make_paper_figures.py
"""
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


# ---- fig1: position x layer relay (Llama familiarity, uncertain -> control)
pm = pd.read_csv("results/circuit_familiarity/position_map.csv.gz")
u2c = pm[pm.direction == "uncertain_to_control"]
tp = pd.read_csv("results/twin_patching_familiarity.csv.gz")
full = tp[tp.direction == "uncertain_to_control"].groupby("layer").logodds_recovery.mean()
fig, ax = plt.subplots(figsize=(4.6, 2.5))
for grp, lab, c in [("entity", "entity tokens", BLUE),
                    ("suffix-1", "token before prefill", AMBER),
                    ("last", "last token", TEAL)]:
    g = u2c[u2c.group == grp].groupby("layer").logodds_rec.mean()
    ax.plot(g.index, g.values, color=c, lw=1.8, label=lab)
ax.plot(full.index, full.values, color=GREY, lw=1.4, ls="--", label="whole last position (195 pairs)")
ax.set_xlabel("layer"); ax.set_ylabel("hedge log-odds recovery")
ax.set_xlim(0, 31); ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.legend(fontsize=7.5, loc="center right")
fig.tight_layout(); save(fig, "fig1")

# ---- fig2: MLP window, three runs
fig, ax = plt.subplots(figsize=(4.6, 2.5))
for path, lab, kw in [
    ("results/circuit_familiarity/mlp_patching.csv", "Llama familiarity", dict(color=BLUE)),
    ("results/circuit_conflict/mlp_patching.csv", "Llama contested", dict(color=AMBER)),
    ("results/second_model/qwen25_7b_instruct/results/circuit_conflict/mlp_patching.csv",
     "Qwen contested", QWEN_STYLE),
]:
    d = pd.read_csv(path)
    g = d[d.positions == "all"].groupby("layer").rec.mean()
    ax.plot(g.index, g.values, lw=1.8, label=lab, **kw)
ax.set_xlabel("layer"); ax.set_ylabel("recovery (MLP output patched)")
ax.set_xlim(0, 31); ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(); save(fig, "fig2")

# ---- fig3: faithfulness bars from the CSVs (30-set null; Qwen 10)
CELLS = [
    ("Llama\nfamiliarity", "results/circuit_familiarity/faithfulness.csv"),
    ("Llama\ncontested", "results/circuit_conflict/faithfulness.csv"),
    ("Qwen\ncontested", "results/second_model/qwen25_7b_instruct/results/circuit_conflict/faithfulness.csv"),
]
runs = []
for label, path in CELLS:
    d = pd.read_csv(path)
    d = d[d.direction == "control_to_uncertain"]
    circ = d[d["set"] == "circuit"].logodds_rec.mean()
    rnd = d[d["set"].str.startswith("random")].groupby("set").logodds_rec.mean().mean()
    runs.append((label, circ, rnd))
fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = range(len(runs))
ax.bar([i - 0.19 for i in x], [r[1] for r in runs], width=0.36, color=BLUE,
       label="circuit (20 heads + 100 neurons)")
ax.bar([i + 0.19 for i in x], [r[2] for r in runs], width=0.36, color=GREY, alpha=0.6,
       label="random sets (mean of 30; Qwen 10)")
for i, (_, c, r) in enumerate(runs):
    ax.text(i - 0.19, c + 0.02, f"{c:.2f}", ha="center", fontsize=8)
    ax.text(i + 0.19, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in runs], fontsize=8)
ax.set_ylabel("held-out log-odds recovery"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); save(fig, "fig3")

# ---- fig4: lens gap trajectories
def gaps(path):
    d = pd.read_csv(path)
    d = d[d.layer >= 0]
    p = d.groupby(["layer", "arm"])[["lens_logodds", "lens_entropy"]].mean().unstack("arm")
    lo = p[("lens_logodds", "uncertain")] - p[("lens_logodds", "control")]
    H = p[("lens_entropy", "uncertain")] - p[("lens_entropy", "control")]
    return lo, H

lens_runs = [("results/circuit_familiarity/jlens_trajectory.csv.gz", "Llama familiarity", dict(color=BLUE)),
             ("results/circuit_conflict/jlens_trajectory.csv.gz", "Llama contested", dict(color=AMBER)),
             ("results/circuit_conflict_qwen/jlens_trajectory.csv.gz", "Qwen contested", QWEN_STYLE)]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
for path, lab, kw in lens_runs:
    lo, H = gaps(path)
    axes[0].plot(lo.index, lo.values / lo.values[-1], lw=1.8, label=lab, **kw)
    axes[1].plot(H.index, H.values / (abs(H.values[-1]) or 1), lw=1.8, label=lab, **kw)
for ax, ttl in zip(axes, ("hedge log-odds gap (fraction of final)", "entropy gap (fraction of |final|)")):
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axhline(0.5, color="0.85", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("layer"); ax.set_title(ttl)
axes[0].axvspan(13, 19, color=BLUE, alpha=0.08)
axes[0].legend(fontsize=7.2, loc="upper left")
fig.tight_layout(); save(fig, "fig4", svg=True)

# ---- fig5: verbalization moment + steering
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
d = pd.read_csv("results/circuit_familiarity/jlens_entity.csv")
med = d.groupby(["arm", "group", "layer"]).rank_unknown.median().reset_index()
for (arm, grp), (c, ls, lw, lab) in {
    ("uncertain", "readout"): (BLUE, "-", 2.2, "uncertain · readout position"),
    ("control", "readout"): (GREY, "-", 1.6, "control · readout"),
    ("uncertain", "entity"): (AMBER, ":", 1.4, "uncertain · entity span"),
    ("control", "entity"): (TEAL, ":", 1.4, "control · entity span"),
}.items():
    g = med[(med.arm == arm) & (med.group == grp)].sort_values("layer")
    axes[0].plot(g.layer, g.rank_unknown, color=c, ls=ls, lw=lw, label=lab)
axes[0].set_yscale("log"); axes[0].set_ylim(1, 2e5)
axes[0].axvspan(13, 19, color=BLUE, alpha=0.08)
axes[0].set_xlabel("layer"); axes[0].set_ylabel("lens rank of ' unknown' (median)")
axes[0].set_title("the verbalization moment (Llama familiarity)")
axes[0].legend(fontsize=7)
s = pd.read_csv("results/circuit_familiarity/jlens_steer.csv")
for a, c in ((-3.0, GREY), (0.0, AMBER), (3.0, BLUE)):
    g = s[s.alpha == a].groupby("layer")["rankunknown"].median()
    axes[1].plot(g.index, g.values, color=c, lw=1.8, label=f"α = {a:+g}σ")
axes[1].set_yscale("log"); axes[1].set_ylim(1, 2e5)
axes[1].set_xlabel("layer (downstream of the L15 push)")
axes[1].set_ylabel("lens rank of ' unknown' (median)")
axes[1].set_title("steering the direction on known twins")
axes[1].legend(fontsize=7.5)
fig.tight_layout(); save(fig, "fig5", svg=True)

# ---- fig6: direction vs set, switch-on vs switch-off (Llama contested, held-out)
dp = pd.read_csv("results/circuit_conflict/direction_patch.csv")
fa = pd.read_csv("results/circuit_conflict/faithfulness.csv")
dir_on = dp[(dp["set"] == "direction") & (dp.direction == "uncertain_to_control")].logodds_rec.mean()
dir_off = dp[(dp["set"] == "direction") & (dp.direction == "control_to_uncertain")].logodds_rec.mean()
set_on = fa[(fa["set"] == "circuit") & (fa.direction == "uncertain_to_control")].logodds_rec.mean()
set_off = fa[(fa["set"] == "circuit") & (fa.direction == "control_to_uncertain")].logodds_rec.mean()
fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = np.arange(2)
ax.bar(x - 0.19, [dir_on, dir_off], width=0.36, color=AMBER, label="rank-1 direction")
ax.bar(x + 0.19, [set_on, set_off], width=0.36, color=BLUE, label="full set (20 heads + 100 neurons)")
for xi, v in zip([x[0] - 0.19, x[1] - 0.19, x[0] + 0.19, x[1] + 0.19],
                 [dir_on, dir_off, set_on, set_off]):
    ax.text(xi, v + 0.03 if v >= 0 else v - 0.09, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["switch hedge ON\n(inject unknown)", "switch hedge OFF\n(remove unknown)"], fontsize=8)
ax.set_ylabel("hedge log-odds recovery")
ax.set_ylim(-0.3, 1.15); ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.legend(fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); save(fig, "fig6")
print(f"  fig6 values: dir on/off {dir_on:.3f}/{dir_off:.3f}, set on/off {set_on:.3f}/{set_off:.3f}")

# ---- fig7: single-neuron effects vs the identified random-neuron null (familiarity)
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
fig, ax = plt.subplots(figsize=(4.6, 2.3))
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
print(f"  fig7: null max {null_v.max():.4f}, candidates beating all nulls: {int(beat_all.sum())}, largest candidate {cand_v.max():.4f}")
