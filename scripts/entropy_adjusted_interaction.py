"""
Entropy-adjusted uncertain-vs-control interaction.

A perturbation of the logits changes entropy MORE when the distribution is already flat, and
uncertain prompts have much higher baseline entropy than their controls. So a raw
uncertain-vs-control interaction can be "flat-distribution sensitivity" rather than
"uncertainty specificity". For every (neuron, category) cell of each ablation run this script
reports, alongside the raw interaction:

  ancova_arm   : arm effect (uncertain - control) from OLS  entropy_shift ~ arm + orig_entropy
  ancova_p     : its p-value (HC3 robust)
  slope_H_unc  : within-uncertain-arm Spearman rho of entropy_shift on orig_entropy
  matched_inter: interaction restricted to the overlap window of baseline entropy
                 (10th-90th percentile of both arms), Welch p and n per arm
  strat_inter  : stratified interaction: mean over baseline-entropy quintile bins (pooled arms)
                 of (uncertain mean - control mean), bins with >=5 prompts per arm, with a
                 permutation p (arm labels permuted within bin)

BH-FDR (alpha 0.01) over ancova_p per run. Writes <run>/entropy_adjusted.csv and a summary.
Usage: python scripts/entropy_adjusted_interaction.py [--runs a,b,c]
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from _common import REPO_ROOT

CATS = ("ambiguity", "lack_of_knowledge", "contradictory_context")
DEFAULT_RUNS = ("results/ablation_bf16_new", "results/ablation_bf16_v3set", "results/ablation_bf16_old15",
                "results/ablation_bf16_keyset_pooled", "results/ablation_bf16_entropy_weights",
                "results/ablation_bf16_frequency_weights", "results/ablation_bf16_entropy_weights_pooled")


def bh(p, alpha=0.01):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o]
    ok = r <= np.arange(1, n + 1) / n * alpha; s = np.zeros(n, bool)
    if ok.any():
        s[o[: np.max(np.where(ok)) + 1]] = True
    return s


def strat_interaction(g, rng, n_perm=500):
    """Stratified interaction over baseline-entropy quintiles (pooled arms): mean over bins with
    >=5 prompts per arm of (uncertain mean - control mean); permutation p by shuffling arm labels
    within bin. Vectorized: per-bin arrays, all permutations at once."""
    bins = pd.qcut(g.orig_entropy, 5, labels=False, duplicates="drop").to_numpy()
    shift = g.entropy_shift.to_numpy(float); unc = (~g.is_control).to_numpy()
    per_bin = []
    for b in np.unique(bins):
        m = bins == b; x = shift[m]; u = unc[m]
        if u.sum() >= 5 and (~u).sum() >= 5:
            per_bin.append((x, u))
    if not per_bin:
        return np.nan, np.nan, 0
    obs = np.mean([x[u].mean() - x[~u].mean() for x, u in per_bin])
    perm_stats = np.zeros(n_perm)
    for x, u in per_bin:
        nu = u.sum()
        idx = np.argsort(rng.random((n_perm, len(x))), axis=1)      # random permutations of positions
        xp = x[idx]                                                  # [n_perm, n]
        perm_stats += xp[:, :nu].mean(1) - xp[:, nu:].mean(1)
    perm_stats /= len(per_bin)
    p = (np.sum(np.abs(perm_stats) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p), len(per_bin)


def analyze_run(run_dir, rng):
    frames = []
    for p in sorted(run_dir.glob("results_*.csv")):  # any category the run wrote (incl. gated arms: familiarity, conflict, ...)
        c = p.stem[len("results_"):]
        d = pd.read_csv(p); d["category"] = c; frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames)
    df["is_control"] = df["is_control"].astype(bool) if "is_control" in df else df.split.eq("control")
    prov = {}
    for p in run_dir.glob("*.provenance.json"):
        try: prov = json.loads(p.read_text()); break
        except Exception: pass
    ctrl_neurons = set(prov.get("control_neurons", []) or [])
    rows = []
    for (nid, cat), g in df.groupby(["neuron_id", "category"]):
        u, k = g[~g.is_control], g[g.is_control]
        if len(u) < 10 or len(k) < 10:
            continue
        raw = u.entropy_shift.mean() - k.entropy_shift.mean()
        raw_p = stats.ttest_ind(u.entropy_shift, k.entropy_shift, equal_var=False).pvalue
        g2 = g.assign(arm=(~g.is_control).astype(int))
        m = smf.ols("entropy_shift ~ arm + orig_entropy", data=g2).fit(cov_type="HC3")
        lo = max(u.orig_entropy.quantile(.1), k.orig_entropy.quantile(.1)); hi = min(u.orig_entropy.quantile(.9), k.orig_entropy.quantile(.9))
        um, km = u[(u.orig_entropy >= lo) & (u.orig_entropy <= hi)], k[(k.orig_entropy >= lo) & (k.orig_entropy <= hi)]
        if len(um) >= 5 and len(km) >= 5:
            mi, mp = um.entropy_shift.mean() - km.entropy_shift.mean(), stats.ttest_ind(um.entropy_shift, km.entropy_shift, equal_var=False).pvalue
        else:
            mi, mp = np.nan, np.nan
        si, sp, nb = strat_interaction(g, rng)
        rows.append(dict(neuron_id=nid, category=cat, is_control_neuron=nid in ctrl_neurons, n_unc=len(u), n_ctrl=len(k),
                         H_unc=u.orig_entropy.mean(), H_ctrl=k.orig_entropy.mean(),
                         raw_inter=raw, raw_p=raw_p, ancova_arm=m.params["arm"], ancova_p=m.pvalues["arm"],
                         ancova_H=m.params["orig_entropy"], ancova_H_p=m.pvalues["orig_entropy"],
                         slope_H_unc=stats.spearmanr(u.orig_entropy, u.entropy_shift).statistic,
                         matched_window=f"[{lo:.2f},{hi:.2f}]", matched_n_unc=len(um), matched_n_ctrl=len(km),
                         matched_inter=mi, matched_p=mp, strat_inter=si, strat_p=sp, strat_bins=nb))
    t = pd.DataFrame(rows)
    if len(t):
        cand = ~t.is_control_neuron
        t["ancova_fdr"] = False; t.loc[cand, "ancova_fdr"] = bh(t.loc[cand, "ancova_p"].fillna(1))
        t["raw_fdr"] = False; t.loc[cand, "raw_fdr"] = bh(t.loc[cand, "raw_p"].fillna(1))
        t["strat_fdr"] = False; t.loc[cand, "strat_fdr"] = bh(t.loc[cand, "strat_p"].fillna(1))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    lines = ["Entropy-adjusted interaction (alpha 0.01 BH within run; candidates only)", ""]
    for run in args.runs.split(","):
        run_dir = REPO_ROOT / run.strip()
        t = analyze_run(run_dir, rng)
        if t is None or not len(t):
            lines.append(f"## {run}: missing"); continue
        t.to_csv(run_dir / "entropy_adjusted.csv", index=False)
        c = t[~t.is_control_neuron]
        lines.append(f"## {run}: {c.neuron_id.nunique()} candidates x {c.category.nunique()} categories")
        lines.append(f"   FDR survivors: raw {int(c.raw_fdr.sum())} | ANCOVA-adjusted {int(c.ancova_fdr.sum())} | stratified {int(c.strat_fdr.sum())}"
                     f"   (uncorrected p<.01: raw {int((c.raw_p < .01).sum())}, ANCOVA {int((c.ancova_p < .01).sum())}, stratified {int((c.strat_p < .01).sum())})")
        top = c.sort_values("ancova_p").head(5)
        for r in top.itertuples():
            lines.append(f"   {r.neuron_id:<11} {r.category[:4]} raw {r.raw_inter:+.4f} (p {r.raw_p:.2g}) | ANCOVA arm {r.ancova_arm:+.4f} (p {r.ancova_p:.2g}; H slope {r.ancova_H:+.4f}) "
                         f"| matched {r.matched_inter:+.4f} (p {r.matched_p:.2g}, n {r.matched_n_unc}/{r.matched_n_ctrl}) | strat {r.strat_inter:+.4f} (p {r.strat_p:.2g}, {r.strat_bins} bins) | rho(shift,H|unc) {r.slope_H_unc:+.2f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (REPO_ROOT / "results/entropy_adjusted_summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
