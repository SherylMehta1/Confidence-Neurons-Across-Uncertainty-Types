"""
Decider experiment 3 -- gated-arm single-neuron specificity with a RANDOM-NEURON null (no ANCOVA).

The baseline-entropy ANCOVA is unidentified on gated arms (disjoint entropy supports between the twin
conditions). This test replaces it with a null that inherits the entropy sensitivity instead of regressing
it away: every statistic computed for a candidate neuron is computed identically for every RANDOM control
neuron ablated on the SAME prompts (run the ablation with --control-neurons 50), and the candidate's
p-value is its empirical rank among the random neurons.

Statistic per neuron: the twin-paired interaction — mean over twin pairs of
(entropy shift on the uncertain twin − entropy shift on the control twin) — plus its paired Cohen's dz.
Reported: empirical two-sided p vs the random-neuron null, studentized z, BH-FDR over candidates per run.

Usage: python scripts/gated_specificity_test.py --runs results/ablation_fam_ctrl50,results/ablation_conf_ctrl50
Outputs: <run>/specificity_random_null.csv and results/gated_specificity_summary.txt
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import glob
import json

import numpy as np
import pandas as pd

from _common import REPO_ROOT


def paired_interaction(df):
    """Twin-paired interaction per neuron: shift(uncertain twin) - shift(control twin), via twin id = prompt_id minus _u/_c suffix."""
    d = df.copy()
    d["twin"] = d.prompt_id.astype(str).str.replace(r"_(u|c)$", "", regex=True)
    u = d[~d.is_control.astype(bool)].set_index(["neuron_id", "twin"]).entropy_shift
    c = d[d.is_control.astype(bool)].set_index(["neuron_id", "twin"]).entropy_shift
    j = pd.concat([u.rename("su"), c.rename("sc")], axis=1).dropna()
    j["diff"] = j.su - j.sc
    g = j.groupby(level="neuron_id")["diff"]
    out = pd.DataFrame(dict(inter=g.mean(), dz=g.mean() / g.std().replace(0, np.nan), n_pairs=g.count()))
    return out


def bh(pvals, alpha=0.01):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p)
    passed = np.zeros(n, bool); thresh = alpha * (np.arange(1, n + 1) / n)
    ok = p[order] <= thresh
    if ok.any():
        passed[order[: np.max(np.nonzero(ok)) + 1]] = True
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    all_lines = [f"Gated-arm specificity vs random-neuron null (alpha {args.alpha}, BH within run)", ""]
    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        run_dir = REPO_ROOT / run
        files = sorted(glob.glob(str(run_dir / "results_*.csv*")))
        provs = sorted(glob.glob(str(run_dir / "results_*.provenance.json")))
        if not files:
            all_lines.append(f"## {run}: missing"); continue
        prov = json.load(open(provs[0])) if provs else {}
        ctrl_neurons = set(prov.get("control_neurons") or [])
        df = pd.concat([pd.read_csv(f) for f in files if not f.endswith(".provenance.json")], ignore_index=True)
        stats = paired_interaction(df)
        cand = stats[~stats.index.isin(ctrl_neurons)].copy()
        null = stats[stats.index.isin(ctrl_neurons)]
        if len(null) < 10:
            all_lines.append(f"## {run}: only {len(null)} control neurons — rerun the ablation with --control-neurons 50");
        mu, sd = null.inter.mean(), null.inter.std()
        cand["z"] = (cand.inter - mu) / (sd if sd > 0 else np.nan)
        cand["p_emp"] = cand.inter.apply(lambda x: (np.sum(np.abs(null.inter.values) >= abs(x)) + 1) / (len(null) + 1))
        cand["fdr_pass"] = bh(cand.p_emp.values, args.alpha)
        cand = cand.sort_values("p_emp")
        cand.to_csv(run_dir / "specificity_random_null.csv")
        all_lines += [f"## {run}: {len(cand)} candidates vs {len(null)} random neurons (null inter {mu:+.4f} +- {sd:.4f})",
                      f"   FDR survivors: {int(cand.fdr_pass.sum())} | min empirical p attainable: {1/(len(null)+1):.3f}"]
        for nid, r in cand.head(6).iterrows():
            all_lines.append(f"   {nid:<12} inter {r.inter:+.4f} dz {r.dz:+.2f} | z vs null {r.z:+.2f} | p_emp {r.p_emp:.3f}{' *FDR*' if r.fdr_pass else ''}")
        all_lines.append("")
    summary = "\n".join(all_lines)
    print(summary)
    (REPO_ROOT / "results/gated_specificity_summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
