"""
Post-run analysis for the clean bf16 rerun (scripts/run_all.sh outputs).

Per (neuron, category) on UNCERTAIN prompts: mean shift, sign-flip permutation p,
paired dz; same on CONTROL prompts; the uncertain-vs-control INTERACTION (Welch t,
two-sided label-permutation p); the per-prompt SLOPE of entropy_shift on
(orig_activation - mean_val); held-out replication; candidate vs random-control-
neuron comparison; BH-FDR per family. Plus frozen-norm fraction, dose-response
monotonicity, induction check, Stolfo summaries, and an NF4-v3 comparison.

Usage: python scripts/analyze_rerun.py [--runs a,b,c] [--out results/rerun_analysis]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
CATS = ("ambiguity", "lack_of_knowledge", "contradictory_context")
PERSONS = ("person_A_ambiguity", "person_B_lack_of_knowledge", "person_C_contradictory_context")
ALPHA = 0.01


def bh(p, alpha=ALPHA):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    passed = ranked <= (np.arange(1, n + 1) / n) * alpha
    sig = np.zeros(n, bool)
    if passed.any():
        sig[order[: np.max(np.where(passed)) + 1]] = True
    return sig


def sign_flip_p(x, rng, n=20000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan
    obs = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float(((np.abs((signs * x).mean(1)) >= obs).sum() + 1) / (n + 1))


def label_perm_p(a, b, rng, n=10000):
    """Two-sided permutation of group labels, Welch-t statistic."""
    a, b = np.asarray(a, float), np.asarray(b, float)

    def welch(x, y):
        return stats.ttest_ind(x, y, equal_var=False).statistic

    obs = abs(welch(a, b))
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        cnt += abs(welch(pool[:na], pool[na:])) >= obs
    return float((cnt + 1) / (n + 1))


def load_run(run_dir):
    frames = []
    for cat in CATS:
        p = run_dir / f"results_{cat}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["category"] = cat
            frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "is_control" not in df:
        df["is_control"] = df["split"].eq("control")
    df["is_control"] = df["is_control"].astype(bool)
    return df


def analyze_run(df, run_name, candidate_ids, rng):
    rows = []
    for (nid, cat), g in df.groupby(["neuron_id", "category"]):
        u = g[~g.is_control]
        c = g[g.is_control]
        uw = u[u.split == "working"]
        uh = u[u.split == "held_out"]
        sd = u.entropy_shift.std(ddof=1)
        r = dict(
            run=run_name, neuron_id=nid, category=cat, is_candidate=nid in candidate_ids,
            n_unc=len(u), n_ctrl=len(c),
            unc_mean=u.entropy_shift.mean(),
            unc_dz=u.entropy_shift.mean() / sd if sd > 0 else np.nan,
            unc_p=sign_flip_p(u.entropy_shift, rng),
            unc_work_mean=uw.entropy_shift.mean(), unc_work_p=sign_flip_p(uw.entropy_shift, rng),
            unc_held_mean=uh.entropy_shift.mean(), unc_held_p=sign_flip_p(uh.entropy_shift, rng),
            ctrl_mean=c.entropy_shift.mean() if len(c) else np.nan,
            ctrl_p=sign_flip_p(c.entropy_shift, rng) if len(c) else np.nan,
            mean_abs_shift=u.entropy_shift.abs().mean(),
        )
        if len(c):
            r["interaction"] = r["unc_mean"] - r["ctrl_mean"]
            r["interaction_t"] = stats.ttest_ind(u.entropy_shift, c.entropy_shift, equal_var=False).statistic
            r["interaction_p"] = label_perm_p(u.entropy_shift.to_numpy(), c.entropy_shift.to_numpy(), rng)
        if "orig_activation" in g and "mean_val" in g:
            x = (u.orig_activation - u.mean_val).to_numpy(float)
            y = u.entropy_shift.to_numpy(float)
            if np.std(x) > 0:
                lr = stats.linregress(x, y)
                r.update(slope=lr.slope, slope_p=lr.pvalue, slope_r=lr.rvalue)
        rows.append(r)
    t = pd.DataFrame(rows)
    for fam in ("unc_p", "ctrl_p", "interaction_p", "slope_p", "unc_held_p"):
        if fam in t:
            t[f"{fam}_fdr"] = bh(t[fam].fillna(1.0))
    return t


def fmt_table(df):
    try:
        return df.to_markdown(index=False, floatfmt=".4g")
    except Exception:
        return df.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/ablation_bf16_new,results/ablation_bf16_old15,results/ablation_bf16_v3set")
    ap.add_argument("--out", default="results/rerun_analysis")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    md = ["# Clean bf16 rerun - analysis", ""]
    all_tables = []

    for run in a.runs.split(","):
        run_dir = REPO / run
        df = load_run(run_dir)
        if df is None:
            md += [f"## {run}", "_missing_", ""]
            continue
        prov = {}
        for p in sorted(run_dir.glob("*.provenance.json")):
            try:
                prov = json.loads(p.read_text())
                break
            except Exception:
                pass
        ctrl_neurons = set(prov.get("control_neurons", []) or [])
        cand_ids = set(df.neuron_id) - ctrl_neurons if ctrl_neurons else set(df.neuron_id)
        t = analyze_run(df, Path(run).name, cand_ids, rng)
        all_tables.append(t)
        t.to_csv(out / f"{Path(run).name}_cells.csv", index=False)
        prec = df["precision"].iloc[0] if "precision" in df else "?"
        src = df["mean_source"].iloc[0] if "mean_source" in df else "?"
        grid = float(np.mean(np.abs(df.orig_entropy * 256 - np.round(df.orig_entropy * 256)) < 1e-9))
        md += [f"## {run}",
               f"precision={prec} mean_source={src} rows={len(df)} neurons={df.neuron_id.nunique()} "
               f"(candidates={len(cand_ids)}) exact-zero shifts={float((df.entropy_shift == 0).mean()):.3f} "
               f"on-1/256-grid={grid:.3f}", ""]
        c = t[t.is_candidate]

        def fam(col):
            return int(c[f"{col}_fdr"].sum()) if f"{col}_fdr" in c else "n/a"

        md.append(f"- FDR({ALPHA}) survivors among {len(c)} candidate cells: uncertain-shift {fam('unc_p')}, "
                  f"control-shift {fam('ctrl_p')}, uncertain-vs-control interaction {fam('interaction_p')}, "
                  f"activation-slope {fam('slope_p')}, held-out-only {fam('unc_held_p')}")
        ip = int((c.interaction_p < .01).sum()) if "interaction_p" in c else "n/a"
        sp = int((c.slope_p < .01).sum()) if "slope_p" in c else "n/a"
        md.append(f"- uncorrected p<.01: uncertain {int((c.unc_p < .01).sum())}, interaction {ip}, slope {sp}")
        if (~t.is_candidate).any() and "interaction" in t:
            k = t[~t.is_candidate]
            md.append(f"- random control NEURONS: mean|shift| {k.mean_abs_shift.mean():.5f} vs candidates "
                      f"{c.mean_abs_shift.mean():.5f}; |interaction| {k.interaction.abs().mean():.5f} vs "
                      f"{c.interaction.abs().mean():.5f}")
        rep = c[(c.unc_work_p < .01) & (c.unc_held_p < .01) & (np.sign(c.unc_work_mean) == np.sign(c.unc_held_mean))]
        names = ", ".join(f"{r.neuron_id}/{r.category}" for r in rep.itertuples())
        md.append(f"- working AND held-out p<.01 same sign: {len(rep)} {names}")
        cols = [x for x in ["neuron_id", "category", "unc_mean", "ctrl_mean", "interaction", "interaction_p",
                            "unc_dz", "slope", "slope_p"] if x in c]
        sort_col = "interaction_p" if "interaction_p" in c else "unc_p"
        md += ["", "Top-5 by " + sort_col + ":", fmt_table(c.sort_values(sort_col).head(5)[cols]), ""]

    if all_tables:
        pd.concat(all_tables).to_csv(out / "all_cells.csv", index=False)

    fz = REPO / "results/frozen_norm_bf16.csv"
    if fz.exists():
        d = pd.read_csv(fz)
        md.append("## Frozen-RMSNorm decomposition")
        for (nid, cat, ic), g in d.groupby(["neuron_id", "category", "is_control"]):
            full, fro = g.entropy_shift.mean(), g.shift_under_frozen_norm.mean()
            frac = 1 - fro / full if full else np.nan
            md.append(f"- {nid}/{cat}{' (controls)' if ic else ''}: full {full:+.5f} "
                      f"(p={sign_flip_p(g.entropy_shift, rng):.3g}), frozen-norm {fro:+.5f} "
                      f"(p={sign_flip_p(g.shift_under_frozen_norm, rng):.3g}) -> norm-mediated fraction {frac:.2f}")
        md.append("")

    dr = REPO / "results/dose_response_bf16.csv"
    if dr.exists():
        d = pd.read_csv(dr)
        md.append("## Dose-response (per-prompt Spearman of clamped entropy vs sigma level)")
        for (nid, cat, ic), g in d.groupby(["neuron_id", "category", "is_control"]):
            rhos = []
            for _, p in g.groupby("prompt_id"):
                if p.sigma_multiplier.nunique() > 2:
                    rho = stats.spearmanr(p.sigma_multiplier, p.clamped_entropy).statistic
                    if not np.isnan(rho):
                        rhos.append(rho)
            rhos = np.array(rhos)
            lvl = g.groupby("sigma_multiplier").entropy_shift.mean()
            levels = ", ".join(f"{k:+g}s:{v:+.4f}" for k, v in lvl.items())
            cons = max((rhos > 0).mean(), (rhos < 0).mean()) if len(rhos) else np.nan
            md.append(f"- {nid}/{cat}{' (controls)' if ic else ''}: mean rho {rhos.mean():+.3f}, "
                      f"sign-consistent {cons:.2f}, mean shift by level {levels}")
        md.append("")

    ind = REPO / "results/induction_check.csv"
    if ind.exists():
        md += ["## Induction check (all prompts)", fmt_table(pd.read_csv(ind)), ""]

    for stem in ("stolfo_bf16_new", "stolfo_v3set", "stolfo_old15"):
        s = REPO / f"results/{stem}_summary.txt"
        if s.exists():
            md += [f"## {stem}", "```", s.read_text()[:3000], "```", ""]

    v3 = []
    for cat, person in zip(CATS, PERSONS):
        p = REPO / person / "results/results_v3.csv"
        if p.exists():
            v3.append(pd.read_csv(p).assign(category=cat))
    new = REPO / "results/ablation_bf16_v3set"
    nv = load_run(new)
    if v3 and nv is not None:
        v3 = pd.concat(v3)
        nv = nv[~nv.is_control]
        a_ = v3.groupby(["neuron_id", "category"]).entropy_shift.mean().rename("nf4_v3")
        b_ = nv.groupby(["neuron_id", "category"]).entropy_shift.mean().rename("bf16_rerun")
        both = pd.concat([a_, b_], axis=1).dropna()
        if len(both) > 2:
            r = np.corrcoef(both.nf4_v3, both.bf16_rerun)[0, 1]
            md.append(f"## NF4 v3 vs bf16 rerun (same neurons; different mean reference): r={r:.3f} over {len(both)} cells")

    text = "\n".join(md)
    (out / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
