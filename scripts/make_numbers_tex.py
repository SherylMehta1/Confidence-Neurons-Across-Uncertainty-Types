"""
Generate paper/numbers.tex -- one \\newcommand per number the manuscript cites, computed from
committed result files, so a citation can never drift from the result behind it.

Performs NO analysis: it only aggregates CSV/JSON that other scripts already produced. Rerun it
after any upstream result regenerates, then recompile. Output is fully overwritten each run.

Two conventions worth knowing:

* Confidence intervals are read from the figure's own committed <fig>_cells.csv where one exists,
  not recomputed. Recomputing would draw a different bootstrap sample than the figure did (same
  estimator, different random stream) and the table and the figure would disagree in the last
  digit -- exactly the drift this file exists to prevent. Where no figure exists the CI is
  computed here, with the same percentile-bootstrap estimator, and the macro is marked `Recomputed`
  in the provenance comment.
* Nothing silently picks a canonical circuit. Every draw is emitted under its own suffix (Old =
  the pre-dedup 195-pair discovery, New = familiarity_v2, VThree = familiarity_v3 at 60 held-out
  pairs, VThreeFull = familiarity_v3 at all 141), plus a range macro. main.tex cites whichever the
  paper settles on; see the report printed at the end.

Usage: python scripts/make_numbers_tex.py [--check]
  --check  exit non-zero if any source file is missing, and write nothing (for CI)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "paper" / "numbers.tex"

macros, sources, missing = {}, {}, []


def add(name, value, source=None):
    macros[name] = value
    if source:
        sources[name] = source


def need(rel):
    """Path to a required input; records it as missing instead of raising, so one run reports
    every gap at once rather than dying on the first."""
    p = REPO_ROOT / rel
    if not p.exists():
        missing.append(rel)
        return None
    return p


def mean_ci(vals, n_boot=10000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    b = vals[idx].mean(axis=1)
    return vals.mean(), *np.percentile(b, [2.5, 97.5])


# =====================================================================
# L0/L1 -- dataset scale, circuit faithfulness, rank-1 direction, bridge
# =====================================================================

for rel, suffix in (("data/familiarity/gate_report.json", "Old"),
                    ("data/familiarity_v2/gate_report.json", "New"),
                    ("data/familiarity_v3/gate_report.json", "VThree")):
    if (p := need(rel)):
        g = json.loads(p.read_text())
        add(f"pairCount{suffix}", g["n_pairs"], rel)
        add(f"workingCount{suffix}", g["n_working"], rel)
        add(f"heldOutCount{suffix}", g["n_held_out"], rel)
        add(f"candidateCount{suffix}", g["n_candidates"], rel)

MTAG = {"logodds_rec": "Hedge", "entropy_rec": "Entropy"}
DTAG = {"control_to_uncertain": "CU", "uncertain_to_control": "UC"}
CELLS_FOR = {  # circuit dir -> the figure cells.csv that plots it, when one exists
    "results/circuit_familiarity_v2": "paper/figures/fig6_familiarity_cells.csv",
    "results/circuit_familiarity_v3": "paper/figures/fig6_familiarity_v3_cells.csv",
}


def recovery_macros(circuit_dir, suffix, csv_name="faithfulness.csv", setname="circuit", prefix="circuit"):
    """Mean + 95% CI per (readout, direction) for one patched set, preferring the figure's own CIs."""
    p = need(f"{circuit_dir}/{csv_name}")
    if p is None:
        return
    df = pd.read_csv(p)
    sub_all = df[df["set"] == setname]
    cells_rel = CELLS_FOR.get(circuit_dir) if csv_name in ("faithfulness.csv", "direction_patch.csv") else None
    cells = pd.read_csv(REPO_ROOT / cells_rel) if cells_rel and (REPO_ROOT / cells_rel).exists() else None
    series = "full set (20h+100n)" if setname == "circuit" else "rank-1 direction"
    for direction, dtag in DTAG.items():
        for metric, mtag in MTAG.items():
            vals = sub_all[sub_all.direction == direction][metric].to_numpy()
            if not len(vals):
                continue
            row = None
            if cells is not None:
                hit = cells[(cells.readout == metric) & (cells.direction == direction) & (cells.series == series)]
                row = hit.iloc[0] if len(hit) else None
            if row is not None:
                m, lo, hi, src = float(row["mean"]), float(row.ci_lo), float(row.ci_hi), cells_rel
            else:
                m, lo, hi = mean_ci(vals)
                src = f"{circuit_dir}/{csv_name} (CI recomputed)"
            add(f"{prefix}{mtag}{dtag}{suffix}", f"{m:.2f}", src)
            add(f"{prefix}{mtag}{dtag}{suffix}CI", f"[{lo:.2f}, {hi:.2f}]", src)
            add(f"{prefix}{mtag}{dtag}{suffix}N", len(vals), src)


recovery_macros("results/circuit_familiarity", "Old")
recovery_macros("results/circuit_familiarity_v2", "New")
recovery_macros("results/circuit_familiarity_v3", "VThree")
recovery_macros("results/circuit_familiarity_v3", "VThreeFull", csv_name="faithfulness_limit141.csv")
recovery_macros("results/circuit_familiarity", "Old", "direction_patch.csv", "direction", "direction")
recovery_macros("results/circuit_familiarity_v2", "New", "direction_patch.csv", "direction", "direction")
recovery_macros("results/circuit_familiarity_v3", "VThree", "direction_patch.csv", "direction", "direction")
recovery_macros("results/circuit_familiarity_v3", "VThreeFull", "direction_patch_limit141.csv", "direction", "direction")

# The hedge c->u recovery differs across the three discovery draws (0.80 / 0.91 / 0.82); quoting any
# one as "the" value overstates precision, so emit the span too.
draws = [macros.get(f"circuitHedgeCU{s}") for s in ("Old", "New", "VThree")]
if all(draws):
    d = sorted(float(x) for x in draws)
    add("circuitHedgeCURange", f"{d[0]:.2f}--{d[-1]:.2f}", "three discovery draws")

# Entropy, removal direction: -0.13 / +0.14 / +0.10 / -0.06 across four estimates. A macro that
# resolves to a signed number invites citing it as a point value, so the citable form is prose.
add("entropyRemovalFam", "near zero and sign-unstable across replications",
    "circuitEntropyCU{Old,New,VThree,VThreeFull}")
ent = [macros.get(f"circuitEntropyCU{s}") for s in ("Old", "New", "VThree", "VThreeFull")]
if all(ent):
    add("entropyRemovalFamSpan", ", ".join(f"{float(x):+.2f}" for x in ent), "four estimates")

if (p := need("results/circuit_familiarity_v2/prune_curve.csv")):
    pr = pd.read_csv(p)
    cleared = pr[pr.recovery > 0.7]
    add("minimalityK", int(cleared.k.min()) if len(cleared) else "n/a", "results/circuit_familiarity_v2/prune_curve.csv")
if (p := need("results/circuit_familiarity_v3/prune_curve.csv")):
    pr = pd.read_csv(p)
    cleared = pr[pr.recovery > 0.7]
    add("minimalityKVThree", int(cleared.k.min()) if len(cleared) else "n/a", "results/circuit_familiarity_v3/prune_curve.csv")

if (p := need("results/bridge_familiarity_v3/cos_bootstrap.csv")):
    boot = pd.read_csv(p)
    r31 = boot[boot.layer == 31].iloc[0]
    src = "results/bridge_familiarity_v3/cos_bootstrap.csv"
    add("alignCosLThirtyOne", f"{r31.cos:.2f}", src)
    add("alignCosLThirtyOneCI", f"[{r31.ci_lo:.2f}, {r31.ci_hi:.2f}]", src)
    add("alignCosFloor", f"{boot.ci_lo.min():.2f}", src)
    add("alignCosRangeLo", f"{boot.cos.min():.2f}", src)
    add("alignCosRangeHi", f"{boot.cos.max():.2f}", src)
if (p := need("results/bridge_familiarity_v3/cos_permutation.csv")):
    perm = pd.read_csv(p)
    add("alignPermPMax", f"{perm.p_one_sided.max():.4f}", "results/bridge_familiarity_v3/cos_permutation.csv")
    add("alignPermNullSD", f"{perm.null_sd.mean():.2f}", "results/bridge_familiarity_v3/cos_permutation.csv")

if (p := need("data/uu_hedge_confab/gate_report.json")):
    u = json.loads(p.read_text())
    add("uuPairs", u["n_pairs"], "data/uu_hedge_confab/gate_report.json")
    add("uuWorking", u["n_working"], "data/uu_hedge_confab/gate_report.json")
if (p := need("data/uu_hedge_confab_v3/gate_report.json")):
    u = json.loads(p.read_text())
    add("uuPairsVThree", u["n_pairs"], "data/uu_hedge_confab_v3/gate_report.json")
    add("uuWorkingVThree", u["n_working"], "data/uu_hedge_confab_v3/gate_report.json")

# =====================================================================
# L2 -- leave-one-token-out, causal timing
# =====================================================================

# loto.csv is keyed by (cell, variant, direction); every macro must pin all three or it averages
# the component set together with the rank-1 direction, and both directions together.
if (p := need("results/circuit_familiarity_v2/loto.csv")):
    lo = pd.read_csv(p)
    src = "results/circuit_familiarity_v2/loto.csv"
    for cell, ctag in (("circuit", "Circuit"), ("direction", "Direction")):
        for direction, dtag in DTAG.items():
            for variant in sorted(lo.variant.unique()):
                hit = lo[(lo.cell == cell) & (lo.direction == direction) & (lo.variant == variant)]
                if not len(hit):
                    continue
                r = hit.iloc[0]
                vtag = "Full" if variant == "full" else "Minus" + variant[len("minus"):].capitalize()
                add(f"loto{ctag}{dtag}{vtag}", f"{r.logodds_rec_mean:.2f}", src)
                add(f"loto{ctag}{dtag}{vtag}CI", f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", src)
    # the dissociation's hedge-side margin (component set minus rank-1), full vs " not" dropped
    for variant, tag in (("full", "Full"), ("minusnot", "MinusNot")):
        c = lo[(lo.cell == "circuit") & (lo.direction == "control_to_uncertain") & (lo.variant == variant)]
        d = lo[(lo.cell == "direction") & (lo.direction == "control_to_uncertain") & (lo.variant == variant)]
        if len(c) and len(d):
            add(f"lotoHedgeMargin{tag}", f"{c.iloc[0].logodds_rec_mean - d.iloc[0].logodds_rec_mean:.2f}", src)
    add("lotoMaxAbsShift",
        f"{max(abs(lo[lo.variant != 'full'].logodds_rec_mean.to_numpy() - lo[lo.variant == 'full'].logodds_rec_mean.mean())):.2f}", src)

# The informative causal-timing run is 0..l at the suffix positions; the plan's literal "from l
# onward, all positions" is degenerate (flat at 1.00) and is not cited.
if (p := need("results/circuit_familiarity_v2/causal_timing_up_to_suffix_summary.csv")):
    t = pd.read_csv(p)
    src = "results/circuit_familiarity_v2/causal_timing_up_to_suffix_summary.csv"
    for direction, dtag in DTAG.items():
        g = t[t.direction == direction].sort_values("layer")
        for metric, mtag in (("logodds_rec_mean", "Hedge"), ("entropy_rec_mean", "Entropy")):
            cross = g[g[metric] > 0.7].layer
            add(f"timing{mtag}Cross{dtag}", int(cross.min()) if len(cross) else "n/a", src)

# =====================================================================
# L3/L4a -- frequency x knowledge factorial
# =====================================================================

for rel, suffix in (("data/factorial_v2_summary.csv", "New"), ("data/factorial_v3_summary.csv", "VThree")):
    if (p := need(rel)):
        f = pd.read_csv(p).set_index("contrast")
        for contrast, ctag in (("pop_given_known", "PopGivenKnown"), ("pop_given_unknown", "PopGivenUnknown"),
                               ("know_given_hi", "KnowGivenHi"), ("know_given_lo", "KnowGivenLo")):
            if contrast not in f.index:
                continue
            r = f.loc[contrast]
            add(f"factorial{ctag}Pairs{suffix}", int(r.n_pairs), rel)
            add(f"factorial{ctag}EntropyGap{suffix}", f"{r.gap_entropy:+.2f}", rel)
            add(f"factorial{ctag}HedgeGap{suffix}", f"{r.gap_hedge_rate:+.2f}", rel)

# =====================================================================
# L2e -- saturation sweep; emitted once the curve exists
# =====================================================================
# saturation_curve.csv is one row per (n, cell, direction, metric); pin all three or .iloc[-1]
# picks an arbitrary series. `recovery` is the prefix mean, i.e. exactly what `--limit n` prints.
sat_rel = "results/circuit_familiarity_v3/saturation_curve.csv"
if (REPO_ROOT / sat_rel).exists():
    sat = pd.read_csv(REPO_ROOT / sat_rel)
    add("saturationMaxN", int(sat.n.max()), sat_rel)
    for cell, ctag in (("circuit", "Circuit"), ("direction", "Direction")):
        for direction, dtag in DTAG.items():
            g = sat[(sat.cell == cell) & (sat.direction == direction) & (sat.metric == "logodds_rec")].sort_values("n")
            if not len(g):
                continue
            full = g.iloc[-1]
            add(f"saturation{ctag}{dtag}AtMaxN", f"{full.recovery:.2f}", sat_rel)
            settled = g[(g.n >= 40)]
            if len(settled):
                add(f"saturation{ctag}{dtag}DriftFromForty", f"{max(abs(settled.recovery - full.recovery)):.3f}", sat_rel)
            narrow = g[g.n <= 60]
            if len(narrow):
                add(f"saturation{ctag}{dtag}BandAtSixty", f"{narrow.iloc[-1].ci_width:.2f}", sat_rel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report missing inputs and write nothing")
    args = ap.parse_args()
    if missing:
        print("MISSING INPUTS (macros for these were skipped):")
        for m in sorted(set(missing)):
            print("  -", m)
    if args.check:
        raise SystemExit(1 if missing else 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED by scripts/make_numbers_tex.py -- do not hand-edit.\n")
        f.write("% Rerun the script after any result regenerates; this file has no memory of its own.\n")
        f.write("% Suffixes: Old = pre-dedup 195-pair discovery, New = familiarity_v2,\n")
        f.write("% VThree = familiarity_v3 (60 held-out), VThreeFull = familiarity_v3 (141 held-out).\n\n")
        for name in sorted(macros):
            src = sources.get(name)
            f.write(f"\\newcommand{{\\{name}}}{{{macros[name]}}}" + (f"  % {src}\n" if src else "\n"))
    print(f"wrote {len(macros)} macros to {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
