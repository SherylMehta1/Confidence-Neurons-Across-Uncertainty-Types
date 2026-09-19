"""
Leave-one-token-out, CPU half: recompute the headline recovery numbers under each hedge-token variant.

Reads loto_cache.csv (from circuit_loto_cache.py) and, for the full hedge set and for each single
hedge token dropped, recomputes hedge/answer log-odds from the cached log-probs
(logsumexp over the retained hedge ids minus logsumexp over the answer ids -- the answer set is held
fixed, the question is about the hedge set), then recovery() with the same clip as everywhere else.
Per (cell, variant, direction): mean recovery over held-out pairs with a percentile-bootstrap 95% CI
(10k resamples, seed 0, the stats_upgrades.boot_ci estimator replicated) and, where the cache has
random sets, the null band (mean +- sd of per-set means).

Sanity check: the full-set variant must reproduce faithfulness.csv (circuit cell) and
direction_patch.csv (direction cell) per pair; the max abs difference is printed and stored.

Writes <dir>/loto.csv and loto_summary.txt. Refuses to overwrite without --overwrite.

  python scripts/loto_recompute.py --dir results/circuit_familiarity_v2
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np
import pandas as pd

from _common import REPO_ROOT, guard_output
from circuit_common import recovery

B, SEED = 10_000, 0
REFERENCE = {"circuit": ("faithfulness.csv", "circuit"), "direction": ("direction_patch.csv", "direction")}


def boot_ci(v, rng):
    v = np.asarray(v, dtype=float)
    idx = rng.integers(0, len(v), (B, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])


def logsumexp(a):
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True)))[:, 0]


def logodds(df, hedge_ids, answer_ids):
    h = df[[f"lp_{i}" for i in hedge_ids]].to_numpy(float)
    a = df[[f"lp_{i}" for i in answer_ids]].to_numpy(float)
    return logsumexp(h) - logsumexp(a)


def per_pair_recovery(cache, cell, setname, direc, hedge_ids, answer_ids):
    """recovery per pair for one (cell, set, direction) under a hedge-id subset."""
    c = cache[cache.cell == cell]
    base_u = c[c.condition == "baseline_u"].set_index("pair")
    base_c = c[c.condition == "baseline_c"].set_index("pair")
    pat = c[(c.set == setname) & (c.direction == direc) & (c.condition == "patched")].set_index("pair")
    lo_u, lo_c, lo_p = (pd.Series(logodds(x, hedge_ids, answer_ids), index=x.index) for x in (base_u, base_c, pat))
    tgt, src = (lo_u, lo_c) if direc == "control_to_uncertain" else (lo_c, lo_u)
    return pd.Series({k: recovery(lo_p[k], tgt[k], src[k]) for k in pat.index}).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/circuit_familiarity_v2")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    d = REPO_ROOT / args.dir
    out_csv = d / "loto.csv"
    guard_output(out_csv, args.overwrite)

    cache = pd.read_csv(d / "loto_cache.csv")
    tok = json.loads((d / "loto_cache_tokens.json").read_text())
    hedge_map, answer_ids = tok["hedge"], sorted(set(tok["answer"].values()))
    variants = [("full", None, sorted(set(hedge_map.values())))]
    for t, i in hedge_map.items():
        kept = sorted({j for s, j in hedge_map.items() if s != t})  # drop the id only if no other hedge token shares it
        variants.append((f"minus{t.strip()}", t, kept))

    rng = np.random.default_rng(SEED)
    rows, checks = [], {}
    for cell in ("circuit", "direction"):
        if not (cache.cell == cell).any():
            continue
        real_set = REFERENCE[cell][1]
        ref = pd.read_csv(d / REFERENCE[cell][0])
        nulls = sorted(s for s in cache[cache.cell == cell].set.unique() if s.startswith("random"))
        for vname, dropped, hedge_ids in variants:
            for direc in ("control_to_uncertain", "uncertain_to_control"):
                rec = per_pair_recovery(cache, cell, real_set, direc, hedge_ids, answer_ids)
                lo, hi = boot_ci(rec.to_numpy(), rng)
                row = dict(cell=cell, variant=vname, dropped_token=dropped or "", direction=direc,
                           logodds_rec_mean=float(rec.mean()), ci_lo=float(lo), ci_hi=float(hi), n_pairs=len(rec))
                if nulls and direc == "control_to_uncertain":
                    means = [per_pair_recovery(cache, cell, s, direc, hedge_ids, answer_ids).mean() for s in nulls]
                    row.update(null_mean=float(np.mean(means)), null_sd=float(np.std(means, ddof=1)), n_null_sets=len(nulls))
                else:
                    row.update(null_mean=np.nan, null_sd=np.nan, n_null_sets=0)
                if vname == "full":
                    r = ref[(ref["set"] == real_set) & (ref.direction == direc)].set_index("pair").logodds_rec.sort_index()
                    checks[(cell, direc)] = float((rec - r.reindex(rec.index)).abs().max())
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    lines = [f"Leave-one-token-out -- {args.dir}; hedge tokens {list(hedge_map)}; answer set fixed",
             "full-set reproduction of the committed CSVs, max |diff| per pair: "
             + ", ".join(f"{c}/{dr.split('_')[0]}->{dr.split('_')[-1]} {v:.2e}" for (c, dr), v in checks.items()), ""]
    for cell in out.cell.unique():
        lines.append(f"[{cell}]")
        for direc in ("control_to_uncertain", "uncertain_to_control"):
            lines.append(f"  {direc}:")
            for _, r in out[(out.cell == cell) & (out.direction == direc)].iterrows():
                null = f"   null {r.null_mean:+.3f} +- {r.null_sd:.3f} ({int(r.n_null_sets)} sets)" if r.n_null_sets else ""
                lines.append(f"    {r.variant:<14} {r.logodds_rec_mean:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]{null}")
    summary = "\n".join(lines)
    print(summary)
    (d / "loto_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
