"""
Read-off for hedge_judge_criterion.md. CPU only.

Inputs  results/hedge_judge/{sample.jsonl, stratum_counts.json, judgments.jsonl}
Output  results/hedge_judge/summary.txt, corrected_cells.csv

Per model and cell (popularity arm x knowledge): precision = P(judge HEDGE | regex hedge), miss = P(judge HEDGE | regex
answer); corrected cell hedge rate = P(regex hedge | cell) * precision + P(regex answer | cell) * miss, with the cell's
regex proportions taken from the full pool (stratum_counts). Factorial effects recomputed on corrected cell rates; 95% CI
from 5,000 bootstraps of the judged generations within each stratum.
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import json

import numpy as np
import pandas as pd

REPO = _Path(__file__).resolve().parents[1]
D = REPO / "results" / "hedge_judge"
CELLS = [("uncertain", "Unknown"), ("uncertain", "HighlyKnown"), ("control", "Unknown"), ("control", "HighlyKnown")]
B = 5000


def effects(rate):
    pop = ((rate["uncertain", "Unknown"] - rate["control", "Unknown"]) + (rate["uncertain", "HighlyKnown"] - rate["control", "HighlyKnown"])) / 2
    know = ((rate["uncertain", "Unknown"] - rate["uncertain", "HighlyKnown"]) + (rate["control", "Unknown"] - rate["control", "HighlyKnown"])) / 2
    return pop, know, pop - know


def main():
    S = pd.DataFrame([json.loads(l) for l in open(D / "sample.jsonl", encoding="utf-8")])
    J = pd.DataFrame([json.loads(l) for l in open(D / "judgments.jsonl", encoding="utf-8")])
    C = pd.DataFrame(json.load(open(D / "stratum_counts.json")))
    S = S.merge(J, on="sid")
    S["jh"] = (S.judge == "HEDGE").astype(float)
    lines = [f"Judge audit (Qwen2.5-32B-Instruct), {len(S)} judged generations; unparsed: {(S.judge == 'UNPARSED').sum()}", ""]
    rng = np.random.default_rng(0)
    rows = []
    for m, g in S.groupby("model", sort=False):
        cm = C[C.model == m]
        strata = {}
        for (a, k, rx), sg in g.groupby(["arm", "slick", "regex"]):
            strata[(a, k, bool(rx))] = sg.jh.to_numpy()
        tot = {(r.arm, r.slick, bool(r.regex)): r.n_total for r in cm.itertuples()}

        def corrected(st):
            rate = {}
            for a, k in CELLS:
                np_, nn = tot.get((a, k, True), 0), tot.get((a, k, False), 0)
                prec = st[(a, k, True)].mean() if (a, k, True) in st and len(st[(a, k, True)]) else 0.0
                miss = st[(a, k, False)].mean() if (a, k, False) in st and len(st[(a, k, False)]) else 0.0
                rate[a, k] = (np_ * prec + nn * miss) / max(np_ + nn, 1)
            return rate

        point = corrected(strata)
        boots = []
        for _ in range(B):
            bs = {s: v[rng.integers(0, len(v), len(v))] for s, v in strata.items()}
            r = corrected(bs)
            boots.append([r[c] for c in CELLS] + list(effects(r)))
        boots = np.array(boots)
        lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
        regex_rate = {(a, k): tot.get((a, k, True), 0) / max(tot.get((a, k, True), 0) + tot.get((a, k, False), 0), 1) for a, k in CELLS}
        # pool-weighted recall: P(regex hedge | judge hedge)
        num = sum(tot.get((a, k, True), 0) * (strata[(a, k, True)].mean() if (a, k, True) in strata else 0) for a, k in CELLS)
        den = sum(tot.get((a, k, True), 0) * (strata[(a, k, True)].mean() if (a, k, True) in strata else 0)
                  + tot.get((a, k, False), 0) * (strata[(a, k, False)].mean() if (a, k, False) in strata else 0) for a, k in CELLS)
        recall = num / den if den else float("nan")
        prec_all = np.average([strata[(a, k, True)].mean() for a, k in CELLS if (a, k, True) in strata],
                              weights=[tot[(a, k, True)] for a, k in CELLS if (a, k, True) in strata])
        e = effects(point)
        ou = point["uncertain", "Unknown"]; ou_hi = hi[0]
        confab = ou < 0.25 and ou_hi < 0.40
        lines.append(f"== {m}: regex recall {recall:.2f} (adequate if >= 0.80: {'yes' if recall >= 0.80 else 'no'}); pool-weighted precision {prec_all:.2f}")
        for i, c in enumerate(CELLS):
            lines.append(f"   {c[0]:>9}/{c[1]:<11} regex {regex_rate[c]:.3f} -> judge-corrected {point[c]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]")
            rows.append(dict(model=m, arm=c[0], slick=c[1], regex_rate=regex_rate[c], corrected=point[c], lo=lo[i], hi=hi[i]))
        lines.append(f"   corrected effects: popularity {e[0]:+.3f} [{lo[4]:+.3f}, {hi[4]:+.3f}]  knowledge {e[1]:+.3f} [{lo[5]:+.3f}, {hi[5]:+.3f}]  "
                     f"pop - know {e[2]:+.3f} [{lo[6]:+.3f}, {hi[6]:+.3f}]  -> condition 1 on corrected rates: {'yes' if lo[6] > 0 else 'no'}")
        lines.append(f"   'confabulator' label (obscure&Unknown corrected < 0.25 with CI upper < 0.40): {'licensed' if confab else 'NOT licensed'}")
        lines.append("")
    pd.DataFrame(rows).to_csv(D / "corrected_cells.csv", index=False)
    (D / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
