"""
Read-off for hedge_measure_v2_criterion.md. CPU only.

Inputs  results/hedge_measure_v2/{items_*, gens_*, judge_qwen32b_*, judge_gemma27b_*}.jsonl
Output  results/hedge_measure_v2/{summary.txt, item_rates.csv}
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import json
import re

import numpy as np
import pandas as pd

REPO = _Path(__file__).resolve().parents[1]
D = REPO / "results" / "hedge_measure_v2"
JUDGES = ["qwen32b", "gemma27b"]
MODELS = ["llama", "qwen", "gemma", "mistral"]
CELLS = [("uncertain", "Unknown"), ("uncertain", "HighlyKnown"), ("control", "Unknown"), ("control", "HighlyKnown")]
B = 10000


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def effects(m):
    pop = ((m["uncertain", "Unknown"] - m["control", "Unknown"]) + (m["uncertain", "HighlyKnown"] - m["control", "HighlyKnown"])) / 2
    know = ((m["uncertain", "Unknown"] - m["uncertain", "HighlyKnown"]) + (m["control", "Unknown"] - m["control", "HighlyKnown"])) / 2
    return pop, know, pop - know


def main():
    src = open(REPO / "scripts/behavioral_test.py", encoding="utf8").read()
    blk = src[src.index("HEDGE_RE = re.compile("):]; ns = {}
    exec("import re\n" + blk[:blk.index("\n)\n") + 3], ns); H0 = ns["HEDGE_RE"]
    lines, rows = [], []
    for m in MODELS:
        if not all((D / f"judge_{j}_{m}.jsonl").exists() for j in JUDGES):
            lines.append(f"== {m}: judge files not yet available; skipped"); lines.append(""); continue
        gens = {g["key"]: g for g in read(D / f"gens_{m}.jsonl")}
        J = {j: {(r["key"], r["s"]): r["label"] for r in read(D / f"judge_{j}_{m}.jsonl")} for j in JUDGES}
        samp = []
        for k, g in gens.items():
            for s, txt in enumerate(g["samples"]):
                a, b = J["qwen32b"].get((k, s)), J["gemma27b"].get((k, s))
                if a is None or b is None:
                    continue
                samp.append(dict(key=k, arm=g["arm"], slick=g["slick"], s=s, a=a, b=b, regex=bool(H0.search(txt)),
                                 prim=(a == "DONT_KNOW" and b == "DONT_KNOW"),
                                 sec=(a in ("DONT_KNOW", "CLARIFY") and b in ("DONT_KNOW", "CLARIFY"))))
        S = pd.DataFrame(samp)
        k_dk = kappa(S.a == "DONT_KNOW", S.b == "DONT_KNOW")
        unp = int(((S.a == "UNPARSED") | (S.b == "UNPARSED")).sum())
        recall = S[S.prim].regex.mean() if S.prim.any() else float("nan")
        it = S.groupby(["key", "arm", "slick"])[["prim", "sec", "regex"]].mean().reset_index()
        it["model"] = m; rows.append(it)
        rng = np.random.default_rng(0)
        out = {}
        for y in ("prim", "sec", "regex"):
            cell = {c: it[(it.arm == c[0]) & (it.slick == c[1])][y].to_numpy() for c in CELLS}
            pt = effects({c: v.mean() for c, v in cell.items()})
            bs = np.array([effects({c: v[rng.integers(0, len(v), len(v))].mean() for c, v in cell.items()}) for _ in range(B)])
            lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
            out[y] = (pt, lo, hi, {c: v.mean() for c, v in cell.items()})
        pt, lo, hi, cm = out["prim"]
        ok_k = k_dk >= 0.60
        if not ok_k:
            verdict = "UNMEASURED (judge agreement kappa < 0.60)"
        elif lo[2] > 0:
            verdict = "FAMILIARITY > KNOWLEDGE"
        elif hi[2] < 0:
            verdict = "KNOWLEDGE > FAMILIARITY"
        else:
            verdict = "NO DIFFERENCE RESOLVED"
        lines.append(f"== {m}: {len(it)} items, {len(S)} samples; kappa(DONT_KNOW) = {k_dk:.2f}; unparsed {unp}; "
                     f"regex recall on 96-token replies = {recall:.2f}")
        for y, name in (("prim", "primary (both DONT_KNOW)"), ("sec", "secondary (+CLARIFY)"), ("regex", "regex on new replies")):
            p, l, h, c = out[y]
            lines.append(f"   {name:<26} cells " + " ".join(f"{a[:3]}/{k[:4]} {v:.3f}" for (a, k), v in c.items())
                         + f" | pop {p[0]:+.3f} [{l[0]:+.3f},{h[0]:+.3f}] know {p[1]:+.3f} [{l[1]:+.3f},{h[1]:+.3f}] pop-know {p[2]:+.3f} [{l[2]:+.3f},{h[2]:+.3f}]")
        lines.append(f"   -> {verdict}")
        lines.append("")
    pd.concat(rows).to_csv(D / "item_rates.csv", index=False)
    (D / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
