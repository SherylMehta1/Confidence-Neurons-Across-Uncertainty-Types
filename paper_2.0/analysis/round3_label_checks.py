"""Post hoc checks requested by review round 3 (methods referee F1-F3). Llama, two-judge items.
(1) Popular-unknown audit: questions whose free replies contain a gold alias may be mislabelled 'unknown'. Recompute the
    popularity-minus-knowledge contrast after dropping, from both unknown cells, questions where any free reply
    contains a gold alias (symmetric rule), and after dropping them from popular-unknown only (worst case).
(2) Label agreement: cross-tab of original vs paraphrase labels with Cohen's kappa (items labelled under both), and the
    contrast on items where both labellings agree.
(3) Alias inside abstentions: share of both-judges-DONT_KNOW replies that contain a gold alias, per cell.
Effects with 10,000 within-cell item bootstraps (seed 0). Writes analysis/round3_label_checks.txt."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = next(p for p in (HERE.parents[1], HERE.parents[1] / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
pool = {}
for l in open(R / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("subj"):
        pool.setdefault(f"{r['prop']}::{r['subj']}", r)


def aliases(r):
    a = r.get("aliases") or json.loads(r.get("possible_answers") or "[]")
    return [norm(x) for x in a if x and len(norm(x)) >= 2]


lab = {}
for j in ("qwen32b", "gemma27b"):
    for l in open(R / f"results/hedge_measure_v2/judge_{j}_llama.jsonl", encoding="utf-8"):
        x = json.loads(l); lab.setdefault((x["key"], x["s"]), []).append(x["label"])
rows = []
for l in open(R / "results/hedge_measure_v2/gens_llama.jsonl", encoding="utf-8"):
    g = json.loads(l)
    al = aliases(pool[g["key"]])
    subj = norm(pool[g["key"]]["subj"])
    for s, txt in enumerate(g["samples"]):
        t = " " + norm(txt) + " "
        hit = any((" " + a + " ") in t for a in al if a not in subj)  # ignore aliases contained in the subject name itself
        rows.append(dict(key=g["key"], arm=g["arm"], slick=g["slick"], s=s, alias=hit,
                         abstain=lab.get((g["key"], s), []) == ["DONT_KNOW", "DONT_KNOW"]))
D = pd.DataFrame(rows)
I = D.groupby(["key", "arm", "slick"]).agg(prim=("abstain", "mean"), any_alias=("alias", "max")).reset_index()
CELLS = [("uncertain", "Unknown"), ("uncertain", "HighlyKnown"), ("control", "Unknown"), ("control", "HighlyKnown")]


def effects(df, know="slick", B=10_000):
    cells = {c: df[(df.arm == c[0]) & (df[know] == c[1])].prim.to_numpy(float) for c in CELLS}

    def st(c):
        m = {k: v.mean() for k, v in c.items()}
        pop = ((m[CELLS[0]] - m[CELLS[2]]) + (m[CELLS[1]] - m[CELLS[3]])) / 2
        kn = ((m[CELLS[0]] - m[CELLS[1]]) + (m[CELLS[2]] - m[CELLS[3]])) / 2
        return pop - kn
    rng = np.random.default_rng(0)
    bs = [st({k: rng.choice(v, len(v)) for k, v in cells.items()}) for _ in range(B)]
    n = {f"{a[:3]}/{k[:4]}": len(v) for (a, k), v in cells.items()}
    m = {f"{a[:3]}/{k[:4]}": round(float(v.mean()), 3) for (a, k), v in cells.items()}
    return st(cells), np.percentile(bs, 2.5), np.percentile(bs, 97.5), n, m


out = []
say = lambda s: (print(s), out.append(s))
say("(1) popularity-minus-knowledge contrast, two judges")
for name, df in (("all items", I),
                 ("drop unknown questions with a gold alias in any free reply (both unknown cells)", I[~((I.slick == "Unknown") & I.any_alias)]),
                 ("drop such questions from popular-unknown only (worst case)", I[~((I.slick == "Unknown") & (I.arm == "control") & I.any_alias)])):
    d, lo, hi, n, m = effects(df)
    say(f"  {name}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  n {n}  means {m}")
say("  share of unknown questions with a gold alias in any free reply: " + "  ".join(
    f"{a[:3]} {I[(I.arm == a) & (I.slick == 'Unknown')].any_alias.mean():.2f}" for a in ("uncertain", "control")))
E = {}
for l in open(R / "results/round3/e1_llama.jsonl", encoding="utf-8"):
    x = json.loads(l); E[x["key"]] = x["slick_para"]
I["para"] = I.key.map(E)
both = I[I.para.isin(["HighlyKnown", "Unknown"])]
po = (both.slick == both.para).mean()
p1, p2 = (both.slick == "Unknown").mean(), (both.para == "Unknown").mean()
pe = p1 * p2 + (1 - p1) * (1 - p2)
say(f"(2) labels under both (n={len(both)}): agreement {po:.3f}, Cohen's kappa {(po - pe) / (1 - pe):.3f}")
say("    cross-tab:\n" + "\n".join("      " + l for l in pd.crosstab(both.slick, both.para).to_string().splitlines()))
agree = both[both.slick == both.para]
d, lo, hi, n, m = effects(agree)
say(f"    contrast on items where both labellings agree: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  n {n}  means {m}")
say("(3) share of both-judges-DONT_KNOW replies containing a gold alias (subject-name echoes excluded): " + "  ".join(
    f"{a[:3]}/{k[:4]} {D[(D.arm == a) & (D.slick == k) & D.abstain].alias.mean():.2f} (n={int(((D.arm == a) & (D.slick == k) & D.abstain).sum())})"
    for a, k in CELLS))
(HERE / "round3_label_checks.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
