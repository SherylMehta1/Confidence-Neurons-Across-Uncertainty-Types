"""X0 shared code: paths, repo functions reused verbatim, bootstraps. CPU only; needs numpy and pandas.

Seed 0 and 10,000 resamples for every interval (95% percentile). Nothing here writes outside OUT (default: this folder).
Cells: OU/OK/PU/PK = obscure/popular x unknown/known. Margin = popularity effect - knowledge effect = OK - PU.
"""
import ast
import json
import os
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("REPO", HERE.parents[1])).resolve()
OUT = Path(os.environ.get("OUT", HERE)).resolve()
OUT.mkdir(parents=True, exist_ok=True)
HM = REPO / "results/hedge_measure_v2"
R3 = REPO / "results/round3"
POOL = REPO / "data/familiarity_v3/all_measured.jsonl"
SEED, B = 0, 10_000
CELLS = ["OK", "OU", "PK", "PU"]
CELL_NAME = {"OK": "obscure known", "OU": "obscure unknown", "PK": "popular known", "PU": "popular unknown"}
CELL_OF = {("uncertain", "HighlyKnown"): "OK", ("uncertain", "Unknown"): "OU", ("control", "HighlyKnown"): "PK", ("control", "Unknown"): "PU"}
BIG3 = ("country", "capital", "sport")


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def repo_functions(path, names, env):
    """Exec only the named top-level defs/assignments of a repo script (avoids importing torch)."""
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    keep = [n for n in tree.body if getattr(n, "name", None) in names
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    exec(compile(ast.Module(keep, []), path, "exec"), env)
    return [env[n] for n in names]


normalize, answer_head, is_correct, slick_class = repo_functions(
    "scripts/build_familiarity_twins.py", ["normalize", "answer_head", "is_correct", "slick_class"], {"re": re, "string": string})
(HEDGE_RE,) = repo_functions("scripts/behavioral_test.py", ["HEDGE_RE"], {"re": re})


def nrm(x):  # alias matching exactly as in paper_2.0/analysis/echo_check.py
    return " " + re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip() + " "


def pool_aliases(r):
    a = r.get("aliases") or json.loads(r.get("possible_answers") or "[]")
    a = a if isinstance(a, list) else json.loads(a)
    return [x for x in a if x]


def kappa(a, b):  # as in scripts/hedge_measure_v2_analysis.py
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


# ---------- bootstraps ----------
def boot_mean(v, rng, size=B, n=None):
    """Means (or, with n given, sums of n[i] draws) of resamples of v with replacement, via multinomial counts."""
    v = np.asarray(v, float)
    vals, cnt = np.unique(v, return_counts=True)
    p = cnt / cnt.sum()
    if n is None:
        return rng.multinomial(len(v), p, size=size) @ vals / len(v)
    return rng.multinomial(n, p) @ vals


def effects(m):
    pop = ((m["OU"] - m["PU"]) + (m["OK"] - m["PK"])) / 2
    know = ((m["OU"] - m["OK"]) + (m["PU"] - m["PK"])) / 2
    return {"popularity effect": pop, "knowledge effect": know, "margin": pop - know,
            "gap among unknown": m["OU"] - m["PU"], "gap among known": m["OK"] - m["PK"]}


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (np.nan, np.nan) if len(x) == 0 else tuple(np.percentile(x, [2.5, 97.5]))


def question_boot(df, y, cells=CELLS, seed=SEED):
    """Resample questions within each cell. Returns point means and bootstrap means per cell."""
    rng = np.random.default_rng(seed)
    v = {c: df.loc[df.cell == c, y].to_numpy(float) for c in cells}
    pt = {c: x.mean() if len(x) else np.nan for c, x in v.items()}
    bs = {c: boot_mean(x, rng) if len(x) else np.full(B, np.nan) for c, x in v.items()}
    return pt, bs


def relation_boot(df, y, two_stage, cells=CELLS, seed=SEED):
    """Resample relations with replacement; if two_stage, then resample questions within each drawn relation x cell.
    Resamples in which a needed cell is empty give nan and are dropped by ci()."""
    rng = np.random.default_rng(seed)
    props = sorted(df.prop.unique())
    W = rng.multinomial(len(props), np.full(len(props), 1 / len(props)), size=B)
    bs = {}
    for c in cells:
        num, den = np.zeros(B), np.zeros(B)
        for j, p in enumerate(props):
            x = df.loc[(df.cell == c) & (df.prop == p), y].to_numpy(float)
            if len(x) == 0:
                continue
            den += W[:, j] * len(x)
            num += boot_mean(x, rng, n=W[:, j] * len(x)) if two_stage else W[:, j] * x.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            bs[c] = num / den
    return bs


def gap_ci(df, y, a="OU", b="PU", seed=SEED):
    pt, bs = question_boot(df, y, cells=[a, b], seed=seed)
    return (pt[a] - pt[b], *ci(bs[a] - bs[b]))


def fmt(e, lo, hi, d=3):
    return f"{e:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


class Log:
    def __init__(self, name):
        self.path, self.lines = OUT / name, []

    def __call__(self, s=""):
        print(s)
        self.lines.append(str(s))

    def save(self):
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def save(df, name, **kw):
    df.to_csv(OUT / name, index=False, float_format="%.6g", **kw)
