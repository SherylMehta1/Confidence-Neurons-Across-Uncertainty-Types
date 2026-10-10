"""Camera-ready tabulations from existing data (chair round 3, items 4 and 8; K21, K23, K25). Post hoc, descriptive.
  python analysis/lockin_checks.py   -> analysis/lockin_checks.txt
(1) known-arm forced-answer entropy and two-judge abstention with 95% item-bootstrap intervals (10,000, seed 0);
(2) two-judge kappa(DONT_KNOW) on Llama with a 95% bootstrap interval over questions (2,000, seed 0), and each judge's
    agreement with the three-judge majority;
(3) per-cell counts of the Llama pool and of the 387-question sample, and the overlap of the steering questions with it."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE.parent.parent, HERE.parent.parent / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
D = REPO / "results/hedge_measure_v2"
OUT = []


def say(s):
    print(s); OUT.append(s)


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def boot(v, B=10_000, seed=0):
    v = np.asarray(v, float); r = np.random.default_rng(seed)
    return v.mean(), *np.percentile(v[r.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5])


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


pool = {}
for r in read(REPO / "data/familiarity_v3/all_measured.jsonl"):
    if r.get("subj"):
        pool.setdefault(f"{r['prop']}::{r['subj']}", r)
items = {r["key"]: r for r in read(D / "items_llama.jsonl")}
say("(1) known arm, Llama")
for arm, nm in (("uncertain", "obscure"), ("control", "popular")):
    e = boot([r["entropy"] for r in pool.values() if r["arm"] == arm and r["slick"] == "HighlyKnown"])
    say(f"  {nm} known forced-answer entropy {e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}]")
J = {}
for jt in ("qwen32b", "gemma27b", "llama70b"):
    for x in read(D / f"judge_{jt}_llama.jsonl"):
        J.setdefault((x["key"], x["s"]), {})[jt] = x["label"] == "DONT_KNOW"
per = {}
for (k, s), v in J.items():
    if "qwen32b" in v and "gemma27b" in v:
        per.setdefault(k, []).append(float(v["qwen32b"] and v["gemma27b"]))
for arm, nm in (("uncertain", "obscure"), ("control", "popular")):
    e = boot([np.mean(per[k]) for k, r in items.items() if r["arm"] == arm and r["slick"] == "HighlyKnown" and k in per])
    say(f"  {nm} known two-judge abstention {e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}]")

say("(2) judges on Llama")
keys = sorted({k for k, _ in J})
by = {k: [v for (kk, _), v in J.items() if kk == k and len(v) == 3] for k in keys}
q = [x["qwen32b"] for k in keys for x in by[k]]; g = [x["gemma27b"] for k in keys for x in by[k]]
k0 = kappa(q, g)
r = np.random.default_rng(0); bs = []
for _ in range(2000):
    s = r.integers(0, len(keys), len(keys))
    qq = [x["qwen32b"] for i in s for x in by[keys[i]]]; gg = [x["gemma27b"] for i in s for x in by[keys[i]]]
    bs.append(kappa(qq, gg))
say(f"  two-judge kappa(DONT_KNOW) {k0:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] over {len(q)} replies with all three labels")
allv = [x for k in keys for x in by[k]]
maj = [sum(x.values()) >= 2 for x in allv]
say("  agreement with the three-judge majority: " + "  ".join(f"{jt} {np.mean([x[jt] == m for x, m in zip(allv, maj)]):.3f}" for jt in ("qwen32b", "gemma27b", "llama70b")))

say("(3) counts")
for arm in ("uncertain", "control"):
    say(f"  pool {arm}: " + "  ".join(f"{lab} {sum(1 for r in pool.values() if r['arm'] == arm and r['slick'] == lab)}" for lab in ("HighlyKnown", "Maybe", "Unknown")))
    say(f"  387-sample {arm}: " + "  ".join(f"{lab} {sum(1 for r in items.values() if r['arm'] == arm and r['slick'] == lab)}" for lab in ("HighlyKnown", "Unknown")))
st = {x["key"] for x in read(REPO / "results/round3/e4/replies_llama.jsonl") if x["split"] == "eval"}
say(f"  steering EVAL questions {len(st)}; also in the 387-question sample {len(st & set(items))}")
(HERE / "lockin_checks.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
