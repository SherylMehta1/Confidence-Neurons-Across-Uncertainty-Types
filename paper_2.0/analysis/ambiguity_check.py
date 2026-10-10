"""Post hoc check (review round 3 attack: referential ambiguity / homonymy). For each of Llama's 387 two-judge items,
count Wikidata entities whose English label or alias equals the subject string exactly (wbsearchentities, up to 50
hits), then: (1) homonym counts by cell; (2) linear probability models of two-judge abstention with relation fixed
effects, without and with log(1 + homonyms) and the E6 entity-knowledge score; (3) within obscure items, abstention
vs whether the model identified the entity (E6). Writes analysis/ambiguity_check.{csv,txt}."""
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = next(p for p in (HERE.parents[1], HERE.parents[1] / "Confidence-Neurons-Across-Uncertainty-Types") if (p / "results").is_dir())
cache = HERE / "ambiguity_check.csv"
J = pd.read_csv(R / "results/hedge_measure_v2/item_rates.csv").query("model == 'llama'")[["key", "arm", "slick", "prim"]]
J["prop"] = J.key.str.split("::").str[0]
J["subj"] = J.key.str.split("::", n=1).str[1]
done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["key", "homonyms"])
have = set(done.key)
rows = done.to_dict("records")
for k, s in zip(J.key, J.subj):
    if k in have:
        continue
    url = ("https://www.wikidata.org/w/api.php?action=wbsearchentities&language=en&type=item&limit=50&format=json&search="
           + urllib.parse.quote(s))
    n = None
    for t in range(6):
        try:
            res = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-paper-pilot/1.0"}), timeout=60).read())
            n = sum(1 for h in res.get("search", []) if h.get("match", {}).get("text", "").lower() == s.lower())
            break
        except Exception as e:
            time.sleep(10 * (t + 1))
    rows.append(dict(key=k, homonyms=n))
    pd.DataFrame(rows).to_csv(cache, index=False)
    time.sleep(1.0)
H = pd.DataFrame(rows).dropna()
D = J.merge(H, on="key")
lab = {}
for l in open(R / "results/round3/e6_judge_0.jsonl", encoding="utf-8"):
    x = json.loads(l); lab.setdefault(x["key"], []).append(x["label"] == "CORRECT")
D["ek"] = D.key.map(lambda k: np.mean(lab[k]) if k in lab else np.nan)
D = D.dropna()
D["obscure"] = (D.arm == "uncertain").astype(float)
D["unknown"] = (D.slick == "Unknown").astype(float)
D["log_hom"] = np.log1p(D.homonyms.astype(float))
D["log_hom_z"] = (D.log_hom - D.log_hom.mean()) / D.log_hom.std()
D["ek_z"] = (D.ek - D.ek.mean()) / D.ek.std()
out = []
say = lambda s: (print(s), out.append(s))
say(f"items with homonym counts and entity knowledge: {len(D)}")
for (a, k), g in D.groupby(["arm", "slick"]):
    say(f"  {('obscure' if a == 'uncertain' else 'popular'):8s} {k:11s} n {len(g):3d}  median homonyms {g.homonyms.median():.0f}  "
        f"share with >= 2 {np.mean(g.homonyms >= 2):.2f}  abstention {g.prim.mean():.3f}")


def lpm(df, y, cols, B=2000):
    X = pd.get_dummies(df[cols + ["prop"]], columns=["prop"], drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    Xv, yv = X.to_numpy(float), df[y].to_numpy(float)
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    rng = np.random.default_rng(0)
    bs = np.array([np.linalg.lstsq(Xv[i], yv[i], rcond=None)[0] for i in (rng.integers(0, len(yv), len(yv)) for _ in range(B))])
    return {c: (beta[X.columns.get_loc(c)], *np.percentile(bs[:, X.columns.get_loc(c)], [2.5, 97.5])) for c in cols}


for cols in (["obscure", "unknown"], ["obscure", "unknown", "log_hom_z"], ["obscure", "unknown", "log_hom_z", "ek_z"]):
    r = lpm(D, "prim", cols)
    say("  LPM " + "  ".join(f"{c} {b:+.3f} [{lo:+.3f},{hi:+.3f}]" for c, (b, lo, hi) in r.items()))
O = D[D.arm == "uncertain"]
say(f"within obscure items (n={len(O)}): abstention when the model identified the entity in >= 4 of 6 completions "
    f"{O[O.ek >= 4 / 6].prim.mean():.3f} (n={int((O.ek >= 4 / 6).sum())}) vs otherwise {O[O.ek < 4 / 6].prim.mean():.3f} (n={int((O.ek < 4 / 6).sum())})")
OK = O[O.slick == "HighlyKnown"]
say(f"obscure known only (n={len(OK)}): identified {OK[OK.ek >= 4 / 6].prim.mean():.3f} (n={int((OK.ek >= 4 / 6).sum())}) "
    f"vs not {OK[OK.ek < 4 / 6].prim.mean():.3f} (n={int((OK.ek < 4 / 6).sum())}); single-homonym items {OK[OK.homonyms <= 1].prim.mean():.3f} (n={int((OK.homonyms <= 1).sum())})")
(HERE / "ambiguity_check.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
