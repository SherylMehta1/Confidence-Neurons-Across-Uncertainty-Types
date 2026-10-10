"""
nohedge_direction_criterion.md: is Llama's familiarity steering effect carried by familiarity itself, or by a
"hedging intent" component? Fits the familiarity direction only on items the model never hedged on (hedge_rate = 0
over 20 stored free generations) and steers exactly as direction reuse v2 does.

  python scripts/direction_nohedge.py --inst-id unsloth/Meta-Llama-3.1-8B-Instruct --inst-labels labels_dr/llama_inst.jsonl \
      --out-dir results/direction_nohedge/llama
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT
from direction_reuse import CELLS, GAP, HEDGE_RE, Steer, cos, directions, generate, log, residuals
from direction_reuse_v2 import free_prompt
from jlens_cascade import item_key
from shared.model_utils import load_model
from shared.provenance import build_provenance, write_provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-id", required=True)
    ap.add_argument("--inst-labels", required=True)
    ap.add_argument("--items", default="data/familiarity_v3/all_measured.jsonl")
    ap.add_argument("--n-steer", type=int, default=150)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in open(REPO_ROOT / args.items, encoding="utf-8") if l.strip()]
    items = [r for r in items if r.get("subj")]
    keys = [item_key(r) for r in items]
    arm = {item_key(r): r["arm"] for r in items}
    hedge = {item_key(r): r["hedge_rate"] for r in items}
    lab = {x["key"]: x["slick"] for x in map(json.loads, open(REPO_ROOT / args.inst_labels, encoding="utf-8"))}
    perm = np.random.default_rng(0).permutation(len(keys))
    working = set(np.array(keys)[perm[: len(keys) // 2]])  # identical split to direction_reuse(_v2)

    model, tok = load_model(args.inst_id)
    model.eval()
    prompts = {k: free_prompt(tok, r["q"], "chat") for k, r in zip(keys, items)}
    log(f"caching residuals for {len(keys)} items", out)
    H = residuals(model, tok, keys, prompts)
    l = args.layer
    idx = lambda a, kn, nohedge: np.array([i for i, k in enumerate(keys) if k in working and arm[k] == a and lab.get(k) == kn
                                           and (not nohedge or hedge[k] == 0)])
    full = directions(H[:, l:l + 1], {c: idx(c[0], c[1], False) for c in CELLS})[0][0]
    cU = {c: idx(c[0], "Unknown", True) for c in (("uncertain", "Unknown"), ("control", "Unknown"))}
    cK = {c: idx(c[0], "HighlyKnown", True) for c in (("uncertain", "HighlyKnown"), ("control", "HighlyKnown"))}
    dU = H[cU["uncertain", "Unknown"], l].mean(0) - H[cU["control", "Unknown"], l].mean(0)
    dK = H[cK["uncertain", "HighlyKnown"], l].mean(0) - H[cK["control", "HighlyKnown"], l].mean(0)
    dboth = (dU + dK) / 2
    log("working-half zero-hedge cell sizes: " + ", ".join(f"{a}/{k}: {len(v)}" for (a, k), v in {**cU, **cK}.items()), out)
    log(f"cos(d_nohedge_U, d_fam) = {cos(dU, full):+.3f}; cos(d_nohedge_both, d_fam) = {cos(dboth, full):+.3f}", out)

    held = [r for r in items if item_key(r) not in working and r["arm"] == "control" and lab.get(item_key(r)) == "HighlyKnown"][: args.n_steer]
    nat = np.linalg.norm(full)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    rr = np.random.default_rng(7)
    conds = {"none": None, "d_fam": unit(full), "d_nohedge_U": unit(dU), "d_nohedge_both": unit(dboth)}
    for j in range(args.n_random):
        conds[f"random{j}"] = unit(rr.standard_normal(full.shape[-1]))
    st = Steer(model, l)
    rows = []
    for c, v in conds.items():
        st.vec = None if v is None else torch.tensor(v * nat, dtype=torch.float32, device=model.device)
        for r in held:
            txt = generate(model, tok, prompts[item_key(r)])
            rows.append(dict(key=item_key(r), cond=c, hedged=bool(HEDGE_RE.search(txt)), hedged_amended=bool(HEDGE_RE.search(txt) or GAP.search(txt)), text=txt))
        log(f"  {c:<15} hedge {np.mean([x['hedged'] for x in rows if x['cond'] == c]):.3f}", out)
    st.vec = None
    S = pd.DataFrame(rows)
    S.to_csv(out / "steer.csv", index=False)
    ph = S.pivot(index="key", columns="cond", values="hedged").astype(float)
    hr = ph.mean()
    rnd = [c for c in ph.columns if c.startswith("random")]
    rand_mean = hr[rnd].mean()
    effU, effF = hr["d_nohedge_U"] - hr["none"], hr["d_fam"] - hr["none"]
    c1 = hr["d_nohedge_U"] - rand_mean >= 0.10
    c2 = effF > 0 and effU >= 0.5 * effF
    n = len(ph); bb = np.random.default_rng(0)
    diff = (ph["d_nohedge_U"] - ph[rnd].mean(1)).values
    ci = np.percentile([diff[bb.integers(0, n, n)].mean() for _ in range(5000)], [2.5, 97.5])
    verdict = "FAMILIARITY PER SE" if (c1 and c2) else "HEDGING INTENT NOT EXCLUDED"
    lines = [f"Llama-3.1-8B-Instruct, layer {l}, natural norm of d_fam = {nat:.2f}, n = {n}",
             "hedge rate: " + "  ".join(f"{c} {v:.3f}" for c, v in hr.items()),
             f"d_fam effect {effF:+.3f} (v2: +0.267); d_nohedge_U effect {effU:+.3f}; d_nohedge_U - random {diff.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]",
             f"condition 1 (>= 0.10 over random): {'yes' if c1 else 'no'}; condition 2 (>= 50% of d_fam effect): {'yes' if c2 else 'no'}",
             f"VERDICT: {verdict}"]
    for x in lines:
        log(x, out)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_provenance(out / "steer.csv", build_provenance(model, script="scripts/direction_nohedge.py", **vars(args)))
    log("NOHEDGE_DONE", out)


if __name__ == "__main__":
    main()
