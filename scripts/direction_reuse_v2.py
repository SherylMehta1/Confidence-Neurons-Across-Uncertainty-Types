"""
Direction reuse, version 2 (direction_reuse_v2_criterion.md): position-matched fit and dose-calibrated steering.

Differences from scripts/direction_reuse.py (version 1, reported as run):
  - directions are fitted at the last position of the prompt that is steered: instruct = format_chat_prompt(q)
    (free generation, no prefill); base = "Question: {q}\nAnswer:" (no prefill);
  - the steering dose k* is the smallest k in {1,2,4,8} x ||d_fam^inst(l*)|| at which the instruct model's OWN
    direction raises hedging >= 0.10 over the mean of 5 random directions at the same k (base direction not used).

  python scripts/direction_reuse_v2.py --pair llama --base-id unsloth/Meta-Llama-3.1-8B --inst-id unsloth/Meta-Llama-3.1-8B-Instruct \
      --base-labels labels_dr/llama_base.jsonl --inst-labels labels_dr/llama_inst.jsonl --out-dir results/direction_reuse_v2/llama
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import gc
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT
from direction_reuse import CELLS, GAP, HEDGE_RE, Steer, cos, directions, first_entropy, generate, labels_for, log, residuals
from jlens_cascade import build_prompt, item_key
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt
from shared.provenance import build_provenance, write_provenance


def free_prompt(tok, q, fmt):
    qq = q.strip() if q.strip().endswith("?") else q.strip() + "?"
    if fmt == "chat":
        return format_chat_prompt(tok, qq)
    bos = tok.bos_token if getattr(tok, "bos_token_id", None) is not None else ""
    return (bos or "") + f"Question: {qq}\nAnswer:"


def degenerate(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return bool(ids) and Counter(ids).most_common(1)[0][1] > 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--base-id", required=True)
    ap.add_argument("--inst-id", required=True)
    ap.add_argument("--base-labels", required=True)
    ap.add_argument("--inst-labels", required=True)
    ap.add_argument("--items", default="data/familiarity_v3/all_measured.jsonl")
    ap.add_argument("--n-steer", type=int, default=150)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--rel-layer", type=float, default=0.55)
    ap.add_argument("--doses", default="1,2,4,8")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    doses = [float(x) for x in args.doses.split(",")]

    items = [json.loads(l) for l in open(REPO_ROOT / args.items, encoding="utf-8") if l.strip()]
    items = [r for r in items if r.get("subj")][: args.limit]
    keys = [item_key(r) for r in items]
    arm = {item_key(r): r["arm"] for r in items}
    perm = np.random.default_rng(0).permutation(len(keys))
    working = set(np.array(keys)[perm[: len(keys) // 2]])

    res = {}
    for tag, mid, fmt, lp in (("base", args.base_id, "base", args.base_labels), ("inst", args.inst_id, "chat", args.inst_labels)):
        model, tok = load_model(mid)
        model.eval()
        lab_prompts = {item_key(r): build_prompt(tok, r["q"], fmt) for r in items}  # label provenance prompts (lens study)
        lab = labels_for(model, tok, items, lab_prompts, lp, out, tag)
        fit_prompts = {item_key(r): free_prompt(tok, r["q"], fmt) for r in items}
        log(f"[{tag}] caching residuals at the free-generation prompt for {len(keys)} items", out)
        H = residuals(model, tok, keys, fit_prompts)
        idx_w = {c: np.array([i for i, k in enumerate(keys) if k in working and (arm[k], lab.get(k)) == c]) for c in CELLS}
        fam, know = directions(H, idx_w)
        res[tag] = dict(H=H, lab=lab, fam=fam, know=know, nl=H.shape[1])
        if tag == "inst":
            res["inst"].update(model=model, tok=tok)
        else:
            del model; gc.collect(); torch.cuda.empty_cache()

    B, I = res["base"], res["inst"]
    L = min(B["nl"], I["nl"])
    ls = int(round(args.rel_layer * (L - 1)))
    geo = pd.DataFrame([dict(layer=l, cos_fam=cos(B["fam"][l], I["fam"][l]), cos_know=cos(B["know"][l], I["know"][l])) for l in range(L)])
    geo.to_csv(out / "geometry.csv", index=False)
    w_idx = np.array([i for i, k in enumerate(keys) if k in working])
    nulls = []
    for p in range(args.n_perm):
        pr = np.random.default_rng(1000 + p)
        a = {keys[i]: arm[keys[i]] for i in w_idx}
        for kn in ("Unknown", "HighlyKnown"):
            ids = [keys[i] for i in w_idx if I["lab"].get(keys[i]) == kn]
            for k, v in zip(ids, pr.permutation([a[k] for k in ids])):
                a[k] = v
        cs = []
        for D in (B, I):
            ic = {c: np.array([i for i in w_idx if (a[keys[i]], D["lab"].get(keys[i])) == c]) for c in CELLS}
            cs.append(directions(D["H"][:, ls:ls + 1], ic)[0][0])
        nulls.append(cos(cs[0], cs[1]))
    g_obs = float(geo.loc[ls, "cos_fam"]); g99 = float(np.percentile(nulls, 99))
    log(f"geometry at l*={ls}: cos = {g_obs:.3f}; null 99th pct {g99:.3f} (mean {np.mean(nulls):.3f})", out)

    model, tok = I["model"], I["tok"]
    held = [r for r in items if item_key(r) not in working and r["arm"] == "control" and I["lab"].get(item_key(r)) == "HighlyKnown"][: args.n_steer]
    nat = np.linalg.norm(I["fam"][ls])
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    rr = np.random.default_rng(7)
    rand = [unit(rr.standard_normal(I["fam"].shape[-1])) for _ in range(args.n_random)]
    st = Steer(model, ls)
    rows = []

    def run(cond, vec, k, with_entropy=False):
        st.vec = None if vec is None else torch.tensor(vec * nat * k, dtype=torch.float32, device=model.device)
        hs = []
        for r in held:
            txt = generate(model, tok, free_prompt(tok, r["q"], "chat"))
            row = dict(key=item_key(r), cond=cond, dose=k, hedged=bool(HEDGE_RE.search(txt)),
                       hedged_amended=bool(HEDGE_RE.search(txt) or GAP.search(txt)), degenerate=degenerate(tok, txt), text=txt)
            if with_entropy:
                row["entropy"] = first_entropy(model, tok, build_prompt(tok, r["q"], "chat"))
            rows.append(row); hs.append(row["hedged"])
        st.vec = None
        h = float(np.mean(hs))
        log(f"  dose {k:g}  {cond:<10} hedge {h:.3f}", out)
        return h

    base_none = run("none", None, 0, with_entropy=True)
    kstar = None
    for k in doses:
        own = run("own_fam", unit(I["fam"][ls]), k)
        rnd = np.mean([run(f"random{j}", v, k) for j, v in enumerate(rand)])
        log(f"dose {k:g}: own - random = {own - rnd:+.3f}", out)
        if own - rnd >= 0.10:
            kstar = k
            break
    if kstar is None:
        verdict = "NOT TESTABLE"
        lines = [f"PAIR {args.pair} (v2): no dose in {doses} gives own - random >= 0.10 -> NOT TESTABLE",
                 f"G cos at l*={ls}: {g_obs:+.3f} (null 99th {g99:+.3f})"]
    else:
        # entropy + remaining conditions at k*
        for c, v in (("own_fam", unit(I["fam"][ls])), ("base_fam", unit(B["fam"][ls])), ("inst_know", unit(I["know"][ls]))):
            run(c + "@k*", v, kstar, with_entropy=True)
        for j, v in enumerate(rand):
            run(f"random{j}@k*", v, kstar, with_entropy=True)
        S = pd.DataFrame(rows)
        ks = S[S.cond.str.endswith("@k*") | (S.cond == "none")].copy()
        ks["c"] = ks.cond.str.replace("@k*", "", regex=False)
        hr = ks.groupby("c").hedged.mean()
        rnd_mean = hr[[c for c in hr.index if c.startswith("random")]].mean()
        own_eff, base_eff = hr["own_fam"] - hr["none"], hr["base_fam"] - hr["none"]
        c1, c2 = g_obs > g99, hr["base_fam"] - rnd_mean >= 0.10
        c3 = own_eff > 0 and base_eff >= 0.5 * own_eff
        verdict = "REUSED" if (c1 and c2 and c3) else "NOT REUSED"
        ph = ks.pivot(index="key", columns="c", values="hedged").astype(float)
        pe = ks.pivot(index="key", columns="c", values="entropy")
        n = len(ph); bb = np.random.default_rng(0)
        boot = lambda f: np.percentile([f(bb.integers(0, n, n)) for _ in range(5000)], [2.5, 97.5])
        dh = lambda ix: (ph["own_fam"].values[ix] - ph["inst_know"].values[ix]).mean()
        de = lambda ix: (pe["inst_know"].values[ix] - pe["own_fam"].values[ix]).mean()
        dh_ci, de_ci = boot(dh), boot(de)
        lines = [f"PAIR {args.pair} (v2): l* = {ls} of {L}; k* = {kstar:g} x natural norm; n = {n}",
                 f"G  cos(d_fam base, inst) = {g_obs:+.3f}; null 99th pct {g99:+.3f} -> condition 1 {'yes' if c1 else 'no'}",
                 "S  hedge at k*: " + "  ".join(f"{c} {v:.3f}" for c, v in hr.items()),
                 f"   random mean {rnd_mean:.3f}; own effect {own_eff:+.3f}; base effect {base_eff:+.3f}; condition 2 {'yes' if c2 else 'no'}; condition 3 {'yes' if c3 else 'no'}",
                 "   degenerate outputs at k*: " + "  ".join(f"{c} {v:.2f}" for c, v in ks.groupby('c').degenerate.mean().items()),
                 "E  entropy at k*: " + "  ".join(f"{c} {v:.3f}" for c, v in pe.mean().items()),
                 f"DD hedge(fam)-hedge(know) {dh(np.arange(n)):+.3f} [{dh_ci[0]:+.3f},{dh_ci[1]:+.3f}]; entropy(know)-entropy(fam) "
                 f"{de(np.arange(n)):+.3f} [{de_ci[0]:+.3f},{de_ci[1]:+.3f}] -> {'yes' if (dh_ci[0] > 0 and de_ci[0] > 0) else 'no'}"]
    lines.append(f"VERDICT {args.pair} (v2): {verdict}")
    for l in lines:
        log(l, out)
    pd.DataFrame(rows).to_csv(out / "steer.csv", index=False)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_provenance(out / "steer.csv", build_provenance(model, script="scripts/direction_reuse_v2.py", **vars(args)))
    log("DIRECTION_REUSE_V2_DONE", out)


if __name__ == "__main__":
    main()
