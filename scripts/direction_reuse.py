"""
Does instruction tuning reuse the base model's familiarity direction? (direction_reuse_criterion.md)

For one base/instruct pair:
  1. For each model, cache the last-position residual of every layer on its lens-study prompt (base:
     "Question: ... Answer: The answer is"; instruct: chat template + "The answer is") for the 2,673 PopQA
     candidates, with that model's SliCK labels (--*-labels files, or labelled here by the builder's procedure).
  2. Fit, on the working half (fixed random half, seed 0), the familiarity direction d_fam (obscure - popular,
     knowledge balanced) and the knowledge direction d_know (unknown - known, popularity balanced) per layer.
  3. Geometry: cos(d_fam^base, d_fam^inst) per layer; permutation null at l* (arm labels permuted within knowledge
     cells, the same permutation in both models).
  4. Steering in the instruct model at l* = round(0.55 * (n_layers - 1)): add a vector (scaled to ||d_fam^inst(l*)||)
     to the residual at the last prompt position and every generated position; greedy 24-token free generation on
     150 held-out popular & HighlyKnown prompts (chat template, no prefill); hedging by behavioral_test.HEDGE_RE.
     Also first-token entropy at the prefill prompt under the same vector.

  python scripts/direction_reuse.py --pair llama --base-id unsloth/Meta-Llama-3.1-8B --inst-id unsloth/Meta-Llama-3.1-8B-Instruct \
      --base-labels labels/llama_base.jsonl --inst-labels labels/llama_inst.jsonl --out-dir results/direction_reuse/llama
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import gc
import json
import re

import numpy as np
import pandas as pd
import torch

from _common import REPO_ROOT
from behavioral_test import HEDGE_RE
from build_familiarity_twins import greedy_answer, is_correct, sample_answers, slick_class
from jlens_cascade import build_prompt, item_key
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt
from shared.provenance import build_provenance, write_provenance

GAP = re.compile(r"\bnot (?:readily|immediately|currently|publicly|entirely|widely) (?:available|clear|known)\b", re.I)
CELLS = [("uncertain", "Unknown"), ("uncertain", "HighlyKnown"), ("control", "Unknown"), ("control", "HighlyKnown")]


def log(msg, out):
    print(msg, flush=True)
    with open(out / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def decoder_layers(model):
    return model.model.layers


def labels_for(model, tok, items, prompts, path, out, tag):
    lab = {}
    if path and (REPO_ROOT / path).exists():
        for l in open(REPO_ROOT / path, encoding="utf-8"):
            if l.strip():
                x = json.loads(l); lab[x["key"]] = x["slick"]
    todo = [r for r in items if item_key(r) not in lab]
    log(f"[{tag}] labels: {len(lab)} loaded, labelling {len(todo)} here", out)
    with open(out / f"labels_{tag}_inrun.jsonl", "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            k, p = item_key(r), prompts[item_key(r)]
            aliases = r["aliases"] if isinstance(r["aliases"], list) else json.loads(r["aliases"])
            g = greedy_answer(model, tok, p)
            s = sample_answers(model, tok, p, 10)
            frac = float(np.mean([is_correct(a, aliases) for a in s]))
            lab[k] = slick_class(is_correct(g, aliases), frac)
            f.write(json.dumps(dict(key=k, slick=lab[k], frac_correct=frac)) + "\n"); f.flush()
            if (i + 1) % 250 == 0:
                log(f"  [{tag}] labelled {i + 1}/{len(todo)}", out)
    return lab


@torch.no_grad()
def residuals(model, tok, keys, prompts):
    H = None
    for i, k in enumerate(keys):
        enc = tok(prompts[k], return_tensors="pt", add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]
        v = torch.stack([h[0, -1] for h in hs]).to(torch.float16).cpu()
        if H is None:
            H = torch.empty((len(keys),) + tuple(v.shape), dtype=torch.float16)
        H[i] = v
    return H.float().numpy()  # [N, L, d]


def directions(H, idx_cells):
    m = {c: H[ix].mean(0) for c, ix in idx_cells.items()}
    fam = ((m["uncertain", "Unknown"] - m["control", "Unknown"]) + (m["uncertain", "HighlyKnown"] - m["control", "HighlyKnown"])) / 2
    know = ((m["uncertain", "Unknown"] - m["uncertain", "HighlyKnown"]) + (m["control", "Unknown"] - m["control", "HighlyKnown"])) / 2
    return fam, know  # [L, d]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class Steer:
    """Adds vec to the output of decoder layer `layer` at the last position of a multi-token pass and at every
    position of single-token (decode) passes."""

    def __init__(self, model, layer):
        self.vec = None
        self.h = decoder_layers(model)[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.vec is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        if hs.shape[1] > 1:
            hs[:, -1, :] += self.vec.to(hs.dtype)
        else:
            hs += self.vec.to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


@torch.no_grad()
def generate(model, tok, prompt, n=24):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    enc.pop("token_type_ids", None)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=pad)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def first_entropy(model, tok, prompt):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    lp = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
    return float(-(lp.exp() * lp).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--base-id", required=True)
    ap.add_argument("--inst-id", required=True)
    ap.add_argument("--base-labels", default=None)
    ap.add_argument("--inst-labels", default=None)
    ap.add_argument("--items", default="data/familiarity_v3/all_measured.jsonl")
    ap.add_argument("--n-steer", type=int, default=150)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--rel-layer", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in open(REPO_ROOT / args.items, encoding="utf-8") if l.strip()]
    items = [r for r in items if r.get("subj")][: args.limit]
    keys = [item_key(r) for r in items]
    arm = {item_key(r): r["arm"] for r in items}
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(keys))
    working = set(np.array(keys)[perm[: len(keys) // 2]])

    res = {}
    for tag, mid, fmt, lp in (("base", args.base_id, "base", args.base_labels), ("inst", args.inst_id, "chat", args.inst_labels)):
        model, tok = load_model(mid)
        model.eval()
        prompts = {item_key(r): build_prompt(tok, r["q"], fmt) for r in items}
        lab = labels_for(model, tok, items, prompts, lp, out, tag)
        log(f"[{tag}] caching residuals for {len(keys)} items", out)
        H = residuals(model, tok, keys, prompts)
        idx_w = {c: np.array([i for i, k in enumerate(keys) if k in working and (arm[k], lab.get(k)) == c]) for c in CELLS}
        log(f"[{tag}] working-half cell sizes {{{', '.join(f'{a}/{b}: {len(v)}' for (a, b), v in idx_w.items())}}}", out)
        fam, know = directions(H, idx_w)
        res[tag] = dict(H=H, lab=lab, fam=fam, know=know, idx_w=idx_w, nl=H.shape[1])
        if tag == "inst":
            res["inst"].update(model=model, tok=tok)
        else:
            del model; gc.collect(); torch.cuda.empty_cache()

    B, I = res["base"], res["inst"]
    L = min(B["nl"], I["nl"])
    ls = int(round(args.rel_layer * (L - 1)))
    geo = pd.DataFrame([dict(layer=l, cos_fam=cos(B["fam"][l], I["fam"][l]), cos_know=cos(B["know"][l], I["know"][l]),
                             cos_fam_know_inst=cos(I["fam"][l], I["know"][l])) for l in range(L)])
    geo.to_csv(out / "geometry.csv", index=False)
    # permutation null at l*: permute arms within knowledge cells of the working half (same permutation for both models)
    w_idx = np.array([i for i, k in enumerate(keys) if k in working])
    nulls = []
    for p in range(args.n_perm):
        pr = np.random.default_rng(1000 + p)
        a = {keys[i]: arm[keys[i]] for i in w_idx}
        for kn in ("Unknown", "HighlyKnown"):
            for D in (I,):  # permutation defined on the instruct labels' knowledge cells, applied to both
                ids = [keys[i] for i in w_idx if D["lab"].get(keys[i]) == kn]
                vals = pr.permutation([a[k] for k in ids])
                for k, v in zip(ids, vals):
                    a[k] = v
        cs = []
        for D in (B, I):
            ic = {c: np.array([i for i in w_idx if (a[keys[i]], D["lab"].get(keys[i])) == c]) for c in CELLS}
            cs.append(directions(D["H"][:, ls:ls + 1], ic)[0][0])
        nulls.append(cos(cs[0], cs[1]))
    g_obs = float(geo.loc[ls, "cos_fam"]); g99 = float(np.percentile(nulls, 99))
    log(f"geometry at l*={ls}: cos(d_fam base, inst) = {g_obs:.3f}; permutation null 99th pct {g99:.3f} (mean {np.mean(nulls):.3f})", out)

    # ---- steering in the instruct model ----
    model, tok = I["model"], I["tok"]
    held = [r for r in items if item_key(r) not in working and r["arm"] == "control" and I["lab"].get(item_key(r)) == "HighlyKnown"]
    held = held[: args.n_steer]
    nat = np.linalg.norm(I["fam"][ls])
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    d = I["fam"].shape[-1]
    rr = np.random.default_rng(7)
    conds = {"none": None, "own_fam": unit(I["fam"][ls]) * nat, "base_fam": unit(B["fam"][ls]) * nat,
             "inst_know": unit(I["know"][ls]) * nat, "own_fam_x2": unit(I["fam"][ls]) * nat * 2,
             "base_fam_x2": unit(B["fam"][ls]) * nat * 2}
    for j in range(args.n_random):
        conds[f"random{j}"] = unit(rr.standard_normal(d)) * nat
    st = Steer(model, ls)
    rows = []
    for c, v in conds.items():
        st.vec = None if v is None else torch.tensor(v, dtype=torch.float32, device=model.device)
        for r in held:
            q = r["q"].strip(); q = q if q.endswith("?") else q + "?"
            txt = generate(model, tok, format_chat_prompt(tok, q))
            H1 = first_entropy(model, tok, build_prompt(tok, r["q"], "chat"))
            rows.append(dict(key=item_key(r), cond=c, hedged=bool(HEDGE_RE.search(txt)),
                             hedged_amended=bool(HEDGE_RE.search(txt) or GAP.search(txt)), entropy=H1, text=txt))
        log(f"  steer {c:<12} hedge {np.mean([x['hedged'] for x in rows if x['cond'] == c]):.3f}  "
            f"entropy {np.mean([x['entropy'] for x in rows if x['cond'] == c]):.3f}", out)
    st.vec = None
    S = pd.DataFrame(rows)
    S.to_csv(out / "steer.csv", index=False)

    # ---- verdicts ----
    piv_h = S.pivot(index="key", columns="cond", values="hedged").astype(float)
    piv_e = S.pivot(index="key", columns="cond", values="entropy")
    rnd = [c for c in piv_h.columns if c.startswith("random")]
    hr = piv_h.mean()
    rand_mean = hr[rnd].mean()
    own_eff, base_eff = hr["own_fam"] - hr["none"], hr["base_fam"] - hr["none"]
    c1 = g_obs > g99
    testable = hr["own_fam"] - rand_mean >= 0.10
    c2 = hr["base_fam"] - rand_mean >= 0.10
    c3 = own_eff > 0 and base_eff >= 0.5 * own_eff
    verdict = ("NOT TESTABLE" if not testable else ("REUSED" if (c1 and c2 and c3) else "NOT REUSED"))
    bb = np.random.default_rng(0)
    n = len(piv_h)

    def boot(f):
        v = [f(bb.integers(0, n, n)) for _ in range(5000)]
        return np.percentile(v, [2.5, 97.5])
    dh = lambda ix: (piv_h["own_fam"].values[ix] - piv_h["inst_know"].values[ix]).mean()
    de = lambda ix: (piv_e["inst_know"].values[ix] - piv_e["own_fam"].values[ix]).mean()
    dh_ci, de_ci = boot(dh), boot(de)
    dd = dh_ci[0] > 0 and de_ci[0] > 0
    lines = [f"PAIR {args.pair}: base {args.base_id} | instruct {args.inst_id} | l* = {ls} of {L} layers | n steer prompts = {n}",
             f"G  cos(d_fam base, inst) at l* = {g_obs:+.3f}; permutation 99th pct {g99:+.3f} -> condition 1 {'yes' if c1 else 'no'}",
             f"   cos at all layers: " + " ".join(f"L{l}:{x:+.2f}" for l, x in zip(geo.layer, geo.cos_fam)),
             "S  hedge rate by condition: " + "  ".join(f"{c} {hr[c]:.3f}" for c in hr.index),
             f"   random mean {rand_mean:.3f}; own effect {own_eff:+.3f}; base effect {base_eff:+.3f}; "
             f"testable {testable}; condition 2 {'yes' if c2 else 'no'}; condition 3 {'yes' if c3 else 'no'}",
             f"   amended detector: " + "  ".join(f"{c} {v:.3f}" for c, v in S.groupby('cond').hedged_amended.mean().items()),
             "E  first-token entropy by condition: " + "  ".join(f"{c} {v:.3f}" for c, v in piv_e.mean().items()),
             f"DD hedge(own_fam) - hedge(inst_know) = {dh(np.arange(n)):+.3f} [{dh_ci[0]:+.3f}, {dh_ci[1]:+.3f}]; "
             f"entropy(inst_know) - entropy(own_fam) = {de(np.arange(n)):+.3f} [{de_ci[0]:+.3f}, {de_ci[1]:+.3f}] -> double dissociation {'yes' if dd else 'no'}",
             f"VERDICT {args.pair}: {verdict}"]
    for l in lines:
        log(l, out)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(dict(pair=args.pair, layer=ls, n_layers=L, cos=g_obs, null99=g99, null_mean=float(np.mean(nulls)), hedge=hr.to_dict(),
                   entropy=piv_e.mean().to_dict(), verdict=verdict, c1=bool(c1), c2=bool(c2), c3=bool(c3), testable=bool(testable),
                   dd=bool(dd), dh=[float(dh(np.arange(n)))] + list(map(float, dh_ci)), de=[float(de(np.arange(n)))] + list(map(float, de_ci))),
              open(out / "result.json", "w"), indent=1)
    write_provenance(out / "steer.csv", build_provenance(model, script="scripts/direction_reuse.py", **vars(args)))
    log("DIRECTION_REUSE_DONE", out)


if __name__ == "__main__":
    main()
