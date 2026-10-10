"""E16 (paper/PREREG/e15_e16_olmo_and_steering.md, sha 2b3e665d): E8 with the abstention direction fitted within the
POPULAR arm (no obscure entities, no popularity contrast) and 20 random directions; Llama only.

  python scripts/e16_steer.py gen --model llama --kstar 4      # k* from E4 (same fam, random dirs and SELECT split)
  python scripts/e16_steer.py judge --model llama --judges qwen32b,gemma27b,llama70b
Outputs in results/round3/e8/."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import gc
import json

import numpy as np
import torch

from _common import REPO_ROOT
from direction_reuse import CELLS, HEDGE_RE, Steer, directions, residuals
from e4_steer import JUDGES, MODELS, degenerate, free_prompt, gen_batch
from hedge_measure_v2 import PROMPT
from jlens_cascade import item_key
from shared.model_utils import load_model

OUT = REPO_ROOT / "results/round3/e16"


def log(msg):
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def gen(a):
    mid, lab_path, pool_path = MODELS[a.model]
    torch.manual_seed(0)
    items = [json.loads(l) for l in open(REPO_ROOT / "data/familiarity_v3/all_measured.jsonl", encoding="utf-8") if l.strip()]
    items = [r for r in items if r.get("subj")]
    keys = [item_key(r) for r in items]
    arm = {item_key(r): r["arm"] for r in items}
    lab = {}
    for l in open(REPO_ROOT / lab_path, encoding="utf-8"):
        if l.strip():
            x = json.loads(l); lab[x["key"]] = x["slick"]
    hrate = {}
    for l in open(REPO_ROOT / pool_path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("subj"):
                hrate[f"{r['prop']}::{r['subj']}"] = r["hedge_rate"]
    # identical splits to E4
    perm = np.random.default_rng(0).permutation(len(keys))
    fit = set(np.array(keys)[perm[: len(keys) // 2]])
    rest = [r for r in items if item_key(r) not in fit and r["arm"] == "control" and lab.get(item_key(r)) == "HighlyKnown"]
    order = np.random.default_rng(0).permutation(len(rest))
    ev = [rest[i] for i in order[150:300]]

    model, tok = load_model(mid)
    model.eval()
    L = model.config.num_hidden_layers
    ls = int(round(0.55 * (L - 1)))
    fit_keys = [k for k in keys if k in fit]
    fp = {item_key(r): free_prompt(tok, r["q"]) for r in items}
    H = residuals(model, tok, fit_keys, fp)[:, ls]
    idx = {c: np.array([i for i, k in enumerate(fit_keys) if (arm[k], lab.get(k)) == c]) for c in CELLS}
    fam, _ = directions(H[:, None, :], idx)
    fam = fam[0]
    # the one change from E4: abstain-minus-answer within the obscure arm
    ab_hi = np.array([i for i, k in enumerate(fit_keys) if arm[k] == "control" and hrate.get(k) is not None and hrate[k] >= 0.5])
    ab_lo = np.array([i for i, k in enumerate(fit_keys) if arm[k] == "control" and hrate.get(k) == 0])
    abst_w = H[ab_hi].mean(0) - H[ab_lo].mean(0)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    uw = unit(abst_w)
    fam_perp_p = fam - (fam @ uw) * uw
    nat = float(np.linalg.norm(fam))
    rr = np.random.default_rng(0)
    dirs = {"fam": unit(fam), "abst_pop": uw, "fam_perp_p": unit(fam_perp_p)}
    rnd = {f"random{j}": unit(rr.standard_normal(fam.shape[-1])) for j in range(20)}
    geo = dict(model=a.model, layer=ls, n_layers=L, kstar=a.kstar, nat_norm_fam=nat, nat_norm_abst_pop=float(np.linalg.norm(abst_w)),
               n_abst_hi=int(len(ab_hi)), n_abst_lo=int(len(ab_lo)), cos_fam_abst_pop=float(unit(fam) @ uw))
    log(f"[{a.model}] geometry {json.dumps(geo)}")
    json.dump(geo, open(OUT / f"geometry_{a.model}.json", "w"), indent=1)

    st = Steer(model, ls)
    rep = open(OUT / f"replies_{a.model}.jsonl", "a", encoding="utf-8")
    conds = [("none", None)] + list(rnd.items()) + list(dirs.items())
    for cond, d in conds:
        k = 0 if d is None else a.kstar
        st.vec = None if d is None else torch.tensor(d * nat * k, dtype=torch.float32, device=model.device)
        outs = gen_batch(model, tok, [fp[item_key(r)] for r in ev])
        st.vec = None
        hs = []
        for r, samples in zip(ev, outs):
            for s, t in enumerate(samples):
                h = bool(HEDGE_RE.search(t)); hs.append(h)
                rep.write(json.dumps(dict(model=a.model, split="eval", cond=cond, dose=k, key=item_key(r), q=r["q"], s=s, text=t,
                                          regex=h, degenerate=degenerate(tok, t))) + "\n")
        rep.flush()
        log(f"  [{a.model}] eval {cond:<11} dose {k:g}: regex {np.mean(hs):.3f}")
    rep.close(); st.h.remove()
    log(f"E16_GEN_DONE {a.model}")


@torch.no_grad()
def judge(a):
    reps = [json.loads(l) for l in open(OUT / f"replies_{a.model}.jsonl", encoding="utf-8") if l.strip()]
    for jt in a.judges.split(","):
        outp = OUT / f"judge_{jt}_{a.model}.jsonl"
        done = {(x["cond"], x["key"], x["s"]) for x in map(json.loads, open(outp, encoding="utf-8"))} if outp.exists() else set()
        todo = [r for r in reps if (r["cond"], r["key"], r["s"]) not in done]
        log(f"[judge {jt} {a.model}] {len(todo)} replies to label")
        model, tok = load_model(JUDGES[jt])
        model.eval()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        with open(outp, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), 16):
                b = todo[i:i + 16]
                texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(question=r["q"], reply=r["text"])}],
                                                 tokenize=False, add_generation_prompt=True) for r in b]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                o = model.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id)
                for r, raw in zip(b, tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    raw = raw.strip().upper()
                    labl = next((c for c in ("DONT_KNOW", "CLARIFY", "ANSWER") if raw.startswith(c) or raw.startswith(c.replace("_", " "))), "UNPARSED")
                    f.write(json.dumps(dict(model=r["model"], cond=r["cond"], key=r["key"], s=r["s"], raw=raw, label=labl)) + "\n")
                f.flush()
        del model; gc.collect(); torch.cuda.empty_cache()
        log(f"JUDGED {jt} {a.model}")
    log(f"E16_JUDGE_DONE {a.model}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--model", choices=list(MODELS), required=True); g.add_argument("--kstar", type=float, required=True)
    j = sub.add_parser("judge"); j.add_argument("--model", choices=list(MODELS), required=True)
    j.add_argument("--judges", default="qwen32b,gemma27b,llama70b")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gen(a) if a.cmd == "gen" else judge(a)


if __name__ == "__main__":
    main()
