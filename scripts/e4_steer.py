"""E4 (paper/PREREG/e4_steering_v3.md, sha 18c74883): steering with held-out dose selection, an abstention-orthogonal
direction, and two-judge scoring.

  python scripts/e4_steer.py gen --model llama   # then --model gemma9b
  python scripts/e4_steer.py judge               # both judges over all EVAL replies
Outputs in results/round3/e4/."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import gc
import json
from collections import Counter

import numpy as np
import torch

from _common import REPO_ROOT
from direction_reuse import CELLS, HEDGE_RE, Steer, directions, residuals
from hedge_measure_v2 import PROMPT
from jlens_cascade import build_prompt, item_key
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt

MODELS = {"llama": ("unsloth/Meta-Llama-3.1-8B-Instruct", "labels_dr/llama_inst.jsonl", "data/familiarity_v3/all_measured.jsonl"),
          "gemma9b": ("unsloth/gemma-2-9b-it", "labels_dr/gemma9b_inst.jsonl", "data/familiarity_v3_gemma/all_measured.jsonl")}
JUDGES = {"qwen32b": "Qwen/Qwen2.5-32B-Instruct", "gemma27b": "unsloth/gemma-2-27b-it",
          "llama70b": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"}  # llama70b: secondary, PREREG e4_amendment_third_judge.md
OUT = REPO_ROOT / "results/round3/e4"
DOSES = [0.5, 1, 2, 4, 8]


def log(msg):
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def free_prompt(tok, q):
    q = q.strip() if q.strip().endswith("?") else q.strip() + "?"
    return format_chat_prompt(tok, q)


def degenerate(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"][:24]
    return bool(ids) and Counter(ids).most_common(1)[0][1] > 8


@torch.no_grad()
def gen_batch(model, tok, prompts, n=3, max_new=96, bs=16):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        enc.pop("token_type_ids", None)
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, num_return_sequences=n,
                           pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs += [txt[j * n:(j + 1) * n] for j in range(len(txt) // n)]
    return outs


@torch.no_grad()
def entropies(model, tok, prompts, bs=16):
    tok.padding_side = "left"
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        enc.pop("token_type_ids", None)
        lp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
        res += (-(lp.exp() * lp).sum(-1)).tolist()
    return res


def gen(a):
    OUT.mkdir(parents=True, exist_ok=True)
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
    perm = np.random.default_rng(0).permutation(len(keys))
    fit = set(np.array(keys)[perm[: len(keys) // 2]])
    rest = [r for r in items if item_key(r) not in fit and r["arm"] == "control" and lab.get(item_key(r)) == "HighlyKnown"]
    order = np.random.default_rng(0).permutation(len(rest))
    sel = [rest[i] for i in order[:150]]
    ev = [rest[i] for i in order[150:300]]
    log(f"[{a.model}] FIT {len(fit)}, popular-known outside FIT {len(rest)}, SELECT {len(sel)}, EVAL {len(ev)}")

    model, tok = load_model(mid)
    model.eval()
    L = model.config.num_hidden_layers
    ls = int(round(0.55 * (L - 1)))
    fit_keys = [k for k in keys if k in fit]
    fp = {item_key(r): free_prompt(tok, r["q"]) for r in items}
    H = residuals(model, tok, fit_keys, fp)[:, ls]  # [N, d] at l*
    idx = {c: np.array([i for i, k in enumerate(fit_keys) if (arm[k], lab.get(k)) == c]) for c in CELLS}
    fam, know = directions(H[:, None, :], idx)
    fam, know = fam[0], know[0]
    ab_hi = np.array([i for i, k in enumerate(fit_keys) if hrate.get(k) is not None and hrate[k] >= 0.5])
    ab_lo = np.array([i for i, k in enumerate(fit_keys) if hrate.get(k) == 0])
    abst = H[ab_hi].mean(0) - H[ab_lo].mean(0)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    ua = unit(abst)
    fam_perp = fam - (fam @ ua) * ua
    nat = float(np.linalg.norm(fam))
    rr = np.random.default_rng(0)
    dirs = {"fam": unit(fam), "know": unit(know), "abst": ua, "fam_perp": unit(fam_perp)}
    dirs.update({f"random{j}": unit(rr.standard_normal(fam.shape[-1])) for j in range(5)})
    geo = dict(model=a.model, layer=ls, n_layers=L, nat_norm_fam=nat, nat_norm_know=float(np.linalg.norm(know)),
               nat_norm_abst=float(np.linalg.norm(abst)), n_abst_hi=int(len(ab_hi)), n_abst_lo=int(len(ab_lo)),
               cos_fam_abst=float(unit(fam) @ ua), cos_fam_know=float(unit(fam) @ unit(know)))
    log(f"[{a.model}] geometry {json.dumps(geo)}")
    json.dump(geo, open(OUT / f"geometry_{a.model}.json", "w"), indent=1)

    st = Steer(model, ls)
    rep = open(OUT / f"replies_{a.model}.jsonl", "a", encoding="utf-8")

    def run(split, rows, cond, k, with_entropy=False):
        vec = None if cond == "none" else dirs[cond] * nat * k
        st.vec = None if vec is None else torch.tensor(vec, dtype=torch.float32, device=model.device)
        outs = gen_batch(model, tok, [fp[item_key(r)] for r in rows])
        ent = entropies(model, tok, [build_prompt(tok, r["q"], "chat") for r in rows]) if with_entropy else [None] * len(rows)
        st.vec = None
        hs = []
        for r, samples, e in zip(rows, outs, ent):
            for s, t in enumerate(samples):
                h = bool(HEDGE_RE.search(t)); hs.append(h)
                rep.write(json.dumps(dict(model=a.model, split=split, cond=cond, dose=k, key=item_key(r), q=r["q"], s=s, text=t,
                                          regex=h, degenerate=degenerate(tok, t), entropy=e)) + "\n")
        rep.flush()
        log(f"  [{a.model}] {split} {cond:<9} dose {k:g}: regex {np.mean(hs):.3f}")
        return float(np.mean(hs))

    # dose-response on SELECT (all directions), then k* from fam vs random mean
    curve = {}
    for k in DOSES:
        for c in dirs:
            curve[(c, k)] = run("select", sel, c, k)
    kstar = next((k for k in DOSES if curve[("fam", k)] - np.mean([curve[(f"random{j}", k)] for j in range(5)]) >= 0.10), None)
    log(f"[{a.model}] k* = {kstar}")
    json.dump(dict(kstar=kstar), open(OUT / f"kstar_{a.model}.json", "w"))
    if kstar is not None:
        run("eval", ev, "none", 0, with_entropy=True)
        for c in dirs:
            run("eval", ev, c, kstar, with_entropy=True)
    rep.close(); st.h.remove()
    log(f"E4_GEN_DONE {a.model}")


@torch.no_grad()
def judge(a):
    reps = []
    for m in MODELS:
        p = OUT / f"replies_{m}.jsonl"
        if p.exists():
            reps += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    reps = [r for r in reps if r["split"] == "eval"]
    for jt in a.judges.split(","):
        jid = JUDGES[jt]
        outp = OUT / f"judge_{jt}.jsonl"
        done = {(x["model"], x["cond"], x["key"], x["s"]) for x in map(json.loads, open(outp, encoding="utf-8"))} if outp.exists() else set()
        todo = [r for r in reps if (r["model"], r["cond"], r["key"], r["s"]) not in done]
        log(f"[judge {jt}] {len(todo)} replies to label")
        model, tok = load_model(jid)
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
    log("E4_JUDGE_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--model", choices=list(MODELS), required=True)
    j = sub.add_parser("judge"); j.add_argument("--judges", default="qwen32b,gemma27b")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gen(a) if a.cmd == "gen" else judge(a)


if __name__ == "__main__":
    main()
