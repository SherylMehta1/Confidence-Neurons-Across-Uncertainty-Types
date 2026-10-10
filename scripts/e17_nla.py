"""E17 (paper/PREREG/e17_nla_qwen.md, sha 2d6712e1; exploratory): natural-language autoencoder readout of
Qwen2.5-7B-Instruct's layer-20 residual stream on its 2x2 question sample.

  python scripts/e17_nla.py extract     # base model -> activations at P1 (last prompt token) and P2 (subject's last token)
  python scripts/e17_nla.py explain     # AV: 3 explanations per activation (T = 1.0, 200 tokens)
  python scripts/e17_nla.py critic      # AR: cosine between each explanation's reconstruction and its activation
  python scripts/e17_nla.py judge       # Llama-3.3-70B labels each explanation (A unfamiliar / B uncertain / C names an answer)
Needs the reference package on the path: NLA_INFERENCE_DIR (default /marimo/work/nla-inference).
Outputs in results/round3/e17/."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import gc
import json
import os
import re

import numpy as np
import torch

from _common import REPO_ROOT
from shared.model_utils import load_model
from shared.prompt_format import format_chat_prompt

_sys.path.insert(0, os.environ.get("NLA_INFERENCE_DIR", "/marimo/work/nla-inference"))
OUT = REPO_ROOT / "results/round3/e17"
BASE = "Qwen/Qwen2.5-7B-Instruct"
AV, AR = "kitft/nla-qwen2.5-7b-L20-av", "kitft/nla-qwen2.5-7b-L20-ar"
LAYER = 20  # residual-stream output of block 20 -> hidden_states[21]
JUDGE = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
JUDGE_PROMPT = (
    "An interpretability tool wrote the explanation below of what a language model was internally representing while it "
    "read the question \"{q}\" (about \"{subj}\").\n"
    "Answer three yes/no questions about the EXPLANATION only.\n"
    "A. Does it say or imply that the entity or name is unfamiliar, obscure, unknown, unrecognised or ambiguous?\n"
    "B. Does it say or imply uncertainty, lack of knowledge, or inability to answer the question?\n"
    "C. Does it name a specific candidate answer to the question?\n"
    "Explanation: {e}\n"
    "Reply in exactly this format: A=yes/no B=yes/no C=yes/no")


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


class ListTokenizer:
    """Newer transformers return a dict from apply_chat_template(tokenize=True); nla_inference expects a list of ids.
    This proxy returns the ids as a list and delegates everything else, so the reference checks run unchanged."""

    def __init__(self, tok):
        self._tok = tok

    def apply_chat_template(self, *a, **k):
        r = self._tok.apply_chat_template(*a, **k)
        if k.get("tokenize", True) and not isinstance(r, (list, str)):
            r = r["input_ids"]
            r = r.tolist() if hasattr(r, "tolist") else list(r)
            if r and isinstance(r[0], list):
                r = r[0]
        return r

    def __call__(self, *a, **k):
        return self._tok(*a, **k)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def items():
    pool = {}
    for r in read(REPO_ROOT / "data/familiarity_v3_qwen/all_measured.jsonl"):
        if r.get("subj"):
            pool.setdefault(f"{r['prop']}::{r['subj']}", r)
    out = []
    for r in read(REPO_ROOT / "results/hedge_measure_v2/items_qwen.jsonl"):
        p = pool.get(r["key"], {})
        al = p.get("aliases") or json.loads(p.get("possible_answers") or "[]")
        out.append(dict(r, subj=r["key"].split("::", 1)[1], aliases=al if isinstance(al, list) else json.loads(al)))
    return out


@torch.no_grad()
def extract(a):
    OUT.mkdir(parents=True, exist_ok=True)
    its = items()
    model, tok = load_model(BASE)
    model.eval()
    rows, vecs = [], []
    for it in its:
        q = it["q"].strip(); q = q if q.endswith("?") else q + "?"
        text = format_chat_prompt(tok, q)
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True).hidden_states[LAYER + 1][0].float().cpu()
        positions = {"P1": hs.shape[0] - 1}
        start = text.rfind(it["subj"])
        if start >= 0:
            end = start + len(it["subj"])
            cand = [i for i, (s, e) in enumerate(offs) if s < end and e > start]
            if cand:
                positions["P2"] = cand[-1]
        for pos, i in positions.items():
            rows.append(dict(key=it["key"], arm=it["arm"], slick=it["slick"], q=it["q"], subj=it["subj"], aliases=it["aliases"],
                             pos=pos, tok_index=i, norm=float(hs[i].norm())))
            vecs.append(hs[i].numpy())
    np.save(OUT / "acts.npy", np.stack(vecs).astype(np.float32))
    with open(OUT / "acts_meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"E17_EXTRACT_DONE {len(rows)} activations ({sum(r['pos'] == 'P2' for r in rows)} at P2)", flush=True)


@torch.no_grad()
def explain(a):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla_inference import EXPLANATION_RE, inject_at_marked_positions, load_nla_config, normalize_activation, resolve_embed_scale
    ckpt = snapshot_download(AV)
    tok = ListTokenizer(AutoTokenizer.from_pretrained(ckpt))
    cfg = load_nla_config(ckpt, tok)
    escale = resolve_embed_scale(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ids = torch.tensor(tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)).unsqueeze(0)
    base = (model.get_input_embeddings()(ids.to(model.device)) * escale).float().cpu()
    acts = torch.from_numpy(np.load(OUT / "acts.npy"))
    meta = read(OUT / "acts_meta.jsonl")
    outp = OUT / "explanations.jsonl"
    done = {x["i"] for x in read(outp)}
    todo = [i for i in range(len(meta)) if i not in done]
    print(f"explain: {len(todo)} activations to do; inj_scale {cfg.injection_scale} embed_scale {escale}", flush=True)
    torch.manual_seed(0)
    bs = 24
    with open(outp, "a", encoding="utf-8") as f:
        for b0 in range(0, len(todo), bs):
            b = todo[b0:b0 + bs]
            embs = []
            for i in b:
                v = normalize_activation(acts[i].float().view(1, -1), cfg.injection_scale)
                embs.append(inject_at_marked_positions(ids, base, v, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                                       cfg.injection_right_neighbor_id))
            E = torch.cat(embs, 0).to(model.device, torch.bfloat16)
            o = model.generate(inputs_embeds=E, attention_mask=torch.ones(E.shape[:2], dtype=torch.long, device=model.device),
                               do_sample=True, temperature=1.0, max_new_tokens=200, num_return_sequences=3,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
            txt = tok.batch_decode(o, skip_special_tokens=False)
            for j, i in enumerate(b):
                for s in range(3):
                    raw = txt[j * 3 + s]
                    m = EXPLANATION_RE.search(raw)
                    f.write(json.dumps(dict(i=i, s=s, explanation=(m.group(1).strip() if m else None), raw=raw[:600]), ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {b0 + len(b)}/{len(todo)}", flush=True)
    print("E17_EXPLAIN_DONE", flush=True)


@torch.no_grad()
def critic(a):
    from huggingface_hub import snapshot_download
    from nla_inference import NLACritic
    c = NLACritic(snapshot_download(AR))
    acts = torch.from_numpy(np.load(OUT / "acts.npy"))
    outp = OUT / "critic.jsonl"
    done = {(x["i"], x["s"]) for x in read(outp)}
    with open(outp, "a", encoding="utf-8") as f:
        for x in read(OUT / "explanations.jsonl"):
            if x["explanation"] is None or (x["i"], x["s"]) in done:
                continue
            mse, cos = c.score(x["explanation"], acts[x["i"]])
            f.write(json.dumps(dict(i=x["i"], s=x["s"], mse=mse, cos=cos)) + "\n")
    print("E17_CRITIC_DONE", flush=True)


@torch.no_grad()
def judge(a):
    meta = read(OUT / "acts_meta.jsonl")
    model, tok = load_model(JUDGE)
    model.eval()
    tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    outp = OUT / "labels.jsonl"
    done = {(x["i"], x["s"]) for x in read(outp)}
    todo = [x for x in read(OUT / "explanations.jsonl") if x["explanation"] and (x["i"], x["s"]) not in done]
    print(f"judge: {len(todo)} explanations", flush=True)
    with open(outp, "a", encoding="utf-8") as f:
        for b0 in range(0, len(todo), 16):
            b = todo[b0:b0 + 16]
            texts = [tok.apply_chat_template([{"role": "user", "content": JUDGE_PROMPT.format(q=meta[x["i"]]["q"], subj=meta[x["i"]]["subj"], e=x["explanation"])}],
                                             tokenize=False, add_generation_prompt=True) for x in b]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            o = tok.batch_decode(model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
            for x, raw in zip(b, o):
                lab = {k: (re.search(k + r"\s*=\s*(yes|no)", raw, re.I).group(1).lower() == "yes" if re.search(k + r"\s*=\s*(yes|no)", raw, re.I) else None)
                       for k in ("A", "B", "C")}
                f.write(json.dumps(dict(i=x["i"], s=x["s"], **lab, raw=raw.strip()[:60])) + "\n")
            f.flush()
    print("E17_JUDGE_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "explain", "critic", "judge"])
    a = ap.parse_args()
    {"extract": extract, "explain": explain, "critic": critic, "judge": judge}[a.cmd](a)
