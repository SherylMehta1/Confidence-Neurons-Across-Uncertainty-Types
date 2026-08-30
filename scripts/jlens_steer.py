"""
Steering x lens loop -- push the uncertainty direction causally, watch the verbalization observationally.

Adds alpha * sigma * d (the layer---L uncertainty direction, computed on working pairs) to the residual stream
at the readout position of layer --steer-layer, then decodes EVERY downstream layer through the prefitted
Jacobian lens and tracks the lens rank of the hedge-initial tokens (" unknown", " I", " not") per layer.
If the causal and observational pictures are the same mechanism, steering should make the hedge tokens climb
the lens rankings across the same layers where they climb naturally on uncertain twins.

Usage: python scripts/jlens_steer.py --category familiarity --steer-layer 15 --lens-file <path>
Outputs: <out-dir>/jlens_steer.csv, jlens_steer_summary.txt (+ provenance).
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
from collections import defaultdict

import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from circuit_common import component_module, encode_pair, load_pairs, run_capture
from shared.provenance import build_provenance, write_provenance

HEDGE_WORDS = [" unknown", " I", " not"]


class SteerHook:
    def __init__(self, model, layer, vec):
        self.model, self.layer, self.vec, self.handles = model, layer, vec, []

    def __enter__(self):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = (h[:, -1, :].float() + self.vec.to(h.device)).to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        self.handles.append(component_module(self.model, self.layer, "resid").register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for x in self.handles:
            x.remove()


def main():
    import jlens

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--steer-layer", type=int, default=15)
    ap.add_argument("--alphas", default="-3,0,3")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", required=True)
    ap.add_argument("--limit", type=int, default=40, help="held-out prompts steered (control twins: does +alpha make a KNOWN prompt verbalize doubt?)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    guard_output(out_dir / "jlens_steer.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    L = args.steer_layer
    word_ids = {w: tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in HEDGE_WORDS}

    work, _ = load_pairs(args.category, None, "working")
    held, _ = load_pairs(args.category, args.limit, "held_out")
    us, cs = [], []
    for u, c in work:
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        _, ru = run_capture(model, enc_u, [L], "resid"); _, rc = run_capture(model, enc_c, [L], "resid")
        us.append(ru[L][0, -1].float().cpu()); cs.append(rc[L][0, -1].float().cpu())
    d = torch.stack(us).mean(0) - torch.stack(cs).mean(0)
    d = d / (d.norm() + 1e-8)
    sigma = float(torch.stack([x @ d for x in us + cs]).std())
    print(f"[{args.category}] direction at L{L}, sigma {sigma:.2f}; steering {len(held)} held-out control twins")

    rows = []
    for k, (u, c) in enumerate(held):
        for a in [float(x) for x in args.alphas.split(",")]:
            vec = (a * sigma) * d
            with SteerHook(model, L, vec):
                lens_logits, model_logits, _ = lens.apply(wrapped, c["chat_formatted_prompt"], positions=[-1])
            for l, lg in lens_logits.items():
                if l <= L:
                    continue
                v = (lg[0] if lg.dim() > 1 else lg).float()
                order = v.argsort(descending=True)
                rank = {w: int((order == i).nonzero()[0]) + 1 for w, i in word_ids.items()}
                rows.append(dict(pair=k, alpha=a, layer=int(l), **{f"rank{w.strip()}": rank[w] for w in HEDGE_WORDS}))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(held)}")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "jlens_steer.csv", index=False)

    import numpy as np
    lines = [f"Steering x lens -- {args.category}; direction at L{L}, alphas {args.alphas}; {len(held)} held-out CONTROL twins; median lens rank of hedge tokens by layer"]
    for w in HEDGE_WORDS:
        col = f"rank{w.strip()}"
        lines.append(f"token '{w}':")
        for a, g in df.groupby("alpha"):
            med = g.groupby("layer")[col].median()
            lines.append(f"  a={a:+g}: " + " ".join(f"L{int(l)}:{int(v)}" for l, v in med.items()))
    summary = "\n".join(lines); print(summary)
    (out_dir / "jlens_steer_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_dir / "jlens_steer.csv", build_provenance(model, script="scripts/jlens_steer.py", category=args.category, steer_layer=L, alphas=args.alphas, n=len(held)))


if __name__ == "__main__":
    main()
