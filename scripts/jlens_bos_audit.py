"""
Audit: did the committed Jacobian-lens results run on a doubled BOS, and does it change their conclusions?

jlens tokenizes prompts itself with a forced BOS. scripts/jlens_trajectory.py passes chat_formatted_prompt, which
already begins with the BOS text, so the model saw [BOS, BOS, ...]. This script recomputes the lens trajectory on
the same pairs twice -- prompt as committed (doubled BOS) and with the leading BOS text stripped (single BOS) --
and reports, per condition, the twins' lens hedge log-odds gap and entropy gap by layer, the first layer reaching
50% of the final gap (the numbers the paper quotes), and how well each condition's final-layer logits match a
plain single-BOS forward pass.

  python scripts/jlens_bos_audit.py --category familiarity --limit 60 \
      --lens-file llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse

import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, load_model_from_args
from circuit_common import Readout, load_pairs


def half_layer(gap):
    final = gap.iloc[-1]
    return next((int(l) for l, v in gap.items() if final != 0 and v / final >= 0.5), None)


@torch.no_grad()
def main():
    import jlens

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out-dir", default="results/jlens_bos_audit")
    args = ap.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_from_args(args)
    model.eval()
    bos = tokenizer.bos_token
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    pairs, _ = load_pairs(args.category, args.limit, None)
    ro = Readout(tokenizer)
    rows, drift = [], {"doubled": [], "single": []}
    for k, (u, c) in enumerate(pairs):
        for arm, rec in (("uncertain", u), ("control", c)):
            p = rec["chat_formatted_prompt"]
            ref_ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(model.device)
            ref = model(**ref_ids).logits[0, -1].float()
            for cond, text in (("doubled", p), ("single", p[len(bos):] if bos and p.startswith(bos) else p)):
                ll, ml, _ = lens.apply(wrapped, text, positions=[-1])
                mv = (ml[0] if ml.dim() > 1 else ml).float().to(ref.device)
                drift[cond].append((mv - ref).abs().max().item())
                for l, lg in list(ll.items()) + [(-1, ml)]:
                    v = (lg[0] if lg.dim() > 1 else lg).float()
                    rows.append(dict(pair=k, arm=arm, cond=cond, layer=int(l), logodds=ro.logodds(v).item(), entropy=ro.entropy(v).item()))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(pairs)}")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"bos_audit_{args.category}.csv", index=False)
    lines = [f"BOS audit -- {args.category}, {len(pairs)} pairs, lens {args.lens_file}",
             f"tokenizer BOS text: {bos!r}; committed prompts start with it: {pairs[0][0]['chat_formatted_prompt'].startswith(bos) if bos else False}"]
    for cond in ("doubled", "single"):
        d = df[(df.cond == cond) & (df.layer >= 0)]
        piv = d.groupby(["layer", "arm"])[["logodds", "entropy"]].mean().unstack("arm")
        g_lo = piv[("logodds", "uncertain")] - piv[("logodds", "control")]
        g_H = piv[("entropy", "uncertain")] - piv[("entropy", "control")]
        fin = df[(df.cond == cond) & (df.layer == -1)].groupby("arm")[["logodds", "entropy"]].mean()
        lines += [f"\n[{cond} BOS]  max |final logits - single-BOS plain forward|: median {pd.Series(drift[cond]).median():.3f}, max {max(drift[cond]):.3f}",
                  f"  model-output gaps: hedge log-odds {fin.loc['uncertain', 'logodds'] - fin.loc['control', 'logodds']:+.2f}, entropy {fin.loc['uncertain', 'entropy'] - fin.loc['control', 'entropy']:+.2f}",
                  f"  first layer at 50% of final lens gap: log-odds L{half_layer(g_lo)}, entropy L{half_layer(g_H)} (final {g_lo.iloc[-1]:+.2f}, {g_H.iloc[-1]:+.2f})",
                  "  lens log-odds gap: " + " ".join(f"L{int(l)}:{v:+.2f}" for l, v in g_lo.items()),
                  "  lens entropy gap:  " + " ".join(f"L{int(l)}:{v:+.2f}" for l, v in g_H.items())]
    s = "\n".join(lines)
    print(s)
    (out / f"bos_audit_{args.category}.txt").write_text(s + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
