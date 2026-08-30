"""
Jacobian-lens corroboration -- where does the hedge decision become linearly transportable?

Uses Anthropic's prefitted Jacobian Lens (neuronpedia/jacobian-lens) to decode the residual stream at the
readout position of every layer into calibrated logits, and computes the hedge-vs-answer log-odds and the
entropy per layer for both twin arms. Where the two arms' lens log-odds separate = where the decision is
linearly readable -- an observational corroboration of the causal L13-19 / L17-21 routing window, using a
method entirely independent of patching. Optionally decodes the per-layer uncertainty direction into its
top promoted tokens (is the direction "disposed to say" hedging words?).

Usage:
  python scripts/jlens_trajectory.py --category familiarity \
    --lens-file llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt
Outputs: <out-dir>/jlens_trajectory.csv, jlens_direction_tokens.txt, jlens_summary.txt (+ provenance).
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
from collections import defaultdict

import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from circuit_common import Readout, encode_pair, load_pairs, run_capture
from shared.provenance import build_provenance, write_provenance


def main():
    import jlens

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--direction-layers", default="10-25", help="layers whose uncertainty direction is decoded")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    guard_output(out_dir / "jlens_trajectory.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    pairs, how = load_pairs(args.category, args.limit, None)
    print(f"[{args.category}] {len(pairs)} pairs ({how}); lens {args.lens_file}")
    ro = Readout(tokenizer)

    rows = []
    X = defaultdict(lambda: defaultdict(list))  # layer -> arm -> last-position residuals (for direction decode)
    from _common import parse_layer_range
    dlayers = parse_layer_range(args.direction_layers)
    for k, (u, c) in enumerate(pairs):
        for arm, rec in (("uncertain", u), ("control", c)):
            lens_logits, model_logits, _ = lens.apply(wrapped, rec["chat_formatted_prompt"], positions=[-1])
            for l, lg in lens_logits.items():
                v = lg[0] if lg.dim() > 1 else lg
                rows.append(dict(pair=k, prompt_id=rec["prompt_id"], arm=arm, layer=int(l),
                                 lens_logodds=ro.logodds(v.float()).item(), lens_entropy=ro.entropy(v.float()).item()))
            v = model_logits[0] if model_logits.dim() > 1 else model_logits
            rows.append(dict(pair=k, prompt_id=rec["prompt_id"], arm=arm, layer=-1,
                             lens_logodds=ro.logodds(v.float()).item(), lens_entropy=ro.entropy(v.float()).item()))
        enc_u, enc_c, al = encode_pair(model, tokenizer, u, c)
        _, ru = run_capture(model, enc_u, dlayers, "resid"); _, rc = run_capture(model, enc_c, dlayers, "resid")
        for l in dlayers:
            X[l]["u"].append(ru[l][0, -1].float().cpu()); X[l]["c"].append(rc[l][0, -1].float().cpu())
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "jlens_trajectory.csv", index=False)

    piv = df[df.layer >= 0].groupby(["layer", "arm"])[["lens_logodds", "lens_entropy"]].mean().unstack("arm")
    gap_lo = piv[("lens_logodds", "uncertain")] - piv[("lens_logodds", "control")]
    gap_H = piv[("lens_entropy", "uncertain")] - piv[("lens_entropy", "control")]
    final_lo, final_H = gap_lo.iloc[-1], gap_H.iloc[-1]
    half_lo = next((int(l) for l, v in gap_lo.items() if final_lo != 0 and v / final_lo >= 0.5), None)
    half_H = next((int(l) for l, v in gap_H.items() if final_H != 0 and v / final_H >= 0.5), None)
    lines = [f"Jacobian-lens trajectory -- {args.category}; {len(pairs)} pairs; readout position",
             "lens hedge log-odds gap (uncertain - control) by layer:",
             "  " + " ".join(f"L{int(l)}:{v:+.2f}" for l, v in gap_lo.items()),
             "lens entropy gap by layer:",
             "  " + " ".join(f"L{int(l)}:{v:+.2f}" for l, v in gap_H.items()),
             f"first layer reaching 50% of the final gap: log-odds L{half_lo}, entropy L{half_H} (final gaps {final_lo:+.2f}, {final_H:+.2f})"]

    # decode the per-layer uncertainty direction through the lens transport, if the API allows it
    dir_lines = []
    try:
        W_U = model.get_output_embeddings().weight.detach().float()
        for l in dlayers:
            d = torch.stack(X[l]["u"]).mean(0) - torch.stack(X[l]["c"]).mean(0)
            d = (d / (d.norm() + 1e-8)).to(model.device)
            t = lens.transport(d[None, :], l) if hasattr(lens, "transport") else None
            if t is None:
                raise AttributeError("no transport")
            t = t[0] if t.dim() > 1 else t
            logits = (W_U.to(t.device) @ t.float())
            top = logits.topk(10).indices.tolist(); bot = (-logits).topk(10).indices.tolist()
            dir_lines.append(f"L{l:>2} +dir: " + " ".join(repr(tokenizer.decode([i])) for i in top))
            dir_lines.append(f"    -dir: " + " ".join(repr(tokenizer.decode([i])) for i in bot))
    except Exception as e:
        dir_lines.append(f"direction decoding skipped: {type(e).__name__}: {e}")
    (out_dir / "jlens_direction_tokens.txt").write_text("\n".join(dir_lines) + "\n", encoding="utf-8")

    summary = "\n".join(lines); print(summary); print("\n".join(dir_lines[:8]))
    (out_dir / "jlens_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_dir / "jlens_trajectory.csv", build_provenance(model, script="scripts/jlens_trajectory.py", category=args.category, lens=f"{args.lens_repo}/{args.lens_file}", n_pairs=len(pairs)))


if __name__ == "__main__":
    main()
