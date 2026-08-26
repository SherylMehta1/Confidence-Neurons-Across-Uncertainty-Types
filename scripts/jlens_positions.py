"""
Entity-span lens maps + interactive-demo export.

(a) For every twin pair: decode ALL positions of both prompts through the prefitted Jacobian lens and track
    the lens rank of " unknown" over the ENTITY span and the readout position, per layer -- where is the
    verbalization born, and when does it move?
(b) For --export-pairs selected pairs: dump the top-k lens tokens per (layer x position) for both twins to
    JSON, for the interactive layer x position demo page.

Usage: python scripts/jlens_positions.py --category familiarity --lens-file <path> [--limit 40 --export-pairs 3]
Outputs: <out-dir>/jlens_entity.csv, jlens_entity_summary.txt, jlens_demo.json (+ provenance).
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import pandas as pd
import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from circuit_common import align, load_pairs
from shared.provenance import build_provenance, write_provenance


def main():
    import jlens

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--category", default="familiarity")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--export-pairs", type=int, default=3)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = REPO_ROOT / (args.out_dir or f"results/circuit_{args.category}")
    guard_output(out_dir / "jlens_entity.csv", args.overwrite)

    model, tokenizer = load_model_from_args(args)
    wrapped = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    unk_id = tokenizer(" unknown", add_special_tokens=False)["input_ids"][0]
    pairs, _ = load_pairs(args.category, args.limit, None)
    print(f"[{args.category}] {len(pairs)} pairs; all-position lens decode")

    rows, demo = [], []
    for k, (u, c) in enumerate(pairs):
        ids_u = tokenizer(u["chat_formatted_prompt"], add_special_tokens=False)["input_ids"]
        ids_c = tokenizer(c["chat_formatted_prompt"], add_special_tokens=False)["input_ids"]
        al = align(ids_u, ids_c)
        spans = dict(u=(al["ent_u"], len(ids_u)), c=(al["ent_c"], len(ids_c)))
        recs = dict(u=u, c=c)
        pair_demo = dict(twin_id=u.get("twin_id"), prompts={}, layers={})
        for arm in ("u", "c"):
            lens_logits, model_logits, _ = lens.apply(wrapped, recs[arm]["chat_formatted_prompt"], positions=None)
            (ent_s, ent_e), n = spans[arm]
            toks = tokenizer.convert_ids_to_tokens(ids_u if arm == "u" else ids_c)
            for l, lg in lens_logits.items():
                m = lg.float()  # [positions, vocab]
                if m.dim() == 1:
                    m = m[None, :]
                order = m.argsort(dim=-1, descending=True)
                for grp, pos_list in (("entity", list(range(ent_s, min(ent_e, m.shape[0])))), ("readout", [m.shape[0] - 1])):
                    for pos in pos_list:
                        rank = int((order[pos] == unk_id).nonzero()[0]) + 1
                        rows.append(dict(pair=k, arm=("uncertain" if arm == "u" else "control"), layer=int(l), group=grp, pos=pos, rank_unknown=rank))
                if k < args.export_pairs:
                    step = max(1, len(lens_logits) // 16)
                    if int(l) % step == 0:
                        top = order[:, : args.topk]
                        pair_demo["layers"].setdefault(str(int(l)), {})[arm] = [[tokenizer.decode([int(t)]) for t in row] for row in top]
        if k < args.export_pairs:
            pair_demo["prompts"] = {"u": u["raw_prompt"], "c": c["raw_prompt"], "u_tokens": tokenizer.convert_ids_to_tokens(ids_u), "c_tokens": tokenizer.convert_ids_to_tokens(ids_c)}
            demo.append(pair_demo)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(pairs)}")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "jlens_entity.csv", index=False)
    (out_dir / "jlens_demo.json").write_text(json.dumps(demo), encoding="utf-8")

    med = df.groupby(["arm", "group", "layer"]).rank_unknown.median().reset_index()
    lines = [f"Entity-span lens map -- {args.category}; median lens rank of ' unknown' by layer"]
    for (arm, grp), g in med.groupby(["arm", "group"]):
        lines.append(f"  {arm:9s} {grp:7s}: " + " ".join(f"L{int(r.layer)}:{int(r.rank_unknown)}" for r in g.itertuples()))
    summary = "\n".join(lines); print(summary)
    (out_dir / "jlens_entity_summary.txt").write_text(summary + "\n", encoding="utf-8")
    write_provenance(out_dir / "jlens_entity.csv", build_provenance(model, script="scripts/jlens_positions.py", category=args.category, n_pairs=len(pairs), export_pairs=args.export_pairs))


if __name__ == "__main__":
    main()
