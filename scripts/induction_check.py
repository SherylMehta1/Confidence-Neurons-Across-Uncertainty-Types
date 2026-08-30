"""
scripts/induction_check.py -- induction-quality check on ALL prompts (not a
25-prompt sample): per category, mean/median next-token entropy and top-1
probability for uncertain vs matched control prompts, both splits, plus a
Welch t-test and Mann-Whitney U on top-1 and entropy.

  python scripts/induction_check.py --categories all

Writes results/induction_check.csv (per-category summary),
results/induction_check_per_prompt.csv (every prompt) and provenance.
"""

import argparse

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe: make _common importable
from _common import REPO_ROOT, add_model_args, data_paths, guard_output, load_category, load_model_from_args, parse_categories


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--categories", default="all")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "induction_check.csv"))
    args = ap.parse_args(argv)
    out = guard_output(args.out, args.overwrite)
    per_prompt_out = out.with_name(out.stem + "_per_prompt.csv")

    from scipy import stats
    from shared.model_utils import get_next_token_probs, compute_entropy, compute_top1_prob
    from shared.provenance import build_provenance, write_provenance, data_file_hashes

    model, tokenizer = load_model_from_args(args)
    cats = parse_categories(args.categories)
    rows, files = [], []
    for cat in cats:
        prompts, controls = load_category(cat)
        files += list(data_paths(cat))
        for recs, is_ctrl in ((prompts, False), (controls, True)):
            for r in recs:
                probs = get_next_token_probs(model, tokenizer, r["chat_formatted_prompt"])
                top_id = int(probs.argmax())
                rows.append({"category": cat, "prompt_id": r["prompt_id"], "split": r["split"],
                             "is_control": is_ctrl, "entropy": compute_entropy(probs),
                             "top1_prob": compute_top1_prob(probs), "top1_token": tokenizer.decode([top_id])})
        print(f"[{cat}] {len(prompts)} uncertain + {len(controls)} control prompts scored")

    pp = pd.DataFrame(rows)
    pp.to_csv(per_prompt_out, index=False)

    summary = []
    for cat in cats:
        d = pp[pp.category == cat]
        u, c = d[~d.is_control], d[d.is_control]
        row = {"category": cat, "n_uncertain": len(u), "n_control": len(c)}
        for m in ("entropy", "top1_prob"):
            row[f"uncertain_mean_{m}"] = u[m].mean()
            row[f"control_mean_{m}"] = c[m].mean() if len(c) else np.nan
            row[f"uncertain_median_{m}"] = u[m].median()
            row[f"control_median_{m}"] = c[m].median() if len(c) else np.nan
            row[f"gap_{m}"] = row[f"uncertain_mean_{m}"] - row[f"control_mean_{m}"]
            if len(c) > 1 and len(u) > 1:
                row[f"welch_p_{m}"] = float(stats.ttest_ind(u[m], c[m], equal_var=False).pvalue)
                row[f"mannwhitney_p_{m}"] = float(stats.mannwhitneyu(u[m], c[m]).pvalue)
                row[f"cohens_d_{m}"] = float((u[m].mean() - c[m].mean()) /
                                             np.sqrt((u[m].var(ddof=1) + c[m].var(ddof=1)) / 2))
        for split in ("working", "held_out"):
            row[f"uncertain_mean_top1_{split}"] = u[u.split == split]["top1_prob"].mean()
            row[f"control_mean_top1_{split}"] = c[c.split == split]["top1_prob"].mean() if len(c) else np.nan
        summary.append(row)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out, index=False)
    write_provenance(out, build_provenance(model, categories=cats, data_file_sha256s=data_file_hashes(files),
                                           per_prompt_csv=str(per_prompt_out)))
    pd.set_option("display.width", 200)
    print(sdf[["category", "n_uncertain", "n_control", "uncertain_mean_top1_prob", "control_mean_top1_prob",
               "gap_top1_prob", "uncertain_mean_entropy", "control_mean_entropy"]].to_string(index=False))
    print(f"wrote {out} and {per_prompt_out}")


if __name__ == "__main__":
    main()
