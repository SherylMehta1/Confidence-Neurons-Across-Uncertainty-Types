"""
Add the first-token hedge-vs-answer readout to every item of a familiarity pool (forward pass only).

build_familiarity_twins.py records first-token entropy and top-1 probability at the "The answer is"
prefill, but not the hedge-vs-answer log-odds that every circuit number is reported on. This script
adds it, using circuit_common.Readout unchanged (direction_bridge.HEDGE_TOKENS vs ANSWER_TOKENS at the
last prompt position), so the factorial can be read on the same scale as the circuit results. It also
records the total probability on the hedge tokens, so a log-odds shift can be judged against how
much mass it actually moves.

Reads  data/<pool>/all_measured.jsonl  (uses each item's stored `chat` prompt verbatim)
Writes data/<pool>/hedge_logodds.jsonl  keyed by (prop, subj), plus a provenance sidecar.

  python scripts/add_hedge_logodds.py --model-id Qwen/Qwen2.5-7B-Instruct --pool familiarity_v3_qwen
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # PYTHONSAFEPATH-safe
import argparse
import json

import torch

from _common import REPO_ROOT, add_model_args, guard_output, load_model_from_args
from circuit_common import Readout
from shared.model_utils import tokenize_prompt
from shared.provenance import build_provenance, write_provenance


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--pool", required=True, help="data/<pool>/all_measured.jsonl")
    args = ap.parse_args()
    src = REPO_ROOT / "data" / args.pool / "all_measured.jsonl"
    out = REPO_ROOT / "data" / args.pool / "hedge_logodds.jsonl"
    guard_output(out, args.overwrite)
    model, tokenizer = load_model_from_args(args)
    model.eval()
    ro = Readout(tokenizer)
    print(f"hedge ids {ro.hedge_ids} answer ids {ro.answer_ids}")
    items = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    with open(out, "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            enc = tokenize_prompt(tokenizer, it["chat"], device=model.device)
            logits = model(**enc).logits[0, -1]
            lp = torch.log_softmax(logits.float(), -1)
            rec = dict(prop=it["prop"], subj=it["subj"], hedge_logodds=float(ro.logodds(logits)),
                       entropy_recomputed=float(ro.entropy(logits)),
                       hedge_mass=float(lp[ro.hedge_ids].exp().sum()), answer_mass=float(lp[ro.answer_ids].exp().sum()))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(items)}")
    write_provenance(out, build_provenance(model, script="scripts/add_hedge_logodds.py", pool=args.pool, n_items=len(items)))
    print(f"wrote {len(items)} rows to {out}")


if __name__ == "__main__":
    main()
