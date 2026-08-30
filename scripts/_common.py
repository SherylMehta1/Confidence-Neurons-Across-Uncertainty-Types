"""Shared helpers for the scripts/ entry points: repo-root resolution (so every
script runs from any CWD), data loading, model loading, output guards."""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CATEGORIES = ("ambiguity", "lack_of_knowledge", "contradictory_context")
DEFAULT_MODEL_ID = os.environ.get("CN_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")  # CN_MODEL_ID overrides


def data_paths(category):
    d = REPO_ROOT / "data" / category
    return d / "prompts.jsonl", d / "controls.jsonl"


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_category(category):
    """(prompts, controls) record lists for a category (controls may be empty)."""
    pp, cp = data_paths(category)
    return load_jsonl(pp), load_jsonl(cp)


def parse_categories(arg):
    cats = [c.strip() for c in (arg or "all").split(",") if c.strip()]
    if cats == ["all"]:
        return list(CATEGORIES)
    for c in cats:
        if c not in CATEGORIES and not (REPO_ROOT / "data" / c / "prompts.jsonl").exists():
            raise SystemExit(f"unknown category {c!r}; choose from {CATEGORIES} or any data/<name>/prompts.jsonl")
    return cats


def parse_neurons(arg):
    """'L31_N2477,L30_N3533' -> [(31, 2477), (30, 3533)]"""
    out = []
    for tok in (arg or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        l, n = tok.split("_")
        out.append((int(l[1:]), int(n[1:])))
    if not out:
        raise SystemExit("--neurons must list at least one neuron like L31_N2477")
    return out


def parse_layer_range(arg):
    """'20-31' -> range(20, 32); '20,21,25' -> [20, 21, 25]"""
    if "-" in arg:
        lo, hi = arg.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in arg.split(",") if x.strip()]


def add_model_args(ap):
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--quantize", action="store_true", help="4-bit NF4 (development only)")
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32", "nf4"), default=None,
                    help="unquantized dtype (default bf16); nf4 == --quantize")
    ap.add_argument("--overwrite", action="store_true")


def load_model_from_args(args):
    import torch
    from shared.model_utils import load_model
    quantize = args.quantize or args.precision == "nf4"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, None: None,
             "nf4": None}[args.precision]
    print(f"Loading {args.model_id} ({'nf4' if quantize else (args.precision or 'bf16')})...")
    model, tokenizer = load_model(model_id=args.model_id, quantize=quantize, dtype=dtype)
    return model, tokenizer


def guard_output(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
