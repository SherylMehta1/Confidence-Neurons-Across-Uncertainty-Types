"""
shared/provenance.py -- one place to build and write the provenance JSON that
every artifact-writing function in this repo emits next to its output.

Contract (see RESULTS_SCHEMA.md): an artifact `foo.csv` / `foo.json` gets a
sibling `foo.provenance.json` containing, where applicable:

    model_id, precision ("bf16" | "nf4"), quant_config, dtype,
    transformers_version, torch_version,
    candidate_file_sha256, data_file_sha256s, baseline_prompt_sha256,
    n_baseline_prompts, seed, layer_range, top_k_per_half, top_k_final,
    min_abs_corr, git_head_sha, timestamp

Use `build_provenance(model=..., **extras)` to fill the generic part and
`write_provenance(artifact_path, prov)` to write the sibling file.
"""

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path):
    """sha256 of a file's bytes, or None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_prompts(prompts):
    """Order-independent sha256 of a list of prompt strings (sorted, JSON-joined).
    Matches the legacy detection hash recipe, but returns the full 64-hex digest
    (the legacy candidate_neurons.json stored only the first 16 characters)."""
    blob = json.dumps(sorted(prompts)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def git_head_sha(repo_root=REPO_ROOT):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def model_precision(model):
    """'nf4' | 'bf16' | 'fp16' | 'fp32' -- prefers model.cn_precision set by load_model."""
    if model is None:
        return None
    p = getattr(model, "cn_precision", None)
    if p:
        return p
    if getattr(getattr(model, "config", None), "quantization_config", None) is not None:
        return "nf4"
    try:
        dt = next(model.parameters()).dtype
    except StopIteration:
        return None
    return {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}.get(str(dt), str(dt))


def model_dtype(model):
    if model is None:
        return None
    try:
        return str(next(model.parameters()).dtype)
    except StopIteration:
        return None


def build_provenance(model=None, **extras):
    """Generic provenance block (model/version/git/timestamp) merged with extras.
    Unknown values are recorded as None rather than omitted, so consumers can
    rely on the key set."""
    import torch
    import transformers

    quant = None
    if model is not None:
        qc = getattr(getattr(model, "config", None), "quantization_config", None)
        if qc is not None:
            try:
                quant = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
            except Exception:
                quant = str(qc)
    model_id = None
    if model is not None:
        model_id = getattr(model, "cn_model_id", None) or getattr(model, "name_or_path", None) \
            or getattr(getattr(model, "config", None), "_name_or_path", None)

    prov = {
        "model_id": model_id,
        "precision": model_precision(model),
        "quant_config": quant,
        "dtype": model_dtype(model),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "candidate_file_sha256": None,
        "data_file_sha256s": None,
        "baseline_prompt_sha256": None,
        "n_baseline_prompts": None,
        "seed": None,
        "layer_range": None,
        "top_k_per_half": None,
        "top_k_final": None,
        "min_abs_corr": None,
        "git_head_sha": git_head_sha(),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    prov.update(extras)
    return prov


def provenance_path(artifact_path):
    p = Path(artifact_path)
    return p.with_name(p.stem + ".provenance.json")


def write_provenance(artifact_path, prov):
    """Write `prov` as the sibling provenance JSON of artifact_path; returns its path."""
    out = provenance_path(artifact_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(prov, f, indent=2, default=str)
    return out


def data_file_hashes(paths):
    return {str(p): sha256_file(p) for p in paths}
