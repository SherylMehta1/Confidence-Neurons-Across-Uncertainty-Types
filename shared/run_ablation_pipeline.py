"""
shared/run_ablation_pipeline.py -- per-category mean-ablation runner
(successor of shared/run_phase34.py, which is now a shim).

    from shared.run_ablation_pipeline import run_category
    df = run_category(model, tokenizer, candidates, prompts, controls, "lack_of_knowledge",
                      baseline_prompts_for_mean=general_baseline_prompts(),
                      out_dir="results/ablation_v4")

Mean policy (mean_source column):
  - baseline_prompts_for_mean given      -> "general_baseline"  (recommended)
  - otherwise (default)                  -> "pooled_controls": working-split
        uncertain prompts + working-split matched controls of this category
  - mean_source="category_working"       -> this category's own working-split
        uncertain prompts only (the legacy convention; only on explicit request)
All candidates' means are computed in ONE forward pass per baseline prompt, and
the same mean_val is used for working / held_out / control rows of a neuron.

Outputs in out_dir (created if needed):
  results_<category>.csv                 v4 schema (RESULT_COLUMNS), one row per
                                         neuron x prompt (controls flagged by
                                         is_control, split in {working, held_out})
  results_<category>.provenance.json     contract provenance
  ablation_means.json                    {category: {mean_source, baseline hash,
                                         means: {neuron_id: mean_val}}}
Writes are incremental (append after each neuron) and resumable: neurons
already present in the CSV are skipped. An existing CSV is never clobbered
unless overwrite=True; resume=False makes an existing file an error.
"""

import json
import os
import warnings
from pathlib import Path

import pandas as pd

from shared.ablation import (RESULT_COLUMNS, MEAN_SOURCES, compute_mean_activations,
                             run_ablation_experiment)
from shared.detection import load_candidate_neurons  # noqa: F401  (re-export for callers)
from shared.logit_lens import direct_effect_score
from shared.provenance import (build_provenance, sha256_prompts, sha256_file, write_provenance,
                               model_precision)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _neuron_pairs(candidates):
    out = []
    for c in candidates:
        if isinstance(c, dict):
            out.append((int(c["layer"]), int(c["neuron_idx"])))
        else:
            out.append((int(c[0]), int(c[1])))
    return out


def resolve_mean_prompts(prompts, controls, baseline_prompts_for_mean=None, mean_source=None):
    """Return (mean_source, list of prompt strings) per the mean policy."""
    if baseline_prompts_for_mean is not None:
        if mean_source not in (None, "general_baseline"):
            raise ValueError("baseline_prompts_for_mean given but mean_source != 'general_baseline'")
        return "general_baseline", list(baseline_prompts_for_mean)
    if mean_source is None:
        mean_source = "pooled_controls"
    if mean_source == "general_baseline":
        raise ValueError("mean_source='general_baseline' requires baseline_prompts_for_mean")
    working = [p["chat_formatted_prompt"] for p in prompts if p.get("split") == "working"]
    if mean_source == "pooled_controls":
        ctrl = [p["chat_formatted_prompt"] for p in (controls or []) if p.get("split") == "working"]
        if not ctrl:
            warnings.warn("pooled_controls requested but no working-split control prompts were given; "
                          "the mean is effectively category_working")
        return "pooled_controls", working + ctrl
    if mean_source == "category_working":
        return "category_working", working
    raise ValueError(f"mean_source must be one of {MEAN_SOURCES}, got {mean_source!r}")


def _load_existing(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    return pd.read_csv(csv_path)


def run_category(model, tokenizer, candidates, prompts, controls, category,
                 baseline_prompts_for_mean=None, out_dir="results/ablation", overwrite=False,
                 resume=True, mean_source=None, control_neurons=None, positions="last",
                 candidates_path=None, data_paths=None, seed=None, verbose=True):
    """
    Run mean-ablation for every candidate (plus optional extra `control_neurons`,
    list of (layer, idx)) on this category's uncertain prompts and matched
    controls. `candidates` is a list of dicts (load_candidate_neurons output) or
    (layer, idx) pairs; `prompts`/`controls` are lists of data records
    (DATA_SOURCES.md). Returns the full results DataFrame for the category.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"results_{category}.csv"
    means_path = out_dir / "ablation_means.json"

    if csv_path.exists() and overwrite:
        csv_path.unlink()
        for extra in (out_dir / f"results_{category}.provenance.json",):
            if extra.exists():
                extra.unlink()
    existing = _load_existing(csv_path)
    if existing is not None and not resume:
        raise FileExistsError(f"{csv_path} exists; pass overwrite=True to redo or resume=True to continue")
    done_ids = set(existing["neuron_id"].astype(str)) if existing is not None else set()

    cand_pairs = _neuron_pairs(candidates)
    ctrl_pairs = _neuron_pairs(control_neurons or [])
    all_pairs = list(dict.fromkeys(cand_pairs + ctrl_pairs))

    mean_source, mean_prompts = resolve_mean_prompts(prompts, controls, baseline_prompts_for_mean, mean_source)
    mean_hash = sha256_prompts(mean_prompts)

    # Reuse means from a previous (resumed) run if they came from the same baseline.
    # ablation_means.json is keyed by category: {category: {mean_source, hash, means}}.
    all_means = {}
    if means_path.exists():
        try:
            with open(means_path) as f:
                all_means = json.load(f)
        except (OSError, ValueError):
            all_means = {}
    means = None
    prev = all_means.get(category) if not overwrite else None
    if prev and prev.get("mean_source") == mean_source and prev.get("baseline_prompt_sha256") == mean_hash             and all(f"L{l}_N{n}" in prev.get("means", {}) for l, n in all_pairs):
        means = {(l, n): float(prev["means"][f"L{l}_N{n}"]) for l, n in all_pairs}
        if verbose:
            print(f"[{category}] reusing {len(means)} means from {means_path}")
    if means is None:
        if verbose:
            print(f"[{category}] computing means for {len(all_pairs)} neurons on "
                  f"{len(mean_prompts)} prompts ({mean_source})...")
        means = compute_mean_activations(model, tokenizer, all_pairs, mean_prompts, verbose=verbose)
        all_means[category] = {
            "mean_source": mean_source,
            "baseline_prompt_sha256": mean_hash,
            "n_baseline_prompts": len(mean_prompts),
            "precision": model_precision(model),
            "means": {f"L{l}_N{n}": v for (l, n), v in means.items()},
        }
        with open(means_path, "w") as f:
            json.dump(all_means, f, indent=2)

    precision = model_precision(model)
    prov = build_provenance(
        model, category=category, mean_source=mean_source, positions=positions, seed=seed,
        baseline_prompt_sha256=mean_hash, n_baseline_prompts=len(mean_prompts),
        candidate_file_sha256=sha256_file(candidates_path) if candidates_path else None,
        data_file_sha256s={str(p): sha256_file(p) for p in data_paths} if data_paths else None,
        candidate_neurons=[f"L{l}_N{n}" for l, n in cand_pairs],
        control_neurons=[f"L{l}_N{n}" for l, n in ctrl_pairs],
        n_prompts=len(prompts), n_controls=len(controls or []),
        results_csv=str(csv_path), schema_version=4, columns=RESULT_COLUMNS,
    )
    write_provenance(csv_path, prov)

    todo = [(l, n) for l, n in all_pairs if f"L{l}_N{n}" not in done_ids]
    if verbose:
        print(f"[{category}] {len(todo)} neurons to run ({len(done_ids)} already in {csv_path.name})")

    header_needed = existing is None
    for j, (layer, idx) in enumerate(todo):
        de = direct_effect_score(model, layer, idx)
        rows = []
        for recs, is_control in ((prompts, False), (controls or [], True)):
            for split in ("working", "held_out"):
                subset = [r for r in recs if r.get("split") == split]
                if not subset:
                    continue
                rows += run_ablation_experiment(
                    model, tokenizer, subset, layer, idx, means[(layer, idx)], category,
                    split=split, is_control=is_control, mean_source=mean_source,
                    direct_effect_score=de, precision=precision, positions=positions,
                )
        df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        df.to_csv(csv_path, mode="a", header=header_needed, index=False)
        header_needed = False
        if verbose:
            print(f"[{category}] neuron {j + 1}/{len(todo)} L{layer}_N{idx} done ({len(rows)} rows)")

    final = _load_existing(csv_path)
    if final is None:
        final = pd.DataFrame(columns=RESULT_COLUMNS)
    return final


def run_category_from_paths(model, tokenizer, category, prompts_path, controls_path, candidates_path,
                            **kwargs):
    """Convenience: load the jsonl/json files and call run_category."""
    prompts = _load_jsonl(prompts_path)
    controls = _load_jsonl(controls_path) if controls_path and os.path.exists(controls_path) else []
    candidates = load_candidate_neurons(candidates_path)
    return run_category(model, tokenizer, candidates, prompts, controls, category,
                        candidates_path=candidates_path,
                        data_paths=[p for p in (prompts_path, controls_path) if p], **kwargs)
