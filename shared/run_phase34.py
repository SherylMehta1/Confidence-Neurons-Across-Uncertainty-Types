"""
shared/run_phase34.py -- generic Phase 3 (mechanism check) + Phase 4 (ablation)
runner, used by all three people against the Phase-3-remediated candidate set
and data. This is exactly the kind of "results-generation script" the repo
audit flagged as missing/never-committed for frozen-norm and Stolfo-criteria
work -- writing it once, shared, means nobody re-derives it ad hoc per person
and nobody forgets to commit it.

Usage (from repo root, inside a Kaggle/RunPod session with model+tokenizer
already loaded):

    from shared.run_phase34 import run_category
    run_category(
        model, tokenizer,
        category="lack_of_knowledge",
        prompts_path="data/lack_of_knowledge/prompts.jsonl",
        controls_path="data/lack_of_knowledge/controls.jsonl",
        candidates_path="candidate_neurons.json",   # the NEW, split-half-validated one
        baseline_prompts_for_mean=<list of working-split chat_formatted_prompt strings>,
        out_results_path="person_B_lack_of_knowledge/results/results_v3.csv",
        out_control_results_path="person_B_lack_of_knowledge/results/control_results_v3.csv",
    )
"""

import json
import pandas as pd

from shared.old_detection import load_candidate_neurons
from shared.ablation import compute_mean_activation, run_ablation_experiment


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_category(
    model, tokenizer, category: str,
    prompts_path: str, controls_path: str, candidates_path: str,
    out_results_path: str, out_control_results_path: str,
):
    prompts = _load_jsonl(prompts_path)
    controls = _load_jsonl(controls_path)
    candidates = load_candidate_neurons(candidates_path)

    # mean activation for ablation is computed from this category's OWN
    # working-split prompts (matches the existing project convention --
    # each category ablates using its own baseline, not a shared one).
    working_prompts = [p["chat_formatted_prompt"] for p in prompts if p["split"] == "working"]

    all_rows, all_control_rows = [], []
    for cand in candidates:
        layer, neuron = cand["layer"], cand["neuron_idx"]
        print(f"[{category}] {cand['neuron_id']}: computing mean activation...")
        mean_val = compute_mean_activation(model, tokenizer, working_prompts, layer, neuron)

        for split_name, split_prompts, target_rows in [
            ("working+held_out", prompts, all_rows),
            ("controls", controls, all_control_rows),
        ]:
            print(f"[{category}] {cand['neuron_id']}: ablating on {split_name} "
                  f"({len(split_prompts)} prompts)...")
            for split_val in (["working", "held_out"] if split_name != "controls" else [None]):
                subset = (
                    [p for p in split_prompts if p["split"] == split_val]
                    if split_val is not None else split_prompts
                )
                if not subset:
                    continue
                rows = run_ablation_experiment(
                    model, tokenizer, subset, layer, neuron, mean_val,
                    category=category,
                    split=(split_val if split_val is not None else "control"),
                )
                target_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_results_path, index=False)
    pd.DataFrame(all_control_rows).to_csv(out_control_results_path, index=False)
    print(f"[{category}] wrote {len(all_rows)} rows to {out_results_path}")
    print(f"[{category}] wrote {len(all_control_rows)} rows to {out_control_results_path}")
    return pd.DataFrame(all_rows), pd.DataFrame(all_control_rows)
