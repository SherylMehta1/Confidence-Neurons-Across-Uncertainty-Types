#!/usr/bin/env bash
# Neuron + circuit tests on the gated familiarity arm (PopQA twins). Run after the gate.
set -euo pipefail  # pipefail matters: stage() pipes through tee, so without it a failed stage would still get a .done flag
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export CN_MODEL_ID="${CN_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
mkdir -p runs/logs
stage() { local name="$1"; shift; if [ -f "runs/$name.done" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee "runs/logs/$name.log"; touch "runs/$name.done"; echo "[done] $name  $(date -u +%H:%M:%S)"; }
stage f10_induction  python scripts/induction_check.py --categories familiarity --overwrite --out results/induction_check_familiarity.csv
stage f20_abl_new    python scripts/run_ablation.py --candidates results/candidate_neurons_bf16.json --categories familiarity --baseline pooled_controls --out-dir results/ablation_bf16_familiarity_new --control-neurons 5 --overwrite
stage f21_abl_dir    python scripts/run_ablation.py --candidates results/candidates_direction_familiarity.json --categories familiarity --baseline pooled_controls --out-dir results/ablation_bf16_familiarity_dir --control-neurons 0 --overwrite
stage f22_abl_key    python scripts/run_ablation.py --candidates results/candidates_keyset.json --categories familiarity --baseline pooled_controls --out-dir results/ablation_bf16_familiarity_key --control-neurons 0 --overwrite
stage f30_twins      python scripts/twin_patching.py --category familiarity --overwrite --out results/twin_patching_familiarity.csv
stage f40_bridge     python scripts/direction_bridge.py --category familiarity --layer-range 20-31 --steer --overwrite --out-dir results/bridge_familiarity
echo "FAMILIARITY_ARM_DONE $(date -u)"; touch runs/FAM_ARM.done
