#!/usr/bin/env bash
# Clean end-to-end rerun of the Confidence-Neurons pipeline in bf16.
# Usage: CN_MODEL_ID=unsloth/Meta-Llama-3.1-8B-Instruct bash scripts/run_all.sh [stage]
# Stages run in order; each writes <stage>.done in runs/ so a rerun resumes. Logs in runs/logs/.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export CN_MODEL_ID="${CN_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
mkdir -p runs/logs; START="${1:-}"
stage() { local name="$1"; shift; if [ -f "runs/$name.done" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee "runs/logs/$name.log"; touch "runs/$name.done"; echo "[done] $name  $(date -u +%H:%M:%S)"; }
SIG="L31_N2477"   # Phase-2 held-out survivor; always included in the mechanism stages

stage 00_tests      python -m pytest tests -q
stage 10_data_A     python person_A_ambiguity/preprocess_ambiguity.py --overwrite
stage 11_screen_B   python person_B_lack_of_knowledge/screen_templates.py
stage 12_data_B     python person_B_lack_of_knowledge/rebuild_lack_of_knowledge_whitelisted.py
stage 13_data_C     python person_C_contradictory_context/preprocess_contradictory_context.py
stage 20_induction  python scripts/induction_check.py --overwrite
stage 30_detect     python scripts/detect.py --overwrite --out results/candidate_neurons_bf16.json --distribution-out results/full_correlation_distribution_bf16.json
stage 40_ablate_new python scripts/run_ablation.py --candidates results/candidate_neurons_bf16.json --baseline general --out-dir results/ablation_bf16_new --control-neurons 5 --overwrite
stage 41_ablate_old python scripts/run_ablation.py --candidates results/candidates_old15.json  --baseline general --out-dir results/ablation_bf16_old15 --control-neurons 5 --overwrite
stage 42_ablate_v3c python scripts/run_ablation.py --candidates candidate_neurons.json           --baseline general --out-dir results/ablation_bf16_v3set --control-neurons 5 --overwrite
stage 50_stolfo_new python analysis/stolfo_criteria.py --candidates results/candidate_neurons_bf16.json --out results/stolfo_bf16_new --overwrite
stage 51_stolfo_v3  python analysis/stolfo_criteria.py --candidates candidate_neurons.json --out results/stolfo_v3set --overwrite
stage 52_stolfo_old python analysis/stolfo_criteria.py --candidates results/candidates_old15.json --out results/stolfo_old15 --overwrite
stage 60_mech_new   python scripts/mechanism_check.py --candidates results/candidate_neurons_bf16.json --overwrite
stage 70_frozen     python scripts/frozen_norm.py --neurons "$SIG" --include-controls --baseline general --overwrite --out results/frozen_norm_bf16.csv
stage 71_dose       python scripts/dose_response.py --neurons "$SIG" --values=-2,-1,0,1,2 --include-controls --overwrite --out results/dose_response_bf16.csv
echo "ALL STAGES COMPLETE $(date -u)"; touch runs/ALL.done
