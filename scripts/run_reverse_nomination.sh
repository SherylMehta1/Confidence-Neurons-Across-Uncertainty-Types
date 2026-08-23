#!/usr/bin/env bash
# E1: reverse nomination -- find entropy / frequency neurons from the weights, then verify them
# causally with exactly the same tests used for the correlation-selected candidates.
# Usage: CN_MODEL_ID=... bash scripts/run_reverse_nomination.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export CN_MODEL_ID="${CN_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
mkdir -p runs/logs
stage() { local name="$1"; shift; if [ -f "runs/$name.done" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee "runs/logs/$name.log"; touch "runs/$name.done"; echo "[done] $name  $(date -u +%H:%M:%S)"; }

stage r00_scan      python scripts/reverse_nomination.py --k 20 --overwrite
# causal verification on both nominee sets, both references
stage r10_abl_ent   python scripts/run_ablation.py --candidates results/candidates_entropy_weights.json   --baseline general         --out-dir results/ablation_bf16_entropy_weights   --control-neurons 5 --overwrite
stage r11_abl_frq   python scripts/run_ablation.py --candidates results/candidates_frequency_weights.json --baseline general         --out-dir results/ablation_bf16_frequency_weights --control-neurons 5 --overwrite
stage r12_abl_entP  python scripts/run_ablation.py --candidates results/candidates_entropy_weights.json   --baseline pooled_controls --out-dir results/ablation_bf16_entropy_weights_pooled --control-neurons 0 --overwrite
# mechanism signatures: frozen-norm (entropy nominees), temperature-matched frequency (both sets), dose-response (entropy nominees, lack-of-knowledge)
ENT=$(python -c "import json;print(','.join(c['neuron_id'] for c in json.load(open('results/candidates_entropy_weights.json'))['candidates']))")
FRQ=$(python -c "import json;print(','.join(c['neuron_id'] for c in json.load(open('results/candidates_frequency_weights.json'))['candidates']))")
stage r20_frozen    python scripts/frozen_norm.py --neurons "$ENT" --categories lack_of_knowledge --include-controls --baseline pooled_controls --overwrite --out results/frozen_norm_entropy_weights.csv
stage r21_dose      python scripts/dose_response.py --neurons "$ENT" --values=-2,-1,0,1,2 --categories lack_of_knowledge --include-controls --overwrite --out results/dose_response_entropy_weights.csv
stage r30_freq_ent  python scripts/frequency_causal.py --neurons "$ENT" --control-neurons 10 --categories lack_of_knowledge --out results/frequency_causal_entropy_weights.csv --overwrite
stage r31_freq_frq  python scripts/frequency_causal.py --neurons "$FRQ" --control-neurons 10 --categories lack_of_knowledge --out results/frequency_causal_frequency_weights.csv --overwrite
stage r40_stolfo    python analysis/stolfo_criteria.py --candidates results/candidates_entropy_weights.json --out results/stolfo_entropy_weights --overwrite
echo "REVERSE_NOMINATION_COMPLETE $(date -u)"; touch runs/REVERSE.done
