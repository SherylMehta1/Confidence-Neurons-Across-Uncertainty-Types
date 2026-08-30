#!/usr/bin/env bash
# Sandbox A chain (Llama-3.1-8B-Instruct, existing data/results already in the clone):
#   E1 reverse nomination (+ causal verification) -> E2 direction bridge (+ steering check)
#   -> E5 twin patching -> E6 seed sweep -> E4 multi-neuron behavioral (needs E1's nominee files)
# Usage: CN_MODEL_ID=unsloth/Meta-Llama-3.1-8B-Instruct bash scripts/run_sandbox_A.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export CN_MODEL_ID="${CN_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
mkdir -p runs/logs
stage() { local name="$1"; shift; if [ -f "runs/$name.done" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee "runs/logs/$name.log"; touch "runs/$name.done"; echo "[done] $name  $(date -u +%H:%M:%S)"; }

stage a00_tests     python -m pytest tests -q
stage a10_reverse   bash scripts/run_reverse_nomination.sh
stage a20_bridge    python scripts/direction_bridge.py --category lack_of_knowledge --layer-range 20-31 --steer --overwrite
stage a30_twins     python scripts/twin_patching.py --category lack_of_knowledge --overwrite
for s in 1 2 3 4 5; do
  stage a4${s}_seed   python scripts/detect.py --seed $s --overwrite --out results/seed_sweep/candidate_neurons_seed$s.json --distribution-out results/seed_sweep/full_correlation_distribution_seed$s.json
done
stage a50_multi     python scripts/multi_neuron_behavioral.py --sets freq=results/candidates_frequency_weights.json entropy=results/candidates_entropy_weights.json corrfreq=results/token_frequency_neurons_top.json key=L31_N11541,L31_N6772 --control-sets 2 --overwrite
echo "SANDBOX_A_COMPLETE $(date -u)"; touch runs/A.done
