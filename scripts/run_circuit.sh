#!/usr/bin/env bash
# Overnight circuit pipeline on the gated twin arms (steps 1-6 of the circuit plan).
#   CN_MODEL_ID=unsloth/Meta-Llama-3.1-8B-Instruct ARMS="familiarity conflict" bash scripts/run_circuit.sh
#   CN_MODEL_ID=Qwen/Qwen2.5-7B-Instruct ARMS="conflict" WINDOW=10-22 bash scripts/run_circuit.sh
# Each stage is idempotent (runs/<stage>.done); logs in runs/logs/. Stages: 1 position map, 2+3 heads, 4 MLP/neurons,
# 5 faithfulness (+6 overlap with the first arm when there are two).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export CN_MODEL_ID="${CN_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
ARMS="${ARMS:-familiarity}"; LIMIT="${LIMIT:-80}"; MAP_LIMIT="${MAP_LIMIT:-60}"; VERIFY="${VERIFY:-30}"; WINDOW="${WINDOW:-12-26}"
N_HEADS="${N_HEADS:-20}"; N_NEURONS="${N_NEURONS:-100}"; TAG="${TAG:-}"
SLUG="$(echo "$CN_MODEL_ID" | tr -c "A-Za-z0-9" "_")${TAG}"  # stage markers are per model + tag, so a second model never skips
mkdir -p runs/logs
stage() { local name="$1"; shift; if [ -f "runs/$name.done" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee "runs/logs/$name.log"; touch "runs/$name.done"; echo "[done] $name  $(date -u +%H:%M:%S)"; }
first=""
for arm in $ARMS; do
  out="results/circuit_${arm}${TAG}"
  stage "x1_map_${arm}_${SLUG}"   python scripts/circuit_position_map.py --category "$arm" --limit "$MAP_LIMIT" --out-dir "$out" --overwrite
  stage "x2_heads_${arm}_${SLUG}" python scripts/circuit_heads.py --category "$arm" --limit "$LIMIT" --verify "$VERIFY" --out-dir "$out" --overwrite
  stage "x4_mlp_${arm}_${SLUG}"   python scripts/circuit_mlp.py --category "$arm" --limit "$LIMIT" --window "$WINDOW" --out-dir "$out" --overwrite
  if [ -n "$first" ]; then cmp="--compare results/circuit_${first}${TAG}"; else cmp=""; fi
  stage "x5_faith_${arm}_${SLUG}" python scripts/circuit_faithfulness.py --category "$arm" --circuit-dir "$out" --n-heads "$N_HEADS" --n-neurons "$N_NEURONS" $cmp --overwrite
  [ -z "$first" ] && first="$arm"
done
echo "CIRCUIT_DONE $(date -u)"; touch runs/CIRCUIT.done
