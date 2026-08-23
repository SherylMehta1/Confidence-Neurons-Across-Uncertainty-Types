#!/usr/bin/env bash
# One-shot overnight run on a fresh sandbox: judge audit of the alias grader, then the circuit pipeline per arm.
#   MODEL=llama bash scripts/run_overnight.sh      # Llama-3.1-8B-Instruct: arms familiarity + conflict, window 12-26
#   MODEL=qwen  bash scripts/run_overnight.sh      # Qwen2.5-7B-Instruct: arm conflict (familiarity has only 24 pairs), window 10-22
# Requires data/<arm>/{prompts,controls}.jsonl for the model in question (for Qwen, copy results/second_model/qwen25_7b_instruct/data/* to data/).
# Idempotent through runs/<stage>.done; everything logs to runs/logs/ and the master log passed by the launcher.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
MODEL="${MODEL:-llama}"
case "$MODEL" in
  llama) export CN_MODEL_ID="${CN_MODEL_ID:-unsloth/Meta-Llama-3.1-8B-Instruct}"; ARMS="${ARMS:-familiarity conflict}"; WINDOW="${WINDOW:-12-26}"; JUDGE_CATS="${JUDGE_CATS:-familiarity,conflict,aleatoric}" ;;
  qwen)  export CN_MODEL_ID="${CN_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"; ARMS="${ARMS:-conflict}"; WINDOW="${WINDOW:-10-22}"; JUDGE_CATS="${JUDGE_CATS:-familiarity,conflict,aleatoric}" ;;
  *) echo "MODEL must be llama or qwen"; exit 2 ;;
esac
export ARMS WINDOW
mkdir -p runs/logs
stage() { local name="$1"; shift; if [ -f "runs/$name.done" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name  $(date -u +%H:%M:%S)"; "$@" 2>&1 | tee "runs/logs/$name.log"; touch "runs/$name.done"; echo "[done] $name  $(date -u +%H:%M:%S)"; }
if [ "${SKIP_JUDGE:-0}" != "1" ]; then
  stage "j1_judge_$MODEL" python scripts/judge_audit.py --categories "$JUDGE_CATS" --judge "${JUDGE:-Qwen/Qwen2.5-32B-Instruct}" --n-per-stratum "${N_PER_STRATUM:-60}" --out-dir results --overwrite
fi
bash scripts/run_circuit.sh
echo "OVERNIGHT_DONE $(date -u)"; touch runs/OVERNIGHT.done
