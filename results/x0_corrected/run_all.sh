#!/bin/sh
# Rebuild X0 with the corrected Gemma-27B labels: sh results/x0_corrected/run_all.sh <python with numpy, pandas, scipy, torch> <empty scratch dir>
# 1. shadow repo with corrected labels and regenerated summaries; 2. all X0 scripts on it. Writes only to results/x0_corrected and the scratch dir.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; TOP="$(cd "$HERE/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
"$1" "$TOP/scripts/rejudge_shadow.py" "$2" corrected > "$HERE/shadow_log.txt"
export REPO="$2" OUT="$HERE"
cp "$TOP/results/x0/prereg_table.csv" "$HERE/"   # plans and hashes: independent of the labels
cd "$HERE"
for s in build x0_analyses x0_pool x0_a8 x0_a10 r0_margin_montecarlo r1_counts r3_within_relation r2_intervals x0_fill x0_rerun_repo; do echo "== $s"; "$1" $s.py > /dev/null; done
