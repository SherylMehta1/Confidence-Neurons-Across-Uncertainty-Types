#!/bin/sh
# Re-run all of X0: sh results/x0/run_all.sh <python with numpy and pandas>. CPU only; writes only to results/x0 (or $OUT).
set -e
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
for s in build x0_analyses x0_pool x0_a8 x0_a10 x0_fill x0_rerun_repo; do echo "== $s"; "${1:-python3}" -W ignore $s.py > /dev/null; done
