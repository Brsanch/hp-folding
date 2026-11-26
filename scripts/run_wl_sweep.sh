#!/bin/bash
# WL sweep across S6, S7, S9, S10 with full telemetry.
set -e
cd "$(dirname "$0")/.."
for SEQ in S6 S7 S9 S10; do
  echo "=== $SEQ WL sweep start ==="
  date
  python3 scripts/wang_landau_parallel.py \
    --sequence "$SEQ" \
    --processes 16 \
    --trials 64 \
    --time-budget 240 \
    --diag-dir "diagnostics/wl" \
    2>&1 | tee "wl_${SEQ}_run.log"
  echo "=== $SEQ WL sweep done ==="
  date
done
