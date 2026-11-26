#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== S6/S10 crossover chain start: $(date) ==="

run() {
  local SEQ=$1; local LVL=$2; local TGT=$3; local PAIRS=$4
  echo ""
  echo "=== $SEQ lvl-$LVL ($(date)) PAIRS=$PAIRS ==="
  python3 -u scripts/basin_crossover.py \
    --sequence "$SEQ" \
    --source-level "$LVL" \
    --processes 14 \
    --max-pairs "$PAIRS" \
    --splice-points-per-pair 3 \
    --n-polish-steps 100000 \
    --target "$TGT" \
    --update-submission \
    > "crossover_${SEQ}_lvl${LVL}_v2.log" 2>&1 || true
}

# S6 (gap -2, target 21 = record)
run S6 18 21 5000
run S6 17 21 5000
run S6 16 21 4000

# S10 (gap -3, target 50 = record)
run S10 42 50 3000
run S10 40 50 3000
run S10 38 50 3000

echo "=== chain complete: $(date) ==="
