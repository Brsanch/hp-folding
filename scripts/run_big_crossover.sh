#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== big crossover chain start: $(date) ==="

# Wait for current crossover to finish
echo "Waiting for any active basin_crossover..."
while pgrep -f "basin_crossover.py" > /dev/null; do
  sleep 30
done
echo "ready at $(date)"

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
    > "crossover_${SEQ}_lvl${LVL}_big.log" 2>&1 || true
}

# S7 (gap -1, big diversity)
run S7 33 36 5000
run S7 32 36 5000
run S7 30 36 4000

# S6 (gap -2, biggest diversity)
run S6 18 21 5000
run S6 17 21 5000
run S6 15 21 4000

# S10 (gap -3, smaller diversity)
run S10 42 50 3000
run S10 40 50 3000

echo "=== chain complete: $(date) ==="
