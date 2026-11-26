#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== S10 deeper chain v3 start: $(date) ==="

# Wait for any S10 basin_crossover to finish (correct pattern)
echo "Waiting for any active basin_crossover S10..."
while pgrep -f "basin_crossover.py --sequence S10" > /dev/null; do
  sleep 30
done
echo "ready at $(date)"

run() {
  local LVL=$1; local PAIRS=$2
  echo ""
  echo "=== S10 lvl-$LVL ($(date)) PAIRS=$PAIRS ==="
  python3 -u scripts/basin_crossover.py \
    --sequence S10 \
    --source-level "$LVL" \
    --processes 14 \
    --max-pairs "$PAIRS" \
    --splice-points-per-pair 3 \
    --n-polish-steps 100000 \
    --target 50 \
    --update-submission \
    > "crossover_S10_lvl${LVL}_v3.log" 2>&1 || true
}

run 36 4000
run 34 4000

# Multibasin from new 48 fold (re-mined just now)
echo ""
echo "=== S10 multibasin from level-46 ==="
python3 -u scripts/multibasin_climb.py \
  --sequence S10 \
  --start-level 46 \
  --n-steps 800000 \
  --processes 14 \
  --target 50 \
  --update-submission > multibasin_S10_v3.log 2>&1 || true

echo "=== chain complete: $(date) ==="
