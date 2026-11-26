#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== S9+S6 chain start: $(date) ==="

# Wait for S10 chain to finish
echo "Waiting for S10 chain..."
while pgrep -f "basin_crossover.py --sequence S10" > /dev/null \
   || pgrep -f "multibasin_climb.py --sequence S10" > /dev/null; do
  sleep 30
done
echo "S10 chain done at $(date)"

run_xover() {
  local SEQ=$1; local LVL=$2; local TGT=$3; local PAIRS=$4
  echo ""
  echo "=== $SEQ crossover lvl-$LVL ($(date)) PAIRS=$PAIRS ==="
  python3 -u scripts/basin_crossover.py \
    --sequence "$SEQ" \
    --source-level "$LVL" \
    --processes 14 \
    --max-pairs "$PAIRS" \
    --splice-points-per-pair 3 \
    --n-polish-steps 100000 \
    --target "$TGT" \
    --update-submission \
    > "crossover_${SEQ}_lvl${LVL}_v4.log" 2>&1 || true
}

run_multi() {
  local SEQ=$1; local LVL=$2; local TGT=$3
  echo ""
  echo "=== $SEQ multibasin lvl-$LVL ($(date)) ==="
  python3 -u scripts/multibasin_climb.py \
    --sequence "$SEQ" \
    --start-level "$LVL" \
    --n-steps 800000 \
    --processes 14 \
    --target "$TGT" \
    --update-submission > "multibasin_${SEQ}_lvl${LVL}_v4.log" 2>&1 || true
}

# S9 attack: gap -3 to record 53. Use diverse stepping stones at 46/45/44
run_multi S9 47 53
run_multi S9 46 53
run_xover S9 46 53 5000
run_xover S9 45 53 5000

# S6 attack from VERY DIFFERENT basins (low contact level)
# Hypothesis: lvl 17-19 all collapse to same family; lvl 10-12 might escape
run_xover S6 12 21 6000
run_xover S6 10 21 6000
run_xover S6 8 21 6000

echo "=== chain complete: $(date) ==="
