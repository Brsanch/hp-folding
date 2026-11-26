#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== crossover chain start: $(date) ==="
# Wait for swap-CSP chain to finish
echo "Waiting for swap-CSP chain..."
while pgrep -f "run_43_swap_chain" > /dev/null || pgrep -f "swap_csp.py" > /dev/null; do
  sleep 60
done
echo "swap-CSP chain done at $(date)"

run_crossover() {
  local SEQ=$1; local LVL=$2; local TGT=$3; local PAIRS=$4
  echo ""
  echo "=== $SEQ crossover lvl-$LVL start ($(date)) ==="
  python3 -u scripts/basin_crossover.py \
    --sequence "$SEQ" \
    --source-level "$LVL" \
    --processes 14 \
    --max-pairs "$PAIRS" \
    --splice-points-per-pair 3 \
    --n-polish-steps 200000 \
    --target "$TGT" \
    --update-submission \
    > "crossover_${SEQ}_lvl${LVL}.log" 2>&1
  echo "=== $SEQ done ($(date)) ==="
}

# Order by smallest gap × diversity
run_crossover S7 34 36 210
run_crossover S7 33 36 2000
run_crossover S6 18 21 3000
run_crossover S6 17 21 2500
run_crossover S10 44 50 100
run_crossover S10 42 50 1500
echo "=== chain complete: $(date) ==="
