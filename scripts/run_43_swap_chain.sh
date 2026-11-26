#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== 43-swap chain start: $(date) ==="
# Process most-distinct basins first (most likely to admit different swap topologies)
for FOLD in basin_036_c42 basin_063_c42 basin_004_c42 basin_003_c42; do
  echo ""
  echo "=== swap-CSP from $FOLD ($(date)) ==="
  python3 -u scripts/swap_csp.py \
    --sequence S8 \
    --starting-fold "results/multibasin_S8_lvl38/${FOLD}.json" \
    --target-contacts 43 \
    --time-per-test 0.5 \
    --processes 14 \
    --max-q-distance 2 \
    --update-submission > "swap_csp_S8_${FOLD}.log" 2>&1 || true
  echo "=== $FOLD done ($(date)) ==="
  # If submission promoted to 43, stop
  CUR=$(python3 -c "
import json, sys
sys.path.insert(0, 'src')
from hp_folding import BENCHMARKS
from hp_folding.verifier import verify
sub = json.loads(open('submissions/S8.json').read())
print(verify(BENCHMARKS['S8'].sequence, [tuple(c) for c in sub['coords']]).contacts)
")
  if [ "$CUR" -ge "43" ]; then
    echo "*** RECORD BROKEN at $CUR ***"
    break
  fi
done
echo "=== chain complete: $(date) ==="
