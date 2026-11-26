#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== multi-basin chain start: $(date) ==="
# S7 (gap -1, 88+ distinct basins at level 37/35) — most promising
# S10 (gap -3, deep room, 47→50 possible)
# S6 (gap -2, 19→21 possible)
# S9 (gap -3, hard since at upper bound)
for SEQ in S7 S10 S6 S9; do
  TARGET=$(python3 -c "
import sys; sys.path.insert(0, 'src')
from hp_folding import BENCHMARKS
print(BENCHMARKS['$SEQ'].best_known)
")
  # Pick start level appropriately for each sequence (a few below current best)
  case $SEQ in
    S7) START=33 ;;
    S10) START=44 ;;
    S6) START=17 ;;
    S9) START=48 ;;
  esac
  echo ""
  echo "=== $SEQ start ($(date)) target=$TARGET start_level=$START ==="
  python3 -u scripts/multibasin_climb.py \
    --sequence "$SEQ" \
    --start-level "$START" \
    --n-steps 600000 \
    --processes 14 \
    --target "$TARGET" \
    --update-submission \
    > "multibasin_${SEQ}.log" 2>&1 || true
  echo "=== $SEQ done ($(date)) ==="
done
echo "=== chain complete: $(date) ==="
