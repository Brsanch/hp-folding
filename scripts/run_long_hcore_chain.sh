#!/bin/bash
# Long-running H-core chain: S7 → S8 → S10 → S6.
# Each gets up to 12h. S5 is provably optimal; S9 is at the theoretical
# upper bound (no improvement possible by H-core packing).
set -e
cd "$(dirname "$0")/.."

# Wait for any existing S7 long_hcore_search to finish before starting next
echo "=== chain start: $(date) ==="

wait_for_done() {
  local SEQ=$1
  while pgrep -f "long_hcore_search.py.*--sequence $SEQ" > /dev/null; do
    sleep 60
  done
}

# S7 is already running; wait for it
echo "Waiting for S7 long_hcore_search to finish..."
wait_for_done S7

for SEQ in S8 S10 S6; do
  echo ""
  echo "=== $SEQ long H-core search start ($(date)) ==="
  python3 -u scripts/long_hcore_search.py \
    --sequence "$SEQ" \
    --processes 14 \
    --time-per-block 30 \
    --h-run-max-placements 50000 \
    --n-blocks 8000 \
    --update-submission \
    --max-wall-hours 12 \
    > "long_hcore_${SEQ}.log" 2>&1
  echo "=== $SEQ done ($(date)) ==="
  wait_for_done "$SEQ"
done

echo ""
echo "=== chain complete: $(date) ==="
