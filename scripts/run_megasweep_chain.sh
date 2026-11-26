#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== megasweep chain start: $(date) ==="

# Wait for S8 megasweep to finish
echo "Waiting for S8 megasweep..."
while pgrep -f "wang_landau_parallel.*S8.*megasweep" > /dev/null; do
  sleep 60
done
echo "S8 finished at $(date)"

# S10 megasweep
echo "=== S10 megasweep start ($(date)) ==="
python3 -u scripts/wang_landau_parallel.py \
  --sequence S10 \
  --processes 16 \
  --trials 1024 \
  --time-budget 180 \
  --diag-dir diagnostics/wl_S10_megasweep \
  --init-from submissions/S10.json \
  --target 50 > wl_S10_megasweep.log 2>&1
echo "=== S10 finished ($(date)) ==="

# S7 megasweep — bigger budget needed since the basin is so dominant
echo "=== S7 megasweep start ($(date)) ==="
python3 -u scripts/wang_landau_parallel.py \
  --sequence S7 \
  --processes 16 \
  --trials 1024 \
  --time-budget 180 \
  --diag-dir diagnostics/wl_S7_megasweep \
  --init-from submissions/S7.json \
  --target 36 > wl_S7_megasweep.log 2>&1
echo "=== S7 finished ($(date)) ==="

echo "=== chain complete: $(date) ==="
