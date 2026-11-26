#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== constraint GA chain start: $(date) ==="

# Wait for s9_s6 chain to finish
echo "Waiting for s9_s6 chain..."
while pgrep -f "run_s9_s6_chain" > /dev/null \
   || pgrep -f "basin_crossover.py --sequence S9" > /dev/null \
   || pgrep -f "basin_crossover.py --sequence S6" > /dev/null \
   || pgrep -f "multibasin_climb.py --sequence S9" > /dev/null; do
  sleep 60
done
echo "Ready at $(date)"

# Constraint GA on S6 (gap -2, hardest sequence — needs different approach)
echo ""
echo "=== S6 constraint GA ($(date)) ==="
python3 -u scripts/constraint_ga.py \
  --sequence S6 \
  --population 60 \
  --generations 50 \
  --n-steps-fitness 50000 \
  --processes 14 \
  --seed-from-current \
  --target 21 \
  --update-submission > constraint_ga_S6.log 2>&1 || true
echo "=== S6 done ($(date)) ==="

# Constraint GA on S9 (gap -3, at upper bound, hard but possible)
echo ""
echo "=== S9 constraint GA ($(date)) ==="
python3 -u scripts/constraint_ga.py \
  --sequence S9 \
  --population 60 \
  --generations 40 \
  --n-steps-fitness 50000 \
  --processes 14 \
  --seed-from-current \
  --target 53 \
  --update-submission > constraint_ga_S9.log 2>&1 || true
echo "=== S9 done ($(date)) ==="

echo "=== chain complete: $(date) ==="
