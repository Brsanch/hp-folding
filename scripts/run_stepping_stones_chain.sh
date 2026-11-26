#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== stepping-stone chain start: $(date) ==="
# Order: S8 (now at 40 after promotion, gap -2), S6 (gap -2 with flexibility), S7 (gap -1), S10 (gap -4)
for SEQ in S8 S6 S7 S10; do
  echo ""
  echo "=== $SEQ start ($(date)) ==="
  python3 -u scripts/stepping_stone_attack.py \
    --sequence "$SEQ" \
    --processes 16 \
    --top-k-stones 240 \
    --n-steps 400000 \
    --T-starts 0.5,1.0,1.5,2.0 \
    --update-submission \
    > "stepping_stone_${SEQ}.log" 2>&1
  echo "=== $SEQ done ($(date)) ==="
done
echo "=== chain complete: $(date) ==="
