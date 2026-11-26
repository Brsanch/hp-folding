#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== mass GPU chain start: $(date) ==="
for SEQ in S7 S8 S10 S6; do
  echo ""
  echo "=== $SEQ start ($(date)) ==="
  python3 -u scripts/mass_gpu_attack.py \
    --sequence "$SEQ" \
    --K-gpu 4096 \
    --gpu-steps 200000 \
    --gpu-rounds 6 \
    --top-k 64 \
    --cpu-procs 14 \
    --cpu-time-per 240 \
    --update-submission \
    > "mass_gpu_${SEQ}.log" 2>&1
  echo "=== $SEQ done ($(date)) ==="
done
echo "=== chain complete: $(date) ==="
