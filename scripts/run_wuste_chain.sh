#!/bin/bash
# Wüst-style attack chain: heavy local_rebridge + corner_flip + crankshaft.
# Runs after S8-from-42 finishes.
set -e
cd "$(dirname "$0")/.."

echo "=== Wüst-style chain start: $(date) ==="

# Wait for S8-from-42 to finish
echo "Waiting for S8-from-42..."
while pgrep -f "wang_landau_parallel.*from42" > /dev/null; do
  sleep 60
done
echo "S8-from-42 finished at $(date)"

# Promote any S8-from-42 result first
python3 -c "
import json, shutil, sys
sys.path.insert(0, 'src')
from hp_folding import BENCHMARKS
from hp_folding.verifier import verify
from pathlib import Path
res_p = Path('results/S8_wang_landau_best.json')
sub_p = Path('submissions/S8.json')
if res_p.exists():
    new = json.loads(res_p.read_text())
    sub = json.loads(sub_p.read_text())
    seq = BENCHMARKS['S8'].sequence
    nc = verify(seq, [tuple(c) for c in new['coords']]).contacts
    sc = verify(seq, [tuple(c) for c in sub['coords']]).contacts
    if nc > sc:
        shutil.copy(res_p, sub_p)
        print(f'PROMOTED S8: {sc} -> {nc}')
    else:
        print(f'S8 no improvement (sub={sc}, new={nc})')
"

# S6 with new move set (records most beatable: gap -2, upper bound 27)
for SEQ in S6 S10 S7 S9; do
  TARGET=$(python3 -c "
from hp_folding import BENCHMARKS
import sys
sys.path.insert(0, 'src')
from hp_folding import BENCHMARKS
print(BENCHMARKS['$SEQ'].best_known)
")
  echo ""
  echo "=== $SEQ Wüst-style attack start ($(date)) ==="
  python3 -u scripts/wang_landau_parallel.py \
    --sequence "$SEQ" \
    --processes 16 \
    --trials 768 \
    --time-budget 240 \
    --diag-dir "diagnostics/wl_${SEQ}_wuste" \
    --init-from "submissions/${SEQ}.json" \
    --target "$TARGET" \
    > "wl_${SEQ}_wuste.log" 2>&1
  echo "=== $SEQ done ($(date)) ==="
done
echo "=== chain complete: $(date) ==="
