#!/bin/bash
# Long-restart WL: starts each trial from the current submission fold,
# 16 procs × 16 trials × 900s/trial = 15 min wall per sequence.
# Chains across S5-S10. Updates submission if improved.
set -e
cd "$(dirname "$0")/.."

for SEQ in S6 S7 S8 S9 S5 S10; do
  SUB="submissions/${SEQ}.json"
  if [ ! -f "$SUB" ]; then
    echo "skip $SEQ — no submission file"
    continue
  fi
  echo "=== $SEQ long-restart start ($(date)) ==="
  python3 scripts/wang_landau_parallel.py \
    --sequence "$SEQ" \
    --processes 16 \
    --trials 16 \
    --time-budget 900 \
    --diag-dir "diagnostics/wl_long" \
    --init-from "$SUB" \
    2>&1 | tee "wl_${SEQ}_long_run.log"

  # Promote if long-restart improved.
  python3 - <<PY
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, "src")
from hp_folding import BENCHMARKS
from hp_folding.verifier import verify

seq_id = "${SEQ}"
new_p = Path(f"results/{seq_id}_wang_landau_best.json")
sub_p = Path(f"submissions/{seq_id}.json")
if not new_p.exists():
    print(f"no result file for {seq_id}"); sys.exit(0)
new = json.loads(new_p.read_text())
sub = json.loads(sub_p.read_text())
new_coords = [tuple(c) for c in new["coords"]]
sub_coords = [tuple(c) for c in sub["coords"]]
seq = BENCHMARKS[seq_id].sequence
v_new = verify(seq, new_coords).contacts
v_sub = verify(seq, sub_coords).contacts
print(f"{seq_id}: long-restart={v_new} submission={v_sub}")
if v_new > v_sub:
    shutil.copy(new_p, sub_p)
    print(f"  PROMOTED ({v_sub} -> {v_new})")
else:
    print(f"  no improvement")
PY
  echo "=== $SEQ long-restart done ($(date)) ==="

  echo "=== $SEQ hybrid GPU+CPU start ($(date)) ==="
  python3 scripts/hybrid_gpu_cpu.py \
    --sequence "$SEQ" \
    --gpu-K 2048 \
    --gpu-steps 100000 \
    --top-k 32 \
    --cpu-procs 16 \
    --cpu-time-per 90 \
    --diag-dir "diagnostics/hybrid" \
    --update-submission \
    2>&1 | tee "hybrid_${SEQ}_run.log"
  echo "=== $SEQ hybrid done ($(date)) ==="
done
