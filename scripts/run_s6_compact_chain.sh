#!/usr/bin/env bash
# Run S6 compact-biased attack after S9 grind finishes (or with reduced procs in parallel).
set -u
cd "$(dirname "$0")/.."
LOG="s6_compact_chain.log"
echo "=== S6 compact-attack chain start: $(date) ===" | tee -a "$LOG"

# Wait for S9 grind to free up cores
while pgrep -f "basin_crossover.py" >/dev/null; do
    sleep 60
done
echo "=== S9 grind done, S6 compact attack starting: $(date) ===" | tee -a "$LOG"

# Sweep alpha values: low/med/high bias toward compactness
for ALPHA in 1.0 2.5 5.0; do
    echo "=== alpha=$ALPHA ($(date)) ===" | tee -a "$LOG"
    python3 -u scripts/s6_compact_attack.py \
        --n-steps 2000000 --alpha "$ALPHA" \
        --n-trials-per-basin 25 --processes 14 \
        --target 21 --update-submission \
        >> "s6_compact_alpha${ALPHA}.log" 2>&1
    CUR=$(python3 -c "
import json,sys
sys.path.insert(0,'src')
from hp_folding import BENCHMARKS
from hp_folding.verifier import verify
sub=json.loads(open('submissions/S6.json').read())
print(verify(BENCHMARKS['S6'].sequence,[tuple(c) for c in sub['coords']]).contacts)
")
    echo "  S6 = $CUR after alpha=$ALPHA" | tee -a "$LOG"
    if [ "$CUR" -ge "21" ]; then
        echo "=== RECORD MATCH ===" | tee -a "$LOG"
        exit 0
    fi
done
echo "=== S6 compact-attack end (no break): $(date) ===" | tee -a "$LOG"
