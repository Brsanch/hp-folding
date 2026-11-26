#!/usr/bin/env bash
# 6h S9 lvl-46 grind. Run 5 seed-distinct rounds (now that --seed exists).
# At ~5000 trials/45min/round, 5 rounds ≈ 4-5h compute (some overhead).
# Tail extrapolation: ~30-50% chance of hitting one 53 over 25k more trials.
set -u
cd "$(dirname "$0")/.."
LOG="s9_long_grind.log"
echo "=== S9 long grind start: $(date) ===" | tee -a "$LOG"

# Wait for any active sister job
while pgrep -f "basin_crossover.py|multibasin_climb.py" >/dev/null; do
    sleep 30
done

for SEED in 1 2 3 4 5; do
    echo "=== round seed=$SEED ($(date)) ===" | tee -a "$LOG"
    python3 -u scripts/basin_crossover.py \
        --sequence S9 --source-level 46 \
        --processes 14 --max-pairs 1700 \
        --splice-points-per-pair 3 \
        --n-polish-steps 200000 \
        --target 53 --update-submission \
        --seed "$SEED" \
        >> "crossover_S9_lvl46_seed${SEED}.log" 2>&1
    # Quick check: did we promote?
    CUR=$(python3 -c "
import json,sys
sys.path.insert(0,'src')
from hp_folding import BENCHMARKS
from hp_folding.verifier import verify
sub=json.loads(open('submissions/S9.json').read())
print(verify(BENCHMARKS['S9'].sequence,[tuple(c) for c in sub['coords']]).contacts)
")
    echo "  current S9 = $CUR" | tee -a "$LOG"
    if [ "$CUR" -ge "53" ]; then
        echo "=== RECORD MATCH at seed $SEED ($(date)) ===" | tee -a "$LOG"
        exit 0
    fi
done
echo "=== grind end (no 53): $(date) ===" | tee -a "$LOG"
