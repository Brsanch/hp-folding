#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
LOG="s6_expanded_chain.log"
echo "=== S6 expanded chain start: $(date) ===" | tee -a "$LOG"

# Wait for fresh SA to finish
while pgrep -f "s6_fresh_sa_with_rebridge" >/dev/null; do
    sleep 30
done
echo "=== fresh SA done: $(date) ===" | tee -a "$LOG"

# Step 1: Collect all distinct 19-folds from results/ + diagnostic stones into a fresh stepping_stone level
python3 -u scripts/collect_s6_19_basins.py >> "$LOG" 2>&1

# Step 2: Crossover attack from expanded basin set
echo "=== S6 crossover lvl-19 EXPANDED ($(date)) ===" | tee -a "$LOG"
python3 -u scripts/basin_crossover.py \
    --sequence S6 --source-level 19 \
    --processes 14 --max-pairs 6000 \
    --splice-points-per-pair 5 \
    --n-polish-steps 200000 \
    --target 21 --update-submission \
    --seed 100 \
    >> crossover_S6_lvl19_expanded.log 2>&1
CUR=$(python3 -c "
import json, sys
sys.path.insert(0, 'src')
from hp_folding import BENCHMARKS
from hp_folding.verifier import verify
sub=json.loads(open('submissions/S6.json').read())
print(verify(BENCHMARKS['S6'].sequence,[tuple(c) for c in sub['coords']]).contacts)
")
echo "  S6 = $CUR after expanded crossover" | tee -a "$LOG"
[ "$CUR" -ge "20" ] && echo "*** BREAKTHROUGH ***" | tee -a "$LOG"

# Step 3: Another round of fresh SA to expand basins further if still stuck
if [ "$CUR" -lt "20" ]; then
    echo "=== second fresh SA round ($(date)) ===" | tee -a "$LOG"
    python3 -u scripts/s6_fresh_sa_with_rebridge.py \
        --n-chains 300 --n-steps 3000000 \
        --processes 14 --target 21 --update-submission \
        > s6_fresh_rebridge_round2.log 2>&1
fi

echo "=== chain end: $(date) ===" | tee -a "$LOG"
