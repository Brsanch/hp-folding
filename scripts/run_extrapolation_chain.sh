#!/usr/bin/env bash
# Extrapolation chain: queue AFTER current overnight chain (S6 lvl-8 + constraint GA).
# (1) S9 lvl-46 crossover, 5000 more trials — tail-extrapolation says ~1 hit at 53.
# (2) S10 lvl-44 multibasin climb — test if S9-style 50→52 jump pattern works.
set -u
cd "$(dirname "$0")/.."
LOG="extrapolation_chain.log"
echo "=== extrapolation chain start: $(date) ===" | tee -a "$LOG"

# Wait for any active basin_crossover / constraint_ga to finish.
while pgrep -f "basin_crossover.py|constraint_ga.py|run_constraint_ga.sh" >/dev/null; do
    sleep 60
done
echo "=== prior chain done: $(date) ===" | tee -a "$LOG"

# (1) S9 lvl-46 crossover ~5000 more trials
echo "=== S9 lvl-46 crossover EXTRA ($(date)) ===" | tee -a "$LOG"
python3 -u scripts/basin_crossover.py \
    --sequence S9 --source-level 46 \
    --processes 14 --max-pairs 1700 \
    --splice-points-per-pair 3 \
    --n-polish-steps 200000 \
    --target 53 --update-submission \
    >> crossover_S9_lvl46_extra_v4.log 2>&1
echo "=== S9 lvl-46 EXTRA done: $(date) ===" | tee -a "$LOG"

# (2) S10 lvl-44 multibasin (15 distinct stepping-stones × 3 temps = 45 climbs)
echo "=== S10 lvl-44 multibasin ($(date)) ===" | tee -a "$LOG"
python3 -u scripts/multibasin_climb.py \
    --sequence S10 --start-level 44 \
    --n-steps 600000 --processes 14 \
    --target 50 --update-submission \
    >> multibasin_S10_lvl44_v4.log 2>&1
echo "=== S10 lvl-44 done: $(date) ===" | tee -a "$LOG"

# (3) If still no S10 progress, try lvl-43 (16 distinct stones, 48 climbs)
echo "=== S10 lvl-43 multibasin ($(date)) ===" | tee -a "$LOG"
python3 -u scripts/multibasin_climb.py \
    --sequence S10 --start-level 43 \
    --n-steps 600000 --processes 14 \
    --target 50 --update-submission \
    >> multibasin_S10_lvl43_v4.log 2>&1
echo "=== chain end: $(date) ===" | tee -a "$LOG"
