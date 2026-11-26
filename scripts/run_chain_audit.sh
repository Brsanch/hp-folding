#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
LOG="audit_chain.log"
echo "=== chain start: $(date) ===" | tee -a "$LOG"

# Wait for speed benchmark
while pgrep -f speed_benchmark >/dev/null; do
    sleep 30
done
echo "=== speed benchmark done; launching new sequence design: $(date) ===" | tee -a "$LOG"
python3 -u scripts/new_benchmark_sequences.py >> new_benchmark_sequences.log 2>&1
echo "=== new sequences done: $(date) ===" | tee -a "$LOG"

# Then run designability pilot
echo "=== designability pilot starting: $(date) ===" | tee -a "$LOG"
python3 -u scripts/designability_pilot.py >> designability_pilot.log 2>&1
echo "=== designability pilot done: $(date) ===" | tee -a "$LOG"

echo "=== chain end: $(date) ===" | tee -a "$LOG"
