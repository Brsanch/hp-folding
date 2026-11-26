#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "=== S10 deeper chain start: $(date) ==="

# Wait for current S10 lvl-38 to finish
echo "Waiting for current S10 lvl-38 to finish..."
while pgrep -f "basin_crossover.*S10.*lvl.*38" > /dev/null; do
  sleep 30
done

# Re-mine diagnostics to capture any new 48-fold
echo "Re-mining diagnostics..."
python3 scripts/mine_diagnostics.py 2>&1 | tail -5

# Re-add the current S10 submission to stepping stones if not present
python3 -c "
import json, sys
sys.path.insert(0, 'src')
from hp_folding import BENCHMARKS
from hp_folding.verifier import verify
NB = ((1,0),(-1,0),(0,1),(0,-1))
def cs(seq, coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    pairs = set()
    for i, (x, y) in enumerate(coords):
        if seq[i] != 'H': continue
        for dx, dy in NB:
            j = pos.get((x+dx, y+dy))
            if j is None or abs(j-i) <= 1: continue
            if seq[j] == 'H':
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    return frozenset(pairs)
seq = BENCHMARKS['S10'].sequence
ss = json.loads(open('diagnostics/stepping_stones.json').read())
sub = json.loads(open('submissions/S10.json').read())
coords = [tuple(c) for c in sub['coords']]
v = verify(seq, coords).contacts
ss.setdefault('S10', {}).setdefault(str(v), [])
existing = set(cs(seq, [tuple(c) for c in f]) for f in ss['S10'][str(v)])
if cs(seq, coords) not in existing:
    ss['S10'][str(v)].append([list(c) for c in coords])
    json.dump(ss, open('diagnostics/stepping_stones.json', 'w'))
    print(f'Added current S10 submission (contacts={v}) to stepping stones')
"

run() {
  local SEQ=$1; local LVL=$2; local TGT=$3; local PAIRS=$4
  echo ""
  echo "=== $SEQ lvl-$LVL ($(date)) PAIRS=$PAIRS ==="
  python3 -u scripts/basin_crossover.py \
    --sequence "$SEQ" \
    --source-level "$LVL" \
    --processes 14 \
    --max-pairs "$PAIRS" \
    --splice-points-per-pair 3 \
    --n-polish-steps 100000 \
    --target "$TGT" \
    --update-submission \
    > "crossover_${SEQ}_lvl${LVL}_v3.log" 2>&1 || true
}

# More aggressive S10 chain
run S10 36 50 4000
run S10 34 50 4000

# Also try multibasin from level-46 (close to current 48)
echo ""
echo "=== S10 multibasin from level-46 ==="
python3 -u scripts/multibasin_climb.py \
  --sequence S10 \
  --start-level 46 \
  --n-steps 800000 \
  --processes 14 \
  --target 50 \
  --update-submission > multibasin_S10_v2.log 2>&1 || true

echo "=== chain complete: $(date) ==="
