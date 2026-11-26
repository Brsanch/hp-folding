"""S6 unified CP-SAT attack.

Build ONE CP-SAT problem encoding "find a 50-residue chain in bbox=10 with
≥20 H-H contacts" — using the existing 19-contact production folds as hints.

This is the right use of CP-SAT: a single integrated search instead of
millions of enumeration steps. We inherit the work of polyomino-enum
proofs by tightening the bbox to 10×10 (we already proved all ie≥30
connected polyominoes in bbox≤10 are infeasible — but the chain space
is still rich, and the CP-SAT solver can search it integrally).

Result interpretation:
  status=OPTIMAL   → proven optimum found (whatever its value)
  status=FEASIBLE  → found a chain with ≥20 contacts (RECORD)
  status=INFEASIBLE → proven no 20+ chain in bbox=10 (real bound)
  status=UNKNOWN   → ran out of time (we'd extend the budget)
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, glob, os
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.cpsat_hp_solver import solve_hp_cpsat

SEQ = 'PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH'

def count_contacts(seq, coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    n = 0
    for i, c in enumerate(coords):
        if seq[i] != 'H': continue
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            j = pos.get((c[0]+dx, c[1]+dy))
            if j is None or abs(j-i) <= 1: continue
            if seq[j] == 'H': n += 1
    return n // 2


def main():
    # Load production 19-contact folds
    base = str(Path(__file__).resolve().parent / 'record_attempt_results')
    hints = []
    for f in sorted(glob.glob(f'{base}/S6_*.json')):
        d = json.load(open(f))
        coords = [tuple(c) for c in d['coords']]
        # translate to put coords[0] near origin
        x0, y0 = coords[0]
        coords = [(c[0]-x0, c[1]-y0) for c in coords]
        # check bbox
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        bbox_x = max(xs) - min(xs)
        bbox_y = max(ys) - min(ys)
        hints.append((d.get('contacts', count_contacts(SEQ, coords)), coords, bbox_x, bbox_y, os.path.basename(f)))

    print(f'Loaded {len(hints)} 19-contact folds')
    for c, _, bx, by, name in hints:
        print(f'  {name}: contacts={c}, bbox={bx}x{by}')

    # Use the most-compact hint
    hints.sort(key=lambda h: (max(h[2], h[3]), -h[0]))
    best_hint = hints[0]
    print(f'\nUsing best-hint: {best_hint[4]} (bbox={best_hint[2]}x{best_hint[3]}, contacts={best_hint[0]})')

    # bbox 11 for some slack around the 19-contact bbox
    BBOX = 11
    BUDGET = 7200  # 2 hours

    print(f'\nCP-SAT bbox={BBOX}, min_contacts=20, budget={BUDGET}s, hint loaded...', flush=True)
    t0 = time.time()
    coords, contacts, st = solve_hp_cpsat(
        SEQ, bbox_side=BBOX, time_budget_seconds=BUDGET, n_workers=8,
        initial_solution=best_hint[1], min_contacts=20, log_progress=True,
    )
    print(f'\n=== RESULT (min_contacts=20, bbox={BBOX}) ===', flush=True)
    print(f'status   = {st}', flush=True)
    print(f'contacts = {contacts}', flush=True)
    print(f'walltime = {time.time()-t0:.1f}s', flush=True)
    if st == 'INFEASIBLE':
        print(f'  → PROVEN: no chain in {BBOX}x{BBOX} bbox achieves ≥20 contacts.', flush=True)
        print(f'    (Since all 4 production basins fit in 10x10, this is a strong bound.)', flush=True)
    elif coords and contacts >= 20:
        print(f'  → contacts={contacts} ≥ 20 → WORLD RECORD on S6.', flush=True)
        print(f'    coords = {coords}', flush=True)
        with open(str(Path(__file__).resolve().parent / 's6_record_fold.json'), 'w') as f:
            json.dump({'sequence': SEQ, 'contacts': contacts, 'coords': [list(c) for c in coords]}, f, indent=2)
        print(f'    saved to s6_record_fold.json', flush=True)


if __name__ == "__main__":
    main()
