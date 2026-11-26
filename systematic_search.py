"""Systematic hcore-shape search for HP folding.

Pipeline:
  1. Compute n_H = sequence.count('H') and adjacent_h_pairs (H-H bonds inside
     H-runs in the sequence — these are internal edges that don't count as
     contacts).
  2. Enumerate polyominoes of size n_H, sorted by internal_edges descending.
  3. For each polyomino with internal_edges >= target_contacts + adjacent_h_pairs,
     call solve_csp to check if the HP sequence threads through it.
  4. Stop at first hit, OR enumerate all hits at the maximum reachable
     internal_edges.

Versus random search:
  - Every hcore shape gets visited exactly once (canonical D4 dedup).
  - Routability is a definitive yes/no per shape (CSP is sound; bounded by
     time_budget).
  - We learn the upper bound: max internal_edges among ROUTABLE polyominoes.
"""
from __future__ import annotations
from pathlib import Path
import sys, time, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.polyomino_enum import enumerate_polyominoes
from hp_folding.cpsat_routability import is_routable_cpsat


def adjacent_h_pairs(sequence: str) -> int:
    """Count i such that sequence[i] == sequence[i+1] == 'H'."""
    return sum(1 for i in range(len(sequence) - 1)
                 if sequence[i] == 'H' and sequence[i+1] == 'H')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--name', default='custom')
    ap.add_argument('--target', type=int, required=True,
                    help='Target contact count (≥ this is success)')
    ap.add_argument('--time-per-csp', type=float, default=8.0)
    ap.add_argument('--max-shapes', type=int, default=1_000_000)
    ap.add_argument('--stop-on-success', action='store_true')
    args = ap.parse_args()

    seq = args.sequence
    n_h = seq.count('H')
    ahp = adjacent_h_pairs(seq)
    min_ie = args.target + ahp
    print(f'== {args.name}: |seq|={len(seq)}, n_H={n_h}, target_contacts={args.target}', flush=True)
    print(f'   adjacent_H_pairs={ahp}, min_internal_edges={min_ie}', flush=True)

    print('\nstreaming polyominoes (sorted by internal_edges DESC)...', flush=True)
    # Collect all polyominoes meeting threshold, then sort.
    t0 = time.time()
    candidates = list(enumerate_polyominoes(n_h, min_internal_edges=min_ie))
    candidates.sort(key=lambda kv: -kv[1])
    print(f'  found {len(candidates)} candidate hcores in {time.time()-t0:.1f}s', flush=True)

    if not candidates:
        print(f'  no polyomino of size {n_h} has internal_edges >= {min_ie}.')
        print(f'  → contacts ≥ {args.target} is GEOMETRICALLY IMPOSSIBLE for this n_H.')
        return

    best_ie = candidates[0][1]
    print(f'  max internal_edges available: {best_ie} → max possible contacts (if routable): {best_ie - ahp}\n', flush=True)

    routable_found = []
    n_tried = 0
    n_unknown = 0
    for canon, ie in candidates[:args.max_shapes]:
        n_tried += 1
        block = frozenset(canon)
        t = time.time()
        try:
            coords, status = is_routable_cpsat(seq, block, time_budget_seconds=args.time_per_csp)
        except Exception as e:
            print(f'  [{n_tried}/{len(candidates)}] ie={ie} ERROR: {e}', flush=True)
            continue
        elapsed = time.time() - t
        if coords is not None:
            # count contacts in the chain
            pos = {c: i for i, c in enumerate(coords)}
            cnt = 0
            for i, c in enumerate(coords):
                if seq[i] != 'H': continue
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    j = pos.get((c[0]+dx, c[1]+dy))
                    if j is None or abs(j - i) <= 1: continue
                    if seq[j] == 'H': cnt += 1
            contacts = cnt // 2
            routable_found.append((contacts, ie, canon))
            best = max(routable_found)
            star = ' ← best' if (best[0], best[1]) == (contacts, ie) else ''
            print(f'  [{n_tried}/{len(candidates)}] ie={ie} contacts={contacts} {status} ({elapsed:.2f}s){star}', flush=True)
            if args.stop_on_success and contacts >= args.target:
                print(f'\nSUCCESS: routable hcore found with contacts={contacts} ≥ target={args.target}', flush=True)
                break
        elif status == 'INFEASIBLE':
            if n_tried <= 20 or n_tried % 100 == 0:
                print(f'  [{n_tried}/{len(candidates)}] ie={ie} INFEASIBLE ({elapsed:.2f}s)', flush=True)
        else:  # UNKNOWN — timeout
            n_unknown += 1
            print(f'  [{n_tried}/{len(candidates)}] ie={ie} UNKNOWN/timeout ({elapsed:.2f}s) [{n_unknown} so far]', flush=True)

    print(f'\n=== RESULT ===', flush=True)
    if routable_found:
        routable_found.sort(reverse=True)
        bc, bi, _ = routable_found[0]
        print(f'  best: contacts={bc} (internal_edges={bi})', flush=True)
        print(f'  total routable shapes: {len(routable_found)}', flush=True)
        if bc >= args.target:
            print(f'  ✓ MEETS target {args.target}', flush=True)
        else:
            print(f'  ✗ below target {args.target} (gap: {args.target - bc})', flush=True)
    else:
        print(f'  No routable hcore found among {n_tried} shapes.', flush=True)
        print(f'  → For target {args.target}: NOT REACHABLE in our routability model.', flush=True)


if __name__ == "__main__":
    main()
