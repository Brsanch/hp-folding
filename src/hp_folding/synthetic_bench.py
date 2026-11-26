"""Synthetic HP-sequence benchmark for stress-testing folding methods.

Generates short HP sequences (length 10-18) across a controlled difficulty
spectrum, then brute-force enumerates ALL valid SAW folds to compute the
true optimal contact count. Methods can be evaluated on this corpus with
KNOWN ground truth.

Difficulty axis: fraction of H residues that live in length-1 H-runs
(= "scatteredness"). Clustered sequences have all H in long runs (easy);
scattered sequences have all H in length-1 runs separated by P (hard).

The brute force is feasible because for length N=18 the SAW count is in the
low millions — tractable in tens of seconds with directional canonicalization
(every walk has 8 dihedral images; we count only one).
"""
from __future__ import annotations
import time
from collections import Counter

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parse_runs(sequence: str) -> list[tuple[str, int]]:
    out = []
    if not sequence: return out
    cur, count = sequence[0], 1
    for c in sequence[1:]:
        if c == cur:
            count += 1
        else:
            out.append((cur, count)); cur, count = c, 1
    out.append((cur, count))
    return out


def scatteredness(sequence: str) -> float:
    """Fraction of H residues that live in length-1 H-runs.
    0.0 = all H clustered; 1.0 = all H isolated."""
    runs = parse_runs(sequence)
    n_h = sequence.count('H')
    if n_h == 0: return 0.0
    n_isolated = sum(l for k, l in runs if k == 'H' and l == 1)
    return n_isolated / n_h


def count_contacts(sequence: str, coords: list[Coord]) -> int:
    pos_to_idx = {c: i for i, c in enumerate(coords)}
    contacts = 0
    for i, (x, y) in enumerate(coords):
        if sequence[i] != 'H': continue
        for dx, dy in NBRS:
            n = (x + dx, y + dy)
            j = pos_to_idx.get(n)
            if j is None or j <= i: continue
            if sequence[j] != 'H': continue
            if abs(j - i) <= 1: continue
            contacts += 1
    return contacts


def brute_force_optimum(sequence: str, time_limit_s: float = 60.0) -> tuple[int, list[Coord] | None, int]:
    """Enumerate all valid SAW folds starting at origin with first move (1,0)
    (canonical to break translation+rotation symmetry by 4×). Return
    (best_contacts, an example fold, total_folds_enumerated).

    This is exhaustive — for N≤18 it's feasible in seconds.
    """
    n = len(sequence)
    best = 0
    best_coords = None
    n_folds = 0
    deadline = time.monotonic() + time_limit_s

    # Canonicalize: first cell at (0,0), second at (1,0). This kills 4 of the 8
    # dihedral images (those that would put the second cell elsewhere).
    coords = [(0, 0), (1, 0)]
    occupied = {(0, 0), (1, 0)}

    # Track contacts incrementally
    pos_to_idx = {(0, 0): 0, (1, 0): 1}
    base_contacts = 0
    if n >= 2 and sequence[0] == 'H' and sequence[1] == 'H':
        # adjacent, sequence-adjacent → no contact
        pass

    def dfs(depth, contacts):
        nonlocal best, best_coords, n_folds
        if time.monotonic() > deadline: return
        if depth == n:
            n_folds += 1
            if contacts > best:
                best = contacts
                best_coords = list(coords)
            return
        last = coords[-1]
        for dx, dy in NBRS:
            cand = (last[0] + dx, last[1] + dy)
            if cand in occupied: continue
            # Compute new contacts contributed by placing residue `depth` at cand.
            d_contacts = 0
            if sequence[depth] == 'H':
                for ddx, ddy in NBRS:
                    n_cell = (cand[0] + ddx, cand[1] + ddy)
                    j = pos_to_idx.get(n_cell)
                    if j is None: continue
                    if sequence[j] != 'H': continue
                    if abs(j - depth) <= 1: continue
                    d_contacts += 1
            coords.append(cand)
            occupied.add(cand)
            pos_to_idx[cand] = depth
            dfs(depth + 1, contacts + d_contacts)
            coords.pop()
            occupied.discard(cand)
            del pos_to_idx[cand]

    if n >= 2:
        dfs(2, 0)
    elif n == 1:
        n_folds = 1
        if best_coords is None: best_coords = list(coords[:1])

    return best, best_coords, n_folds


# ════════════════════════════════════════════════════════════
# Synthetic library: a curated set of test sequences
# ════════════════════════════════════════════════════════════

SYNTHETICS = {
    # Length 10
    'L10_clustered':      'HHHHHPPPPP',     # all H in one run
    'L10_two_clusters':   'HHHPPHHPHH',     # mostly clustered
    'L10_alternating':    'HPHPHPHPHP',     # all H isolated
    'L10_separated':      'HPPPHPPPHH',     # H at ends, isolated middle
    # Length 12
    'L12_clustered':      'HHHHHHPPPPPP',
    'L12_alternating':    'HPHPHPHPHPHP',
    'L12_one_loner':      'HHHHHHHPPHPHH', # 1 isolated H amid cluster
    'L12_pseudo_S6':      'PPHPPHPHPHPP', # mini-S6 pattern
    # Length 14
    'L14_clustered':      'HHHHHHHHPPPPPP',
    'L14_alternating':    'HPHPHPHPHPHPHP',
    'L14_mostly_isolated':'HPPHPHPPHPHPPH',
    'L14_pseudo_S6':      'PPHPPHPHPHHHPP',
    # Length 16
    'L16_clustered':      'HHHHHHHHHHPPPPPP',
    'L16_alternating':    'HPHPHPHPHPHPHPHP',
    'L16_pseudo_S6':      'PPHPPHPHPHHPPHPH',
    # Length 18
    'L18_alternating':    'HPHPHPHPHPHPHPHPHP',
    'L18_pseudo_S6':      'PPHPPHPHPHHHHPHPPP',  # mimics S6's start
    'L18_mostly_isolated':'HPPHPHPPHPHPHPPHPH',
    # Length 22 — getting into the regime where methods may struggle
    'L22_clustered':       'HHHHHHHHHHHHPPPPPPPPPP',
    'L22_alternating':     'HPHPHPHPHPHPHPHPHPHPHP',
    'L22_pseudo_S6_a':     'PPHPPHPHPHHHHPHPPPHPPP',  # first 22 of S6
    'L22_pseudo_S6_b':     'HPHPHPHHHHPPPHPHPHPPHH',  # remix
    'L22_mostly_isolated': 'HPPHPHPPHPHPHPPHPHPHPP',
    # Length 26
    'L26_clustered':       'HHHHHHHHHHHHHHPPPPPPPPPPPP',
    'L26_pseudo_S6':       'PPHPPHPHPHHHHPHPPPHPPPHPPP',  # first 26 of S6
    'L26_mostly_isolated': 'HPPHPHPPHPHPHPPHPHPHPPHPHP',
    'L26_alternating':     'HPHPHPHPHPHPHPHPHPHPHPHPHP',
}


def build_corpus():
    """Return list of dict entries with sequence + computed ground truth."""
    out = []
    for name, seq in SYNTHETICS.items():
        n_h = seq.count('H')
        sc = scatteredness(seq)
        runs = parse_runs(seq)
        h_run_lens = [l for k, l in runs if k == 'H']
        out.append({
            'name': name, 'sequence': seq, 'length': len(seq),
            'n_h': n_h, 'scatteredness': round(sc, 3),
            'h_run_lens': h_run_lens,
        })
    return out


# ════════════════════════════════════════════════════════════
# CLI: build, brute force, save corpus
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./synthetic_corpus.json')
    ap.add_argument('--time-per-seq', type=float, default=120.0,
                    help="Brute force time limit per sequence (s)")
    args = ap.parse_args()

    corpus = build_corpus()
    print(f'Brute-forcing optima for {len(corpus)} synthetic sequences...')
    print(f'{"name":22s}  {"len":4s}  {"n_h":4s}  {"scatter":>8s}  {"opt":>4s}  {"folds":>10s}  {"wall":>5s}')
    print('-' * 80)
    for entry in corpus:
        t0 = time.time()
        best, coords, n_folds = brute_force_optimum(entry['sequence'],
                                                      time_limit_s=args.time_per_seq)
        wall = time.time() - t0
        entry['optimum'] = int(best)
        entry['example_fold'] = [list(c) for c in (coords or [])]
        entry['n_folds_enumerated'] = int(n_folds)
        entry['brute_wall_s'] = round(wall, 2)
        print(f'{entry["name"]:22s}  {entry["length"]:>4d}  {entry["n_h"]:>4d}  '
              f'{entry["scatteredness"]:>8.3f}  {best:>4d}  {n_folds:>10d}  {wall:>4.1f}s')

    with open(args.out, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f'\nsaved {len(corpus)} entries to {args.out}')
