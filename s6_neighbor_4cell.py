"""S6 hcore-neighborhood hunt extending to 3-cell modifications.

Previous: 165 + 2941 1-cell + 2-cell mods at ie≥27 — all infeasible.
Now: 3-cell mods (capped at ~50K total, sampled diversely).

Theoretical max for S6 (n_H=22): contacts ≤ 27 (Harary-Harborth).
Current best: 19. Gap to 20 is reachable in principle (ie=27 ≤ 34=ie_max).
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, glob, os, random
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges
from hp_folding import sequences

SEQ = sequences.BENCHMARKS['S6'].sequence
ADJ_HH = sum(1 for i in range(len(SEQ)-1) if SEQ[i]==SEQ[i+1]=='H')
NBRS = ((1,0),(-1,0),(0,1),(0,-1))


def is_connected(cells):
    cells = set(cells)
    if not cells: return True
    stack = [next(iter(cells))]; seen = {stack[0]}
    while stack:
        x, y = stack.pop()
        for dx, dy in NBRS:
            nb = (x+dx, y+dy)
            if nb in cells and nb not in seen:
                seen.add(nb); stack.append(nb)
    return len(seen) == len(cells)


def one_cell_neighbors(hc):
    hc = set(hc)
    seen = set()
    for cell_to_remove in list(hc):
        smaller = hc - {cell_to_remove}
        if not is_connected(smaller): continue
        boundary = set()
        for x, y in smaller:
            for dx, dy in NBRS:
                nb = (x+dx, y+dy)
                if nb not in smaller:
                    boundary.add(nb)
        for cell_to_add in boundary:
            if cell_to_add == cell_to_remove: continue
            new_hc = smaller | {cell_to_add}
            if not is_connected(new_hc): continue
            canon = canonicalize(new_hc)
            if canon in seen: continue
            seen.add(canon)
            yield canon


def k_cell_neighbors(hc, k=3, max_yield=50000, seed=0):
    """k-cell neighbors: BFS in hcore-graph, k modifications deep."""
    rng = random.Random(seed)
    frontier = {canonicalize(hc)}
    visited = set(frontier)
    yielded = set()
    for depth in range(k):
        new_frontier = set()
        for canon in frontier:
            cells = set(canon)
            for nb_canon in one_cell_neighbors(cells):
                if nb_canon in visited: continue
                visited.add(nb_canon)
                new_frontier.add(nb_canon)
                if depth == k - 1:
                    yielded.add(nb_canon)
                    if len(yielded) >= max_yield:
                        return yielded
        # subsample frontier if it explodes
        if len(new_frontier) > 5000:
            new_frontier = set(rng.sample(list(new_frontier), 5000))
        frontier = new_frontier
        if not frontier:
            break
    return yielded


def count_chain_contacts(seq, coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    n = 0
    for i, c in enumerate(coords):
        if seq[i] != 'H': continue
        for dx, dy in NBRS:
            j = pos.get((c[0]+dx, c[1]+dy))
            if j is None or abs(j-i) <= 1: continue
            if seq[j] == 'H': n += 1
    return n // 2


def hcore_canon_from_fold(coords, seq):
    h_cells = [tuple(coords[i]) for i, r in enumerate(seq) if r == 'H']
    return canonicalize(h_cells)


def main():
    base = str(Path(__file__).resolve().parent)
    print(f'Collecting 19-contact S6 folds...', flush=True)
    unique_hcores = {}
    for f in sorted(glob.glob(f'{base}/record_attempt_results/S6_*.json')):
        d = json.load(open(f))
        if d.get('sequence') != SEQ: continue
        coords = [tuple(c) for c in d['coords']]
        canon = hcore_canon_from_fold(coords, SEQ)
        unique_hcores.setdefault(canon, []).append(os.path.basename(f))
    print(f'  Found {len(unique_hcores)} distinct 19-contact basins', flush=True)

    print(f'\n=== Phase: 4-cell modifications (target ie ≥ 27) ===', flush=True)
    overall_seen = set()
    candidates = []
    t0 = time.time()
    for i, hc in enumerate(unique_hcores.keys()):
        per = 0
        for canon in k_cell_neighbors(hc, k=4, max_yield=30000, seed=i):
            if canon in overall_seen: continue
            overall_seen.add(canon)
            ie = internal_edges(set(canon))
            if ie >= 27:
                candidates.append((ie, canon))
                per += 1
        print(f'  Hcore #{i+1}: +{per} new ie≥27 candidates (cumulative: {len(candidates)})', flush=True)
    print(f'  total: {len(candidates)} 4-cell candidates  enum_time={time.time()-t0:.1f}s', flush=True)

    if candidates:
        print(f'\n=== CP-SAT route on 4-cell candidates ===', flush=True)
        candidates.sort(key=lambda kv: -kv[0])
        n_inf = 0; n_unk = 0
        for idx, (ie, canon) in enumerate(candidates):
            coords, status = is_routable_cpsat(SEQ, frozenset(canon),
                                                 time_budget_seconds=10)
            if coords is not None:
                contacts = count_chain_contacts(SEQ, coords)
                print(f'  [{idx+1}/{len(candidates)}] ie={ie} ROUTABLE → contacts={contacts}', flush=True)
                if contacts >= 20:
                    print(f'\n*** RECORD: contacts={contacts} on S6 ***', flush=True)
                    with open(f'{base}/s6_record_fold.json', 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': contacts,
                                    'coords': [list(c) for c in coords],
                                    'note': '4-cell mod from 19-contact S6 basin'}, f, indent=2)
                    return
            else:
                if status == 'INFEASIBLE': n_inf += 1
                else: n_unk += 1
            if (idx+1) % 200 == 0:
                print(f'  [{idx+1}/{len(candidates)}] ... {n_inf} infeas, {n_unk} unk so far', flush=True)
        print(f'  → all 4-cell candidates exhausted with no record  '
              f'({n_inf} infeas, {n_unk} unk)', flush=True)

    print(f'\n=== EXHAUSTED 4-cell sweep ===', flush=True)


if __name__ == "__main__":
    main()
