"""S6 deep neighborhood sweep: depths 5-8 from 4 known basins.

Each depth pushes BFS in hcore-modification space one further. Subsampling
caps growth; total candidates per depth roughly 5K-15K.
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, glob, os, random, signal
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges
from hp_folding import sequences

SEQ = sequences.BENCHMARKS['S6'].sequence
ADJ_HH = 7
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


def k_cell_neighbors_at_depth(hc, target_depth, max_yield, frontier_cap, seed):
    """BFS to exact depth target_depth, returning ONLY hcores at that depth."""
    rng = random.Random(seed)
    frontier = {canonicalize(hc)}
    visited = set(frontier)
    yielded = set()
    for depth in range(target_depth):
        new_frontier = set()
        for canon in frontier:
            cells = set(canon)
            for nb_canon in one_cell_neighbors(cells):
                if nb_canon in visited: continue
                visited.add(nb_canon)
                new_frontier.add(nb_canon)
                if depth == target_depth - 1:
                    yielded.add(nb_canon)
                    if len(yielded) >= max_yield:
                        return yielded
        if len(new_frontier) > frontier_cap:
            new_frontier = set(rng.sample(list(new_frontier), frontier_cap))
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
    print(f'  {len(unique_hcores)} distinct 19-contact basins', flush=True)

    # Cumulative proven-infeasible canonical hcores. Skip any we've already proven.
    proven_infeasible = set()

    overall_infeasible = 0
    overall_routable = 0

    for depth in [5, 6, 7, 8]:
        print(f'\n=== Depth {depth} ===', flush=True)
        candidates = []
        t0 = time.time()
        for i, hc in enumerate(unique_hcores.keys()):
            shapes_at_depth = k_cell_neighbors_at_depth(
                hc, target_depth=depth,
                max_yield=20000, frontier_cap=8000, seed=i + depth*100)
            kept = 0
            for canon in shapes_at_depth:
                if canon in proven_infeasible: continue
                ie = internal_edges(set(canon))
                if ie >= 27:
                    candidates.append((ie, canon))
                    kept += 1
            print(f'  basin #{i+1}: +{kept} new ie≥27 candidates (total: {len(candidates)})', flush=True)
        print(f'  enum_time={time.time()-t0:.0f}s, total candidates: {len(candidates)}', flush=True)

        if not candidates:
            print(f'  no candidates at depth {depth}', flush=True)
            continue

        print(f'  CP-SAT routability...', flush=True)
        t0 = time.time()
        candidates.sort(key=lambda kv: -kv[0])
        n_inf = n_unk = 0
        for idx, (ie, canon) in enumerate(candidates):
            coords, status = is_routable_cpsat(SEQ, frozenset(canon),
                                                 time_budget_seconds=8)
            if coords is not None:
                contacts = count_chain_contacts(SEQ, coords)
                print(f'  [{idx+1}/{len(candidates)}] ie={ie} ROUTABLE → contacts={contacts}', flush=True)
                if contacts >= 20:
                    print(f'\n*** RECORD: contacts={contacts} on S6 (depth-{depth} mod) ***', flush=True)
                    with open(f'{base}/s6_record_fold.json', 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': contacts,
                                    'coords': [list(c) for c in coords],
                                    'note': f'depth-{depth} mod from 19-contact S6 basin'}, f, indent=2)
                    return
                overall_routable += 1
            else:
                if status == 'INFEASIBLE':
                    n_inf += 1
                    proven_infeasible.add(canon)
                else: n_unk += 1
            if (idx+1) % 500 == 0:
                print(f'  [{idx+1}/{len(candidates)}] ... {n_inf} infeas, {n_unk} unk  ({time.time()-t0:.0f}s)', flush=True)
        overall_infeasible += n_inf
        print(f'  depth {depth} done: {n_inf} infeas, {n_unk} unk  '
              f'({time.time()-t0:.0f}s)', flush=True)
        print(f'  cumulative proven infeasible: {len(proven_infeasible)}', flush=True)

    print(f'\n=== DEEP SWEEP COMPLETE ===', flush=True)
    print(f'Cumulative proven infeasible: {len(proven_infeasible)}', flush=True)
    print(f'Routable (any depth): {overall_routable}', flush=True)


if __name__ == "__main__":
    main()
