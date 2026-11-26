"""S6 hcore-neighborhood hunt.

We have 4 known connected-polyomino hcores at ie=26 (= 19 contacts) for S6.
Generate all 1-cell modifications of each (replace one H-cell with an
adjacent empty cell, keep size 22, keep connected). For each modification
with ie ≥ 27, check CP-SAT routability for S6's sequence.

If any routable → 20+ contact = world record.
If none routable from any of the 4 starting hcores → expand to 2-cell mods.
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, glob, os
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges

SEQ = 'PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH'
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
    """Yield distinct connected-polyomino neighbors of size-|hc| reachable
    by removing one cell + adding one cell from the resulting boundary."""
    hc = set(hc)
    seen = set()
    for cell_to_remove in list(hc):
        smaller = hc - {cell_to_remove}
        if not is_connected(smaller): continue
        # Find cells adjacent to `smaller` not already in `smaller`
        boundary = set()
        for x, y in smaller:
            for dx, dy in NBRS:
                nb = (x+dx, y+dy)
                if nb not in smaller:
                    boundary.add(nb)
        for cell_to_add in boundary:
            if cell_to_add == cell_to_remove: continue  # no-op
            new_hc = smaller | {cell_to_add}
            if not is_connected(new_hc): continue
            canon = canonicalize(new_hc)
            if canon in seen: continue
            seen.add(canon)
            yield canon


def two_cell_neighbors(hc, max_yield=20000):
    """1-cell mod, then another 1-cell mod. Capped to avoid explosion."""
    hc = set(hc)
    seen_1 = set([canonicalize(hc)])
    yielded = set()
    for first in one_cell_neighbors(hc):
        first_set = set(first)
        seen_1.add(canonicalize(first_set))
        for second in one_cell_neighbors(first_set):
            if second in seen_1 or second in yielded: continue
            yielded.add(second)
            yield second
            if len(yielded) >= max_yield: return


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


def hcore_canonical_from_fold(coords, seq):
    h_cells = [tuple(coords[i]) for i, r in enumerate(seq) if r == 'H']
    return canonicalize(h_cells)


def main():
    # Load 19-contact production folds, dedupe canonical hcores
    base = str(Path(__file__).resolve().parent / 'record_attempt_results')
    unique_hcores = {}
    for f in sorted(glob.glob(f'{base}/S6_*.json')):
        d = json.load(open(f))
        coords = [tuple(c) for c in d['coords']]
        canon = hcore_canonical_from_fold(coords, SEQ)
        unique_hcores.setdefault(canon, []).append(os.path.basename(f))
    print(f'Found {len(unique_hcores)} distinct 19-contact hcores', flush=True)
    for i, (hc, names) in enumerate(unique_hcores.items()):
        print(f'  Hcore #{i+1}: ie={internal_edges(set(hc))}, {len(names)} runs', flush=True)

    # Phase 1: 1-cell modifications
    print(f'\n=== Phase 1: 1-cell modifications ===', flush=True)
    overall_seen = set()
    candidates_high_ie = []   # (ie, canon) for ie≥27
    t0 = time.time()
    for i, hc in enumerate(unique_hcores.keys()):
        n_total = 0; n_kept = 0
        for canon in one_cell_neighbors(hc):
            n_total += 1
            if canon in overall_seen: continue
            overall_seen.add(canon)
            ie = internal_edges(set(canon))
            if ie >= 27:
                candidates_high_ie.append((ie, canon))
                n_kept += 1
        print(f'  Hcore #{i+1}: {n_total} unique 1-cell neighbors, {n_kept} with ie≥27 (cumulative)', flush=True)
    print(f'  total 1-cell candidates with ie≥27: {len(candidates_high_ie)}', flush=True)
    print(f'  enumeration: {time.time()-t0:.1f}s', flush=True)

    if candidates_high_ie:
        print(f'\n=== CP-SAT route on 1-cell candidates ===', flush=True)
        candidates_high_ie.sort(key=lambda kv: -kv[0])
        for idx, (ie, canon) in enumerate(candidates_high_ie):
            coords, status = is_routable_cpsat(SEQ, frozenset(canon), time_budget_seconds=10)
            if coords is not None:
                contacts = count_chain_contacts(SEQ, coords)
                print(f'  [{idx+1}/{len(candidates_high_ie)}] ie={ie} ROUTABLE → contacts={contacts}', flush=True)
                if contacts >= 20:
                    print(f'\n*** WORLD RECORD: contacts={contacts} on S6 ***', flush=True)
                    print(f'    hcore canonical: {canon}', flush=True)
                    print(f'    chain coords: {coords}', flush=True)
                    with open(str(Path(__file__).resolve().parent / 's6_record_fold.json'), 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': contacts,
                                    'coords': [list(c) for c in coords],
                                    'note': '1-cell mod from production 19-contact basin'}, f, indent=2)
                    print(f'    saved to s6_record_fold.json', flush=True)
                    return
            else:
                print(f'  [{idx+1}/{len(candidates_high_ie)}] ie={ie} {status} (0.0s)', flush=True)
        print(f'  → all 1-cell candidates exhausted with no record', flush=True)

    # Phase 2: 2-cell modifications
    print(f'\n=== Phase 2: 2-cell modifications ===', flush=True)
    candidates_2 = []
    t0 = time.time()
    for i, hc in enumerate(unique_hcores.keys()):
        per_hcore_kept = 0
        for canon in two_cell_neighbors(hc, max_yield=10000):
            if canon in overall_seen: continue
            overall_seen.add(canon)
            ie = internal_edges(set(canon))
            if ie >= 27:
                candidates_2.append((ie, canon))
                per_hcore_kept += 1
        print(f'  Hcore #{i+1}: 2-cell mod sweep: +{per_hcore_kept} new ie≥27 candidates', flush=True)
    print(f'  total 2-cell candidates with ie≥27: {len(candidates_2)}', flush=True)
    print(f'  enumeration: {time.time()-t0:.1f}s', flush=True)

    if candidates_2:
        print(f'\n=== CP-SAT route on 2-cell candidates ===', flush=True)
        candidates_2.sort(key=lambda kv: -kv[0])
        for idx, (ie, canon) in enumerate(candidates_2):
            coords, status = is_routable_cpsat(SEQ, frozenset(canon), time_budget_seconds=10)
            if coords is not None:
                contacts = count_chain_contacts(SEQ, coords)
                if contacts >= 20:
                    print(f'  [{idx+1}/{len(candidates_2)}] ie={ie} ROUTABLE → contacts={contacts} *** RECORD ***', flush=True)
                    with open(str(Path(__file__).resolve().parent / 's6_record_fold.json'), 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': contacts,
                                    'coords': [list(c) for c in coords],
                                    'note': '2-cell mod from production 19-contact basin'}, f, indent=2)
                    return
                # report only the high-contact hits
                if contacts >= 19:
                    print(f'  [{idx+1}/{len(candidates_2)}] ie={ie} contacts={contacts}', flush=True)
            if (idx+1) % 200 == 0:
                print(f'  [{idx+1}/{len(candidates_2)}] ... ', flush=True)
        print(f'  → all 2-cell candidates exhausted with no record', flush=True)

    print(f'\n=== EXHAUSTED 1+2-cell neighborhoods ===', flush=True)


if __name__ == "__main__":
    main()
