"""S9 hcore-neighborhood hunt. Same shape as the S6 analysis."""
from __future__ import annotations
from pathlib import Path
import sys, time, json, glob, os
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges
from hp_folding import sequences

SEQ = sequences.BENCHMARKS['S9'].sequence
ADJ_HH = sum(1 for i in range(len(SEQ)-1) if SEQ[i]==SEQ[i+1]=='H')  # 49
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


def two_cell_neighbors(hc, max_yield=20000):
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


def hcore_canon_from_fold(coords, seq):
    h_cells = [tuple(coords[i]) for i, r in enumerate(seq) if r == 'H']
    return canonicalize(h_cells)


def main():
    base = str(Path(__file__).resolve().parent)
    # Collect ALL 52-contact S9 folds across results dirs
    print(f'Collecting 52-contact S9 folds...', flush=True)
    unique_hcores = {}
    n_folds = 0
    for f in glob.glob(f'{base}/**/*S9*.json', recursive=True):
        if 'submission' in f.lower(): continue  # use prod-grade folds
        try:
            d = json.load(open(f))
            if d.get('sequence') != SEQ: continue
            coords = d.get('coords') or d.get('best_coords')
            c = d.get('contacts') or (count_chain_contacts(SEQ, coords) if coords else 0)
            if c == 52 and coords:
                n_folds += 1
                canon = hcore_canon_from_fold(coords, SEQ)
                unique_hcores.setdefault(canon, []).append(os.path.basename(f))
        except: pass
    # Also include the submission file
    for f in [f'{base}/submissions/S9.json', f'{base}/submissions/2D85.json']:
        try:
            d = json.load(open(f))
            if d.get('sequence') != SEQ: continue
            coords = d.get('coords')
            c = d.get('contacts') or count_chain_contacts(SEQ, coords)
            if c == 52 and coords:
                n_folds += 1
                canon = hcore_canon_from_fold(coords, SEQ)
                unique_hcores.setdefault(canon, []).append(os.path.basename(f))
        except: pass
    print(f'  Found {n_folds} 52-contact S9 folds, {len(unique_hcores)} distinct canonical hcores', flush=True)
    for i, (hc, names) in enumerate(unique_hcores.items()):
        ie = internal_edges(set(hc))
        print(f'  Hcore #{i+1}: ie={ie}, {len(names)} runs: {names[:3]}{"..." if len(names) > 3 else ""}', flush=True)

    if not unique_hcores:
        print('  No 52-contact S9 folds found', flush=True)
        return

    print(f'\n=== Phase 1: 1-cell modifications (target ie ≥ {53+ADJ_HH}) ===', flush=True)
    overall_seen = set()
    candidates = []
    t0 = time.time()
    for i, hc in enumerate(unique_hcores.keys()):
        n_kept = 0
        for canon in one_cell_neighbors(hc):
            if canon in overall_seen: continue
            overall_seen.add(canon)
            ie = internal_edges(set(canon))
            if ie >= 53 + ADJ_HH:
                candidates.append((ie, canon))
                n_kept += 1
        print(f'  Hcore #{i+1}: +{n_kept} new ie≥{53+ADJ_HH} candidates (cumulative)', flush=True)
    print(f'  total: {len(candidates)} candidates  enum_time={time.time()-t0:.1f}s', flush=True)

    if candidates:
        print(f'\n=== CP-SAT route on 1-cell candidates ===', flush=True)
        candidates.sort(key=lambda kv: -kv[0])
        for idx, (ie, canon) in enumerate(candidates):
            coords, status = is_routable_cpsat(SEQ, frozenset(canon),
                                                 time_budget_seconds=20)
            if coords is not None:
                contacts = count_chain_contacts(SEQ, coords)
                print(f'  [{idx+1}/{len(candidates)}] ie={ie} ROUTABLE → contacts={contacts}', flush=True)
                if contacts >= 53:
                    print(f'\n*** WORLD RECORD: contacts={contacts} on S9 ***', flush=True)
                    with open(f'{base}/s9_record_fold.json', 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': contacts,
                                    'coords': [list(c) for c in coords],
                                    'note': '1-cell mod from 52-contact S9 basin'}, f, indent=2)
                    return
            elif idx < 20 or idx % 50 == 0:
                print(f'  [{idx+1}/{len(candidates)}] ie={ie} {status}', flush=True)
        print(f'  → all 1-cell candidates exhausted with no record', flush=True)

    # Phase 2: 2-cell mods
    print(f'\n=== Phase 2: 2-cell modifications ===', flush=True)
    cands_2 = []
    t0 = time.time()
    for i, hc in enumerate(unique_hcores.keys()):
        per = 0
        for canon in two_cell_neighbors(hc, max_yield=10000):
            if canon in overall_seen: continue
            overall_seen.add(canon)
            ie = internal_edges(set(canon))
            if ie >= 53 + ADJ_HH:
                cands_2.append((ie, canon))
                per += 1
        print(f'  Hcore #{i+1}: +{per} new candidates (cumulative)', flush=True)
    print(f'  total 2-cell: {len(cands_2)}  enum_time={time.time()-t0:.1f}s', flush=True)

    if cands_2:
        print(f'\n=== CP-SAT route on 2-cell candidates ===', flush=True)
        cands_2.sort(key=lambda kv: -kv[0])
        for idx, (ie, canon) in enumerate(cands_2):
            coords, status = is_routable_cpsat(SEQ, frozenset(canon),
                                                 time_budget_seconds=15)
            if coords is not None:
                contacts = count_chain_contacts(SEQ, coords)
                if contacts >= 53:
                    print(f'  [{idx+1}/{len(cands_2)}] ie={ie} contacts={contacts} *** RECORD ***', flush=True)
                    with open(f'{base}/s9_record_fold.json', 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': contacts,
                                    'coords': [list(c) for c in coords],
                                    'note': '2-cell mod from S9 basin'}, f, indent=2)
                    return
            if (idx+1) % 200 == 0:
                print(f'  [{idx+1}/{len(cands_2)}] progress', flush=True)
        print(f'  → all 2-cell candidates exhausted with no record', flush=True)

    print(f'\n=== EXHAUSTED 1+2-cell neighborhoods on S9 ===', flush=True)


if __name__ == "__main__":
    main()
