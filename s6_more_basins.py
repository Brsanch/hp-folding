"""S6: discover more 19-contact basins, then run hcore-neighborhood hunt
on EACH new basin. Strategy mimics S9 success: more starting points →
wider coverage → maybe one neighborhood contains a routable 20+.
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, glob, os, random, math
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hp_folding import sequences, search, fold as fold_mod, moves
from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges

SEQ = sequences.BENCHMARKS['S6'].sequence
NBRS = ((1,0),(-1,0),(0,1),(0,-1))


def macro_plus(coords, rng):
    seq = SEQ
    u = rng.random()
    if   u < 0.20: return moves.pull_move_v2(coords, rng)
    elif u < 0.32: return moves.lmw_pull_full(coords, rng)
    elif u < 0.45: return moves.pivot(coords, rng)
    elif u < 0.55: return moves.local_rebridge(coords, rng)
    elif u < 0.65: return moves.compact_pivot(coords, rng, seq)
    elif u < 0.73: return moves.bond_rebridge(coords, rng)
    elif u < 0.79: return moves.cluster_merge(coords, rng, seq)
    elif u < 0.85: return moves.fragment_random_walk(coords, rng)
    elif u < 0.91: return moves.subchain_reflect(coords, rng, seq)
    elif u < 0.96: return moves.crankshaft(coords, rng)
    else:          return moves.end_flip(coords, rng)


def is_connected(cells):
    cells = set(cells)
    if not cells: return True
    stk = [next(iter(cells))]; seen = {stk[0]}
    while stk:
        x,y = stk.pop()
        for dx,dy in NBRS:
            nb = (x+dx, y+dy)
            if nb in cells and nb not in seen:
                seen.add(nb); stk.append(nb)
    return len(seen) == len(cells)


def hcore_canon(coords):
    h = [tuple(coords[i]) for i, r in enumerate(SEQ) if r == 'H']
    return canonicalize(h)


def count_contacts(coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    n = 0
    for i, c in enumerate(coords):
        if SEQ[i] != 'H': continue
        for dx, dy in NBRS:
            j = pos.get((c[0]+dx, c[1]+dy))
            if j is None or abs(j-i) <= 1: continue
            if SEQ[j] == 'H': n += 1
    return n // 2


def one_cell_neighbors(hc):
    hc = set(hc); seen = set()
    for cell_to_remove in list(hc):
        smaller = hc - {cell_to_remove}
        if not is_connected(smaller): continue
        boundary = set()
        for x, y in smaller:
            for dx, dy in NBRS:
                nb = (x+dx, y+dy)
                if nb not in smaller: boundary.add(nb)
        for cell_to_add in boundary:
            if cell_to_add == cell_to_remove: continue
            new_hc = smaller | {cell_to_add}
            if not is_connected(new_hc): continue
            canon = canonicalize(new_hc)
            if canon in seen: continue
            seen.add(canon)
            yield canon


def two_cell_neighbors(hc, max_yield=10000):
    hc = set(hc); seen_1 = set([canonicalize(hc)])
    yielded = set()
    for first in one_cell_neighbors(hc):
        first_set = set(first)
        seen_1.add(canonicalize(first_set))
        for second in one_cell_neighbors(first_set):
            if second in seen_1 or second in yielded: continue
            yielded.add(second)
            yield second
            if len(yielded) >= max_yield: return


def main():
    print(f'S6 basin discovery + neighborhood hunt', flush=True)
    # Load current 4 known basins
    base = str(Path(__file__).resolve().parent)
    known_basins = set()
    for f in sorted(glob.glob(f'{base}/record_attempt_results/S6_*.json')):
        d = json.load(open(f))
        if d.get('sequence') != SEQ: continue
        known_basins.add(hcore_canon([tuple(c) for c in d['coords']]))
    print(f'  Starting with {len(known_basins)} known 19-contact basins', flush=True)

    # Discover more basins via replica exchange × N seeds × 5 min each
    new_basins = set()
    n_temps = 8
    log_tmin = math.log(0.10); log_tmax = math.log(3.5)
    temps = tuple(math.exp(log_tmin + (log_tmax-log_tmin)*i/(n_temps-1)) for i in range(n_temps))

    for seed in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]:
        t0 = time.time()
        # Random initial folds (no warm start)
        try:
            res = search.replica_exchange(
                SEQ, temperatures=temps, iterations=4_000_000,
                exchange_every=1000, seed=seed, move_fn=macro_plus,
                initial_folds=None,
            )
            elapsed = time.time() - t0
            if res.contacts == 19:
                canon = hcore_canon(res.fold.coords)
                if canon in known_basins or canon in new_basins:
                    print(f'  seed {seed}: 19 contacts, hcore already known ({elapsed:.0f}s)', flush=True)
                else:
                    new_basins.add(canon)
                    print(f'  seed {seed}: 19 contacts, NEW BASIN (#{len(new_basins)}) ({elapsed:.0f}s)', flush=True)
            elif res.contacts > 19:
                print(f'  seed {seed}: contacts={res.contacts}!! ({elapsed:.0f}s)', flush=True)
                if res.contacts >= 20:
                    print(f'\n*** RECORD: contacts={res.contacts} on S6 (RE seed {seed}) ***', flush=True)
                    with open(f'{base}/s6_record_fold.json', 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': res.contacts,
                                    'coords': [list(c) for c in res.fold.coords]}, f, indent=2)
                    return
            else:
                print(f'  seed {seed}: contacts={res.contacts} (below 19) ({elapsed:.0f}s)', flush=True)
        except Exception as e:
            print(f'  seed {seed}: ERROR {e}', flush=True)

    all_basins = known_basins | new_basins
    print(f'\n  Total basins now: {len(all_basins)} (was {len(known_basins)}, +{len(new_basins)})', flush=True)

    if not new_basins:
        print(f'  No new basins found. The 4 we have may exhaust the 19-basin set.', flush=True)
        return

    # Run 1-cell + 2-cell hcore mods on EACH new basin
    print(f'\n=== Hcore-neighborhood hunt on {len(new_basins)} new basins ===', flush=True)
    overall_seen = set()
    candidates = []
    for i, hc in enumerate(new_basins):
        for canon in one_cell_neighbors(hc):
            if canon in overall_seen: continue
            overall_seen.add(canon)
            ie = internal_edges(set(canon))
            if ie >= 27: candidates.append((ie, canon))
        for canon in two_cell_neighbors(hc, max_yield=5000):
            if canon in overall_seen: continue
            overall_seen.add(canon)
            ie = internal_edges(set(canon))
            if ie >= 27: candidates.append((ie, canon))
        print(f'  basin #{i+1}: cumulative candidates = {len(candidates)}', flush=True)

    print(f'  total candidates: {len(candidates)}', flush=True)
    if candidates:
        candidates.sort(key=lambda kv: -kv[0])
        for idx, (ie, canon) in enumerate(candidates):
            coords, status = is_routable_cpsat(SEQ, frozenset(canon),
                                                 time_budget_seconds=10)
            if coords is not None:
                contacts = count_contacts(coords)
                print(f'  [{idx+1}/{len(candidates)}] ie={ie} ROUTABLE → contacts={contacts}', flush=True)
                if contacts >= 20:
                    print(f'\n*** RECORD: contacts={contacts} on S6 (1/2-cell mod from new basin) ***', flush=True)
                    with open(f'{base}/s6_record_fold.json', 'w') as f:
                        json.dump({'sequence': SEQ, 'contacts': contacts,
                                    'coords': [list(c) for c in coords]}, f, indent=2)
                    return
            if (idx+1) % 200 == 0:
                print(f'  [{idx+1}/{len(candidates)}] progress', flush=True)
        print(f'  → all candidates exhausted with no record', flush=True)


if __name__ == "__main__":
    main()
