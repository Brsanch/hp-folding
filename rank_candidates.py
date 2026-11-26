"""Rank fold candidates by depth × ridge_height × cross-ridge yield.

For each unique fold in our results corpus:
  - depth = current contacts (the basin floor)
  - ridge_height = smallest +ΔE move (= largest contact count among neighbors
                                         that is < current)
  - cross_ridge_yield = max contact count after polishing each "worse" neighbor
                         (i.e., what's on the OTHER side of the lowest ridge)

A candidate is valuable when ALL three are high:
  - high depth = locally optimal
  - high ridge_height = real basin structure (not just a plateau)
  - high cross_ridge_yield = the basin across the ridge is also deep

Composite score: depth * (depth - ridge_height + 1) * cross_ridge_yield.

Lower scores = either shallow (no structure) or trap (other side worse than us).
"""
from __future__ import annotations
import sys, json, os, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding import sequences, moves
from hp_folding.verifier import count_contacts


def basin_id(coords, sequence):
    """Canonical hcore polyomino — coarse basin identifier."""
    h = [coords[i] for i, r in enumerate(sequence) if r == 'H']
    if not h: return tuple(coords)
    mx_, my = min(p[0] for p in h), min(p[1] for p in h)
    return tuple(sorted((p[0]-mx_, p[1]-my) for p in h))


def enumerate_neighbors(coords, sequence, rng, n_samples=200):
    """Sample distinct one-move neighbors with their contact counts."""
    seen = {tuple(coords)}
    out = []
    fns = [moves.corner_flip, moves.pivot, moves.pull_move_v2,
           moves.local_rebridge, moves.end_flip, moves.crankshaft]
    for _ in range(n_samples):
        for fn in fns:
            try:
                m = fn(list(coords), rng)
            except Exception:
                m = None
            if m is None: continue
            t = tuple(m)
            if t in seen: continue
            seen.add(t)
            out.append((m, count_contacts(sequence, m)))
    return out


def polish_steps(coords, sequence, rng, n_steps=15, n_samples=150):
    """Greedy descent: take best contact-improving neighbor each step."""
    cur = list(coords); cur_c = count_contacts(sequence, cur)
    for _ in range(n_steps):
        nbrs = enumerate_neighbors(cur, sequence, rng, n_samples=n_samples)
        better = [(m, c) for m, c in nbrs if c > cur_c]
        if not better: return cur, cur_c
        m, c = max(better, key=lambda x: x[1])
        cur, cur_c = m, c
    return cur, cur_c


def evaluate_fold(coords, sequence, rng, n_neighbor_samples=300, n_cross_samples=12):
    """Compute (depth, ridge_height, cross_ridge_yield) for one fold."""
    cur_c = count_contacts(sequence, coords)
    nbrs = enumerate_neighbors(coords, sequence, rng, n_samples=n_neighbor_samples)
    if not nbrs:
        return cur_c, 0, cur_c
    # Worse-energy neighbors = those with FEWER contacts.
    worse = sorted([(m, c) for m, c in nbrs if c < cur_c],
                   key=lambda x: -x[1])  # least-bad first (smallest +ΔE)
    if not worse:
        return cur_c, 0, cur_c
    # Ridge height = smallest +ΔE = (cur_c - best_worse_contacts)
    ridge_height = cur_c - worse[0][1]
    # Cross-ridge yield: polish from the top-K least-bad neighbors, take best.
    yields = []
    for m, c in worse[:n_cross_samples]:
        _, polished_c = polish_steps(m, sequence, rng, n_steps=10, n_samples=120)
        yields.append(polished_c)
    cross_yield = max(yields) if yields else cur_c
    return cur_c, ridge_height, cross_yield


def collect_unique_folds():
    """Walk every results dir, gather (sid, contacts, coords) for each unique
    fold (deduped by basin_id)."""
    out = {}   # sid -> {basin_id: (contacts, coords, source)}
    base = Path(__file__).resolve().parent
    dirs = ['perm_results', 'basin_hop_results', 'replica_exchange_results',
            'record_attempt_results', 'saddle_hop_results']
    for d in dirs:
        full = base / d
        if not full.is_dir(): continue
        for f in full.iterdir():
            if not f.suffix == '.json': continue
            try:
                data = json.load(open(f))
                sid = data.get('sequence_id')
                seq = data.get('sequence')
                coords = [tuple(c) for c in data.get('coords', [])]
                contacts = data.get('contacts')
                if not (sid and seq and coords and contacts is not None): continue
                if sid not in out: out[sid] = {}
                bid = basin_id(coords, seq)
                if bid not in out[sid] or out[sid][bid][0] < contacts:
                    out[sid][bid] = (contacts, coords, str(f.name))
            except Exception:
                continue
    return out


def main():
    rng = random.Random(0)
    folds_by_sid = collect_unique_folds()

    print(f'{"sid":4s}  {"src":40s}  {"D":>3s} {"R":>3s} {"Y":>3s}  {"score":>7s}  notes')
    print('-' * 90)

    for sid in ['S6', 'S9', 'S10']:
        if sid not in folds_by_sid: continue
        s = sequences.get(sid)
        rows = []
        # Take top 8 unique folds per sid (by contacts) to keep wall time reasonable.
        items = sorted(folds_by_sid[sid].values(), key=lambda x: -x[0])[:8]
        for contacts, coords, source in items:
            d, r, y = evaluate_fold(coords, s.sequence, rng,
                                      n_neighbor_samples=200, n_cross_samples=8)
            score = d * max(1, r) * y   # composite
            rows.append((source, d, r, y, score))
        rows.sort(key=lambda x: -x[4])
        for source, d, r, y, score in rows:
            print(f'{sid:4s}  {source[:38]:40s}  {d:>3d} {r:>3d} {y:>3d}  {score:>7d}')
        print()

    print('Legend: D=depth(contacts), R=ridge_height(smallest +ΔE), Y=cross-ridge yield')
    print('        score = D × max(1,R) × Y')
    print('  HIGH SCORE = deep basin, real ridge structure, polished other side is also good')


if __name__ == "__main__":
    main()
