"""S6 systematic polyomino + CP-SAT search at ie levels 32 down to 27.

We already know:
  - ie=33 and ie=34 (from earlier run): 0/592 routable (all proven INFEASIBLE)
  - Production methods cap at 19 contacts (ie=26)
  - Target: 20 contacts = ie=27

This script walks ie down from 32 to 27, enumerating ALL canonical connected
polyominoes of size 22 at each ie level and CP-SAT-checking routability.
First routable shape at ie≥27 = world record on S6.
First proven sweep with no hits at ie≥27 = proof that connected-polyomino
hcores can't reach 20 contacts.
"""
from __future__ import annotations
from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.polyomino_enum import enumerate_polyominoes
from hp_folding.cpsat_routability import is_routable_cpsat

SEQ = 'PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH'
N_H = SEQ.count('H')  # 22
ADJ_HH = sum(1 for i in range(len(SEQ)-1) if SEQ[i]==SEQ[i+1]=='H')  # 7
print(f'S6: |seq|={len(SEQ)}, n_H={N_H}, adj_HH={ADJ_HH}')
print(f'  contacts = ie - {ADJ_HH}.  target ≥20 = ie≥27', flush=True)


def count_chain_contacts(seq, coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    n = 0
    for i, c in enumerate(coords):
        if seq[i] != 'H': continue
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            j = pos.get((c[0]+dx, c[1]+dy))
            if j is None or abs(j-i) <= 1: continue
            if seq[j] == 'H': n += 1
    return n // 2


def search_at_ie(target_ie: int, max_bbox: int = 10):
    print(f'\n=== ie = {target_ie} (would give contacts = {target_ie - ADJ_HH}) ===', flush=True)
    t_enum_start = time.time()
    # Enumerate exactly those at this ie
    candidates = []
    for canon, ie in enumerate_polyominoes(N_H, min_internal_edges=target_ie, max_bbox=max_bbox):
        if ie == target_ie:
            candidates.append(canon)
    enum_time = time.time() - t_enum_start
    print(f'  enumerated {len(candidates)} polyominoes at ie={target_ie} in {enum_time:.0f}s', flush=True)

    if not candidates:
        return None

    t_csp = time.time()
    n_inf = n_unk = 0
    for idx, canon in enumerate(candidates):
        coords, status = is_routable_cpsat(SEQ, frozenset(canon), time_budget_seconds=5)
        if coords is not None:
            contacts = count_chain_contacts(SEQ, coords)
            print(f'  [{idx+1}/{len(candidates)}] ie={target_ie}: ROUTABLE → contacts={contacts}', flush=True)
            return ('found', target_ie, contacts, canon, coords)
        if status == 'INFEASIBLE': n_inf += 1
        else: n_unk += 1
        if (idx+1) % 1000 == 0:
            print(f'  [{idx+1}/{len(candidates)}] ... {n_inf} infeasible, {n_unk} unknown so far '
                  f'({(time.time()-t_csp):.0f}s elapsed)', flush=True)
    print(f'  ALL {len(candidates)} infeasible at ie={target_ie}  '
          f'(n_inf={n_inf}, n_unk={n_unk}, csp_time={time.time()-t_csp:.0f}s)', flush=True)
    return ('exhausted', target_ie, 0, None, None)


def main():
    # Walk ie down from 32 (we already know ie=33,34 are all infeasible)
    for ie in [32, 31, 30, 29, 28, 27]:
        result = search_at_ie(ie, max_bbox=10)
        if result is None:
            print(f'  no polyominoes at ie={ie} with bbox≤10', flush=True)
            continue
        kind, target_ie, contacts, canon, coords = result
        if kind == 'found' and contacts >= 20:
            print(f'\n*** RECORD: contacts={contacts} on S6 at ie={target_ie} ***', flush=True)
            print(f'  hcore canon = {canon}', flush=True)
            print(f'  chain coords = {coords}', flush=True)
            return
    print(f'\n=== EXHAUSTED ie=32 down to ie=27 with no routable hcore ===', flush=True)
    print(f'    PROVEN: no connected-polyomino hcore for S6 with bbox≤10 admits ≥20 contacts.', flush=True)


if __name__ == "__main__":
    main()
