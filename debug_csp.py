"""Sanity test: take a known optimal fold, extract its hcore, run CSP on it.
If CSP fails to find any routing, the bug is in the CSP, not the pipeline."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding.hcore_csp import solve_csp
from hp_folding.polyomino_enum import canonicalize, internal_edges
from hp_folding import perm

SEQ = 'PPHPPHPHPHHPPHPH'

# Run PERM to get a known-optimal fold
res = perm.perm(SEQ, n_tours=2000, temperature=0.5, seed=42, timeout_seconds=4.0)
fold = res.fold
print(f'PERM best contacts = {res.contacts}')
print(f'coords = {fold.coords}')

# Extract hcore from this fold
h_cells = [fold.coords[i] for i, r in enumerate(SEQ) if r == 'H']
print(f'h_cells = {h_cells}')
print(f'len = {len(h_cells)}, set len = {len(set(h_cells))}')

# Canonicalize
canon = canonicalize(h_cells)
print(f'canonical hcore = {canon}')
print(f'internal_edges = {internal_edges(set(canon))}')

# Now feed canonical hcore back into solve_csp
print('\nRunning solve_csp on this hcore...')
sol = solve_csp(SEQ, frozenset(canon), time_budget_seconds=10.0)
if sol is None:
    print('  ✗ CSP says NOT routable — BUG (we just got it from a real optimum!)')
else:
    print(f'  ✓ CSP found contacts={sol.contacts}')
    print(f'  coords = {sol.coords}')
