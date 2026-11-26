"""Direct CP-SAT formulation of the HP folding maximum-contact problem.

Decision vars:
  c_i ∈ {cell IDs in bbox}    for i = 0..N-1

Constraints:
  - AllDifferent(c_0..c_{N-1})        # SAW
  - (c_i, c_{i+1}) ∈ adjacency_pairs  # unit-step bonds, i = 0..N-2

Objective:
  maximize Σ_{i<j, both H, |i-j|>1}  contact_ij
  where contact_ij = (c_i, c_j) ∈ adjacency_pairs

This is the entire HP folding problem encoded directly. CP-SAT is an exact
solver: a returned status of OPTIMAL means we have a mathematical proof that
no chain in the chosen bbox achieves more contacts. (FEASIBLE = a valid
chain found but not proven optimal; UNKNOWN = time budget exceeded.)

Bbox sizing:
  - Larger bbox = larger search but allows more shapes.
  - Default: side = ceil(2*sqrt(N)) which fits any compact fold.
  - For record-breaking, the optimal bbox is small (compact folds dominate
    high contact counts), so a tight bbox is both faster and sufficient.
"""
from __future__ import annotations
import math
from typing import Optional

from ortools.sat.python import cp_model

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def solve_hp_cpsat(
    sequence: str,
    *,
    bbox_side: int | None = None,
    time_budget_seconds: float = 60.0,
    n_workers: int = 8,
    initial_solution: list[Coord] | None = None,
    min_contacts: int | None = None,
    log_progress: bool = False,
) -> tuple[Optional[list[Coord]], int, str]:
    """Solve HP folding directly via CP-SAT.

    Returns (coords, contacts, status). status ∈ {'OPTIMAL','FEASIBLE','INFEASIBLE','UNKNOWN'}.
    'OPTIMAL' = proven max contacts; 'FEASIBLE' = a chain found but not proven optimal;
    'UNKNOWN' = time budget exceeded with no chain found.
    """
    n = len(sequence)
    if bbox_side is None:
        bbox_side = max(4, math.ceil(2 * math.sqrt(n)))
    # Center bbox at origin
    half = bbox_side // 2
    cells = [(x, y) for x in range(-half, half + 1) for y in range(-half, half + 1)]
    cell_id = {c: i for i, c in enumerate(cells)}
    n_cells = len(cells)

    # Bipartite parity: cells split into two parities (x+y mod 2). The chain
    # alternates parities as it walks. So residue i sits on parity p_i = (i + p_0) mod 2.
    # We can prune c_i to cells of correct parity by fixing the parity of c_0.
    # We try both p_0 = 0 and p_0 = 1 — but actually the chain has D4 symmetry
    # so without loss of generality we can fix c_0's parity to 0.
    parity_cells = {0: [cid for c, cid in cell_id.items() if (c[0]+c[1]) % 2 == 0],
                     1: [cid for c, cid in cell_id.items() if (c[0]+c[1]) % 2 == 1]}

    model = cp_model.CpModel()

    # x[i] = cell id assigned to residue i, restricted to correct parity
    x = []
    for i in range(n):
        parity = i % 2  # since c_0 is parity 0 (WLOG)
        domain = parity_cells[parity]
        v = model.NewIntVarFromDomain(cp_model.Domain.FromValues(sorted(domain)), f'x_{i}')
        x.append(v)

    # SAW
    model.AddAllDifferent(x)

    # Build adjacency table once
    adjacency_pairs = []
    for c, cid in cell_id.items():
        for dx, dy in NBRS:
            nb = (c[0] + dx, c[1] + dy)
            nbid = cell_id.get(nb)
            if nbid is not None:
                adjacency_pairs.append((cid, nbid))

    # Bond constraints
    for i in range(n - 1):
        model.AddAllowedAssignments([x[i], x[i+1]], adjacency_pairs)

    # Contact indicators: for each pair (i, j) where both H and |j-i| > 1
    contact_pairs = []
    for i in range(n):
        if sequence[i] != 'H': continue
        for j in range(i + 2, n):
            if sequence[j] != 'H': continue
            # parity check: H-H contact requires opposite parity (since lattice is bipartite)
            if (i + j) % 2 == 0:
                continue  # same parity → can't be lattice-adjacent
            contact_pairs.append((i, j))

    contact_vars = []
    for i, j in contact_pairs:
        b = model.NewBoolVar(f'c_{i}_{j}')
        # b == True iff (x_i, x_j) ∈ adjacency_pairs
        model.AddAllowedAssignments([x[i], x[j]], adjacency_pairs).OnlyEnforceIf(b)
        # If b == False, no constraint on (x_i, x_j) — they may or may not be adjacent
        # But the maximization will set b=True when possible. We need:
        # adjacent → b = True, NOT adjacent → b = False.
        # For maximization to be sound, we need the IFF, not just one-way.
        # So: also encode (x_i, x_j) NOT adjacent → b = False.
        # Equivalently: b = True ↔ adjacent.
        # Use the fact that AllowedAssignments OnlyEnforceIf(b) means
        # b → (x_i, x_j) ∈ pairs. We need the converse:
        # (x_i, x_j) ∈ pairs → b. We add the negation:
        not_adjacency_pairs = [
            (a, c) for a in range(n_cells) for c in range(n_cells)
            if (a, c) not in set(adjacency_pairs) and a != c
        ]
        # The negation set is HUGE (n_cells^2). Don't expand. Use a different trick:
        # Constrain (x_i, x_j) ∈ pairs → b via reified OnlyEnforceIf(b.Not()):
        # b.Not() → (x_i, x_j) ∉ pairs.
        model.AddForbiddenAssignments([x[i], x[j]], adjacency_pairs).OnlyEnforceIf(b.Not())
        contact_vars.append(b)

    # Optional lower-bound constraint: turns the problem into "find any
    # solution with ≥ min_contacts, OR prove none exists". This decouples
    # finding from proving and dramatically speeds up record-breaking
    # attempts when the existing record is known.
    if min_contacts is not None:
        model.Add(sum(contact_vars) >= min_contacts)

    # Objective: maximize total contacts
    model.Maximize(sum(contact_vars))

    # Pin x_0 = origin to remove translation symmetry
    model.Add(x[0] == cell_id[(0, 0)])
    # Pin first move direction to remove rotation/reflection symmetry
    # Try x_1 to be (1, 0)
    model.Add(x[1] == cell_id[(1, 0)])
    # And first non-collinear move to be in y > 0 (removes reflection)
    # For each i from 2 onward, find first move that's not (1, 0) — can't easily do
    # symbolically. Skip this; CP-SAT handles it with parallel workers.

    # Initial hint
    if initial_solution is not None and len(initial_solution) == n:
        if all(c in cell_id for c in initial_solution):
            for i, c in enumerate(initial_solution):
                model.AddHint(x[i], cell_id[c])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_seconds
    solver.parameters.num_search_workers = n_workers
    solver.parameters.log_search_progress = log_progress

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        coords = [cells[solver.Value(x[i])] for i in range(n)]
        contacts = sum(solver.Value(b) for b in contact_vars)
        st = 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'
        return coords, int(contacts), st
    if status == cp_model.INFEASIBLE:
        return None, 0, 'INFEASIBLE'
    return None, 0, 'UNKNOWN'


if __name__ == "__main__":
    import time
    # Sanity: small test sequence
    seq = 'HHPHPHPH'  # n_H=5
    t0 = time.time()
    coords, contacts, st = solve_hp_cpsat(seq, time_budget_seconds=10)
    print(f'HHPHPHPH: contacts={contacts} status={st} ({time.time()-t0:.2f}s)')
    print(f'  coords={coords}')

    # H8 — known max is 3 (2x4 rect)
    seq = 'HHHHHHHH'
    t0 = time.time()
    coords, contacts, st = solve_hp_cpsat(seq, time_budget_seconds=10)
    print(f'HHHHHHHH: contacts={contacts} status={st} ({time.time()-t0:.2f}s)')
    # Note: HHHHHHHH has 7 sequence-adjacent H pairs. Internal edges in 2x4 = 10.
    # Contacts = ie - adj_pairs = 10 - 7 = 3. Yes.
