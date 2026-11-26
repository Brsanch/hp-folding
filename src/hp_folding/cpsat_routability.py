"""CP-SAT routability oracle for HP folding.

Given a sequence and a candidate hcore (set of lattice cells), decide
whether there exists a self-avoiding walk c_0..c_{N-1} on Z² such that
each H-residue lands in hcore and each P-residue lands outside hcore.

Formulation as a CP-SAT problem (OR-tools):
  - Each cell in the bbox is assigned an integer ID.
  - For residue i, variable x_i ∈ {valid cell IDs by residue type}.
  - AllDifferent over (x_0..x_{N-1}).
  - Adjacency: for each consecutive (i, i+1), enforce
       (x_i, x_{i+1}) ∈ {(c, c') : c, c' adjacent on lattice}.

CP-SAT handles AllDifferent and table constraints natively and prunes
much harder than the hand-rolled Python DFS. Empirically the order-of-
magnitude speedup makes n_H=40+ tractable.

Returns: (list[Coord], 'FEASIBLE'|'OPTIMAL') if routable, (None, 'INFEASIBLE')
if proven unroutable, or (None, 'UNKNOWN') if the time budget expires.
"""
from __future__ import annotations
from typing import Optional

from ortools.sat.python import cp_model

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def is_routable_cpsat(
    sequence: str,
    hcore: frozenset[Coord],
    *,
    time_budget_seconds: float = 30.0,
    bbox_pad: int = 4,
    n_workers: int = 8,
    max_p_offset: int | None = None,
) -> tuple[Optional[list[Coord]], str]:
    """Decide routability via CP-SAT.

    Returns (coords, status) where status ∈ {'OPTIMAL','FEASIBLE','INFEASIBLE','UNKNOWN'}.
    coords is the SAW if status is OPTIMAL or FEASIBLE, else None.
    """
    n = len(sequence)
    n_h_seq = sequence.count('H')
    if n_h_seq != len(hcore):
        return None, 'INFEASIBLE'

    # Build cell domain: bbox around hcore, padded
    xs = [c[0] for c in hcore]; ys = [c[1] for c in hcore]
    xmin = min(xs) - bbox_pad
    xmax = max(xs) + bbox_pad
    ymin = min(ys) - bbox_pad
    ymax = max(ys) + bbox_pad
    if max_p_offset is not None:
        # tighter bbox: P residues stay within max_p_offset of any hcore cell
        xmin = min(xs) - max_p_offset
        xmax = max(xs) + max_p_offset
        ymin = min(ys) - max_p_offset
        ymax = max(ys) + max_p_offset

    cells = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
    cell_id = {c: i for i, c in enumerate(cells)}
    hcore_ids = frozenset(cell_id[c] for c in hcore if c in cell_id)
    if len(hcore_ids) != len(hcore):
        # hcore extends past bbox — infeasible (shouldn't happen)
        return None, 'INFEASIBLE'

    h_domain = sorted(hcore_ids)
    p_domain = sorted(set(range(len(cells))) - hcore_ids)

    # Adjacency table: for each cell id, list of neighbor ids
    adj: list[list[int]] = [[] for _ in range(len(cells))]
    for i, (x, y) in enumerate(cells):
        for dx, dy in NBRS:
            nb = (x + dx, y + dy)
            if nb in cell_id:
                adj[i].append(cell_id[nb])

    model = cp_model.CpModel()

    # x[i] = cell id assigned to residue i
    x = []
    for i, r in enumerate(sequence):
        domain = h_domain if r == 'H' else p_domain
        v = model.NewIntVarFromDomain(cp_model.Domain.FromValues(domain), f'x_{i}')
        x.append(v)

    # SAW: all cells distinct
    model.AddAllDifferent(x)

    # Adjacency: (x_i, x_{i+1}) ∈ adjacency_pairs
    adjacency_pairs = []
    for c, neighbors in enumerate(adj):
        for nb in neighbors:
            adjacency_pairs.append((c, nb))
    for i in range(n - 1):
        model.AddAllowedAssignments([x[i], x[i+1]], adjacency_pairs)

    # No residue-level symmetry pinning: any H residue (not just the first)
    # may land at the lex-smallest hcore cell. The bbox itself sits inside a
    # finite domain so translation symmetry is bounded; CP-SAT handles the
    # remaining D4 freedom efficiently with parallel workers.

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_seconds
    solver.parameters.num_search_workers = n_workers
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        coords = [cells[solver.Value(x[i])] for i in range(n)]
        st = 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'
        return coords, st
    if status == cp_model.INFEASIBLE:
        return None, 'INFEASIBLE'
    return None, 'UNKNOWN'  # timeout or model error


if __name__ == "__main__":
    import time
    # H8 sanity
    seq = 'HHHHHHHH'
    hc = frozenset([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1)])  # 4x2
    t0 = time.time()
    coords, st = is_routable_cpsat(seq, hc, time_budget_seconds=10)
    print(f'H8 on 4x2: {st} coords={coords} ({time.time()-t0:.2f}s)')

    # L16 sanity
    seq = 'PPHPPHPHPHHPPHPH'
    # Pick an arbitrary hcore matching n_H=7 — try a connected polyomino
    hc = frozenset([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2)])  # P-pentomino + 2
    t0 = time.time()
    coords, st = is_routable_cpsat(seq, hc, time_budget_seconds=10)
    print(f'L16 on size-7 polyomino: {st} ({time.time()-t0:.2f}s)')
    if coords:
        print(f'  coords={coords}')
