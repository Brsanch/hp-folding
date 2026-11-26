"""Grid-occupancy CP-SAT formulation for exact 2D HP lattice folding.

Uses 0-1 cell occupancy variables p[i, c] (residue i at lattice cell c),
H-occupancy h[c], and lattice-edge contact variables z[u, v] <= h[u], h[v].
Compared to residue-pair table encodings, this gives:
  - Pure linear inequalities (strong LP relaxation + fast clause propagation)
  - Exact per-H-residue degree cuts (capturing parity & backbone P-neighbors)
  - Exact WLOG symmetry breaking by pinning a sequential bond at the grid center
  - Manhattan-distance reachability pruning from the pinned bond
  - Support for BOTH connected and disconnected H-cores automatically.
"""
from __future__ import annotations

import time
from typing import Optional

from ortools.sat.python import cp_model

from hp_folding.verifier import verify

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _max_hcore_deg(seq: str, i: int) -> int:
    """Upper bound on H-core graph degree of H-residue i."""
    n = len(seq)
    seq_h = (i > 0 and seq[i - 1] == "H") + (i < n - 1 and seq[i + 1] == "H")
    max_nonseq = 3 if i in (0, n - 1) else 2
    return seq_h + max_nonseq


def solve_grid_cpsat(
    sequence: str,
    *,
    grid_w: int = 11,
    grid_h: int = 11,
    min_contacts: int | None = None,
    pin_index: int | None = None,
    time_budget_seconds: float = 60.0,
    n_workers: int = 8,
    hint_coords: list[Coord] | None = None,
    log_progress: bool = False,
) -> tuple[Optional[list[Coord]], int, str, dict]:
    """Solve 2D HP folding on a bond-centered grid_w x grid_h box using occupancy CP-SAT.

    Pins the selected bond (pin_index, pin_index + 1) near the grid center to fix
    translation and 4-fold rotation. Note: because the finite grid boundary breaks
    translation invariance, an INFEASIBLE result refutes all folds fitting inside
    the fixed box relative to the pinned bond (unconditionally covering all bounding
    boxes up to floor((grid_w+1)/2) x floor((grid_h+1)/2) when centered).
    """
    t0 = time.time()
    n = len(sequence)
    adj_hh = sum(1 for i in range(n - 1) if sequence[i] == "H" and sequence[i + 1] == "H")

    # Choose pinned bond (pin_idx, pin_idx + 1) near center of sequence
    if pin_index is None:
        # Prefer an HH bond near n // 2 if one exists, else n // 2
        mid = n // 2
        hh_bonds = [i for i in range(n - 1) if sequence[i] == "H" and sequence[i + 1] == "H"]
        if hh_bonds:
            pin_index = min(hh_bonds, key=lambda i: abs(i - mid))
        else:
            pin_index = mid

    cx, cy = grid_w // 2, grid_h // 2
    # Ensure parity of (cx, cy) matches pin_index % 2
    if (cx + cy) % 2 != pin_index % 2:
        cx += 1
    p0_cell = (cx, cy)
    p1_cell = (cx + 1, cy)

    cells = [(x, y) for x in range(grid_w) for y in range(grid_h)]
    cell_set = set(cells)
    nbrs_of: dict[Coord, list[Coord]] = {
        c: [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in cell_set]
        for c in cells
    }

    # Valid reachable cells for each residue i (parity + Manhattan reachability from pin)
    valid_cells: list[list[Coord]] = []
    for i in range(n):
        if i == pin_index:
            valid_cells.append([p0_cell])
        elif i == pin_index + 1:
            valid_cells.append([p1_cell])
        else:
            d0 = abs(i - pin_index)
            d1 = abs(i - (pin_index + 1))
            par = i % 2
            vc = [
                c
                for c in cells
                if (c[0] + c[1]) % 2 == par
                and abs(c[0] - p0_cell[0]) + abs(c[1] - p0_cell[1]) <= d0
                and abs(c[0] - p1_cell[0]) + abs(c[1] - p1_cell[1]) <= d1
            ]
            valid_cells.append(vc)

    model = cp_model.CpModel()

    # p[(i, c)] = 1 iff residue i is at cell c
    p: dict[tuple[int, Coord], cp_model.IntVar] = {}
    cell_to_res: dict[Coord, list[int]] = {c: [] for c in cells}
    for i in range(n):
        for c in valid_cells[i]:
            v = model.NewBoolVar(f"p_{i}_{c[0]}_{c[1]}")
            p[(i, c)] = v
            cell_to_res[c].append(i)

    # 1. Exactly one cell per residue
    for i in range(n):
        model.AddExactlyOne([p[(i, c)] for c in valid_cells[i]])

    # 2. At most one residue per cell (self-avoidance)
    for c, rlist in cell_to_res.items():
        if len(rlist) > 1:
            model.AddAtMostOne([p[(i, c)] for i in rlist])

    # 3. Chain adjacency (both directions for tight propagation)
    for i in range(n - 1):
        vc_i = set(valid_cells[i])
        vc_next = set(valid_cells[i + 1])
        for c in valid_cells[i]:
            nbs = [nb for nb in nbrs_of[c] if nb in vc_next]
            if not nbs:
                model.Add(p[(i, c)] == 0)
            else:
                model.Add(p[(i, c)] <= sum(p[(i + 1, nb)] for nb in nbs))
        for c in valid_cells[i + 1]:
            nbs = [nb for nb in nbrs_of[c] if nb in vc_i]
            if not nbs:
                model.Add(p[(i + 1, c)] == 0)
            else:
                model.Add(p[(i + 1, c)] <= sum(p[(i, nb)] for nb in nbs))

    # 4. H-occupancy per cell
    h_var: dict[Coord, cp_model.IntVar] = {}
    h_res_at: dict[Coord, list[int]] = {}
    for c in cells:
        h_list = [i for i in cell_to_res[c] if sequence[i] == "H"]
        if h_list:
            hv = model.NewBoolVar(f"h_{c[0]}_{c[1]}")
            model.Add(hv == sum(p[(i, c)] for i in h_list))
            h_var[c] = hv
            h_res_at[c] = h_list

    # 5. Lattice edge variables z[(u, v)] for u even, v odd, both capable of holding H
    z_vars: list[cp_model.IntVar] = []
    inc_edges: dict[Coord, list[cp_model.IntVar]] = {c: [] for c in h_var}
    for u, hu in h_var.items():
        if (u[0] + u[1]) % 2 != 0:
            continue
        for v in nbrs_of[u]:
            if v not in h_var:
                continue
            hv = h_var[v]
            ze = model.NewBoolVar(f"z_{u[0]}_{u[1]}_{v[0]}_{v[1]}")
            model.Add(ze <= hu)
            model.Add(ze <= hv)
            z_vars.append(ze)
            inc_edges[u].append(ze)
            inc_edges[v].append(ze)

    # 6. Per-cell H-degree valid cuts (tightens LP relaxation to parity UB at root!)
    for c, edges in inc_edges.items():
        if not edges:
            continue
        rhs = sum(_max_hcore_deg(sequence, i) * p[(i, c)] for i in h_res_at[c])
        model.Add(sum(edges) <= rhs)

    total_ie = sum(z_vars)
    if min_contacts is not None:
        model.Add(total_ie >= min_contacts + adj_hh)
    else:
        model.Maximize(total_ie)

    # Optional warm-start hint (align pinned bond to p0_cell -> p1_cell)
    if hint_coords is not None and len(hint_coords) == n:
        c0 = hint_coords[pin_index]
        c1 = hint_coords[pin_index + 1]
        dx, dy = c1[0] - c0[0], c1[1] - c0[1]
        # Rotate (dx, dy) to (1, 0)
        if (dx, dy) == (1, 0):
            rot = lambda x, y: (x, y)
        elif (dx, dy) == (-1, 0):
            rot = lambda x, y: (-x, -y)
        elif (dx, dy) == (0, 1):
            rot = lambda x, y: (y, -x)
        else:
            rot = lambda x, y: (-y, x)
        aligned = []
        for x, y in hint_coords:
            rx, ry = rot(x - c0[0], y - c0[1])
            aligned.append((rx + p0_cell[0], ry + p0_cell[1]))
        if all((i, aligned[i]) in p for i in range(n)):
            for i in range(n):
                for c in valid_cells[i]:
                    model.AddHint(p[(i, c)], 1 if c == aligned[i] else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_seconds
    solver.parameters.num_search_workers = n_workers
    solver.parameters.log_search_progress = log_progress

    status = solver.Solve(model)
    elapsed = time.time() - t0
    stats = {
        "elapsed": round(elapsed, 3),
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
        "best_bound": solver.BestObjectiveBound() - adj_hh if min_contacts is None else None,
        "grid": f"{grid_w}x{grid_h}",
        "pin_index": pin_index,
        "n_vars": len(p),
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        coords: list[Coord] = []
        for i in range(n):
            chosen = next(c for c in valid_cells[i] if solver.Value(p[(i, c)]) == 1)
            coords.append(chosen)
        v = verify(sequence, coords)
        assert v.valid, f"CP-SAT returned invalid fold: {v.error}"
        st = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        return coords, v.contacts, st, stats
    if status == cp_model.INFEASIBLE:
        return None, 0, "INFEASIBLE", stats
    return None, 0, "UNKNOWN", stats


if __name__ == "__main__":
    from hp_folding.sequences import BENCHMARKS

    for sid, gw in [("S1", 9), ("S4", 11)]:
        b = BENCHMARKS[sid]
        coords, contacts, st, stats = solve_grid_cpsat(
            b.sequence, grid_w=gw, grid_h=gw, time_budget_seconds=30, n_workers=8
        )
        print(f"{sid} ({gw}x{gw}): status={st}, contacts={contacts} (best={b.best_known}), stats={stats}")
