"""Spatial-Window CP-SAT Large Neighborhood Search (LNS) for 2D HP folding.

Starts from a valid fold and repeatedly:
  1. Selects a spatial window (or ball) on Z^2 containing ~18-32 residues
     (often spanning multiple distant sequence strands simultaneously).
  2. Freezes all residues outside the window and builds an exact grid-occupancy
     CP-SAT subproblem for the freed residues with exact parity & Manhattan
     reachability pruning.
  3. Maximizes `1000 * delta_contacts + changed_h_cells`, so it:
     - Immediately accepts any strict contact improvement (+1 or more), and
     - On equal-contact plateaus, actively shifts H-core cells to hop across
       structurally distinct basins.
"""
from __future__ import annotations

import random
import time
from typing import Optional

from ortools.sat.python import cp_model

from hp_folding.verifier import verify

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def select_spatial_window(
    sequence: str,
    coords: list[Coord],
    rng: random.Random,
    *,
    min_free: int = 16,
    max_free: int = 28,
) -> set[int]:
    """Pick a spatial ball around a random H-residue containing [min_free, max_free] residues."""
    n = len(coords)
    h_indices = [i for i, r in enumerate(sequence) if r == "H"]
    center_idx = rng.choice(h_indices) if h_indices else rng.randrange(n)
    cx, cy = coords[center_idx]

    # Sort all residues by Chebyshev + slight jittered Manhattan distance from center
    order = sorted(
        range(n),
        key=lambda i: (
            max(abs(coords[i][0] - cx), abs(coords[i][1] - cy)),
            abs(coords[i][0] - cx) + abs(coords[i][1] - cy),
            rng.random(),
        ),
    )
    k = rng.randint(min_free, min(max_free, n))
    return set(order[:k])


def lns_step_cpsat(
    sequence: str,
    coords: list[Coord],
    free_indices: set[int],
    *,
    pad: int = 2,
    time_budget_seconds: float = 3.0,
    n_workers: int = 8,
    require_improvement: bool = False,
    maximize_diversity: bool = True,
) -> tuple[Optional[list[Coord]], int, str, dict]:
    """Exactly re-optimize `free_indices` via grid-occupancy CP-SAT."""
    t0 = time.time()
    n = len(sequence)
    base_v = verify(sequence, coords)
    assert base_v.valid, f"Invalid input fold: {base_v.error}"

    fixed_indices = set(range(n)) - free_indices
    fixed_cells = {coords[i] for i in fixed_indices}
    fixed_h_cells = {coords[i] for i in fixed_indices if sequence[i] == "H"}

    # Build bounding box around freed residues
    fxs = [coords[i][0] for i in free_indices]
    fys = [coords[i][1] for i in free_indices]
    xmin, xmax = min(fxs) - pad, max(fxs) + pad
    ymin, ymax = min(fys) - pad, max(fys) + pad

    free_grid = [
        (x, y)
        for x in range(xmin, xmax + 1)
        for y in range(ymin, ymax + 1)
        if (x, y) not in fixed_cells
    ]
    free_grid_set = set(free_grid)
    nbrs_of = {
        c: [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in free_grid_set]
        for c in free_grid
    }

    # Decompose free_indices into maximal contiguous runs [a..b] for exact reachability
    sorted_free = sorted(free_indices)
    runs: list[tuple[int, int]] = []
    start = prev = sorted_free[0]
    for idx in sorted_free[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))

    valid_cells: dict[int, list[Coord]] = {}
    for a, b in runs:
        left_anchor = coords[a - 1] if a > 0 else None
        right_anchor = coords[b + 1] if b < n - 1 else None
        for i in range(a, b + 1):
            par = (coords[i][0] + coords[i][1]) % 2
            vc = []
            for c in free_grid:
                if (c[0] + c[1]) % 2 != par:
                    continue
                if left_anchor is not None:
                    if abs(c[0] - left_anchor[0]) + abs(c[1] - left_anchor[1]) > i - (a - 1):
                        continue
                if right_anchor is not None:
                    if abs(c[0] - right_anchor[0]) + abs(c[1] - right_anchor[1]) > (b + 1) - i:
                        continue
                vc.append(c)
            valid_cells[i] = vc

    model = cp_model.CpModel()
    p: dict[tuple[int, Coord], cp_model.IntVar] = {}
    cell_to_res: dict[Coord, list[int]] = {c: [] for c in free_grid}
    for i in sorted_free:
        for c in valid_cells[i]:
            p[(i, c)] = model.NewBoolVar(f"p_{i}_{c[0]}_{c[1]}")
            cell_to_res[c].append(i)

    # 1. Exactly one cell per free residue
    for i in sorted_free:
        model.AddExactlyOne([p[(i, c)] for c in valid_cells[i]])

    # 2. At most one free residue per free cell
    for c, rlist in cell_to_res.items():
        if len(rlist) > 1:
            model.AddAtMostOne([p[(i, c)] for i in rlist])

    # 3. Chain adjacency within free runs and at fixed boundary anchors
    for a, b in runs:
        if a > 0:
            anc = coords[a - 1]
            allowed = {
                (anc[0] + dx, anc[1] + dy) for dx, dy in NBRS
            }
            for c in valid_cells[a]:
                if c not in allowed:
                    model.Add(p[(a, c)] == 0)
        if b < n - 1:
            anc = coords[b + 1]
            allowed = {
                (anc[0] + dx, anc[1] + dy) for dx, dy in NBRS
            }
            for c in valid_cells[b]:
                if c not in allowed:
                    model.Add(p[(b, c)] == 0)
        for i in range(a, b):
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

    # 4. H-occupancy on free cells
    h_var: dict[Coord, cp_model.IntVar] = {}
    for c in free_grid:
        hl = [i for i in cell_to_res[c] if sequence[i] == "H"]
        if hl:
            hv = model.NewBoolVar(f"h_{c[0]}_{c[1]}")
            model.Add(hv == sum(p[(i, c)] for i in hl))
            h_var[c] = hv

    # 5. Free-fixed and free-free H-H edges
    edge_terms = []
    for c, hv in h_var.items():
        w_fix = sum(1 for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in fixed_h_cells)
        if w_fix > 0:
            edge_terms.append(w_fix * hv)

    for u, hu in h_var.items():
        if (u[0] + u[1]) % 2 != 0:
            continue
        for v in nbrs_of[u]:
            if v not in h_var:
                continue
            ze = model.NewBoolVar(f"z_{u[0]}_{u[1]}_{v[0]}_{v[1]}")
            model.Add(ze <= hu)
            model.Add(ze <= h_var[v])
            edge_terms.append(ze)

    # Compute baseline value of edge_terms on the current fold
    cur_free_h = {coords[i] for i in free_indices if sequence[i] == "H"}
    cur_free_fixed = sum(
        1
        for c in cur_free_h
        for dx, dy in NBRS
        if (c[0] + dx, c[1] + dy) in fixed_h_cells
    )
    cur_free_free = sum(
        1
        for u in cur_free_h
        if (u[0] + u[1]) % 2 == 0
        for dx, dy in NBRS
        if (u[0] + dx, u[1] + dy) in cur_free_h
    )
    base_sub_edges = cur_free_fixed + cur_free_free

    sub_edges = sum(edge_terms)
    if require_improvement:
        model.Add(sub_edges >= base_sub_edges + 1)
    else:
        model.Add(sub_edges >= base_sub_edges)

    if maximize_diversity:
        # Reward moving H-cells to new locations on equal-contact plateaus
        unchanged_h = sum(h_var[c] for c in cur_free_h if c in h_var)
        moved_h = len(cur_free_h) - unchanged_h
        model.Maximize(1000 * (sub_edges - base_sub_edges) + moved_h)
    else:
        model.Maximize(sub_edges)

    # Warm-start hint with current coordinates
    for i in sorted_free:
        for c in valid_cells[i]:
            model.AddHint(p[(i, c)], 1 if c == coords[i] else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_seconds
    solver.parameters.num_search_workers = n_workers
    status = solver.Solve(model)
    elapsed = time.time() - t0
    stats = {
        "elapsed": round(elapsed, 3),
        "n_free": len(free_indices),
        "n_runs": len(runs),
        "n_vars": len(p),
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        new_coords = list(coords)
        for i in sorted_free:
            new_coords[i] = next(c for c in valid_cells[i] if solver.Value(p[(i, c)]) == 1)
        v = verify(sequence, new_coords)
        assert v.valid, f"LNS produced invalid fold: {v.error}"
        new_h = {new_coords[i] for i in range(n) if sequence[i] == "H"}
        old_h = {coords[i] for i in range(n) if sequence[i] == "H"}
        stats["h_symdiff"] = len(new_h ^ old_h) // 2
        return new_coords, v.contacts, solver.StatusName(status), stats

    return None, base_v.contacts, solver.StatusName(status), stats


def select_two_strand_window(
    sequence: str,
    coords: list[Coord],
    rng: random.Random,
    *,
    strand_len: int = 16,
    n_strands: int = 2,
) -> set[int]:
    """Pick 1-2 spatially adjacent contiguous strands (keeps runs <= 2 for high freedom)."""
    n = len(coords)
    # Pick first strand center on a random H residue
    h_indices = [i for i, r in enumerate(sequence) if r == "H"]
    c1 = rng.choice(h_indices) if h_indices else rng.randrange(n)
    half = strand_len // 2
    a1 = max(0, min(n - strand_len, c1 - half))
    free = set(range(a1, min(n, a1 + strand_len)))

    if n_strands >= 2:
        # Find a second residue spatially close to strand 1 but distant in sequence
        s1_cells = [coords[i] for i in free]
        candidates = []
        for j in h_indices:
            if j in free or abs(j - c1) < strand_len:
                continue
            d = min(abs(coords[j][0] - x) + abs(coords[j][1] - y) for x, y in s1_cells)
            if d <= 4:
                candidates.append((d, j))
        if candidates:
            candidates.sort()
            _, c2 = rng.choice(candidates[: min(12, len(candidates))])
            a2 = max(0, min(n - strand_len, c2 - half))
            free |= set(range(a2, min(n, a2 + strand_len)))
    return free


def run_spatial_lns(
    sequence: str,
    start_coords: list[Coord],
    *,
    n_iters: int = 30,
    min_free: int = 18,
    max_free: int = 28,
    pad: int = 2,
    step_budget: float = 3.0,
    seed: int = 42,
    destroy_mode: str = "hybrid",
    strand_len: int = 16,
    verbose: bool = True,
) -> tuple[list[Coord], int, dict]:
    """Run iterative Spatial-Window CP-SAT LNS with plateau diversity hopping."""
    rng = random.Random(seed)
    cur_coords = list(start_coords)
    best_coords = list(start_coords)
    cur_c = best_c = verify(sequence, cur_coords).contacts
    seen_hcores: set[frozenset[Coord]] = {
        frozenset(cur_coords[i] for i, r in enumerate(sequence) if r == "H")
    }
    t0 = time.time()

    for it in range(1, n_iters + 1):
        if destroy_mode == "strands" or (destroy_mode == "hybrid" and it % 2 == 0):
            free = select_two_strand_window(sequence, cur_coords, rng, strand_len=strand_len)
        else:
            free = select_spatial_window(sequence, cur_coords, rng, min_free=min_free, max_free=max_free)
        new_coords, new_c, st, stats = lns_step_cpsat(
            sequence,
            cur_coords,
            free,
            pad=pad,
            time_budget_seconds=step_budget,
            maximize_diversity=True,
        )
        if new_coords is not None and new_c >= cur_c:
            cur_coords = new_coords
            cur_c = new_c
            hc = frozenset(cur_coords[i] for i, r in enumerate(sequence) if r == "H")
            seen_hcores.add(hc)
            if new_c > best_c:
                best_c = new_c
                best_coords = list(new_coords)
                if verbose:
                    print(
                        f"  [iter {it:3d}] NEW BEST contacts={best_c}!! "
                        f"({st}, {stats['elapsed']}s, free={stats['n_free']}, runs={stats['n_runs']})",
                        flush=True,
                    )
            elif verbose and stats.get("h_symdiff", 0) > 0:
                print(
                    f"  [iter {it:3d}] plateau hop contacts={cur_c} "
                    f"(h_moved={stats['h_symdiff']}, unique_hcores={len(seen_hcores)}, "
                    f"{st}, {stats['elapsed']}s, runs={stats['n_runs']})",
                    flush=True,
                )
    return best_coords, best_c, {
        "elapsed": round(time.time() - t0, 2),
        "unique_hcores": len(seen_hcores),
        "iters": n_iters,
    }
