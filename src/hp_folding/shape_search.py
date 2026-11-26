"""Shape-constrained SAW search for HP folding.

Rationale
---------
Diagnostic analysis of the 21-contact folds we keep landing in shows that
our folds already have near-optimally
compact H-cores (Rg² within 10--40% of the theoretical minimum for the
H-count). What they lack is the specific lattice *shape* that would
maximise non-sequential H-H adjacencies. For S5, placing all 25 H
residues in a 5×5 block (40 internal lattice edges) and subtracting the
17 sequence-forced HH adjacencies gives an analytic ceiling of
**23 contacts**, which matches the open record.

Instead of running more Monte Carlo that already converges to 21, we
search directly for a valid chain whose H-residues occupy every cell of
a target shape. The search is a constrained DFS:

  - The chain is built residue by residue.
  - At each step, the next position is an unoccupied lattice neighbour
    of the previous.
  - H residues must land inside the target shape.
  - P residues must land outside the target shape.
  - At the end, every shape cell must be occupied (so the H-count must
    match the shape size).

Subtle points:
  - We pick the starting position from a small anchor set (points just
    outside the shape, or any shape cell if sequence[0] is H).
  - We use iterative deepening / timeout since complete enumeration is
    exponential.
  - Symmetry reduction: fix the shape's bounding box at origin and
    iterate over lattice translations is unnecessary because the shape
    position and orientation are arbitrary; we use the shape as given.

For S5 the target shape is a 5×5 block of 25 cells.
"""

from __future__ import annotations

import time

from hp_folding.fold import Fold
from hp_folding.verifier import count_contacts

Coord = tuple[int, int]


# ---- target shapes ----------------------------------------------------------

def rectangle_shape(rows: int, cols: int) -> frozenset[Coord]:
    """Return a rows × cols rectangle of lattice points anchored at (0, 0)."""
    return frozenset((i, j) for i in range(rows) for j in range(cols))


def shape_5x5() -> frozenset[Coord]:
    """The canonical 25-cell 5×5 block, anchored at (0,0)...(4,4)."""
    return rectangle_shape(5, 5)


def shape_6x7_plus_1(side: str = "right") -> frozenset[Coord]:
    """6×7 rectangle (42 cells) plus one adjacent cell for 43 total.
    ``side`` ∈ {"right", "top", "bottom", "left"} picks where the extra
    cell goes. Near-minimum perimeter = 28 for 43 cells."""
    base = set(rectangle_shape(6, 7))
    if side == "right":
        extra = (0, 7)
    elif side == "top":
        extra = (6, 0)
    elif side == "left":
        extra = (0, -1)
    elif side == "bottom":
        extra = (-1, 0)
    else:
        raise ValueError(f"unknown side {side!r}")
    base.add(extra)
    return frozenset(base)


def shape_7x7_minus_6() -> frozenset[Coord]:
    """7x7 minus 6 cells (43 cells). Placement: remove 6 corners-ish to keep
    a compact bulk. One convenient removal: top row 6 cells -> bottom row 1 cell
    (actually 7x7 - 6 = 43; we remove the top row leaving 6 rows of 7 + 1 cell)."""
    s = set(rectangle_shape(7, 7))
    # Remove the top-right diagonal 6 cells to keep shape compact
    for i, j in [(6, 6), (6, 5), (6, 4), (5, 6), (5, 5), (4, 6)]:
        s.discard((i, j))
    return frozenset(s)


def shape_7x8_plus_3() -> frozenset[Coord]:
    """7x8 (56 cells) + 3 extra adjacent cells = 59 cells for S9."""
    base = set(rectangle_shape(7, 8))
    base.add((0, 8))
    base.add((1, 8))
    base.add((-1, 0))
    return frozenset(base)


def rect_minus_plus(rows: int, cols: int,
                    remove: Coord, add: Coord) -> frozenset[Coord]:
    """Rectangle with one cell removed and one adjacent cell added.

    The resulting polyomino has the same cell count as the original
    rectangle but gains (at least) one concave region, which is what
    makes shape packing feasible for sequences containing length-1
    P-bridges between H-runs.

    Constraints: the ``remove`` cell must be in the rectangle, and the
    ``add`` cell must be exterior to the rectangle and adjacent to at
    least one rectangle cell (for connectivity).
    """
    s = set(rectangle_shape(rows, cols))
    if remove not in s:
        raise ValueError(f"remove cell {remove} is not in the {rows}x{cols} rectangle")
    if add in s:
        raise ValueError(f"add cell {add} is already in the rectangle")
    s.discard(remove)
    s.add(add)
    # Connectivity check
    if not _connected(s):
        raise ValueError("resulting shape is disconnected")
    return frozenset(s)


def _connected(shape: set[Coord]) -> bool:
    if not shape:
        return True
    first = next(iter(shape))
    seen = {first}
    stack = [first]
    while stack:
        x, y = stack.pop()
        for dx, dy in NEIGHBOUR_OFFSETS:
            n = (x + dx, y + dy)
            if n in shape and n not in seen:
                seen.add(n)
                stack.append(n)
    return seen == shape


def generate_42cell_concave_shapes() -> dict[str, frozenset[Coord]]:
    """Catalogue of 42-cell concave shapes derived from the 6×7 rectangle.

    We create concavities by removing an edge (non-corner) cell — the
    resulting empty cell is an exterior position with 3 in-shape neighbours,
    the concavity we need for single-P-bridge re-entry. To keep the shape
    at 42 cells, we add a protrusion on a different edge.

    Each variant produces one inlet (the removed edge cell) and one bump
    (the added protrusion).
    """
    shapes: dict[str, frozenset[Coord]] = {}
    # Remove an edge-midpoint cell → inlet with 3 in-shape neighbours
    # Add a protrusion on a different edge to restore cell count to 42
    variants: list[tuple[str, Coord, Coord]] = []

    top_edge = [(0, c) for c in range(1, 6)]
    bot_edge = [(5, c) for c in range(1, 6)]
    left_edge = [(r, 0) for r in range(1, 5)]
    right_edge = [(r, 6) for r in range(1, 5)]

    protrusions_for_removed = {
        # For each removed cell, valid protrusion spots that don't collide
        "top": [(6, c) for c in range(0, 7)],     # protrude below
        "bot": [(-1, c) for c in range(0, 7)],    # protrude above
        "left": [(r, 7) for r in range(0, 6)],    # protrude right
        "right": [(r, -1) for r in range(0, 6)],  # protrude left
    }

    for rem in top_edge:
        for add in protrusions_for_removed["top"]:
            variants.append((f"top{rem[1]}_bot{add[1]}", rem, add))
    for rem in bot_edge:
        for add in protrusions_for_removed["bot"]:
            variants.append((f"bot{rem[1]}_top{add[1]}", rem, add))
    for rem in left_edge:
        for add in protrusions_for_removed["left"]:
            variants.append((f"left{rem[0]}_right{add[0]}", rem, add))
    for rem in right_edge:
        for add in protrusions_for_removed["right"]:
            variants.append((f"right{rem[0]}_left{add[0]}", rem, add))

    for name, rem, add in variants:
        try:
            shapes[name] = rect_minus_plus(6, 7, rem, add)
        except ValueError:
            pass
    return shapes


def shape_7x9_minus_4corners() -> frozenset[Coord]:
    """59-cell shape: 7×9 rectangle with the 4 corners removed.

    Perimeter: 32 (same as 7×9), internal edges: 102.
    Each removed corner creates a small concavity (2 in-shape neighbours).
    Theoretical contact ceiling for S9 (49 forced HH): 53.
    """
    s = set(rectangle_shape(7, 9))
    for corner in [(0, 0), (0, 8), (6, 0), (6, 8)]:
        s.discard(corner)
    return frozenset(s)


def shape_6x8_minus_5() -> frozenset[Coord]:
    """43-cell shape: 6×8 rectangle with 5 cells removed (for S7).

    Remove 4 corners + 1 edge cell. Internal edges: 71.
    Theoretical contact ceiling for S7 (35 forced HH): 36.
    """
    s = set(rectangle_shape(6, 8))
    for cell in [(0, 0), (0, 7), (5, 0), (5, 7), (0, 4)]:
        s.discard(cell)
    return frozenset(s)


def shape_8x7_minus_4corners() -> frozenset[Coord]:
    """52-cell shape — used for combinatorics tests, not a benchmark target."""
    s = set(rectangle_shape(8, 7))
    for corner in [(0, 0), (0, 6), (7, 0), (7, 6)]:
        s.discard(corner)
    return frozenset(s)


def shape_s8_octagon_v1() -> frozenset[Coord]:
    """42-cell octagonal shape modelled after the 2D64 ground-state fold
    visible in Wüst & Landau 2012 Fig. 4.

    Starts from a 7×7 rectangle and removes 7 boundary cells (4 corners
    + 3 edge cells) to give 42 cells matching S8's H count.
    The shape has corner notches consistent with Wüst & Landau's fold.
    """
    s = set(rectangle_shape(7, 7))
    for cell in [(0, 0), (0, 6), (6, 0), (6, 6),  # corners
                 (0, 5), (6, 5), (0, 1)]:          # 3 more near corners
        s.discard(cell)
    return frozenset(s)


def shape_s8_octagon_v2() -> frozenset[Coord]:
    """42-cell octagon variant 2: symmetric corner chamfers."""
    s = set(rectangle_shape(7, 7))
    for cell in [(0, 0), (0, 1), (1, 0),   # TL corner chamfer
                 (0, 6), (0, 5), (1, 6),   # TR corner chamfer
                 (6, 0)]:                   # single BL corner
        s.discard(cell)
    return frozenset(s)


def shape_s8_octagon_v3() -> frozenset[Coord]:
    """42-cell octagon variant 3: 4 corners + 3 asymmetric edge cells."""
    s = set(rectangle_shape(7, 7))
    for cell in [(0, 0), (0, 6), (6, 0), (6, 6),
                 (0, 3), (6, 3), (3, 0)]:
        s.discard(cell)
    return frozenset(s)


def shape_s8_6x8_minus_6() -> frozenset[Coord]:
    """42-cell shape: 6×8 rectangle minus 6 cells, preserving symmetry."""
    s = set(rectangle_shape(6, 8))
    for cell in [(0, 0), (0, 7), (5, 0), (5, 7),
                 (0, 4), (5, 4)]:
        s.discard(cell)
    return frozenset(s)


def generate_42cell_two_corner_shapes() -> dict[str, frozenset[Coord]]:
    """42-cell shapes with TWO corners shifted — for sequences with
    multiple single-P bridges (e.g., S8 has bridges at residues 12 and 49,
    both single-P, so the shape needs concavities for both).

    Start with 6×7, remove 2 corners, add 2 protrusions elsewhere.
    Cell count stays 42; perimeter grows to ~30; internal edges ~69;
    S8 ceiling ~41 (slightly below record 42 — but may be the only
    feasible family).
    """
    shapes: dict[str, frozenset[Coord]] = {}
    corners = [(0, 0), (0, 6), (5, 0), (5, 6)]

    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            rem1, rem2 = corners[i], corners[j]
            base = set(rectangle_shape(6, 7))
            base.discard(rem1)
            base.discard(rem2)
            # Enumerate pairs of add positions
            possible_adds = set()
            for cell in base:
                for dx, dy in NEIGHBOUR_OFFSETS:
                    a = (cell[0] + dx, cell[1] + dy)
                    if a not in base and a not in (rem1, rem2):
                        possible_adds.add(a)
            possible_adds_list = sorted(possible_adds)
            for ai, add1 in enumerate(possible_adds_list):
                for add2 in possible_adds_list[ai + 1:]:
                    s = set(base)
                    s.add(add1)
                    s.add(add2)
                    if len(s) != 42 or not _connected(s):
                        continue
                    name = (f"rm{rem1[0]},{rem1[1]}+rm{rem2[0]},{rem2[1]}"
                            f"_add{add1[0]},{add1[1]}+{add2[0]},{add2[1]}")
                    shapes[name] = frozenset(s)
    return shapes


def generate_42cell_corner_shift_shapes() -> dict[str, frozenset[Coord]]:
    """Enumerate 42-cell shapes from 6×7 by removing one of 4 corners
    and adding one cell adjacent to a non-removed part of the rectangle.

    These shapes have perimeter 28 and 70 internal edges, giving a
    contact ceiling of exactly 42 for S8 (with its 28 HH pairs).

    The shape's concavity is the exterior cell that replaces the removed
    corner — that cell has exactly 2 in-shape neighbours, giving the
    minimum concavity needed for length-1 P-bridge re-entry.
    """
    shapes: dict[str, frozenset[Coord]] = {}
    corners = [(0, 0), (0, 6), (5, 0), (5, 6)]

    for rem in corners:
        # Enumerate add positions: any exterior cell adjacent to a non-removed
        # cell in the 6x7 rectangle
        base = set(rectangle_shape(6, 7))
        base.discard(rem)
        adds: list[Coord] = []
        for cell in base:
            for dx, dy in NEIGHBOUR_OFFSETS:
                a = (cell[0] + dx, cell[1] + dy)
                if a not in base and a != rem:
                    adds.append(a)
        # Deduplicate
        adds = sorted(set(adds))
        for add in adds:
            name = f"rm{rem[0]},{rem[1]}_add{add[0]},{add[1]}"
            try:
                shapes[name] = rect_minus_plus(6, 7, rem, add)
            except ValueError:
                pass
    return shapes


def generate_42cell_double_concave_shapes() -> dict[str, frozenset[Coord]]:
    """42-cell shapes with two concavities (two separate indents).

    Used when a sequence has two well-separated single-P bridges that need
    distinct re-entry points.
    """
    shapes: dict[str, frozenset[Coord]] = {}
    # Remove two edge cells, add two protrusions.
    variants = [
        ("top_bottom_mid",
         [(0, 3), (5, 3)], [(-1, 3), (6, 3)]),
        ("left_right_mid",
         [(2, 0), (3, 6)], [(2, -1), (3, 7)]),
        ("TL_BR_corners",
         [(0, 0), (5, 6)], [(-1, 0), (6, 6)]),
    ]
    for name, removes, adds in variants:
        s = set(rectangle_shape(6, 7))
        ok = True
        for r in removes:
            if r in s:
                s.discard(r)
            else:
                ok = False
                break
        for a in adds:
            if a not in s:
                s.add(a)
            else:
                ok = False
                break
        if ok and _connected(s) and len(s) == 42:
            shapes[name] = frozenset(s)
    return shapes


def shape_4x6_minus_corner() -> frozenset[Coord]:
    """24 cells: 4x6 rectangle minus one corner — useful for H-count 23."""
    base = set(rectangle_shape(4, 6))
    base.discard((0, 5))
    return frozenset(base)


def shape_6x5_minus_5() -> frozenset[Coord]:
    """25 cells in a 6×5 box with the bottom row truncated to 5 cells (same as 5x5)."""
    return shape_5x5()


def shape_staircase_25() -> frozenset[Coord]:
    """25-cell staircase: a compact non-rectangular shape to try as an alternative."""
    # 5+5+5+5+5 rows, each shifted by 1 — L-shape style. Simplification: we return 5x5.
    return shape_5x5()


# ---- shape-constrained DFS --------------------------------------------------

NEIGHBOUR_OFFSETS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def anchor_positions(shape: frozenset[Coord]) -> list[Coord]:
    """Lattice points adjacent to the shape — good starting anchors for a
    chain that begins with a P residue."""
    out: set[Coord] = set()
    for x, y in shape:
        for dx, dy in NEIGHBOUR_OFFSETS:
            p = (x + dx, y + dy)
            if p not in shape:
                out.add(p)
    return sorted(out)


def inner_positions(shape: frozenset[Coord]) -> list[Coord]:
    """Lattice points of the shape (for starting anchor when sequence[0] is H)."""
    return sorted(shape)


def _count_contacts_in_shape(
    sequence: str, coords: list[Coord], shape: frozenset[Coord]
) -> int:
    """Fast contact count — same as verifier but assumes fold is a valid SAW."""
    return count_contacts(sequence, coords)


def _count_reachable_shape_cells(
    pos: Coord, occupied: set[Coord], shape: frozenset[Coord], max_steps: int
) -> int:
    """BFS-bounded count of shape cells reachable from ``pos`` within
    ``max_steps`` moves on unoccupied lattice sites.

    Used as an admissible upper bound on the number of H residues that can
    still be placed. Cheap (O(max_steps²) cells visited in the worst case)
    and tight enough to prune far more aggressively than unrestricted DFS.
    """
    if max_steps <= 0:
        return 0
    count = 0
    visited = {pos}
    frontier = [pos]
    for _ in range(max_steps):
        new_frontier: list[Coord] = []
        for x, y in frontier:
            for dx, dy in NEIGHBOUR_OFFSETS:
                n = (x + dx, y + dy)
                if n in visited or n in occupied:
                    continue
                visited.add(n)
                new_frontier.append(n)
                if n in shape:
                    count += 1
        frontier = new_frontier
        if not frontier:
            break
    return count


def search_shape_constrained(
    sequence: str,
    shape: frozenset[Coord],
    *,
    timeout_seconds: float = 30.0,
    min_contacts_to_stop: int | None = None,
    p_may_enter_shape: bool = False,
    seed: int | None = None,
    telemetry=None,
    initial_coords: list[Coord] | None = None,
    forward_check: bool = True,
) -> tuple[list[Coord] | None, int, int]:
    """DFS for a valid SAW of ``sequence`` whose H residues occupy every
    cell of ``shape``.

    Parameters
    ----------
    sequence
        HP string.
    shape
        Set of lattice positions. Must have size ``>=`` count of H in sequence.
    timeout_seconds
        Abort the search after this wall-clock budget.
    min_contacts_to_stop
        If given, stop early once a fold with at least this many non-sequential
        H-H contacts is found (typically the open record).
    p_may_enter_shape
        If True, P residues are allowed to occupy shape cells (and those cells
        are then not filled by H's — i.e. a partial shape fit). Default False,
        i.e. strict shape occupancy by H only.

    Returns
    -------
    (coords, contacts, nodes)
        ``coords`` is ``None`` on timeout/failure; ``contacts`` is the best
        contact count seen; ``nodes`` is the number of DFS nodes expanded.
    """
    seq = sequence.upper()
    n = len(seq)
    h_mask = [c == "H" for c in seq]
    h_count = sum(h_mask)
    if len(shape) < h_count:
        raise ValueError(
            f"shape has {len(shape)} cells but sequence has {h_count} H residues"
        )
    if not p_may_enter_shape and len(shape) != h_count:
        raise ValueError(
            f"strict mode requires shape size ({len(shape)}) == H count ({h_count})"
        )

    best_coords: list[Coord] | None = None
    best_contacts = -1
    nodes = 0

    import random
    rng = random.Random(seed)

    start_time = time.monotonic()
    timed_out = False

    # We thread the anchor through the recursion so telemetry can attribute
    # both successes and failures to the starting anchor.
    current_anchor: list[Coord | None] = [None]

    # Precompute anchor set (may be shuffled if seed is given)
    anchor_set = (inner_positions(shape) if h_mask[0]
                  else anchor_positions(shape))
    if seed is not None:
        anchor_set = list(anchor_set)
        rng.shuffle(anchor_set)

    # Precompute H-count remaining from each chain position
    h_remaining_from = [0] * (n + 1)
    for k in range(n - 1, -1, -1):
        h_remaining_from[k] = h_remaining_from[k + 1] + (1 if h_mask[k] else 0)

    # If the caller provided a seed partial fold, validate it.
    seed_depth = 0
    seed_coords: list[Coord] = []
    seed_occupied: set[Coord] = set()
    seed_h_placed = 0
    if initial_coords is not None:
        for k, c in enumerate(initial_coords):
            c = tuple(c)
            if c in seed_occupied:
                raise ValueError(f"seed coord {c} appears twice at index {k}")
            if k > 0 and (abs(c[0] - seed_coords[-1][0]) + abs(c[1] - seed_coords[-1][1])) != 1:
                raise ValueError(f"seed coord {c} at index {k} not adjacent to previous")
            if h_mask[k] and c not in shape:
                raise ValueError(f"seed coord {c} at index {k} is H but outside shape")
            if not h_mask[k] and c in shape and not p_may_enter_shape:
                raise ValueError(f"seed coord {c} at index {k} is P but inside shape (strict mode)")
            seed_coords.append(c)
            seed_occupied.add(c)
            if h_mask[k]:
                seed_h_placed += 1
        seed_depth = len(seed_coords)

    def dfs(i: int, coords: list[Coord], occupied: set[Coord],
            h_placed: int) -> bool:
        """Return True if we should abort (stop-early or timeout), False otherwise."""
        nonlocal best_coords, best_contacts, nodes, timed_out
        nodes += 1
        if nodes % 32768 == 0:
            if time.monotonic() - start_time > timeout_seconds:
                timed_out = True
                return True

        if telemetry is not None and current_anchor[0] is not None:
            telemetry.record_anchor_depth(current_anchor[0], i)

        if i == n:
            if h_placed != h_count:
                if telemetry is not None:
                    telemetry.record_fail(i)
                return False  # shouldn't happen since we prune on shape constraint
            c = count_contacts(seq, coords)
            if telemetry is not None and current_anchor[0] is not None:
                telemetry.record_success(current_anchor[0], coords, c)
            if c > best_contacts:
                best_contacts = c
                best_coords = list(coords)
                if min_contacts_to_stop is not None and c >= min_contacts_to_stop:
                    return True
            return False

        prev = coords[-1] if coords else None
        if prev is None:
            # First residue: choose an anchor.
            for p in anchor_set:
                current_anchor[0] = p
                coords.append(p)
                occupied.add(p)
                new_h = h_placed + (1 if h_mask[0] else 0)
                if dfs(1, coords, occupied, new_h):
                    return True
                if timed_out:
                    return True
                coords.pop()
                occupied.remove(p)
            current_anchor[0] = None
            return False

        # Forward-check pruning: if the remaining chain cannot possibly fit
        # enough H residues into the remaining unvisited shape cells, prune.
        # We use a cheap BFS-based upper bound on reachable shape cells from
        # the current position within the remaining steps.
        if forward_check:
            remaining_steps = n - i
            h_needed = h_remaining_from[i]
            if h_needed > 0:
                reachable = _count_reachable_shape_cells(
                    prev, occupied, shape, remaining_steps
                )
                if reachable < h_needed:
                    if telemetry is not None:
                        telemetry.record_fail(i)
                    return False

        # Extend from prev, trying neighbours in a randomised order so that
        # multi-seed restarts explore different subtrees.
        offsets = list(NEIGHBOUR_OFFSETS)
        if seed is not None:
            rng.shuffle(offsets)
        valid_move_count = 0
        for dx, dy in offsets:
            nxt = (prev[0] + dx, prev[1] + dy)
            if nxt in occupied:
                continue
            in_shape = nxt in shape
            if h_mask[i]:
                if not in_shape:
                    continue  # H must land in shape
            else:
                if in_shape and not p_may_enter_shape:
                    continue  # P must stay out of shape in strict mode

            valid_move_count += 1
            coords.append(nxt)
            occupied.add(nxt)
            new_h = h_placed + (1 if h_mask[i] else 0)
            if dfs(i + 1, coords, occupied, new_h):
                return True
            if timed_out:
                return True
            coords.pop()
            occupied.remove(nxt)

        if telemetry is not None:
            telemetry.record_valid_moves(i, valid_move_count)
            telemetry.record_fail(i)
            if valid_move_count == 0:
                telemetry.record_out_of_moves(i)
        return False

    if seed_depth > 0:
        # Start DFS from the seeded partial fold
        current_anchor[0] = seed_coords[0]
        dfs(seed_depth, list(seed_coords), set(seed_occupied), seed_h_placed)
    else:
        dfs(0, [], set(), 0)
    return best_coords, best_contacts, nodes


def search_s5_5x5(timeout_seconds: float = 60.0) -> tuple[Fold | None, int, int]:
    """Convenience: search S5 under strict 5×5 H-shape."""
    from hp_folding.sequences import BENCHMARKS
    bench = BENCHMARKS["S5"]
    shape = shape_5x5()
    coords, contacts, nodes = search_shape_constrained(
        bench.sequence, shape,
        timeout_seconds=timeout_seconds,
        min_contacts_to_stop=23,
    )
    if coords is None:
        return None, contacts, nodes
    return Fold(sequence=bench.sequence, coords=coords), contacts, nodes
