"""H-core CSP solver with proper variable ordering and propagation.

This is the run-aware backtracking solver the analogy actually describes:
each H-run is a CSP variable, its domain is the set of length-L SAW
placements inside the suite block, and the binary constraints between
adjacent H-runs in chain order are P-run routability (a length-K SAW
exists outside the block from one H-run's endpoint to the next H-run's
startpoint).

Key design choices:
  - Variable ordering: assign the longest H-run first (smallest effective
    domain after propagation). After that, expand outward to chain
    neighbors. This avoids the chain-order DFS pathology where the long
    run is placed last and rarely fits.
  - Constraint check on commit: when assigning var v, verify P-bridges
    to all already-assigned chain neighbors of v.
  - Final assembly: once all H-runs have placements that mutually satisfy
    bridge constraints, ROUTE the actual P-paths to build the chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from hp_folding.hcore import (
    Coord,
    Run,
    enumerate_run_paths_in_block,
    parse_runs,
    _bbox_around_block,
    _NEIGHBOURS,
)


@dataclass
class CSPSolution:
    coords: list[Coord]
    contacts: int


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bridge_feasible(
    end_a: Coord,
    start_b: Coord,
    p_length: int,
    block: frozenset[Coord],
    occupied_block: set[Coord],
    occupied_outside: set[Coord],
    bbox: tuple[int, int, int, int],
    deadline: float,
) -> bool:
    """Cheap feasibility: is there a length-`p_length` SAW from a neighbour
    of `end_a` to a neighbour of `start_b`, all cells outside `block`,
    none in `occupied_outside`, inside `bbox`?

    For the chain to be valid: bond `end_a` -> P_0 (so P_0 is neighbour of
    end_a, outside block); bond P_{K-1} -> start_b (so P_{K-1} is neighbour
    of start_b, outside block). The path P_0..P_{K-1} is K cells of SAW
    outside block.

    For p_length == 0: only feasible if `end_a` and `start_b` are
    lattice-adjacent (which is impossible here because both are in the
    block and the chain would mean two H-runs in a row — no such case
    exists if the run decomposition is correct).
    """
    if p_length <= 0:
        # Two H-runs with no P between them shouldn't exist after run
        # decomposition. Treat as adjacency: end_a manhattan-adjacent to
        # start_b would mean they should have been merged into one H-run.
        return False

    bx0, bx1, by0, by1 = bbox

    # Manhattan + parity necessary condition.
    # We need K cells on the lattice such that:
    #   step 1: end_a -> P_0 (1 hop)
    #   steps 2..K: P_0 -> P_1 -> ... -> P_{K-1} (K-1 hops within P-path)
    #   step K+1: P_{K-1} -> start_b (1 hop)
    # Total hops = K + 1. Parity: |dx|+|dy| from end_a to start_b == K+1 mod 2.
    d = _manhattan(end_a, start_b)
    if d > p_length + 1:
        return False
    if (d + p_length + 1) % 2 != 0:
        return False

    # BFS feasibility: from each candidate P_0 (neighbour of end_a outside
    # block, not occupied), check if we can reach a candidate P_{K-1}
    # (neighbour of start_b similarly) in exactly K-1 more steps. We use
    # bounded DFS instead of bidirectional BFS because the path length
    # is small (≤ ~12 typically).
    target_neighbours = []
    for dx, dy in _NEIGHBOURS:
        nb = (start_b[0] + dx, start_b[1] + dy)
        if nb in block or nb in occupied_outside or nb == end_a:
            continue
        x, y = nb
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            continue
        target_neighbours.append(nb)
    if not target_neighbours:
        return False

    visited: set[Coord] = set()

    def dfs(node: Coord, remaining: int) -> bool:
        if time.monotonic() > deadline:
            return True  # bail out, signal upward
        if remaining == 0:
            return node in target_neighbours
        # Reachability prune
        for tgt in target_neighbours:
            d_remain = _manhattan(node, tgt)
            if d_remain <= remaining and (d_remain + remaining) % 2 == 0:
                break
        else:
            return False
        for dx, dy in _NEIGHBOURS:
            nb = (node[0] + dx, node[1] + dy)
            if nb in block or nb in occupied_outside or nb in visited:
                continue
            x, y = nb
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            visited.add(nb)
            if dfs(nb, remaining - 1):
                visited.discard(nb)
                return True
            visited.discard(nb)
        return False

    # Try each P_0 neighbour of end_a.
    for dx, dy in _NEIGHBOURS:
        p0 = (end_a[0] + dx, end_a[1] + dy)
        if p0 in block or p0 in occupied_outside:
            continue
        x, y = p0
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            continue
        if p0 == start_b:
            continue
        # We need to walk K-1 more cells from p0, ending at a target_neighbour,
        # without revisiting.
        if p_length == 1:
            # single P cell == p0; need p0 ∈ target_neighbours
            if p0 in target_neighbours:
                return True
            continue
        visited.clear()
        visited.add(p0)
        if dfs(p0, p_length - 1):
            return True
    return False


def _route_p_path(
    end_a: Coord,
    start_b: Coord,
    p_length: int,
    block: frozenset[Coord],
    occupied_outside: set[Coord],
    bbox: tuple[int, int, int, int],
    deadline: float,
) -> list[Coord] | None:
    """Concrete path from neighbour of end_a to neighbour of start_b,
    p_length cells of SAW outside the block, avoiding occupied_outside."""
    bx0, bx1, by0, by1 = bbox
    if p_length <= 0:
        return None
    target_neighbours = []
    for dx, dy in _NEIGHBOURS:
        nb = (start_b[0] + dx, start_b[1] + dy)
        if nb in block or nb in occupied_outside or nb == end_a:
            continue
        x, y = nb
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            continue
        target_neighbours.append(nb)
    if not target_neighbours:
        return None

    visited: set[Coord] = set()
    path: list[Coord] = []

    def dfs(node: Coord, remaining: int) -> bool:
        if time.monotonic() > deadline:
            return False
        if remaining == 0:
            return node in target_neighbours
        for tgt in target_neighbours:
            d_remain = _manhattan(node, tgt)
            if d_remain <= remaining and (d_remain + remaining) % 2 == 0:
                break
        else:
            return False
        for dx, dy in _NEIGHBOURS:
            nb = (node[0] + dx, node[1] + dy)
            if nb in block or nb in occupied_outside or nb in visited:
                continue
            x, y = nb
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            visited.add(nb)
            path.append(nb)
            if dfs(nb, remaining - 1):
                return True
            path.pop()
            visited.discard(nb)
        return False

    for dx, dy in _NEIGHBOURS:
        p0 = (end_a[0] + dx, end_a[1] + dy)
        if p0 in block or p0 in occupied_outside:
            continue
        x, y = p0
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            continue
        if p0 == start_b:
            continue
        path.clear()
        path.append(p0)
        if p_length == 1:
            if p0 in target_neighbours:
                return list(path)
            path.clear()
            continue
        visited.clear()
        visited.add(p0)
        if dfs(p0, p_length - 1):
            return list(path)
        path.clear()
    return None


def _enumerate_bridge_paths(
    end_a: Coord,
    start_b: Coord,
    p_length: int,
    block: frozenset[Coord],
    occupied_outside: set[Coord],
    bbox: tuple[int, int, int, int],
    deadline: float,
    *,
    max_paths: int = 400,
) -> list[list[Coord]]:
    """All length-K SAW paths from a neighbour of end_a to a neighbour of
    start_b, outside block, avoiding occupied_outside."""
    bx0, bx1, by0, by1 = bbox
    out: list[list[Coord]] = []
    if p_length <= 0:
        return out
    target_neighbours = []
    for dx, dy in _NEIGHBOURS:
        nb = (start_b[0] + dx, start_b[1] + dy)
        if nb in block or nb in occupied_outside or nb == end_a:
            continue
        x, y = nb
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            continue
        target_neighbours.append(nb)
    if not target_neighbours:
        return out

    visited: set[Coord] = set()
    path: list[Coord] = []

    def dfs(node: Coord, remaining: int) -> bool:
        if time.monotonic() > deadline:
            return True
        if remaining == 0:
            if node in target_neighbours:
                out.append(list(path))
                if len(out) >= max_paths:
                    return True
            return False
        ok = False
        for tgt in target_neighbours:
            d = _manhattan(node, tgt)
            if d <= remaining and (d + remaining) % 2 == 0:
                ok = True
                break
        if not ok:
            return False
        for dx, dy in _NEIGHBOURS:
            nb = (node[0] + dx, node[1] + dy)
            if nb in block or nb in occupied_outside or nb in visited:
                continue
            x, y = nb
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            visited.add(nb)
            path.append(nb)
            if dfs(nb, remaining - 1):
                return True
            path.pop()
            visited.discard(nb)
        return False

    for dx, dy in _NEIGHBOURS:
        p0 = (end_a[0] + dx, end_a[1] + dy)
        if p0 in block or p0 in occupied_outside:
            continue
        x, y = p0
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            continue
        if p0 == start_b:
            continue
        path.clear()
        path.append(p0)
        if p_length == 1:
            if p0 in target_neighbours:
                out.append(list(path))
                if len(out) >= max_paths:
                    return out
            path.clear()
            continue
        visited.clear()
        visited.add(p0)
        if dfs(p0, p_length - 1):
            return out
        path.clear()
    return out


def _enumerate_terminal_paths(
    anchor: Coord,
    p_length: int,
    block: frozenset[Coord],
    occupied_outside: set[Coord],
    bbox: tuple[int, int, int, int],
    deadline: float,
    *,
    going_in: bool,
    max_paths: int = 400,
) -> list[list[Coord]]:
    """For first or last P-run with only one anchor."""
    bx0, bx1, by0, by1 = bbox
    out: list[list[Coord]] = []
    if p_length <= 0:
        return out
    visited: set[Coord] = set()
    path: list[Coord] = []

    if going_in:
        target_neighbours = []
        for dx, dy in _NEIGHBOURS:
            nb = (anchor[0] + dx, anchor[1] + dy)
            if nb in block or nb in occupied_outside:
                continue
            x, y = nb
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            target_neighbours.append(nb)
        if not target_neighbours:
            return out

        def dfs(node: Coord, remaining: int) -> bool:
            if time.monotonic() > deadline:
                return True
            if remaining == 0:
                if node in target_neighbours:
                    out.append(list(path))
                    if len(out) >= max_paths:
                        return True
                return False
            ok = False
            for tgt in target_neighbours:
                d = _manhattan(node, tgt)
                if d <= remaining and (d + remaining) % 2 == 0:
                    ok = True
                    break
            if not ok:
                return False
            for dx, dy in _NEIGHBOURS:
                nb = (node[0] + dx, node[1] + dy)
                if nb in block or nb in occupied_outside or nb in visited:
                    continue
                x, y = nb
                if x < bx0 or x > bx1 or y < by0 or y > by1:
                    continue
                visited.add(nb)
                path.append(nb)
                if dfs(nb, remaining - 1):
                    return True
                path.pop()
                visited.discard(nb)
            return False

        for x in range(bx0, bx1 + 1):
            for y in range(by0, by1 + 1):
                start = (x, y)
                if start in block or start in occupied_outside:
                    continue
                if not any(_manhattan(start, c) <= p_length + 1 for c in block):
                    continue
                visited.clear()
                visited.add(start)
                path.clear()
                path.append(start)
                if p_length == 1:
                    if start in target_neighbours:
                        out.append(list(path))
                        if len(out) >= max_paths:
                            return out
                    continue
                if dfs(start, p_length - 1):
                    return out
                if time.monotonic() > deadline:
                    return out
        return out
    else:
        def dfs(node: Coord, remaining: int) -> bool:
            if time.monotonic() > deadline:
                return True
            if remaining == 0:
                out.append(list(path))
                if len(out) >= max_paths:
                    return True
                return False
            for dx, dy in _NEIGHBOURS:
                nb = (node[0] + dx, node[1] + dy)
                if nb in block or nb in occupied_outside or nb in visited:
                    continue
                x, y = nb
                if x < bx0 or x > bx1 or y < by0 or y > by1:
                    continue
                visited.add(nb)
                path.append(nb)
                if dfs(nb, remaining - 1):
                    return True
                path.pop()
                visited.discard(nb)
            return False

        for dx, dy in _NEIGHBOURS:
            first = (anchor[0] + dx, anchor[1] + dy)
            if first in block or first in occupied_outside:
                continue
            x, y = first
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            visited.clear()
            visited.add(first)
            path.clear()
            path.append(first)
            if p_length == 1:
                out.append(list(path))
                if len(out) >= max_paths:
                    return out
                continue
            if dfs(first, p_length - 1):
                return out
        return out


def _route_terminal_p_path(
    anchor: Coord,
    p_length: int,
    block: frozenset[Coord],
    occupied_outside: set[Coord],
    bbox: tuple[int, int, int, int],
    deadline: float,
    *,
    going_in: bool,
) -> list[Coord] | None:
    """Route the chain's first or last P-run (which only has one anchor).

    going_in == True: this is the FIRST run (PPPHH...). The chain enters
        from outside; we need a length-K SAW path whose LAST cell is a
        neighbour of `anchor` (the first H-run's start cell). The first
        cell can be anywhere outside.
    going_in == False: this is the LAST run (...HHPPP). The chain exits;
        path's FIRST cell is a neighbour of `anchor` (last H-run's end),
        last cell can be anywhere outside.
    """
    bx0, bx1, by0, by1 = bbox
    if p_length <= 0:
        return None
    visited: set[Coord] = set()
    path: list[Coord] = []

    if going_in:
        target_neighbours = []
        for dx, dy in _NEIGHBOURS:
            nb = (anchor[0] + dx, anchor[1] + dy)
            if nb in block or nb in occupied_outside:
                continue
            x, y = nb
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            target_neighbours.append(nb)
        if not target_neighbours:
            return None

        def dfs(node: Coord, remaining: int) -> bool:
            if time.monotonic() > deadline:
                return False
            if remaining == 0:
                return node in target_neighbours
            # parity prune to nearest target
            ok = False
            for tgt in target_neighbours:
                d = _manhattan(node, tgt)
                if d <= remaining and (d + remaining) % 2 == 0:
                    ok = True
                    break
            if not ok:
                return False
            for dx, dy in _NEIGHBOURS:
                nb = (node[0] + dx, node[1] + dy)
                if nb in block or nb in occupied_outside or nb in visited:
                    continue
                x, y = nb
                if x < bx0 or x > bx1 or y < by0 or y > by1:
                    continue
                visited.add(nb)
                path.append(nb)
                if dfs(nb, remaining - 1):
                    return True
                path.pop()
                visited.discard(nb)
            return False

        # Iterate over plausible start cells (within bbox, near the block).
        for x in range(bx0, bx1 + 1):
            for y in range(by0, by1 + 1):
                start = (x, y)
                if start in block or start in occupied_outside:
                    continue
                # Heuristic: only start near block (within p_length+1 manhattan)
                near = any(_manhattan(start, c) <= p_length + 1 for c in block)
                if not near:
                    continue
                visited.clear()
                visited.add(start)
                path.clear()
                path.append(start)
                if p_length == 1:
                    if start in target_neighbours:
                        return list(path)
                    continue
                if dfs(start, p_length - 1):
                    return list(path)
                if time.monotonic() > deadline:
                    return None
        return None
    else:
        # path[0] is neighbour of anchor; path can end anywhere outside
        def dfs(node: Coord, remaining: int) -> bool:
            if time.monotonic() > deadline:
                return False
            if remaining == 0:
                return True
            for dx, dy in _NEIGHBOURS:
                nb = (node[0] + dx, node[1] + dy)
                if nb in block or nb in occupied_outside or nb in visited:
                    continue
                x, y = nb
                if x < bx0 or x > bx1 or y < by0 or y > by1:
                    continue
                visited.add(nb)
                path.append(nb)
                if dfs(nb, remaining - 1):
                    return True
                path.pop()
                visited.discard(nb)
            return False

        for dx, dy in _NEIGHBOURS:
            first = (anchor[0] + dx, anchor[1] + dy)
            if first in block or first in occupied_outside:
                continue
            x, y = first
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            visited.clear()
            visited.add(first)
            path.clear()
            path.append(first)
            if p_length == 1:
                return list(path)
            if dfs(first, p_length - 1):
                return list(path)
        return None


def _count_contacts(sequence: str, coords: list[Coord]) -> int:
    pos = {c: i for i, c in enumerate(coords)}
    c = 0
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue
        for dx, dy in _NEIGHBOURS:
            nb = (x + dx, y + dy)
            j = pos.get(nb)
            if j is None:
                continue
            if abs(j - i) <= 1:
                continue
            if sequence[j] == "H":
                c += 1
    return c // 2


def solve_csp(
    sequence: str,
    suite_block: frozenset[Coord],
    *,
    time_budget_seconds: float = 30.0,
    h_run_max_placements: int = 5000,
) -> CSPSolution | None:
    """Solve H-core CSP with proper variable ordering and propagation.

    Returns the highest-contact valid threading or None if infeasible.
    """
    runs = parse_runs(sequence)
    h_runs = [(i, r) for i, r in enumerate(runs) if r.kind == "H"]
    n_h = sum(1 for c in sequence if c == "H")
    if len(suite_block) != n_h:
        raise ValueError("suite_block size mismatch")

    n_chain = len(sequence)
    pad = max(4, n_chain - n_h)
    bbox = _bbox_around_block(suite_block, pad)
    deadline = time.monotonic() + time_budget_seconds

    # Pre-enumerate H-run placements per length, balanced across start cells.
    placements_by_length: dict[int, list[tuple[Coord, ...]]] = {}
    for _, r in h_runs:
        if r.length not in placements_by_length:
            per_start = max(1, h_run_max_placements // max(1, len(suite_block)))
            placements_by_length[r.length] = enumerate_run_paths_in_block(
                suite_block, r.length,
                max_paths=h_run_max_placements,
                max_per_start=per_start * 4,
            )

    # Variables: indexed in chain-order (0..H-1).
    # var v corresponds to the v-th H-run, which is h_runs[v].
    h_count = len(h_runs)
    domains: list[list[tuple[Coord, ...]]] = [
        list(placements_by_length[r.length]) for _, r in h_runs
    ]

    # P-run lengths between consecutive H-runs in chain order.
    # p_between[v] = length of P-run between H-run v and H-run v+1.
    p_between: list[int] = []
    for v in range(h_count - 1):
        chain_idx_a = h_runs[v][0]
        chain_idx_b = h_runs[v + 1][0]
        # The P-run between them is at chain index chain_idx_a + 1 .. chain_idx_b - 1
        # All non-H runs between them (typically just one P-run).
        p_len = sum(r.length for r in runs[chain_idx_a + 1: chain_idx_b])
        p_between.append(p_len)
    leading_p_length = sum(r.length for r in runs[: h_runs[0][0]]) if h_count else 0
    trailing_p_length = (
        sum(r.length for r in runs[h_runs[-1][0] + 1:]) if h_count else 0
    )

    # Variable order: longest H-run first (smallest expected domain),
    # tiebreak by chain-position (early first for stability).
    var_order = sorted(
        range(h_count),
        key=lambda v: (-h_runs[v][1].length, v),
    )

    assigned: dict[int, tuple[Coord, ...]] = {}
    occupied_block: set[Coord] = set()
    best_solution: CSPSolution | None = None

    # Bridge feasibility cache: maps (end_a, start_b, p_len, occupied_block_frozen)
    # -> bool. We freeze the relevant occupied state into a frozenset to make
    # the key hashable. For speed we use a coarser key without occupied state
    # (best-case feasibility), then re-check at commit time.
    _bridge_static_cache: dict[tuple[Coord, Coord, int], bool] = {}

    def bridge_static(end_a: Coord, start_b: Coord, p_len: int) -> bool:
        """Best-case bridge feasibility ignoring inter-bridge collisions."""
        key = (end_a, start_b, p_len)
        if key in _bridge_static_cache:
            return _bridge_static_cache[key]
        r = _bridge_feasible(
            end_a, start_b, p_len, suite_block,
            occupied_block, set(), bbox, deadline,
        )
        _bridge_static_cache[key] = r
        return r

    def all_assigned() -> bool:
        return len(assigned) == h_count

    def neighbours_in_chain(v: int):
        return [w for w in (v - 1, v + 1) if 0 <= w < h_count]

    def consistent_with_assigned_neighbours(v: int, value: tuple[Coord, ...]) -> bool:
        """Check P-bridge feasibility between value (placement of v) and
        every chain-neighbor of v that's already assigned. Uses cached
        best-case bridge feasibility."""
        for w in neighbours_in_chain(v):
            if w not in assigned:
                continue
            if w == v - 1:
                end_a = assigned[w][-1]
                start_b = value[0]
                p_len = p_between[w]
            else:
                end_a = value[-1]
                start_b = assigned[w][0]
                p_len = p_between[v]
            if not bridge_static(end_a, start_b, p_len):
                return False
        return True

    def assemble_chain() -> CSPSolution | None:
        """All H-runs assigned. Backtracking DFS over P-run routings."""
        coords: list[Coord] = [None] * n_chain  # type: ignore
        outside_used: set[Coord] = set()
        h_var_of_run_idx = {ci: v for v, (ci, _) in enumerate(h_runs)}

        # P-runs in chain order (only the P-run runs)
        p_run_indices = [i for i, r in enumerate(runs) if r.kind == "P"]

        # For each P-run, enumerate ALL valid path candidates.
        # We then DFS over choices.
        def candidates_for(pi: int) -> list[list[Coord]]:
            run = runs[pi]
            if pi == 0:
                next_v = h_var_of_run_idx[pi + 1]
                anchor = assigned[next_v][0]
                return _enumerate_terminal_paths(
                    anchor, run.length, suite_block, outside_used,
                    bbox, deadline, going_in=True, max_paths=400,
                )
            elif pi == len(runs) - 1:
                prev_v = h_var_of_run_idx[pi - 1]
                anchor = assigned[prev_v][-1]
                return _enumerate_terminal_paths(
                    anchor, run.length, suite_block, outside_used,
                    bbox, deadline, going_in=False, max_paths=400,
                )
            else:
                prev_v = h_var_of_run_idx[pi - 1]
                next_v = h_var_of_run_idx[pi + 1]
                return _enumerate_bridge_paths(
                    assigned[prev_v][-1],
                    assigned[next_v][0],
                    run.length, suite_block, outside_used,
                    bbox, deadline, max_paths=400,
                )

        # Place all H-runs first.
        for i, run in enumerate(runs):
            if run.kind == "H":
                v = h_var_of_run_idx[i]
                placement = assigned[v]
                for k, c in enumerate(placement):
                    coords[run.start + k] = c

        best_local: list[CSPSolution | None] = [None]

        def route_dfs(p_idx_in_list: int) -> bool:
            if time.monotonic() > deadline:
                return False
            if p_idx_in_list == len(p_run_indices):
                # Done — validate and score
                if any(c is None for c in coords):
                    return False
                # SAW check
                seen_cells: set[Coord] = set()
                for i, c in enumerate(coords):
                    if c in seen_cells:
                        return False
                    seen_cells.add(c)
                    if i > 0 and _manhattan(coords[i - 1], c) != 1:
                        return False
                contacts = _count_contacts(sequence, coords)
                if best_local[0] is None or contacts > best_local[0].contacts:
                    best_local[0] = CSPSolution(coords=list(coords), contacts=contacts)
                return False  # explore alternatives for higher contacts

            pi = p_run_indices[p_idx_in_list]
            run = runs[pi]
            cands = candidates_for(pi)
            for path in cands:
                # check no overlap with outside_used (rebuild check since DFS may
                # have changed it)
                if any(c in outside_used for c in path):
                    continue
                for k, c in enumerate(path):
                    coords[run.start + k] = c
                    outside_used.add(c)
                route_dfs(p_idx_in_list + 1)
                for k, c in enumerate(path):
                    coords[run.start + k] = None  # type: ignore
                    outside_used.discard(c)
                if best_local[0] is not None:
                    # Found a complete fold; we keep searching but only briefly
                    # for higher contacts.
                    pass
            return False

        route_dfs(0)
        return best_local[0]

    # Live domains, updated by forward checking.
    domains_live: list[list[tuple[Coord, ...]]] = [list(d) for d in domains]
    # Stack of (var_index, saved_domain) for restoring on backtrack.
    domain_save_stack: list[tuple[int, list[tuple[Coord, ...]]]] = []

    def filter_domain_after_commit(v: int, value: tuple[Coord, ...]) -> bool:
        """After committing v=value, prune domains of unassigned chain
        neighbors. Returns False if any neighbor's domain becomes empty."""
        for w in neighbours_in_chain(v):
            if w in assigned:
                continue
            new_dom: list[tuple[Coord, ...]] = []
            for p in domains_live[w]:
                if any(c in occupied_block for c in p):
                    continue
                if w == v - 1:
                    if not bridge_static(p[-1], value[0], p_between[w]):
                        continue
                else:  # w == v + 1
                    if not bridge_static(value[-1], p[0], p_between[v]):
                        continue
                new_dom.append(p)
            domain_save_stack.append((w, domains_live[w]))
            domains_live[w] = new_dom
            if not new_dom:
                return False
        return True

    def restore_domains_to(target_len: int) -> None:
        while len(domain_save_stack) > target_len:
            w, saved = domain_save_stack.pop()
            domains_live[w] = saved

    def select_var() -> int:
        # Prefer LONGEST H-run first (most constraining placement); within
        # same length, prefer smallest live domain.
        unassigned = [v for v in range(h_count) if v not in assigned]
        return min(unassigned, key=lambda v: (
            -h_runs[v][1].length, len(domains_live[v]), v
        ))

    def backtrack() -> bool:
        nonlocal best_solution
        if time.monotonic() > deadline:
            return False
        if all_assigned():
            sol = assemble_chain()
            if sol is not None and (
                best_solution is None or sol.contacts > best_solution.contacts
            ):
                best_solution = sol
            return False  # keep searching for higher-contact alternatives

        v = select_var()
        # If domain is empty, fail.
        if not domains_live[v]:
            return False

        # Iterate over a SNAPSHOT — recursion may further prune domains_live[v]
        # via filter_domain_after_commit on its way down.
        for value in list(domains_live[v]):
            if any(c in occupied_block for c in value):
                continue
            # Adjacency to ALREADY ASSIGNED chain-neighbors is checked by
            # bridge_static.
            ok = True
            for w in neighbours_in_chain(v):
                if w in assigned:
                    if w == v - 1:
                        if not bridge_static(assigned[w][-1], value[0], p_between[w]):
                            ok = False; break
                    else:
                        if not bridge_static(value[-1], assigned[w][0], p_between[v]):
                            ok = False; break
            if not ok:
                continue

            assigned[v] = value
            for c in value:
                occupied_block.add(c)
            saved_stack_len = len(domain_save_stack)
            if filter_domain_after_commit(v, value):
                backtrack()
            restore_domains_to(saved_stack_len)
            for c in value:
                occupied_block.discard(c)
            del assigned[v]
            if time.monotonic() > deadline:
                return False
        return False

    backtrack()
    return best_solution
