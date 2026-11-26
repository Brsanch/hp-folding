"""H-core constrained folding (CHCC-style, run-aware CSP version).

Strategy: every high-contact HP fold has its H residues clustered in a
compact lattice region (the "suite block"). The chain decomposes into
alternating H-runs (consecutive H residues, must form a SAW path inside
the block) and P-runs (consecutive P residues, must form a SAW path
outside the block, connecting two block boundary cells).

Pipeline:
  1. Parse sequence into runs (H-runs and P-runs of fixed lengths).
  2. Enumerate candidate suite blocks: compact polyominoes of size N_H,
     ranked by internal-edge count, filtered by minimum concave-corner
     count to handle short P-runs.
  3. For each suite block:
     a. For each H-run, enumerate all length-L SAW paths inside the
        block (cached per (block, length) but recomputed when occupied
        cells change during search).
     b. CSP-solve: pick one placement per H-run (in chain order) such
        that all P-runs are routable as length-K SAW paths outside the
        block.
  4. Among successful threadings, return the one with the most
     non-sequential H-H contacts.

The CSP solver uses chain-order backtracking. After committing the
i-th H-run's placement, we check P-run i is routable to any candidate
start of H-run i+1 *before* enumerating i+1's placements. This prunes
massively earlier than residue-by-residue DFS.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

Coord = tuple[int, int]
_NEIGHBOURS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ---------------------------------------------------------------------------
# Sequence run-decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    kind: str       # "H" or "P"
    length: int
    start: int      # chain index of first residue in this run


def parse_runs(sequence: str) -> list[Run]:
    runs: list[Run] = []
    i = 0
    n = len(sequence)
    while i < n:
        kind = sequence[i]
        j = i
        while j < n and sequence[j] == kind:
            j += 1
        runs.append(Run(kind=kind, length=j - i, start=i))
        i = j
    return runs


# ---------------------------------------------------------------------------
# Suite-block enumeration
# ---------------------------------------------------------------------------


def min_perimeter(n: int) -> int:
    if n <= 0:
        return 0
    a = math.isqrt(n)
    if a * a == n:
        return 4 * a
    if a * a + a >= n:
        return 4 * a + 2
    return 4 * a + 4


def _rect(a: int, b: int) -> frozenset[Coord]:
    return frozenset((x, y) for x in range(a) for y in range(b))


def _internal_edges(cells: frozenset[Coord]) -> int:
    e = 0
    for x, y in cells:
        if (x + 1, y) in cells:
            e += 1
        if (x, y + 1) in cells:
            e += 1
    return e


def _is_connected(cells: frozenset[Coord]) -> bool:
    if not cells:
        return True
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in _NEIGHBOURS:
            nb = (x + dx, y + dy)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _canon_translate(cells: frozenset[Coord]) -> frozenset[Coord]:
    if not cells:
        return cells
    minx = min(x for x, _ in cells)
    miny = min(y for _, y in cells)
    return frozenset((x - minx, y - miny) for x, y in cells)


def _concave_corner_count(cells: frozenset[Coord]) -> int:
    """A concave corner: an empty lattice cell adjacent to two block cells
    in perpendicular directions. These are exactly the points where a
    length-1 P-run can sit while bridging two block cells.
    """
    if not cells:
        return 0
    minx = min(x for x, _ in cells) - 1
    maxx = max(x for x, _ in cells) + 1
    miny = min(y for _, y in cells) - 1
    maxy = max(y for _, y in cells) + 1
    count = 0
    for x in range(minx, maxx + 1):
        for y in range(miny, maxy + 1):
            if (x, y) in cells:
                continue
            # is this empty cell adjacent to >=2 block cells in perpendicular
            # directions?
            nbrs_in_block = []
            for dx, dy in _NEIGHBOURS:
                nb = (x + dx, y + dy)
                if nb in cells:
                    nbrs_in_block.append((dx, dy))
            if len(nbrs_in_block) < 2:
                continue
            # check perpendicularity
            for i in range(len(nbrs_in_block)):
                for j in range(i + 1, len(nbrs_in_block)):
                    a, b = nbrs_in_block[i], nbrs_in_block[j]
                    if a[0] * b[0] + a[1] * b[1] == 0:  # perpendicular
                        count += 1
                        break
                else:
                    continue
                break
    return count


def _seed_blocks(n_h: int) -> list[frozenset[Coord]]:
    """Generate a few small near-square seed polyominoes of size n_h."""
    seeds: list[frozenset[Coord]] = []
    a = math.isqrt(n_h)
    # Try (a, a), (a, a+1), (a+1, a+1) base rectangles, removing extras
    # from corners.
    for da, db in ((0, 0), (0, 1), (1, 1), (-1, 1), (1, 2)):
        a2, b2 = a + da, a + db
        if a2 < 1 or b2 < 1 or a2 * b2 < n_h:
            continue
        extras = a2 * b2 - n_h
        if extras > 2 * a2:  # don't try removing too many
            continue
        base = _rect(a2, b2)
        if extras == 0:
            seeds.append(base)
            continue
        # Remove `extras` cells from one corner systematically.
        for corner in ((0, 0), (a2 - 1, 0), (0, b2 - 1), (a2 - 1, b2 - 1)):
            remaining = extras
            cur = set(base)
            cx, cy = corner
            # Remove cells in expanding shell from this corner until we've
            # taken `extras` cells.
            shell_order = sorted(
                cur,
                key=lambda c: abs(c[0] - cx) + abs(c[1] - cy),
            )
            taken = 0
            for cell in shell_order:
                if taken >= extras:
                    break
                cur.discard(cell)
                if not _is_connected(frozenset(cur)):
                    cur.add(cell)
                    continue
                taken += 1
            if taken == extras and len(cur) == n_h:
                seeds.append(frozenset(cur))
    # Dedup
    out: list[frozenset[Coord]] = []
    seen: set[frozenset[Coord]] = set()
    for s in seeds:
        c = _canon_translate(s)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _outside_cells_of(block: frozenset[Coord]) -> list[Coord]:
    """All lattice cells not in `block` but adjacent to a block cell."""
    out: set[Coord] = set()
    for x, y in block:
        for dx, dy in _NEIGHBOURS:
            nb = (x + dx, y + dy)
            if nb not in block:
                out.add(nb)
    return list(out)


def _boundary_cells(block: frozenset[Coord]) -> list[Coord]:
    return [
        c for c in block
        if any((c[0] + dx, c[1] + dy) not in block for dx, dy in _NEIGHBOURS)
    ]


def _single_mutations(
    block: frozenset[Coord],
) -> list[frozenset[Coord]]:
    """Generate all polyominoes of the same size as `block` reachable by:
       remove one boundary cell + add one outside cell that's still
       lattice-adjacent to (block \\ {removed cell}). Result must remain
       connected and translation-canonical."""
    out: list[frozenset[Coord]] = []
    seen: set[frozenset[Coord]] = set()
    boundary = _boundary_cells(block)
    for rm in boundary:
        b1 = block - {rm}
        if not _is_connected(b1):
            continue
        # candidates to add: outside cells adjacent to b1
        for ad in _outside_cells_of(b1):
            if ad == rm:
                continue
            b2 = b1 | {ad}
            if not _is_connected(b2):
                continue
            canon = _canon_translate(b2)
            if canon in seen:
                continue
            seen.add(canon)
            out.append(canon)
    return out


def enumerate_suite_blocks(
    n_h: int,
    *,
    max_extra_perimeter: int = 6,
    min_concave_corners: int = 0,
    max_blocks: int = 200,
) -> list[frozenset[Coord]]:
    """Compact polyominoes of size n_h, ranked by (concave_corners,
    internal_edges).

    Mutation-based BFS: start from near-square seed polyominoes, expand
    via single-cell mutations (remove one boundary, add one adjacent
    outside). Prune by perimeter excess relative to the analytic minimum.
    """
    import heapq
    p_min = min_perimeter(n_h)
    seen: set[frozenset[Coord]] = set()
    scored: list[tuple[int, int, frozenset[Coord]]] = []
    # Priority queue: highest ie first. Use negative ie as priority for min-heap.
    # Tiebreak by counter for deterministic ordering.
    pq: list[tuple[int, int, frozenset[Coord]]] = []
    counter = 0

    for s in _seed_blocks(n_h):
        if s in seen:
            continue
        seen.add(s)
        ie = _internal_edges(s)
        heapq.heappush(pq, (-ie, counter, s))
        counter += 1

    target = max_blocks * 4
    explored_cap = max(50_000, max_blocks * 500)  # hard cap to prevent runaway BFS
    while pq and len(scored) < target and len(seen) < explored_cap:
        neg_ie, _, s = heapq.heappop(pq)
        ie = -neg_ie
        perim = 4 * n_h - 2 * ie
        if perim > p_min + max_extra_perimeter:
            continue
        cc = _concave_corner_count(s)
        if cc >= min_concave_corners:
            scored.append((cc, ie, s))
        # Expand via single-cell mutations
        for m in _single_mutations(s):
            if m in seen:
                continue
            seen.add(m)
            ie_m = _internal_edges(m)
            perim_m = 4 * n_h - 2 * ie_m
            if perim_m > p_min + max_extra_perimeter + 2:
                continue
            heapq.heappush(pq, (-ie_m, counter, m))
            counter += 1

    # Sort by internal-edge count desc (most contacts possible),
    # tiebreak by concave-corner count desc (more flexible threading).
    scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
    return [s for _, _, s in scored[:max_blocks]]


# ---------------------------------------------------------------------------
# H-run placement enumeration (length-L SAWs inside a block)
# ---------------------------------------------------------------------------


def enumerate_run_paths_in_block(
    block: frozenset[Coord],
    length: int,
    *,
    max_paths: int | None = None,
    max_per_start: int | None = None,
) -> list[tuple[Coord, ...]]:
    """All length-L SAW paths within `block`. Balanced across starting cells
    so that no single start dominates the cap.

    If `max_paths` is given, total returned ≤ max_paths.
    If `max_per_start` is given, at most that many paths per starting cell.
    """
    out: list[tuple[Coord, ...]] = []
    block_list = list(block)
    # Default per-start cap so all starts get some representation.
    if max_per_start is None and max_paths is not None:
        max_per_start = max(1, max_paths // max(1, len(block_list)))

    cur: list[tuple[Coord, ...]]

    def dfs(path: list[Coord], start_count: list[int]) -> bool:
        if len(path) == length:
            out.append(tuple(path))
            start_count[0] += 1
            if max_paths is not None and len(out) >= max_paths:
                return True
            if max_per_start is not None and start_count[0] >= max_per_start:
                return True
            return False
        last = path[-1]
        for dx, dy in _NEIGHBOURS:
            nb = (last[0] + dx, last[1] + dy)
            if nb in block and nb not in path:
                path.append(nb)
                if dfs(path, start_count):
                    path.pop()
                    return True
                path.pop()
        return False

    for start in block_list:
        if max_paths is not None and len(out) >= max_paths:
            break
        start_count = [0]
        dfs([start], start_count)
    return out


# ---------------------------------------------------------------------------
# P-run routing (length-K SAW outside block from start to end)
# ---------------------------------------------------------------------------


def route_p_run(
    start: Coord,
    end: Coord,
    length: int,
    block: frozenset[Coord],
    forbidden: set[Coord],
    bbox: tuple[int, int, int, int],
    *,
    deadline: float | None = None,
) -> list[Coord] | None:
    """Find a SAW of `length` cells starting from `start`, ending at a
    cell adjacent to `end`, where:
      - `start` is the last cell of the previous H-run (in `forbidden`)
      - `end` is the first cell of the next H-run (in `forbidden`)
      - all `length` middle cells are NOT in `block`, NOT in `forbidden`,
        within `bbox`.
      - The chain bond from start -> first P, last P -> end must each be
        manhattan distance 1.
    """
    bx0, bx1, by0, by1 = bbox
    if length == 0:
        # Direct chain bond between H-runs requires manhattan(start, end) == 1
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) == 1:
            return []
        return None

    visited: set[Coord] = set()
    path: list[Coord] = []

    def dfs(last: Coord, depth: int) -> bool:
        if deadline is not None and time.monotonic() > deadline:
            return False
        if depth == length:
            return abs(last[0] - end[0]) + abs(last[1] - end[1]) == 1
        # Manhattan reachability prune: remaining cells must allow reaching
        # `end` in (length - depth + 1) lattice steps (via one final hop).
        remaining = length - depth
        # We need (remaining - 1) more P cells then a final step into `end`
        # Manhattan distance from `last` to `end` <= remaining (parity may matter)
        d_to_end = abs(last[0] - end[0]) + abs(last[1] - end[1])
        if d_to_end > remaining:
            return False
        # Parity: each step changes manhattan distance to `end` by ±1.
        # From `last` we need exactly `remaining` more cells before stepping to end.
        # The step lands on a cell at distance 1 from end; we then take the final
        # step. So after `remaining` cells (the last placed), we need d == 1.
        # Total parity from `last` to end through `remaining` cells: d_to_end
        # and remaining must have the same parity (with offset: end isn't placed,
        # we just need the LAST P cell to be neighbor of end, then final hop).
        # Effectively: after placing `remaining` more P cells and then taking
        # one bond into `end`, we've taken `remaining + 1` lattice steps from
        # `last`. Manhattan can change by at most ± (remaining+1) per step,
        # parity = (remaining+1) mod 2. So need d_to_end and (remaining + 1)
        # have same parity.
        if (d_to_end + remaining + 1) % 2 != 0:
            return False
        for dx, dy in _NEIGHBOURS:
            nb = (last[0] + dx, last[1] + dy)
            if nb in block:
                continue
            if nb in forbidden:
                continue
            if nb in visited:
                continue
            x, y = nb
            if x < bx0 or x > bx1 or y < by0 or y > by1:
                continue
            visited.add(nb)
            path.append(nb)
            if dfs(nb, depth + 1):
                return True
            path.pop()
            visited.discard(nb)
        return False

    if dfs(start, 0):
        return list(path)
    return None


# ---------------------------------------------------------------------------
# CSP solver: thread the chain through a candidate suite block
# ---------------------------------------------------------------------------


@dataclass
class HCoreSolution:
    coords: list[Coord]
    contacts: int


def _bbox_around_block(block: frozenset[Coord], pad: int) -> tuple[int, int, int, int]:
    xs = [x for x, _ in block]
    ys = [y for _, y in block]
    return (min(xs) - pad, max(xs) + pad,
            min(ys) - pad, max(ys) + pad)


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


def solve_csp_for_block(
    sequence: str,
    block: frozenset[Coord],
    *,
    time_budget_seconds: float = 30.0,
    h_run_max_placements: int = 2000,
) -> HCoreSolution | None:
    """Run-aware CSP solver. Returns the highest-contact threading found,
    or None if no threading exists for this block in the time budget."""
    runs = parse_runs(sequence)
    n_h = sum(1 for c in sequence if c == "H")
    if len(block) != n_h:
        raise ValueError(
            f"block has {len(block)} cells but sequence has {n_h} H residues"
        )

    h_runs = [r for r in runs if r.kind == "H"]
    # Pre-enumerate all length-L SAW paths inside the block, per H-run length.
    # For repeated lengths we share placements.
    placements_by_length: dict[int, list[tuple[Coord, ...]]] = {}
    for r in h_runs:
        if r.length in placements_by_length:
            continue
        placements_by_length[r.length] = enumerate_run_paths_in_block(
            block, r.length, max_paths=h_run_max_placements
        )

    pad = max(4, len(sequence) - n_h)
    bbox = _bbox_around_block(block, pad)

    coords: list[Coord] = [None] * len(sequence)  # type: ignore
    occupied: set[Coord] = set()
    deadline = time.monotonic() + time_budget_seconds

    best: HCoreSolution | None = None

    def commit_path(start_index: int, path: tuple[Coord, ...] | list[Coord]):
        for k, c in enumerate(path):
            coords[start_index + k] = c
            occupied.add(c)

    def uncommit_path(start_index: int, path: tuple[Coord, ...] | list[Coord]):
        for k, c in enumerate(path):
            coords[start_index + k] = None  # type: ignore
            occupied.discard(c)

    def solve(run_idx: int) -> bool:
        nonlocal best
        if time.monotonic() > deadline:
            return False
        # All runs placed
        if run_idx == len(runs):
            c = _count_contacts(sequence, coords)
            if best is None or c > best.contacts:
                best = HCoreSolution(coords=list(coords), contacts=c)
            return False  # keep searching for higher-contact alternatives

        r = runs[run_idx]
        if r.kind == "H":
            placements = placements_by_length[r.length]
            # If this is not the first run, the placement's first cell must
            # be lattice-adjacent to the last placed cell, and must not
            # overlap occupied.
            if run_idx == 0:
                cands = placements
            else:
                last_cell = coords[r.start - 1]
                lcx, lcy = last_cell  # type: ignore
                cands = [
                    p for p in placements
                    if abs(p[0][0] - lcx) + abs(p[0][1] - lcy) == 1
                ]
            for path in cands:
                if any(c in occupied for c in path):
                    continue
                commit_path(r.start, path)
                if solve(run_idx + 1):
                    uncommit_path(r.start, path)
                    return True
                uncommit_path(r.start, path)
                if time.monotonic() > deadline:
                    return False
            return False
        else:
            # P-run: route from coords[r.start - 1] (last H-run end) to
            # the first cell of the NEXT H-run, which we don't know yet.
            # Strategy: route to ANY block-boundary cell, then constrain the
            # next H-run to start adjacent.
            #
            # Practical approach: enumerate length-K SAW paths starting from
            # coords[r.start - 1], ending at a cell adjacent to *some* free
            # block cell. We commit just the path; the next H-run is then
            # constrained by the path's last cell.
            assert run_idx > 0, "sequence cannot start with a P-run handled here"
            start = coords[r.start - 1]
            # The next run is necessarily an H-run (alternation).
            next_run = runs[run_idx + 1] if run_idx + 1 < len(runs) else None
            # We need the LAST P cell to be lattice-adjacent to the first cell
            # of the next H-run, which must be inside the block. If this is
            # the FINAL run (no next), no block-adjacency required.
            paths = _enumerate_p_paths(
                start, r.length, block, occupied, bbox, deadline,
                require_last_block_adjacent=(next_run is not None),
            )
            for ppath in paths:
                # ppath has length r.length, all outside block
                # last cell must be adjacent to a block cell that's free
                lx, ly = ppath[-1]
                has_free_block_nb = False
                for dx, dy in _NEIGHBOURS:
                    nb = (lx + dx, ly + dy)
                    if nb in block and nb not in occupied:
                        has_free_block_nb = True
                        break
                if not has_free_block_nb and next_run is not None:
                    continue
                commit_path(r.start, ppath)
                if solve(run_idx + 1):
                    uncommit_path(r.start, ppath)
                    return True
                uncommit_path(r.start, ppath)
                if time.monotonic() > deadline:
                    return False
            return False

    # Sequence must start with run 0. If sequence starts with a P-run, we'd
    # need extra logic; defer for now (record-targeted sequences S6-S10 all
    # start with P or H — we handle both).
    r0 = runs[0]
    if r0.kind == "H":
        for path in placements_by_length[r0.length]:
            commit_path(r0.start, path)
            solve(1)
            uncommit_path(r0.start, path)
            if time.monotonic() > deadline:
                break
    else:
        # P-run start: chain begins outside the block. Enumerate starting
        # cells in a small ring around the block.
        ring = _outside_ring(block, bbox)
        for s in ring:
            paths = _enumerate_p_paths_from_free(
                s, r0.length, block, set(), bbox, deadline
            )
            for ppath in paths:
                commit_path(r0.start, ppath)
                solve(1)
                uncommit_path(r0.start, ppath)
                if time.monotonic() > deadline:
                    break
            if time.monotonic() > deadline:
                break

    return best


def _enumerate_p_paths(
    start: Coord,
    length: int,
    block: frozenset[Coord],
    forbidden: set[Coord],
    bbox: tuple[int, int, int, int],
    deadline: float,
    *,
    max_paths: int = 2000,
    require_last_block_adjacent: bool = True,
) -> list[list[Coord]]:
    """All SAW paths of `length` cells starting at a neighbour of `start`,
    avoiding `block` and `forbidden`, staying inside `bbox`. If
    require_last_block_adjacent (default), the last cell of each path
    must be lattice-adjacent to a free (non-occupied, non-`forbidden`)
    block cell so the next H-run can connect."""
    bx0, bx1, by0, by1 = bbox
    out: list[list[Coord]] = []
    visited: set[Coord] = set()
    path: list[Coord] = []

    def is_block_adj_free(c: Coord) -> bool:
        for dx, dy in _NEIGHBOURS:
            nb = (c[0] + dx, c[1] + dy)
            if nb in block and nb not in forbidden:
                return True
        return False

    def dfs(last: Coord, remaining: int) -> bool:
        if time.monotonic() > deadline:
            return True
        if remaining == 0:
            if require_last_block_adjacent and not is_block_adj_free(last):
                return False
            out.append(list(path))
            return len(out) >= max_paths
        for dx, dy in _NEIGHBOURS:
            nb = (last[0] + dx, last[1] + dy)
            if nb in block or nb in forbidden or nb in visited:
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

    # First step: from `start`, into a free non-block neighbour.
    for dx, dy in _NEIGHBOURS:
        nb = (start[0] + dx, start[1] + dy)
        if nb in block or nb in forbidden:
            continue
        x, y = nb
        if x < bx0 or x > bx1 or y < by0 or y > by1:
            continue
        visited.add(nb)
        path.append(nb)
        if dfs(nb, length - 1):
            break
        path.pop()
        visited.discard(nb)
    return out


def _enumerate_p_paths_from_free(
    start: Coord,
    length: int,
    block: frozenset[Coord],
    forbidden: set[Coord],
    bbox: tuple[int, int, int, int],
    deadline: float,
    *,
    max_paths: int = 1000,
    require_last_block_adjacent: bool = True,
) -> list[list[Coord]]:
    """SAW paths of `length` cells starting AT `start` (P-run, sequence
    begins outside block). If require_last_block_adjacent, the last cell
    of the path must be lattice-adjacent to a non-occupied block cell so
    the next H-run can connect."""
    bx0, bx1, by0, by1 = bbox
    out: list[list[Coord]] = []
    visited: set[Coord] = {start}
    path: list[Coord] = [start]

    def is_block_adj(c: Coord) -> bool:
        for dx, dy in _NEIGHBOURS:
            if (c[0] + dx, c[1] + dy) in block:
                return True
        return False

    def dfs(last: Coord, remaining: int) -> bool:
        if time.monotonic() > deadline:
            return True
        if remaining == 0:
            if require_last_block_adjacent and not is_block_adj(last):
                return False
            out.append(list(path))
            return len(out) >= max_paths
        for dx, dy in _NEIGHBOURS:
            nb = (last[0] + dx, last[1] + dy)
            if nb in block or nb in forbidden or nb in visited:
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

    if start in block:
        return []
    dfs(start, length - 1)
    return out


def _outside_ring(block: frozenset[Coord], bbox: tuple[int, int, int, int]) -> list[Coord]:
    """Cells immediately adjacent to the block (one ring out)."""
    out = set()
    for x, y in block:
        for dx, dy in _NEIGHBOURS:
            nb = (x + dx, y + dy)
            if nb not in block:
                out.add(nb)
    return list(out)
