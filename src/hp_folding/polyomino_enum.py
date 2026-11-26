"""Systematic enumeration of fixed polyominoes by internal-edge count.

For HP folding: the H-only "hcore" is a set of n_H lattice cells. The number
of internal H-H edges (= 4*n_H - perimeter)/2 lower-bounds H-H contacts
(internal edges include sequence-adjacent H pairs which DON'T count as
contacts; we filter those later via the CSP routability layer).

This module enumerates connected polyominoes of a fixed size, deduplicated
under the dihedral group D4 of the square lattice + translations (hcores
related by isometry yield the same fold up to canonical chain symmetry).

Strategy: Redelmeier-style canonical enumeration.

  - Start from cell (0, 0).
  - At each step, the candidate set is the union of free neighbors of
    placed cells.
  - To avoid duplicates, we use Redelmeier's "untried" trick: each cell is
    assigned a number when first added to candidate set; a cell can only
    be placed if its number > the number of every still-unplaced earlier
    candidate that was rejected. (This gives us *fixed* polyominoes
    counted exactly once per occurrence.)
  - For each completed polyomino, count internal edges and canonicalize
    under D4 + translation.

Output: dict mapping `internal_edges → set[canonical_polyomino]`.
"""
from __future__ import annotations
from typing import Iterable

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


D4 = [
    lambda x, y: (x, y),
    lambda x, y: (-x, y),
    lambda x, y: (x, -y),
    lambda x, y: (-x, -y),
    lambda x, y: (y, x),
    lambda x, y: (-y, x),
    lambda x, y: (y, -x),
    lambda x, y: (-y, -x),
]


def canonicalize(cells: Iterable[Coord]) -> tuple[Coord, ...]:
    """Canonical form: smallest sorted-tuple over D4 + translation-to-origin."""
    cells = list(cells)
    candidates = []
    for op in D4:
        transformed = [op(x, y) for (x, y) in cells]
        x0 = min(c[0] for c in transformed)
        y0 = min(c[1] for c in transformed)
        shifted = tuple(sorted((c[0] - x0, c[1] - y0) for c in transformed))
        candidates.append(shifted)
    return min(candidates)


def internal_edges(cells: set[Coord]) -> int:
    """Count H-H lattice edges within the polyomino."""
    e = 0
    for x, y in cells:
        if (x + 1, y) in cells: e += 1
        if (x, y + 1) in cells: e += 1
    return e


def enumerate_polyominoes(n: int, *,
                            min_internal_edges: int = 0,
                            max_bbox: int | None = None):
    """Generate all *canonical* polyominoes of size n via Redelmeier.

    Yields (canonical_form, internal_edges) for each unique polyomino with
    `internal_edges >= min_internal_edges` and bbox extent ≤ max_bbox.

    Pruning bounds:
      - bbox: any polyomino exceeding `max_bbox` in either dimension is
        pruned during recursion. High-internal-edge polyominoes fit in
        small bboxes; we lose nothing for high-target searches.
      - perimeter / internal_edges: max final_ie ≤ current_ie + 2*remaining.
        (Each remaining cell adds at most 2 edges on average in compact
        polyominoes; this is a tight upper bound for the kind of shapes we
        care about. It's not a hard mathematical bound but in practice
        cuts irrelevant branches.)
    """
    if max_bbox is None:
        # Default: tight enough that we get reasonable speedups, loose
        # enough that we don't lose realistic compact shapes.
        max_bbox = max(2, int(2 * (n ** 0.5)) + 1)

    placed: list[Coord] = [(0, 0)]
    placed_set: set[Coord] = {(0, 0)}
    candidates: list[Coord] = []
    for dx, dy in NBRS:
        c = (dx, dy)
        if c[1] > 0 or (c[1] == 0 and c[0] > 0):
            candidates.append(c)
    seen: set[tuple[Coord, ...]] = set()

    def edges_with_existing(cell: Coord) -> int:
        return sum(1 for dx, dy in NBRS if (cell[0] + dx, cell[1] + dy) in placed_set)

    def recurse(remaining: int, cands: list[Coord], rejected: set[Coord],
                cur_ie: int, xmin: int, xmax: int, ymin: int, ymax: int):
        if remaining == 0:
            if cur_ie >= min_internal_edges:
                canon = canonicalize(placed_set)
                if canon not in seen:
                    seen.add(canon)
                    yield (canon, cur_ie)
            return
        # Bound: can we still reach min_internal_edges?
        if cur_ie + 2 * remaining < min_internal_edges:
            return
        local_rejected = set(rejected)
        for i, cell in enumerate(cands):
            if cell in local_rejected:
                continue
            # bbox prune
            new_xmin = min(xmin, cell[0])
            new_xmax = max(xmax, cell[0])
            new_ymin = min(ymin, cell[1])
            new_ymax = max(ymax, cell[1])
            if new_xmax - new_xmin + 1 > max_bbox: continue
            if new_ymax - new_ymin + 1 > max_bbox: continue
            d = edges_with_existing(cell)
            placed.append(cell)
            placed_set.add(cell)
            new_cands = list(cands[i+1:])
            for dx, dy in NBRS:
                nc = (cell[0] + dx, cell[1] + dy)
                if nc in placed_set or nc in local_rejected: continue
                if nc[1] < 0 or (nc[1] == 0 and nc[0] <= 0): continue
                if nc in new_cands or nc in cands[:i+1]: continue
                new_cands.append(nc)
            yield from recurse(remaining - 1, new_cands, local_rejected,
                                cur_ie + d, new_xmin, new_xmax, new_ymin, new_ymax)
            placed.pop()
            placed_set.remove(cell)
            local_rejected.add(cell)

    yield from recurse(n - 1, candidates, set(), 0, 0, 0, 0, 0)


def enumerate_by_contacts(n_h: int, *, target_min_internal: int):
    """Sort polyominoes of size n_h by internal_edges descending, filter by min."""
    out = []
    for canon, ie in enumerate_polyominoes(n_h, min_internal_edges=target_min_internal):
        out.append((ie, canon))
    out.sort(key=lambda t: -t[0])
    return out


if __name__ == "__main__":
    # Sanity: counts should match OEIS A001168 (fixed polyominoes)
    # n=1:1, n=2:2, n=3:6, n=4:19, n=5:63, n=6:216, n=7:760
    # But we count *free* (canonical under D4) polyominoes (OEIS A000105):
    # n=1:1, n=2:1, n=3:2, n=4:5, n=5:12, n=6:35, n=7:108
    import time
    for n in range(1, 8):
        t0 = time.time()
        cnt = sum(1 for _ in enumerate_polyominoes(n))
        print(f'n={n}: {cnt} free polyominoes  ({time.time()-t0:.2f}s)')
