"""Move operators for Monte Carlo / simulated annealing on HP folds.

All moves take a list of lattice coordinates describing a valid self-avoiding
walk, plus a `random.Random` instance, and return either:
  - a new list of coordinates describing an equal-length valid SAW, or
  - ``None`` if no legal move of the requested type is available.

The input fold is never modified.

Implemented moves:

end_flip
    Rotate the first or last residue to a free lattice neighbour of its
    interior neighbour.

pivot
    Pick an interior residue and rotate the tail by 90, 180, or 270 degrees
    around it. Rejected if the result self-intersects.

pull_move
    The Lesh-Mitzenmacher-Whitesides (2003) pull move: slide residue `i` onto
    a free lattice site diagonally adjacent to residue `i+1`, then cascade the
    remaining chain backward to preserve connectivity. This is the canonical
    "large-scale" move for HP folding and is ergodic on SAWs, unlike pivots
    which leave some configurations unreachable on the 2D square lattice.
"""

from __future__ import annotations

import random

Coord = tuple[int, int]
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _is_valid_saw(coords: list[Coord]) -> bool:
    if len(coords) != len(set(coords)):
        return False
    for i in range(len(coords) - 1):
        if _manhattan(coords[i], coords[i + 1]) != 1:
            return False
    return True


def end_flip(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Rotate one of the two end residues to a free lattice neighbour of its
    interior neighbour. Returns ``None`` if no legal flip exists."""
    n = len(coords)
    if n < 2:
        return None
    # Pick head (0) or tail (n-1) uniformly.
    end = 0 if rng.random() < 0.5 else n - 1
    interior = 1 if end == 0 else n - 2
    anchor = coords[interior]
    occupied = set(coords)
    occupied.discard(coords[end])
    candidates = [
        (anchor[0] + dx, anchor[1] + dy)
        for dx, dy in _NEIGHBOURS
        if (anchor[0] + dx, anchor[1] + dy) != coords[end]
        and (anchor[0] + dx, anchor[1] + dy) not in occupied
    ]
    if not candidates:
        return None
    new = list(coords)
    new[end] = rng.choice(candidates)
    return new


def pivot(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Rotate the tail of the chain (after a random interior pivot) by 90,
    180, or 270 degrees. Returns ``None`` if the rotated fold self-intersects
    or if the chain is too short."""
    n = len(coords)
    if n < 4:
        return None
    k = rng.randrange(1, n - 1)
    angle = rng.choice((90, 180, 270))
    cx, cy = coords[k]
    new = list(coords[: k + 1])
    for x, y in coords[k + 1 :]:
        rx, ry = x - cx, y - cy
        if angle == 90:
            rx, ry = -ry, rx
        elif angle == 180:
            rx, ry = -rx, -ry
        else:  # 270
            rx, ry = ry, -rx
        new.append((cx + rx, cy + ry))
    if len(set(new)) != n:
        return None
    return new


def corner_flip(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Flip a 90-degree corner. Pick an interior residue i where coords[i-1],
    coords[i], coords[i+1] form an L-shape (the two non-coords[i] residues
    are at manhattan distance 2 with perpendicular offsets, and coords[i] is
    at one of the two corner cells completing the rectangle). Move coords[i]
    to the OTHER corner cell if free.

    This is the simplest and one of the most effective local moves for HP
    folding — chain bonds (i-1, i) and (i, i+1) are preserved by construction.
    """
    n = len(coords)
    if n < 3:
        return None
    occupied = set(coords)
    # Pick a random interior residue with corner geometry
    indices = list(range(1, n - 1))
    rng.shuffle(indices)
    for i in indices:
        prev = coords[i - 1]
        nxt = coords[i + 1]
        # prev and nxt at manhattan 2, perpendicular (not collinear)
        dx = nxt[0] - prev[0]
        dy = nxt[1] - prev[1]
        if abs(dx) + abs(dy) != 2:
            continue
        if dx == 0 or dy == 0:
            continue  # collinear
        # Two corner cells: (prev[0]+dx, prev[1]) and (prev[0], prev[1]+dy).
        # One of them is coords[i]; the other is the flip target.
        cur = coords[i]
        candidate1 = (prev[0] + dx, prev[1])
        candidate2 = (prev[0], prev[1] + dy)
        target = candidate2 if cur == candidate1 else candidate1
        if target in occupied and target != cur:
            continue
        new_coords = list(coords)
        new_coords[i] = target
        return new_coords
    return None


def local_rebridge(
    coords: list[Coord],
    rng: random.Random,
    *,
    seg_len: int = 4,
) -> list[Coord] | None:
    """Pick a sub-segment of `seg_len` consecutive interior residues and
    replace its lattice path with a different length-`seg_len` SAW between
    the same two endpoints, avoiding all other chain cells.

    For seg_len in [2..6], this is the natural "rebridge" move that
    enables topological reorganization a single corner-flip can't reach.
    """
    n = len(coords)
    if n < seg_len + 2:
        return None
    occupied = set(coords)

    starts = list(range(0, n - seg_len - 1))
    rng.shuffle(starts)
    for i in starts:
        # Segment is coords[i+1..i+seg_len] (seg_len cells), endpoints
        # coords[i] (fixed before) and coords[i+seg_len+1] (fixed after).
        a = coords[i]
        b = coords[i + seg_len + 1]
        # Manhattan distance must be reachable in seg_len+1 steps with parity.
        d = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if d > seg_len + 1 or (d + seg_len + 1) % 2 != 0:
            continue
        # Forbidden cells: occupied EXCEPT the current segment cells (which
        # we're about to free up).
        seg_cells = {coords[i + 1 + k] for k in range(seg_len)}
        forbidden = (occupied - seg_cells) | {a, b}
        # Search for an alternative path of length seg_len from a to b.
        path = _find_alt_path(
            a, b, seg_len, forbidden, current=tuple(coords[i + 1: i + 1 + seg_len]),
            rng=rng,
        )
        if path is None:
            continue
        new_coords = list(coords)
        for k in range(seg_len):
            new_coords[i + 1 + k] = path[k]
        # Validate SAW (defensive)
        if len(set(new_coords)) != n:
            continue
        return new_coords
    return None


def _find_alt_path(
    a: Coord,
    b: Coord,
    seg_len: int,
    forbidden: set[Coord],
    *,
    current: tuple[Coord, ...],
    rng: random.Random,
) -> list[Coord] | None:
    """Find a length-`seg_len` path of cells (each lattice-adjacent to the
    next) from a neighbour of `a`, ending at a neighbour of `b`, avoiding
    `forbidden`. Returns a path different from `current` if possible."""
    visited: set[Coord] = set()
    path: list[Coord] = []
    found: list[list[Coord]] = []

    def dfs(node: Coord, remaining: int) -> bool:
        if remaining == 0:
            if abs(node[0] - b[0]) + abs(node[1] - b[1]) == 1:
                p = list(path)
                if tuple(p) != current:
                    found.append(p)
                    return True  # halt search at first different path
            return False
        # Manhattan + parity prune
        d = abs(node[0] - b[0]) + abs(node[1] - b[1])
        if d > remaining + 1 or (d + remaining + 1) % 2 != 0:
            return False
        # Iterate neighbours in random order for diversity
        nbrs = [(node[0] + dx, node[1] + dy) for dx, dy in _NEIGHBOURS]
        rng.shuffle(nbrs)
        for nb in nbrs:
            if nb in forbidden or nb in visited:
                continue
            visited.add(nb)
            path.append(nb)
            if dfs(nb, remaining - 1):
                return True
            path.pop()
            visited.discard(nb)
        return False

    # First step: from a, into a neighbour
    nbrs = [(a[0] + dx, a[1] + dy) for dx, dy in _NEIGHBOURS]
    rng.shuffle(nbrs)
    for nb in nbrs:
        if nb in forbidden:
            continue
        visited.add(nb)
        path.append(nb)
        if seg_len == 1:
            if abs(nb[0] - b[0]) + abs(nb[1] - b[1]) == 1 and tuple([nb]) != current:
                return list(path)
            path.pop()
            visited.discard(nb)
            continue
        if dfs(nb, seg_len - 1):
            return found[0] if found else None
        path.pop()
        visited.discard(nb)
    return None


def segment_rotate(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Rotate (reflect) a sub-chain through the line connecting its endpoints.

    Pick i, j with i < j-1 such that coords[i] and coords[j] share an x or y
    coordinate (axis-aligned endpoints). The segment coords[i+1..j-1] is
    reflected through the line through the endpoints. Endpoint bonds are
    preserved by construction (distance preserved by reflection through a
    line containing both endpoints); interior chain bonds are also
    preserved (rigid reflection). The resulting chain is valid iff the
    reflected interior cells don't collide with the unchanged prefix/suffix.

    This is a global-class move — much bigger than crankshaft (which is just
    a 2-cell case of this).
    """
    n = len(coords)
    if n < 4:
        return None
    # Pick (i, j) randomly across allowed pairs.
    # Try a sample of lengths and starts.
    starts = list(range(0, n - 3))
    rng.shuffle(starts)
    for i in starts:
        # Random length, biased toward shorter
        max_seg_inner = min(20, n - 2 - i)
        if max_seg_inner < 2:
            continue
        seg_lens = list(range(2, max_seg_inner + 1))
        rng.shuffle(seg_lens)
        for inner_len in seg_lens:
            j = i + inner_len + 1
            if j >= n:
                continue
            a = coords[i]
            b = coords[j]
            # Axis-aligned endpoints?
            if a[0] == b[0]:
                # Vertical line through x = a[0]; reflect x of interior
                new_coords = list(coords)
                for m in range(i + 1, j):
                    x, y = coords[m]
                    new_coords[m] = (2 * a[0] - x, y)
            elif a[1] == b[1]:
                new_coords = list(coords)
                for m in range(i + 1, j):
                    x, y = coords[m]
                    new_coords[m] = (x, 2 * a[1] - y)
            else:
                continue
            # Skip identity moves (axis-aligned segment ON the axis already)
            if all(new_coords[m] == coords[m] for m in range(i + 1, j)):
                continue
            # Validate SAW
            if len(set(new_coords)) != n:
                continue
            return new_coords
    return None


def crankshaft(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Crankshaft move on a 4-residue U-shape. Pick i where coords[i-1],
    coords[i], coords[i+1], coords[i+2] form a U: coords[i-1] and
    coords[i+2] are at manhattan distance 1, and coords[i], coords[i+1]
    are the two cells "outside" the line connecting them. Reflect the U
    to the opposite side.
    """
    n = len(coords)
    if n < 4:
        return None
    occupied = set(coords)
    indices = list(range(1, n - 2))
    rng.shuffle(indices)
    for i in indices:
        a = coords[i - 1]
        d = coords[i + 2]
        if _manhattan(a, d) != 1:
            continue
        b = coords[i]
        c = coords[i + 1]
        # New positions: reflect b, c to other side of segment a-d.
        dx = d[0] - a[0]
        dy = d[1] - a[1]
        # Perpendicular offset of b from a:
        ob_x = b[0] - a[0]
        ob_y = b[1] - a[1]
        # Reflect: the "side" is determined by the perpendicular component.
        # Simplest: try both perpendicular directions.
        # Compute the two candidates by rotating b around the a-d line.
        # b is at offset (ob_x, ob_y) from a; the new b would be at
        # (-ob_x, -ob_y) flipped along the axis (dx, dy).
        # Easier: each of b and c is at perpendicular distance 1 from
        # segment a-d. Reflect by negating the perpendicular component.
        # If a-d is horizontal (dy=0), perpendicular is y-axis: reflect y.
        # If a-d is vertical (dx=0), perpendicular is x-axis: reflect x.
        if dy == 0:
            new_b = (b[0], 2 * a[1] - b[1])
            new_c = (c[0], 2 * a[1] - c[1])
        else:
            new_b = (2 * a[0] - b[0], b[1])
            new_c = (2 * a[0] - c[0], c[1])
        if new_b == b or new_c == c:
            continue
        if new_b in occupied and new_b not in (b, c):
            continue
        if new_c in occupied and new_c not in (b, c):
            continue
        if new_b == new_c:
            continue
        new_coords = list(coords)
        new_coords[i] = new_b
        new_coords[i + 1] = new_c
        if _is_valid_saw(new_coords):
            return new_coords
    return None


def pull_move_v2(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Pull move with cascade. Pick interior residue i, move to a free
    neighbor cell, then cascade backward repairing broken chain bonds by
    shifting each preceding residue to the predecessor's old position.

    Implements the working subset of L-M-W pull moves: cases where the new
    cell + cascade chain remain a valid SAW. Returns None if cascade fails
    (free position is not adjacent to the cell being repaired).
    """
    n = len(coords)
    if n < 3:
        return None
    i = rng.randrange(1, n - 1)
    direction = rng.choice((+1, -1))

    occupied = set(coords)
    nbs = list(_NEIGHBOURS)
    rng.shuffle(nbs)

    for dx, dy in nbs:
        new_pos = (coords[i][0] + dx, coords[i][1] + dy)
        if new_pos in occupied:
            continue
        new_coords = list(coords)
        new_coords[i] = new_pos
        # Check chain bond on direction side first (e.g., (i, i+1) for direction=+1)
        side_bond_ok = (
            _manhattan(new_pos, coords[i + direction]) == 1
        ) if 0 <= i + direction < n else True
        if not side_bond_ok:
            continue
        # Cascade backward: free_pos starts at the cell residue i just left
        free_pos = coords[i]
        j = i - direction
        cascade_ok = True
        while 0 <= j < n:
            target = new_coords[j + direction]
            if _manhattan(new_coords[j], target) == 1:
                break
            # Need to shift residue j to free_pos
            if _manhattan(free_pos, target) != 1:
                cascade_ok = False
                break
            old_j = new_coords[j]
            new_coords[j] = free_pos
            free_pos = old_j
            j -= direction
        if not cascade_ok:
            continue
        # Validate SAW
        if len(set(new_coords)) != n:
            continue
        all_bonds = True
        for k in range(n - 1):
            if _manhattan(new_coords[k], new_coords[k + 1]) != 1:
                all_bonds = False
                break
        if not all_bonds:
            continue
        return new_coords
    return None


def pull_move(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Lesh-Mitzenmacher-Whitesides pull move.

    Pick a residue ``i`` (1 <= i < n-1) and a direction. Find:
      - C: a lattice cell at manhattan-distance 2 from coords[i] (diagonal),
        lattice-adjacent to coords[i + direction] (the "anchor"), and
        unoccupied (except possibly by coords[i - direction]).
      - L: a lattice cell that is BOTH adjacent to coords[i - direction]
        AND adjacent to C, unoccupied (except possibly by coords[i]).
    Then move residue i to C and residue (i - direction) to L. Cascade
    residues further back along the chain by shifting each to the old
    position of its successor in the cascade direction, until a residue's
    new position is already adjacent to its (newly-placed) chain-neighbor.

    Returns the new SAW or ``None`` if no legal pull is found for the
    randomly chosen i and direction.

    References
    ----------
    Lesh, Mitzenmacher, Whitesides (2003), *A Complete and Effective Move Set
    for Simplified Protein Folding*, RECOMB, pp. 188-195.
    """
    n = len(coords)
    if n < 3:
        return None

    direction = rng.choice((+1, -1))
    # We need both i-direction and i+direction to be valid chain indices.
    if direction == +1:
        i = rng.randrange(1, n - 1)
    else:
        i = rng.randrange(1, n - 1)
    anchor = coords[i + direction]
    j = i - direction
    if not (0 <= j < n):
        return None
    prev = coords[j]

    occupied = set(coords)
    ix, iy = coords[i]
    # Enumerate (C, L) pairs.
    pairs: list[tuple[Coord, Coord]] = []
    for dxc, dyc in _NEIGHBOURS:
        c = (anchor[0] + dxc, anchor[1] + dyc)
        if c == coords[i]:
            continue
        if abs(c[0] - ix) + abs(c[1] - iy) != 2:
            continue
        if c[0] == ix or c[1] == iy:
            continue
        if c in occupied and c != prev:
            continue
        # L is adjacent to both prev and c, and unoccupied (or == coords[i]).
        for dxl, dyl in _NEIGHBOURS:
            l = (prev[0] + dxl, prev[1] + dyl)
            if abs(l[0] - c[0]) + abs(l[1] - c[1]) != 1:
                continue
            if l == prev or l == c:
                continue
            if l in occupied and l != coords[i]:
                continue
            pairs.append((c, l))

    if not pairs:
        return None
    c_new, l_new = rng.choice(pairs)

    new_coords = list(coords)
    new_coords[i] = c_new
    new_coords[j] = l_new

    # Cascade: residues at indices further from i (in direction opposite to
    # `direction`) shift one slot toward the original chain. Each takes the
    # old position of its predecessor in the cascade.
    old_pos = coords[j]  # old position of residue j; now residue (j - direction) may take it
    k = j - direction
    while 0 <= k < n:
        if _manhattan(new_coords[k], new_coords[k + direction]) == 1:
            break
        next_old = coords[k]
        new_coords[k] = old_pos
        old_pos = next_old
        k -= direction

    if _is_valid_saw(new_coords):
        return new_coords
    return None


def _h_clusters(coords: list[Coord], seq: str) -> list[set[int]]:
    """Connected components of H-beads under lattice adjacency. Returns
    list of sets of chain-residue indices."""
    h_pos: dict[Coord, int] = {}
    for i, c in enumerate(coords):
        if seq[i] == "H":
            h_pos[tuple(c)] = i
    visited: set[int] = set()
    clusters: list[set[int]] = []
    for start_pos, start_idx in h_pos.items():
        if start_idx in visited:
            continue
        comp: set[int] = set()
        stack = [start_pos]
        while stack:
            p = stack.pop()
            idx = h_pos.get(p)
            if idx is None or idx in comp:
                continue
            comp.add(idx)
            for dx, dy in _NEIGHBOURS:
                nb = (p[0] + dx, p[1] + dy)
                if nb in h_pos and h_pos[nb] not in comp:
                    stack.append(nb)
        visited |= comp
        clusters.append(comp)
    return clusters


def _centroid(coords: list[Coord], indices) -> tuple[float, float]:
    if not indices:
        return (0.0, 0.0)
    sx = sum(coords[i][0] for i in indices)
    sy = sum(coords[i][1] for i in indices)
    return (sx / len(indices), sy / len(indices))


def cluster_merge(
    coords: list[Coord], rng: random.Random, seq: str
) -> list[Coord] | None:
    """Hinge-fold designed to merge fragmented H-clusters.

    1. Identify H-clusters (lattice-connected components of H-beads).
    2. If only one cluster exists, return None (nothing to merge).
    3. Pick a hinge index k between the two largest clusters along the chain.
    4. Try all 8 D4 transforms (4 rotations × 2 reflections) of coords[k+1:]
       around coords[k]. Score each by post-transform inter-centroid
       distance (lower = better merge). Return the lowest-distance valid
       SAW that strictly reduces inter-cluster distance, else None.
    """
    n = len(coords)
    if n < 4:
        return None
    clusters = _h_clusters(coords, seq)
    if len(clusters) < 2:
        return None
    # Pick the two largest clusters
    clusters.sort(key=len, reverse=True)
    A, B = clusters[0], clusters[1]
    # Hinge: pick a chain index between max(A) and min(B), or between min(A)
    # and max(B), whichever range is non-empty.
    a_min, a_max = min(A), max(A)
    b_min, b_max = min(B), max(B)
    if a_max < b_min:
        lo, hi = a_max + 1, b_min
    elif b_max < a_min:
        lo, hi = b_max + 1, a_min
    else:
        # Interleaved — pick a hinge in the smaller cluster's edge
        lo = max(min(a_min, b_min) + 1, 1)
        hi = min(max(a_max, b_max), n - 1)
    if lo >= hi:
        return None
    k = rng.randrange(lo, hi)
    head = coords[: k + 1]
    tail = coords[k + 1:]
    if not tail:
        return None
    pivot_pt = coords[k]
    head_set = set(head)
    head_centroid = _centroid(head, [i for i in A if i <= k] or list(range(k + 1)))

    # Current inter-cluster distance baseline
    cur_a = _centroid(coords, A)
    cur_b = _centroid(coords, B)
    cur_dist = abs(cur_a[0] - cur_b[0]) + abs(cur_a[1] - cur_b[1])

    best = None
    best_dist = cur_dist - 0.01  # require strict improvement

    for refl in (False, True):
        for rot in range(4):
            new_tail = []
            ok = True
            tail_set: set[Coord] = set()
            for x, y in tail:
                # translate to origin (relative to pivot)
                rx, ry = x - pivot_pt[0], y - pivot_pt[1]
                if refl:
                    ry = -ry
                for _ in range(rot):
                    rx, ry = -ry, rx
                nx, ny = rx + pivot_pt[0], ry + pivot_pt[1]
                if (nx, ny) in head_set or (nx, ny) in tail_set:
                    ok = False
                    break
                tail_set.add((nx, ny))
                new_tail.append((nx, ny))
            if not ok:
                continue
            cand = head + new_tail
            # Identity transform → reject (same fold)
            if not refl and rot == 0:
                continue
            # Inter-cluster distance after transform
            new_a = _centroid(cand, A)
            new_b = _centroid(cand, B)
            new_dist = abs(new_a[0] - new_b[0]) + abs(new_a[1] - new_b[1])
            if new_dist < best_dist:
                best_dist = new_dist
                best = cand
    return best


def _h_bbox_area(coords: list[Coord], seq: str) -> int:
    h = [coords[i] for i in range(len(coords)) if seq[i] == "H"]
    if not h:
        return 0
    xs = [c[0] for c in h]
    ys = [c[1] for c in h]
    return (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)


def compact_pivot(
    coords: list[Coord], rng: random.Random, seq: str
) -> list[Coord] | None:
    """Pivot biased toward H-bounding-box compaction.

    Picks a random interior pivot k, tries all 8 D4 transforms of the tail
    coords[k+1:], and returns the one that minimizes post-transform
    H-bbox area (with strict improvement over current). SAW-valid only.
    Bypasses SA's contact-only acceptance to escape elongated minima.
    """
    n = len(coords)
    if n < 4:
        return None
    cur_area = _h_bbox_area(coords, seq)
    if cur_area == 0:
        return None
    k = rng.randrange(1, n - 1)
    head = coords[: k + 1]
    tail = coords[k + 1:]
    pivot_pt = coords[k]
    head_set = set(head)

    best = None
    best_area = cur_area  # require strict improvement

    for refl in (False, True):
        for rot in range(4):
            if not refl and rot == 0:
                continue
            new_tail: list[Coord] = []
            tail_set: set[Coord] = set()
            ok = True
            for x, y in tail:
                rx, ry = x - pivot_pt[0], y - pivot_pt[1]
                if refl:
                    ry = -ry
                for _ in range(rot):
                    rx, ry = -ry, rx
                nx, ny = rx + pivot_pt[0], ry + pivot_pt[1]
                if (nx, ny) in head_set or (nx, ny) in tail_set:
                    ok = False
                    break
                tail_set.add((nx, ny))
                new_tail.append((nx, ny))
            if not ok:
                continue
            cand = head + new_tail
            area = _h_bbox_area(cand, seq)
            if area < best_area:
                best_area = area
                best = cand
    return best


def subchain_reflect(
    coords: list[Coord], rng: random.Random, seq: str
) -> list[Coord] | None:
    """Pick a chain segment [i, j] and reflect it about its endpoint axis.

    The first and last residues of the segment stay fixed; intermediate
    residues are reflected perpendicular to the chord. After reflection,
    bond lengths along the segment are preserved (rigid transform), so the
    only failure mode is self-intersection or a broken splice.

    Returns the reflected SAW only if it strictly reduces H-bbox area.
    """
    n = len(coords)
    if n < 6:
        return None
    cur_area = _h_bbox_area(coords, seq)
    # Pick segment length 4..min(n-2, 20)
    L = rng.randint(4, min(n - 2, 20))
    i = rng.randrange(0, n - L)
    j = i + L
    p_i = coords[i]
    p_j = coords[j]
    # Reflect coords[i+1..j-1] across the perpendicular bisector of the chord
    # In 2D lattice, only certain chords (axis-aligned or diagonal of equal
    # |dx|=|dy|) admit lattice-preserving reflection. Use the chord direction.
    dx = p_j[0] - p_i[0]
    dy = p_j[1] - p_i[1]
    # Reflection across the line through p_i with direction (dx, dy):
    # new = p_i + 2 * proj(v - p_i, (dx,dy)) - (v - p_i)
    # Only integer-preserving when (dx*dx + dy*dy) divides 2*(dx*vx + dy*vy)*dx etc.
    # Easiest case: reflect across vertical (dx==0) or horizontal (dy==0) lines
    new_seg: list[Coord] = []
    if dx == 0 and dy != 0:
        # chord is vertical → reflect across the vertical line x = p_i[0]
        for v in coords[i + 1: j]:
            new_seg.append((2 * p_i[0] - v[0], v[1]))
    elif dy == 0 and dx != 0:
        # chord is horizontal → reflect across horizontal line y = p_i[1]
        for v in coords[i + 1: j]:
            new_seg.append((v[0], 2 * p_i[1] - v[1]))
    else:
        # Diagonal chord: skip (would not stay on lattice for arbitrary L)
        return None

    candidate = coords[: i + 1] + new_seg + coords[j:]
    if len(set(candidate)) != len(candidate):
        return None
    if not _is_valid_saw(candidate):
        return None
    new_area = _h_bbox_area(candidate, seq)
    if new_area >= cur_area:
        return None
    return candidate


def bond_rebridge(
    coords: list[Coord], rng: random.Random
) -> list[Coord] | None:
    """Wüst-Landau-style internal bond-rebridging by segment reversal.

    Finds chain indices k, j (k+1 < j) such that:
      coords[k]   is lattice-adjacent to coords[j], AND
      coords[k+1] is lattice-adjacent to coords[j+1].
    Reversing coords[k+1..j] then yields a valid SAW with two NEW chain
    bonds (k→old-j and old-k+1→j+1) and the same number of residues. This
    can drastically rewire chain topology in a single move; it is the
    canonical Wang-Landau move for HP lattice folding.

    Returns the rewired SAW, or None if no valid (k, j) pair exists.
    """
    n = len(coords)
    if n < 5:
        return None
    pos: dict[Coord, int] = {tuple(c): i for i, c in enumerate(coords)}
    candidates: list[tuple[int, int]] = []
    for k in range(n - 3):
        ax, ay = coords[k]
        bx, by = coords[k + 1]
        # j must satisfy: coords[j] adj coords[k] AND coords[j+1] adj coords[k+1]
        for dx, dy in _NEIGHBOURS:
            nb = (ax + dx, ay + dy)
            j = pos.get(nb)
            if j is None or j <= k + 1 or j >= n - 1:
                continue
            cj1x, cj1y = coords[j + 1]
            if abs(bx - cj1x) + abs(by - cj1y) == 1:
                candidates.append((k, j))
    if not candidates:
        return None
    k, j = candidates[rng.randrange(len(candidates))]
    # Reverse inclusive segment [k+1 .. j]
    new_coords = coords[: k + 1] + coords[k + 1: j + 1][::-1] + coords[j + 1:]
    # Validate (positions are a permutation, bond lengths preserved by reversal
    # except at the two splice points, which we already verified)
    if len(set(new_coords)) != n:
        return None
    return new_coords


def end_rebridge(
    coords: list[Coord], rng: random.Random
) -> list[Coord] | None:
    """Prefix or suffix bond-rebridging via reversal.

    Picks k such that coords[0] is lattice-adjacent to coords[k+1]; reversing
    coords[0..k] yields a SAW with a new bond (old-coords[0] → coords[k+1]).
    Symmetric variant works on the suffix.
    """
    n = len(coords)
    if n < 4:
        return None
    pos: dict[Coord, int] = {tuple(c): i for i, c in enumerate(coords)}
    if rng.random() < 0.5:
        # Prefix reversal: coords[0..k], new bond is coords[0] -> coords[k+1]
        head = coords[0]
        candidates = []
        for dx, dy in _NEIGHBOURS:
            nb = (head[0] + dx, head[1] + dy)
            kp1 = pos.get(nb)
            if kp1 is None or kp1 < 2:
                continue
            candidates.append(kp1 - 1)  # this is k; reverse coords[0..k]
        if not candidates:
            return None
        k = candidates[rng.randrange(len(candidates))]
        new_coords = coords[: k + 1][::-1] + coords[k + 1:]
    else:
        # Suffix reversal: coords[k..n-1], new bond is coords[k-1] -> old-coords[n-1]
        tail = coords[-1]
        candidates = []
        for dx, dy in _NEIGHBOURS:
            nb = (tail[0] + dx, tail[1] + dy)
            km1 = pos.get(nb)
            if km1 is None or km1 >= n - 2:
                continue
            candidates.append(km1 + 1)
        if not candidates:
            return None
        k = candidates[rng.randrange(len(candidates))]
        new_coords = coords[: k] + coords[k:][::-1]
    if len(set(new_coords)) != n:
        return None
    return new_coords


def lmw_pull(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Lesh-Mitzenmacher-Whitesides pull move (clean implementation, 2003).

    The "pull" move:
      - Pick interior residue i and direction d ∈ {+1, -1}.
      - Identify a diagonal lattice cell c that is adjacent to coords[i+d]
        and unoccupied (or contains coords[i-d], which we'll then move).
      - Identify cell l adjacent to c AND adjacent to coords[i-2d]
        (the chain neighbor of i-d that stays put).
      - Move coords[i] → c, coords[i-d] → l.
      - Cascade: each residue further from i (in direction -d) takes the
        OLD position of its successor, until adjacency is satisfied.
    Returns the resulting valid SAW or None.
    """
    n = len(coords)
    if n < 4:
        return None
    occupied = set(coords)

    # Choose i and d randomly; try a few combinations to escape sparse layouts
    tried = set()
    for _ in range(min(n * 2, 32)):
        i = rng.randrange(1, n - 1)
        d = rng.choice((-1, +1))
        if (i, d) in tried:
            continue
        tried.add((i, d))
        if i + d < 0 or i + d >= n or i - d < 0 or i - d >= n:
            continue

        pos_i = coords[i]
        pos_next = coords[i + d]
        pos_prev = coords[i - d]

        # Diagonal cells adjacent to pos_next: rotate (pos_i→pos_next) by ±90°
        bx, by = pos_next[0] - pos_i[0], pos_next[1] - pos_i[1]
        c_candidates = [
            (pos_next[0] + by, pos_next[1] - bx),
            (pos_next[0] - by, pos_next[1] + bx),
        ]
        rng.shuffle(c_candidates)

        for c in c_candidates:
            if c == pos_i:
                continue
            if c in occupied and c != pos_prev:
                continue
            # Find l: must be adjacent to c, adjacent to coords[i-2d] if exists.
            if 0 <= i - 2 * d < n:
                anchor_far = coords[i - 2 * d]
            else:
                anchor_far = None

            l_candidates = []
            for dx, dy in _NEIGHBOURS:
                l = (c[0] + dx, c[1] + dy)
                if l == c or l == pos_next:
                    continue
                if l in occupied and l != pos_i:
                    continue
                if anchor_far is not None:
                    if abs(l[0] - anchor_far[0]) + abs(l[1] - anchor_far[1]) != 1:
                        continue
                l_candidates.append(l)
            if not l_candidates:
                continue
            l = rng.choice(l_candidates)

            new_coords = list(coords)
            new_coords[i] = c
            new_coords[i - d] = l

            # Cascade: residues at i-2d, i-3d, ... need to shift if not yet adjacent
            cascade_idx = i - 2 * d
            old_pos = pos_prev
            cascade_ok = True
            while 0 <= cascade_idx < n:
                # Check adjacency to chain neighbor in direction d
                neighbor_pos = new_coords[cascade_idx + d]
                cur_pos = new_coords[cascade_idx]
                if abs(cur_pos[0] - neighbor_pos[0]) + abs(cur_pos[1] - neighbor_pos[1]) == 1:
                    break
                next_old = coords[cascade_idx]
                new_coords[cascade_idx] = old_pos
                old_pos = next_old
                cascade_idx -= d
            else:
                # Reached end of chain without satisfying adjacency? OK if we exhausted
                pass

            if _is_valid_saw(new_coords):
                return new_coords
    return None


def lmw_pull_full(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Full Lesh-Mitzenmacher-Whitesides 2003 pull move WITH cascade.

    Differs from `lmw_pull` by lifting the constraint that L must be adjacent
    to s_{i-2d}. When that adjacency does not hold, the move triggers a
    cascade: residues s_{i-2d}, s_{i-3d}, ... each take the OLD position of
    their successor along direction +d, until chain adjacency is restored or
    we run off the chain end. This is the canonical LMW move; the restricted
    `lmw_pull` above is a strict subset that excludes all cascade-requiring
    pulls.
    """
    n = len(coords)
    if n < 4:
        return None
    occupied = set(coords)

    tried = set()
    for _ in range(min(n * 2, 32)):
        i = rng.randrange(1, n - 1)
        d = rng.choice((-1, +1))
        if (i, d) in tried:
            continue
        tried.add((i, d))
        if i + d < 0 or i + d >= n or i - d < 0 or i - d >= n:
            continue

        pos_i = coords[i]
        pos_next = coords[i + d]
        pos_prev = coords[i - d]

        bx, by = pos_next[0] - pos_i[0], pos_next[1] - pos_i[1]
        c_candidates = [
            (pos_next[0] + by, pos_next[1] - bx),
            (pos_next[0] - by, pos_next[1] + bx),
        ]
        rng.shuffle(c_candidates)

        for c in c_candidates:
            if c == pos_i:
                continue
            if c in occupied and c != pos_prev:
                continue

            l_candidates = []
            for dx, dy in _NEIGHBOURS:
                l = (c[0] + dx, c[1] + dy)
                if l == c or l == pos_next:
                    continue
                if l in occupied and l != pos_i:
                    continue
                l_candidates.append(l)
            if not l_candidates:
                continue
            rng.shuffle(l_candidates)

            for l in l_candidates:
                new_coords = list(coords)
                new_coords[i] = c
                new_coords[i - d] = l

                # Cascade: residues i-2d, i-3d, ... each take the OLD position
                # of their +d-successor until chain adjacency is restored.
                cascade_ok = True
                cascade_idx = i - 2 * d
                # The "old" position about to be taken next is pos_prev (old s_{i-d}).
                old_successor_pos = pos_prev
                while 0 <= cascade_idx < n:
                    cur_pos = new_coords[cascade_idx]
                    neighbor_pos = new_coords[cascade_idx + d]
                    if _manhattan(cur_pos, neighbor_pos) == 1:
                        break
                    # Pull cascade_idx into old_successor_pos, but not if it
                    # collides with another already-placed residue.
                    if old_successor_pos in (
                        new_coords[j] for j in range(n) if j != cascade_idx
                    ):
                        cascade_ok = False
                        break
                    next_old = coords[cascade_idx]
                    new_coords[cascade_idx] = old_successor_pos
                    old_successor_pos = next_old
                    cascade_idx -= d

                if cascade_ok and _is_valid_saw(new_coords):
                    return new_coords
    return None


def fragment_random_walk(
    coords: list[Coord], rng: random.Random, max_attempts: int = 12,
) -> list[Coord] | None:
    """Davis 2016 Fragment Random Walk (FRW) move.

    1. Pick a random monomer m ∈ {2, N-1} (1-indexed) → 0-indexed [1, N-2].
    2. With 50% probability, regrow monomers at indices > m, else < m.
    3. Run a self-avoiding random walk for the regrown fragment from m
       outward, with no endpoint constraint at the far end.
    4. If a step has no free lattice neighbor, abort and try another seed
       up to ``max_attempts`` times.

    Returns the new SAW or None if all attempts fail.

    This is the canonical "non-local" move from Davis 2016 thesis §3.5.1
    that we previously lacked. It allows entire chain halves to take a
    completely different path, far more disruptive than local_rebridge
    which has both endpoints fixed.
    """
    n = len(coords)
    if n < 4:
        return None

    occupied = set(coords)

    for _ in range(max_attempts):
        # Pick a random pivot monomer (interior)
        m = rng.randrange(1, n - 1)
        # Decide which side to regrow
        regrow_high = rng.random() < 0.5

        if regrow_high:
            # Regrow indices m+1, m+2, ..., n-1 starting from coords[m]
            new_coords = list(coords[: m + 1])
            kept = set(new_coords)
            current = coords[m]
            ok = True
            for i in range(m + 1, n):
                # Pick a random free lattice neighbor of current
                nbrs = [(current[0] + dx, current[1] + dy)
                        for dx, dy in _NEIGHBOURS]
                rng.shuffle(nbrs)
                placed = False
                for nb in nbrs:
                    if nb not in kept:
                        new_coords.append(nb)
                        kept.add(nb)
                        current = nb
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok and len(new_coords) == n:
                return new_coords
        else:
            # Regrow indices 0, 1, ..., m-1 from coords[m] outward
            # We build the regrown prefix in reverse (from m-1 down to 0),
            # then reverse it to get correct chain order.
            new_prefix = []  # will be reversed at end
            kept = set(coords[m:])
            current = coords[m]
            ok = True
            for i in range(m - 1, -1, -1):
                nbrs = [(current[0] + dx, current[1] + dy)
                        for dx, dy in _NEIGHBOURS]
                rng.shuffle(nbrs)
                placed = False
                for nb in nbrs:
                    if nb not in kept:
                        new_prefix.append(nb)
                        kept.add(nb)
                        current = nb
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if ok and len(new_prefix) == m:
                new_coords = list(reversed(new_prefix)) + list(coords[m:])
                if len(new_coords) == n:
                    return new_coords
    return None


def random_move(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
    """Propose a random move by sampling from the implemented move set.

    Weights: 20% end_flip, 30% pivot, 50% pull_move. These weights are a
    reasonable default; solvers are free to tune them.
    """
    u = rng.random()
    if u < 0.20:
        return end_flip(coords, rng)
    if u < 0.50:
        return pivot(coords, rng)
    return pull_move(coords, rng)


def make_move_fn(
    end_flip_weight: float = 0.20,
    pivot_weight: float = 0.30,
    pull_weight: float = 0.50,
):
    """Return a move proposer that samples end_flip / pivot / pull_move with
    the given probabilities. Weights must be non-negative and sum to 1."""
    total = end_flip_weight + pivot_weight + pull_weight
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"move weights must sum to 1 (got {total})")
    if min(end_flip_weight, pivot_weight, pull_weight) < 0:
        raise ValueError("move weights must be non-negative")

    c1 = end_flip_weight
    c2 = end_flip_weight + pivot_weight

    def _move(coords: list[Coord], rng: random.Random) -> list[Coord] | None:
        u = rng.random()
        if u < c1:
            return end_flip(coords, rng)
        if u < c2:
            return pivot(coords, rng)
        return pull_move(coords, rng)

    return _move


MOVE_STRATEGIES: dict[str, tuple[float, float, float]] = {
    "mixed": (0.20, 0.30, 0.50),
    "pull_only": (0.0, 0.0, 1.0),
    "pull_heavy": (0.05, 0.05, 0.90),
    "pivot_heavy": (0.10, 0.60, 0.30),
    "pivot_only": (0.0, 1.0, 0.0),
    "pivot_dominant": (0.05, 0.80, 0.15),
    "balanced": (0.33, 0.34, 0.33),
    "no_pull": (0.40, 0.60, 0.0),
}
