"""Gap-targeted move-sequence polish (Rubik's-cube-style algorithm search).

Given a reference fold, identify "near-contact" H pairs (manhattan distance 2)
and search for short move sequences that turn them into actual contacts WHILE
preserving every existing H-H contact. This is the analog of cube speedsolving:
recognize a pattern, apply a memorized algorithm that fixes it without
disturbing already-solved layers.

Strategy:
  - BFS over move sequences (corner_flip, crankshaft, pivot, end_flip).
  - At each frontier state, check if any near-contact pair has closed AND no
    existing contact has been lost.
  - Bounded depth (default 5) and bounded breadth (sample top-K candidate
    moves) to keep cost tractable.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from hp_folding.moves import (
    corner_flip,
    crankshaft,
    pivot,
    end_flip,
)
from hp_folding.verifier import count_contacts

Coord = tuple[int, int]
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _h_contact_set(sequence: str, coords: list[Coord]) -> frozenset[tuple[int, int]]:
    """Set of (i, j) pairs (i < j, |i-j|>1) that are H-H lattice-adjacent."""
    pos = {c: i for i, c in enumerate(coords)}
    pairs: set[tuple[int, int]] = set()
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue
        for dx, dy in _NEIGHBOURS:
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1:
                continue
            if sequence[j] != "H":
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return frozenset(pairs)


def _near_contact_h_pairs(
    sequence: str, coords: list[Coord]
) -> list[tuple[int, int]]:
    """H-H pairs at manhattan distance 2 (one move from being a contact)."""
    out: list[tuple[int, int]] = []
    for i in range(len(sequence)):
        if sequence[i] != "H":
            continue
        for j in range(i + 2, len(sequence)):
            if sequence[j] != "H":
                continue
            if _manhattan(coords[i], coords[j]) == 2:
                out.append((i, j))
    return out


@dataclass
class GapPolishResult:
    coords: list[Coord]
    contacts: int
    moves_applied: int
    closed_pairs: list[tuple[int, int]]


def _enumerate_corner_flips(coords: list[Coord]) -> list[list[Coord]]:
    """All valid corner flips."""
    out = []
    n = len(coords)
    occupied = set(coords)
    for i in range(1, n - 1):
        prev = coords[i - 1]
        nxt = coords[i + 1]
        dx = nxt[0] - prev[0]
        dy = nxt[1] - prev[1]
        if abs(dx) + abs(dy) != 2 or dx == 0 or dy == 0:
            continue
        cur = coords[i]
        cand1 = (prev[0] + dx, prev[1])
        cand2 = (prev[0], prev[1] + dy)
        target = cand2 if cur == cand1 else cand1
        if target in occupied and target != cur:
            continue
        new_coords = list(coords)
        new_coords[i] = target
        out.append(new_coords)
    return out


def _enumerate_crankshafts(coords: list[Coord]) -> list[list[Coord]]:
    """All valid crankshafts."""
    out = []
    n = len(coords)
    occupied = set(coords)
    for i in range(1, n - 2):
        a = coords[i - 1]
        d = coords[i + 2]
        if _manhattan(a, d) != 1:
            continue
        b = coords[i]
        c = coords[i + 1]
        dy = d[1] - a[1]
        if dy == 0:
            new_b = (b[0], 2 * a[1] - b[1])
            new_c = (c[0], 2 * a[1] - c[1])
        else:
            new_b = (2 * a[0] - b[0], b[1])
            new_c = (2 * a[0] - c[0], c[1])
        if new_b == b or new_c == c or new_b == new_c:
            continue
        if new_b in occupied and new_b not in (b, c):
            continue
        if new_c in occupied and new_c not in (b, c):
            continue
        new_coords = list(coords)
        new_coords[i] = new_b
        new_coords[i + 1] = new_c
        # Validate SAW
        if len(set(new_coords)) != n:
            continue
        out.append(new_coords)
    return out


def _enumerate_pivots(coords: list[Coord]) -> list[list[Coord]]:
    """All valid 90/180/270 pivot rotations."""
    out = []
    n = len(coords)
    for k in range(1, n - 1):
        cx, cy = coords[k]
        for angle in (90, 180, 270):
            new_coords = list(coords[: k + 1])
            for x, y in coords[k + 1:]:
                rx, ry = x - cx, y - cy
                if angle == 90:
                    rx, ry = -ry, rx
                elif angle == 180:
                    rx, ry = -rx, -ry
                else:
                    rx, ry = ry, -rx
                new_coords.append((cx + rx, cy + ry))
            if len(set(new_coords)) != n:
                continue
            out.append(new_coords)
    return out


def _candidate_moves(
    coords: list[Coord], rng: random.Random, sample_k: int = 8
) -> list[list[Coord]]:
    """Exhaustively enumerate valid candidate moves from this state."""
    out: list[list[Coord]] = []
    out.extend(_enumerate_corner_flips(coords))
    out.extend(_enumerate_crankshafts(coords))
    out.extend(_enumerate_pivots(coords))
    # Shuffle deterministically by rng
    rng.shuffle(out)
    return out


def gap_polish_pass(
    sequence: str,
    coords: list[Coord],
    *,
    max_depth: int = 5,
    breadth: int = 8,
    seed: int = 0,
) -> GapPolishResult:
    """One pass of gap-targeted polish. For each near-contact H pair, search
    for a move sequence that closes it without losing existing contacts. If
    found, accept and continue.

    Returns the polished fold + how many extra contacts were added.
    """
    rng = random.Random(seed)
    cur = list(coords)
    cur_contacts = count_contacts(sequence, cur)
    cur_set = _h_contact_set(sequence, cur)
    closed_pairs: list[tuple[int, int]] = []
    total_moves = 0

    progress = True
    while progress:
        progress = False
        # Relaxed search: BFS over move sequences from current fold,
        # accept any sequence where final contacts > current.
        improvement = _bfs_improve_contacts(
            sequence, cur, cur_contacts,
            max_depth=max_depth, breadth=breadth, rng=rng,
        )
        if improvement is not None:
            new_coords, new_contacts, depth = improvement
            cur = list(new_coords)
            cur_contacts = new_contacts
            total_moves += depth
            progress = True
    return GapPolishResult(
        coords=cur,
        contacts=cur_contacts,
        moves_applied=total_moves,
        closed_pairs=closed_pairs,
    )


def _bfs_improve_contacts(
    sequence: str,
    start_coords: list[Coord],
    start_contacts: int,
    *,
    max_depth: int,
    breadth: int,
    rng: random.Random,
) -> tuple[list[Coord], int, int] | None:
    """BFS over move sequences. Find shortest sequence that improves contacts.
    Returns (final_coords, final_contacts, depth_used) or None.
    """
    seen: set[tuple[Coord, ...]] = {tuple(start_coords)}
    # Each frontier: (coords, depth)
    frontier: deque = deque()
    frontier.append((start_coords, 0))

    best_found: tuple[list[Coord], int, int] | None = None

    while frontier:
        cur, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        cands = _candidate_moves(cur, rng, sample_k=breadth)
        # Only explore breadth candidates
        for cand in cands[:breadth]:
            tup = tuple(cand)
            if tup in seen:
                continue
            seen.add(tup)
            cand_contacts = count_contacts(sequence, cand)
            if cand_contacts > start_contacts:
                # Found improvement at this depth; return it (BFS gives shortest)
                if best_found is None or cand_contacts > best_found[1]:
                    best_found = (cand, cand_contacts, depth + 1)
                # Don't return immediately — let BFS finish current depth to find
                # the BEST improvement at this depth.
            frontier.append((cand, depth + 1))
        # If we have a result and frontier is moving past current depth, return.
        if best_found is not None and (not frontier or frontier[0][1] > depth):
            return best_found
    return best_found


def _bfs_close_pair(
    sequence: str,
    start_coords: list[Coord],
    target_i: int,
    target_j: int,
    must_preserve: frozenset[tuple[int, int]],
    *,
    max_depth: int,
    breadth: int,
    rng: random.Random,
) -> list[list[Coord]] | None:
    """BFS for a move sequence (length ≤ max_depth) that brings target_i and
    target_j to lattice distance 1 while preserving every contact in
    must_preserve. Returns the sequence of intermediate coords (last is the
    successful state), or None."""
    start_tup = tuple(start_coords)
    seen: set[tuple[Coord, ...]] = {start_tup}
    # Each frontier entry: (coords, sequence_of_states_so_far)
    frontier: deque[tuple[list[Coord], list[list[Coord]]]] = deque()
    frontier.append((start_coords, []))

    while frontier:
        cur, hist = frontier.popleft()
        if len(hist) >= max_depth:
            continue
        # Generate candidate moves
        for cand in _candidate_moves(cur, rng, sample_k=breadth):
            tup = tuple(cand)
            if tup in seen:
                continue
            seen.add(tup)
            # Check preservation
            cand_set = _h_contact_set(sequence, cand)
            if not must_preserve.issubset(cand_set):
                continue
            # Check goal
            if _manhattan(cand[target_i], cand[target_j]) == 1:
                # Must also be lattice-adjacent and form a contact (which it
                # will since both are H).
                return hist + [cand]
            new_hist = hist + [cand]
            if len(new_hist) < max_depth:
                frontier.append((cand, new_hist))
    return None
