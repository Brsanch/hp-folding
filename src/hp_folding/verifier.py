"""Verifier for HP-lattice folds on the 2D square lattice.

Implements the validation and scoring rules specified by the CAISc 2026
HP Protein Folding problem page. Given a sequence and integer lattice
coordinates, determines whether the fold is valid and computes its score.

Validity (all must hold):
  1. coords length equals sequence length
  2. all coordinates are unique (self-avoidance)
  3. consecutive residues are lattice-adjacent (Manhattan distance == 1)

Score (for valid folds only):
  - contacts: number of non-sequential H-H pairs at lattice distance 1
  - radius_of_gyration_squared: mean squared distance from the centroid
    (used as compactness tiebreaker when contacts tie)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Coord = tuple[int, int]


@dataclass(frozen=True)
class Verdict:
    """Result of verifying a fold."""

    valid: bool
    contacts: int
    radius_of_gyration_squared: float
    error: str | None = None

    def __bool__(self) -> bool:
        return self.valid


def verify(sequence: str, coords: Sequence[Coord]) -> Verdict:
    """Verify a fold and compute its contact count and compactness.

    Parameters
    ----------
    sequence
        HP sequence, a string of 'H' and 'P' characters (case-insensitive).
    coords
        Integer lattice coordinates for each residue, in sequence order.

    Returns
    -------
    Verdict
        ``valid=True`` with contact count and radius-of-gyration-squared
        if the fold passes all checks, else ``valid=False`` with an
        ``error`` message describing the first failure.
    """
    seq = sequence.strip().upper()
    if not seq:
        return Verdict(False, 0, 0.0, "sequence is empty")
    if any(ch not in "HP" for ch in seq):
        return Verdict(False, 0, 0.0, "sequence contains characters other than H/P")

    try:
        coords_tup: list[Coord] = [(int(c[0]), int(c[1])) for c in coords]
    except (TypeError, ValueError, IndexError):
        return Verdict(False, 0, 0.0, "each coord must be a pair of integers")

    if len(coords_tup) != len(seq):
        return Verdict(
            False, 0, 0.0,
            f"coords length {len(coords_tup)} does not match "
            f"sequence length {len(seq)}",
        )

    if len(set(coords_tup)) != len(coords_tup):
        return Verdict(
            False, 0, 0.0,
            "coords are not unique (chain crosses itself)",
        )

    for i in range(len(coords_tup) - 1):
        if _manhattan(coords_tup[i], coords_tup[i + 1]) != 1:
            return Verdict(
                False, 0, 0.0,
                f"residues {i} and {i + 1} are not lattice-adjacent "
                f"(distance {_manhattan(coords_tup[i], coords_tup[i + 1])})",
            )

    contacts = count_contacts(seq, coords_tup)
    rg2 = radius_of_gyration_squared(coords_tup)
    return Verdict(True, contacts, rg2)


def count_contacts(sequence: str, coords: Sequence[Coord]) -> int:
    """Count non-sequential H-H contacts at lattice distance 1.

    Assumes the fold is already a valid self-avoiding walk.
    """
    seq = sequence.upper()
    position_to_index = {tuple(c): i for i, c in enumerate(coords)}
    contacts = 0
    for i, ci in enumerate(coords):
        if seq[i] != "H":
            continue
        x, y = ci
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = position_to_index.get((x + dx, y + dy))
            if j is None or j <= i:
                continue
            if j - i > 1 and seq[j] == "H":
                contacts += 1
    return contacts


def count_contacts_delta(
    sequence: str,
    old_coords: Sequence[Coord],
    new_coords: Sequence[Coord],
    current_count: int,
) -> int:
    """Compute the new contact count given the old count and a move that
    changed some residue positions. O(k) where k = number of changed
    residues, vs O(n²) for `count_contacts` from scratch.

    Correctness: identifies the set of changed residues, then for each
    pair (i, j) with i ∈ changed and seq[i] == seq[j] == 'H' and |i-j|>1:
      - was contact at old? (manhattan(old_coords[i], old_coords[j]) == 1)
      - is contact at new? (manhattan(new_coords[i], new_coords[j]) == 1)
    Net delta = #(was→not) - #(not→was), counted once per ordered (i<j) pair.

    For pair (i, j) with both i, j ∈ changed, we consider it once via i.
    For pair (i, j) with only i ∈ changed, we still need to check from i's
    perspective AND we need to compare pair status using BOTH old and new
    positions of the ENDPOINTS (j's position is unchanged, so old_coords[j]
    == new_coords[j]).
    """
    seq = sequence.upper()
    changed = [i for i in range(len(new_coords))
               if tuple(new_coords[i]) != tuple(old_coords[i])]
    if not changed:
        return current_count

    old_pos = {tuple(c): i for i, c in enumerate(old_coords)}
    new_pos = {tuple(c): i for i, c in enumerate(new_coords)}

    delta = 0
    counted: set[tuple[int, int]] = set()
    NB = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    for i in changed:
        if seq[i] != "H":
            continue
        # Pair candidates: H residues j (|i-j|>1) that are H-neighbours of
        # i in OLD or NEW positions. Use both old & new neighborhoods.
        candidates: set[int] = set()
        ox, oy = tuple(old_coords[i])
        for dx, dy in NB:
            j = old_pos.get((ox + dx, oy + dy))
            if j is not None and abs(j - i) > 1 and seq[j] == "H":
                candidates.add(j)
        nx, ny = tuple(new_coords[i])
        for dx, dy in NB:
            j = new_pos.get((nx + dx, ny + dy))
            if j is not None and abs(j - i) > 1 and seq[j] == "H":
                candidates.add(j)

        for j in candidates:
            pair = (i, j) if i < j else (j, i)
            if pair in counted:
                continue
            counted.add(pair)
            old_was = manhattan(tuple(old_coords[i]), tuple(old_coords[j])) == 1
            new_is = manhattan(tuple(new_coords[i]), tuple(new_coords[j])) == 1
            if old_was and not new_is:
                delta -= 1
            elif new_is and not old_was:
                delta += 1

    return current_count + delta


class IncrementalContactCounter:
    """Stateful contact counter that maintains the (coords, pos_dict, count)
    invariant across moves. Each call to `apply` is O(k) where k is the
    number of changed residues — typically 1-2 for corner_flip/crankshaft.
    """

    __slots__ = ("sequence", "coords", "pos", "count")

    NB = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, sequence: str, coords: Sequence[Coord]):
        self.sequence = sequence.upper()
        self.coords = [tuple(c) for c in coords]
        self.pos: dict[tuple[int, int], int] = {
            tuple(c): i for i, c in enumerate(coords)
        }
        # Initial full count
        c = 0
        seq = self.sequence
        for i, ci in enumerate(self.coords):
            if seq[i] != "H":
                continue
            x, y = ci
            for dx, dy in self.NB:
                j = self.pos.get((x + dx, y + dy))
                if j is None or j <= i:
                    continue
                if j - i > 1 and seq[j] == "H":
                    c += 1
        self.count = c

    def apply(self, new_coords: Sequence[Coord]) -> int:
        """Update internal state to reflect new_coords. Return new count.
        Assumes new_coords is a valid SAW (caller's responsibility).
        """
        seq = self.sequence
        old_coords = self.coords
        # Identify changed indices
        changed: list[int] = []
        for i in range(len(new_coords)):
            if tuple(new_coords[i]) != old_coords[i]:
                changed.append(i)
        if not changed:
            return self.count

        # Phase 1: collect contact pairs that involve any changed residue.
        # Pair (i, j) involves a changed residue if i ∈ changed OR j is at
        # a position that was a neighbour of a changed residue's old/new pos.
        # We directly enumerate H-neighbours of each changed residue at
        # OLD positions (lost contacts) and NEW positions (gained contacts).
        delta = 0
        counted: set[tuple[int, int]] = set()

        # Phase 1a: subtract old contacts that involve any changed residue.
        for i in changed:
            if seq[i] != "H":
                continue
            ox, oy = old_coords[i]
            for dx, dy in self.NB:
                j = self.pos.get((ox + dx, oy + dy))
                if j is None or abs(j - i) <= 1 or seq[j] != "H":
                    continue
                key = (i, j) if i < j else (j, i)
                if key in counted:
                    continue
                counted.add(key)
                # Was a contact (since both occupy adjacent old positions, and
                # both are H, and chain-distance > 1).
                # Will it still be a contact at new positions?
                ni = tuple(new_coords[i])
                nj = tuple(new_coords[j])
                if abs(ni[0] - nj[0]) + abs(ni[1] - nj[1]) != 1:
                    delta -= 1

        # Phase 1b: add new contacts that involve any changed residue.
        # Update pos dict incrementally first so neighbour lookups use new state.
        for i in changed:
            del self.pos[old_coords[i]]
        for i in changed:
            self.pos[tuple(new_coords[i])] = i

        # Now scan new neighbourhoods of changed residues.
        new_counted: set[tuple[int, int]] = set()
        for i in changed:
            if seq[i] != "H":
                continue
            nx, ny = new_coords[i]
            for dx, dy in self.NB:
                j = self.pos.get((nx + dx, ny + dy))
                if j is None or abs(j - i) <= 1 or seq[j] != "H":
                    continue
                key = (i, j) if i < j else (j, i)
                if key in new_counted:
                    continue
                new_counted.add(key)
                # Check if it WAS a contact at old positions
                oi = old_coords[i]
                oj = old_coords[j] if j not in changed else old_coords[j]
                if abs(oi[0] - oj[0]) + abs(oi[1] - oj[1]) != 1:
                    delta += 1

        self.coords = [tuple(c) for c in new_coords]
        self.count += delta
        return self.count


def radius_of_gyration_squared(coords: Sequence[Coord]) -> float:
    """Mean squared distance from the centroid (compactness tiebreaker)."""
    n = len(coords)
    if n == 0:
        return 0.0
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    return sum((c[0] - cx) ** 2 + (c[1] - cy) ** 2 for c in coords) / n


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
