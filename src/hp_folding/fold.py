"""Fold data structure: an HP sequence placed on the 2D lattice."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from hp_folding.verifier import Verdict, verify

Coord = tuple[int, int]


@dataclass
class Fold:
    """A placement of an HP sequence on the 2D square lattice.

    Attributes
    ----------
    sequence
        HP string being folded.
    coords
        Lattice coordinates for each residue, in sequence order.
    """

    sequence: str
    coords: list[Coord] = field(default_factory=list)

    @classmethod
    def linear(cls, sequence: str) -> Fold:
        """Return a fully-extended straight-line fold (zero contacts)."""
        return cls(sequence=sequence, coords=[(i, 0) for i in range(len(sequence))])

    @classmethod
    def random_saw(
        cls, sequence: str, rng, max_attempts: int = 500
    ) -> Fold:
        """Return a random self-avoiding walk by greedy stepwise growth.

        Restarts from scratch if a growth attempt paints itself into a corner.
        Falls back to a straight-line fold if no attempt succeeds within
        ``max_attempts``.
        """
        n = len(sequence)
        for _ in range(max_attempts):
            coords: list[tuple[int, int]] = [(0, 0)]
            occupied = {(0, 0)}
            for _ in range(n - 1):
                x, y = coords[-1]
                options = [
                    (x + dx, y + dy)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (x + dx, y + dy) not in occupied
                ]
                if not options:
                    coords = None  # type: ignore[assignment]
                    break
                nxt = rng.choice(options)
                coords.append(nxt)
                occupied.add(nxt)
            if coords is not None and len(coords) == n:
                return cls(sequence=sequence, coords=coords)
        return cls.linear(sequence)

    def verify(self) -> Verdict:
        """Run the CAISc verifier on this fold."""
        return verify(self.sequence, self.coords)

    def contacts(self) -> int:
        """Return the number of H-H contacts, or raise if the fold is invalid."""
        v = self.verify()
        if not v.valid:
            raise ValueError(f"invalid fold: {v.error}")
        return v.contacts

    def occupied(self) -> set[Coord]:
        """Return the set of occupied lattice positions."""
        return set(self.coords)

    def copy(self) -> Fold:
        return Fold(sequence=self.sequence, coords=list(self.coords))

    def __iter__(self) -> Iterator[Coord]:
        return iter(self.coords)

    def __len__(self) -> int:
        return len(self.coords)
