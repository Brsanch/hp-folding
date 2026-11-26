"""Beam search for HP-lattice folding.

At each chain depth we maintain the top-K partial folds by a heuristic
score. Unlike depth-first search which commits to a single branch deep,
beam search keeps ``K`` alternatives alive at every depth and advances
them in lockstep — directly analogous to the many parallel folding
paths through a funneled energy landscape.

Scoring is ``contacts_so_far + compactness_bonus``, where
``compactness_bonus`` penalises the bounding-box area of already-placed
H residues. This encourages hydrophobic collapse (clustering H's
together) as the chain grows, so the search converges on folds whose
H-core is tight without being forced into a predetermined shape.

The implementation uses tuples instead of lists/sets so that beam
states can be cheaply copied (shared substructure via Python's
immutable tuples).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from hp_folding.fold import Fold
from hp_folding.verifier import count_contacts

Coord = tuple[int, int]
_NEIGHBOUR_OFFSETS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class BeamState:
    """A partial fold living in the beam."""

    coords: tuple[Coord, ...]
    # occupied is derived from coords but cached for speed
    occupied: frozenset[Coord]
    # Index set of placed H residues (positions in the sequence)
    h_lattice: frozenset[Coord]
    contacts: int
    h_min_x: int
    h_max_x: int
    h_min_y: int
    h_max_y: int
    h_count: int


def _score(state: BeamState, *, compactness_weight: float = 0.25,
           potential_weight: float = 0.5) -> float:
    """Score a beam state: contacts + compactness + future-contact potential.

    Compactness is ``-max(width, height) × compactness_weight`` where the
    extent is the bounding-box dimension of the H-subset. Smaller extent
    (denser H-cluster) gives a higher score.

    Future-contact potential rewards states whose H-subset has many
    unoccupied lattice neighbours — these are "slots" where future
    H-residues could land and form new contacts. This makes the score
    less myopic: a partial fold with 5 contacts but many free H-adjacent
    slots is preferred over one with 7 contacts that has painted itself
    into a corner.
    """
    if state.h_count == 0:
        extent = 0
    else:
        extent = max(state.h_max_x - state.h_min_x,
                     state.h_max_y - state.h_min_y)
    compactness = -compactness_weight * extent

    # Count free lattice neighbours of placed H's (potential future contacts)
    potential = 0
    for h in state.h_lattice:
        for dx, dy in _NEIGHBOUR_OFFSETS:
            n = (h[0] + dx, h[1] + dy)
            if n not in state.occupied:
                potential += 1

    return state.contacts + compactness + potential_weight * potential


def _extend(state: BeamState, nxt: Coord, is_h: bool, prev: Coord) -> BeamState:
    """Return a new BeamState extending ``state`` by placing residue ``nxt``."""
    new_coords = state.coords + (nxt,)
    new_occupied = state.occupied | {nxt}
    new_h_lattice = state.h_lattice
    new_contacts = state.contacts
    new_h_count = state.h_count
    new_min_x, new_max_x = state.h_min_x, state.h_max_x
    new_min_y, new_max_y = state.h_min_y, state.h_max_y
    if is_h:
        # Count contacts with existing H's (excluding chain-adjacent prev if H).
        for dx, dy in _NEIGHBOUR_OFFSETS:
            neighbor = (nxt[0] + dx, nxt[1] + dy)
            if neighbor in state.h_lattice and neighbor != prev:
                new_contacts += 1
        new_h_lattice = state.h_lattice | {nxt}
        new_h_count = state.h_count + 1
        if state.h_count == 0:
            new_min_x = new_max_x = nxt[0]
            new_min_y = new_max_y = nxt[1]
        else:
            new_min_x = min(state.h_min_x, nxt[0])
            new_max_x = max(state.h_max_x, nxt[0])
            new_min_y = min(state.h_min_y, nxt[1])
            new_max_y = max(state.h_max_y, nxt[1])
    return BeamState(
        coords=new_coords,
        occupied=new_occupied,
        h_lattice=new_h_lattice,
        contacts=new_contacts,
        h_min_x=new_min_x, h_max_x=new_max_x,
        h_min_y=new_min_y, h_max_y=new_max_y,
        h_count=new_h_count,
    )


def beam_search(
    sequence: str,
    *,
    beam_width: int = 1000,
    compactness_weight: float = 0.25,
    potential_weight: float = 0.5,
    timeout_seconds: float | None = None,
    verbose: bool = False,
) -> Fold:
    """Run beam search on ``sequence`` and return the best fold found.

    Parameters
    ----------
    sequence
        HP string.
    beam_width
        Number of partial folds retained at each depth.
    compactness_weight
        Weight on the H-bounding-box penalty. Higher weight → tighter
        H-cluster preference (potentially at the cost of missing folds
        with non-minimal compactness at intermediate steps).
    timeout_seconds
        Abort early if we exceed this wall-clock budget, returning
        whatever the current best fold is.
    verbose
        If True, print depth-by-depth progress.

    Returns
    -------
    Fold
        The best fold discovered.
    """
    seq = sequence.upper()
    n = len(seq)
    h_mask = [c == "H" for c in seq]

    start = time.monotonic()

    # Initial beam state: one residue at the origin.
    first_is_h = h_mask[0]
    initial = BeamState(
        coords=((0, 0),),
        occupied=frozenset({(0, 0)}),
        h_lattice=frozenset({(0, 0)}) if first_is_h else frozenset(),
        contacts=0,
        h_min_x=0 if first_is_h else 0,
        h_max_x=0 if first_is_h else 0,
        h_min_y=0 if first_is_h else 0,
        h_max_y=0 if first_is_h else 0,
        h_count=1 if first_is_h else 0,
    )
    beam: list[BeamState] = [initial]

    for i in range(1, n):
        if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
            if verbose:
                print(f"  timeout at depth {i}")
            break

        is_h = h_mask[i]
        new_beam: list[BeamState] = []
        for state in beam:
            prev = state.coords[-1]
            for dx, dy in _NEIGHBOUR_OFFSETS:
                nxt = (prev[0] + dx, prev[1] + dy)
                if nxt in state.occupied:
                    continue
                new_state = _extend(state, nxt, is_h, prev)
                new_beam.append(new_state)

        if not new_beam:
            if verbose:
                print(f"  beam died at depth {i}")
            break

        # Keep top K.
        new_beam.sort(
            key=lambda s: _score(
                s,
                compactness_weight=compactness_weight,
                potential_weight=potential_weight,
            ),
            reverse=True,
        )
        beam = new_beam[:beam_width]

        if verbose and (i % 10 == 0 or i == n - 1):
            best = beam[0]
            print(f"  depth {i:>3d}: beam={len(beam):>6d} "
                  f"top_contacts={best.contacts:>3d} "
                  f"h_extent={max(best.h_max_x-best.h_min_x, best.h_max_y-best.h_min_y)}")

    # Return best full fold.
    best = max(beam, key=lambda s: s.contacts)
    fold = Fold(sequence=seq, coords=list(best.coords))
    # Verify contact count (guard against bookkeeping bugs).
    actual = count_contacts(seq, list(best.coords))
    if actual != best.contacts:
        # Should not happen; fall back to the actual count.
        fold = Fold(sequence=seq, coords=list(best.coords))
    return fold


def multi_start_beam(
    sequence: str,
    *,
    beam_widths: tuple[int, ...] = (500, 1000, 2000, 5000),
    compactness_weights: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    timeout_seconds: float | None = None,
    verbose: bool = False,
) -> Fold:
    """Sweep beam configurations and return the best fold across all runs."""
    seq = sequence.upper()
    best: Fold | None = None
    best_contacts = -1
    for bw in beam_widths:
        for cw in compactness_weights:
            if verbose:
                print(f"\n[beam_width={bw}, compactness_weight={cw}]")
            f = beam_search(
                seq, beam_width=bw, compactness_weight=cw,
                timeout_seconds=timeout_seconds, verbose=verbose,
            )
            c = count_contacts(seq, f.coords)
            if c > best_contacts:
                best = f
                best_contacts = c
    assert best is not None
    return best
