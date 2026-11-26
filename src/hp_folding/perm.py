"""PERM — Pruned-Enriched Rosenbluth Method for HP lattice protein folding.

PERM grows a chain one residue at a time, sampling each next step from the
Boltzmann distribution of local energy contributions. Two weight-based
population-control events keep compute focused on productive regions:

* **Prune**: if the current Rosenbluth weight falls below a lower threshold,
  discard the chain with probability 1/2 (and double the weight of any
  surviving continuation).
* **Enrich**: if the weight rises above an upper threshold, spawn two
  copies of the chain (each with half the weight) and grow them
  independently. This is the recursive structure that gives PERM its
  power — it naturally allocates compute to trajectories that are
  already finding high-quality folds.

Reference: Grassberger 1997, "Pruned-enriched Rosenbluth method: Simulations
of theta polymers of chain length up to 1,000,000" (Phys Rev E).
Grassberger & Hsu 2002 for further HP-model refinements.

The implementation here is recursive but uses an iterative stack-based
traversal to avoid Python's recursion depth limits on 100-residue chains.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from hp_folding.fold import Fold
from hp_folding.verifier import count_contacts

Coord = tuple[int, int]
_NEIGHBOUR_OFFSETS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class PERMResult:
    """Outcome of a PERM run."""

    fold: Fold
    contacts: int
    total_tours: int
    complete_tours: int
    walltime_seconds: float
    best_history: list[int] = field(default_factory=list)


def _delta_contacts(
    sequence: str,
    new_pos: Coord,
    occupied_hs: set[Coord],
    prev: Coord,
    idx: int,
) -> int:
    """Count new H-H contacts formed by placing residue ``idx`` at ``new_pos``.

    Assumes residue ``idx`` is the one being placed. ``occupied_hs`` is the
    set of already-placed H-residue positions; ``prev`` is the chain-adjacent
    previous residue's position (exclude from contact count).
    """
    if sequence[idx] != "H":
        return 0
    d = 0
    for dx, dy in _NEIGHBOUR_OFFSETS:
        n = (new_pos[0] + dx, new_pos[1] + dy)
        if n in occupied_hs and n != prev:
            d += 1
    return d


def perm(
    sequence: str,
    *,
    temperature: float = 0.5,
    n_tours: int = 10_000,
    weight_lower: float | None = None,
    weight_upper: float | None = None,
    adapt_weights: bool = True,
    seed: int | None = None,
    timeout_seconds: float | None = None,
    verbose: bool = False,
) -> PERMResult:
    """Run PERM on the given HP sequence and return the best fold found.

    Parameters
    ----------
    sequence
        HP string.
    temperature
        Boltzmann temperature. Lower T biases more strongly toward
        contact-forming moves. Typical HP values: 0.3–1.5.
    n_tours
        Number of independent tours to attempt (each tour starts a new
        chain from scratch; enrichment events generate additional chains
        within each tour).
    weight_lower, weight_upper
        Prune / enrich thresholds. If None, set adaptively to
        (1/3, 3) × running geometric mean of completed-chain weights.
    adapt_weights
        If True, update thresholds every 100 completed tours based on the
        running weight distribution.
    seed
        PRNG seed. None → system entropy.
    timeout_seconds
        Abort early if wall-clock budget is exceeded, returning the best
        fold found so far.
    verbose
        If True, print progress every 1000 tours.

    Returns
    -------
    PERMResult
        Best fold + statistics.
    """
    rng = random.Random(seed)
    seq = sequence.upper()
    n = len(seq)
    inv_T = 1.0 / max(temperature, 1e-6)

    # Default thresholds (adapt later)
    if weight_lower is None:
        weight_lower = 0.1
    if weight_upper is None:
        weight_upper = 10.0

    best_contacts = -1
    best_coords: list[Coord] | None = None
    best_history: list[int] = []
    total_tours = 0
    complete_tours = 0
    weight_samples: list[float] = []

    start = time.monotonic()

    # Stack-based iterative traversal.
    # Each stack entry is a chain state ready to extend from index `i`:
    # (i, coords, occupied, h_positions, weight)
    def initial_state() -> tuple[int, list[Coord], set[Coord], set[Coord], float]:
        first = (0, 0)
        coords = [first]
        occupied = {first}
        h_positions = {first} if seq[0] == "H" else set()
        return (1, coords, occupied, h_positions, 1.0)

    for tour_idx in range(n_tours):
        if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
            break

        total_tours += 1
        stack = [initial_state()]

        while stack:
            i, coords, occupied, h_positions, weight = stack.pop()

            if i == n:
                complete_tours += 1
                c = count_contacts(seq, coords)
                weight_samples.append(max(weight, 1e-12))
                if c > best_contacts:
                    best_contacts = c
                    best_coords = list(coords)
                    best_history.append((total_tours, c))
                    if verbose:
                        elapsed = time.monotonic() - start
                        print(f"  tour {total_tours:>5d}: new best {c} contacts "
                              f"(wall {elapsed:.1f}s)")
                continue

            if (timeout_seconds is not None
                    and time.monotonic() - start > timeout_seconds):
                break

            # Enumerate candidate next positions with their delta-contact values.
            prev = coords[-1]
            candidates: list[tuple[Coord, int]] = []
            for dx, dy in _NEIGHBOUR_OFFSETS:
                nxt = (prev[0] + dx, prev[1] + dy)
                if nxt in occupied:
                    continue
                dc = _delta_contacts(seq, nxt, h_positions, prev, i)
                candidates.append((nxt, dc))

            if not candidates:
                # Chain trapped; discard.
                continue

            # Rosenbluth local weight: sum of Boltzmann factors.
            ws = [math.exp(dc * inv_T) for (_, dc) in candidates]
            W_local = sum(ws)
            new_weight = weight * W_local

            # Population control BEFORE choosing the next residue: this
            # ensures the weight being compared is for the current step.
            if new_weight < weight_lower:
                # Prune with probability 1/2.
                if rng.random() < 0.5:
                    continue
                new_weight *= 2.0
            elif new_weight > weight_upper:
                # Enrich: spawn two offspring with halved weight.
                half = new_weight * 0.5
                # Choose two next positions (with replacement) biased by Boltzmann.
                for _ in range(2):
                    probs = [w / W_local for w in ws]
                    idx_pick = _sample_index(probs, rng)
                    nxt, _dc = candidates[idx_pick]
                    new_coords = list(coords)
                    new_coords.append(nxt)
                    new_occupied = set(occupied)
                    new_occupied.add(nxt)
                    new_h = set(h_positions)
                    if seq[i] == "H":
                        new_h.add(nxt)
                    stack.append(
                        (i + 1, new_coords, new_occupied, new_h, half)
                    )
                continue

            # Normal step: sample one next position.
            probs = [w / W_local for w in ws]
            idx_pick = _sample_index(probs, rng)
            nxt, _dc = candidates[idx_pick]
            new_coords = list(coords)
            new_coords.append(nxt)
            new_occupied = set(occupied)
            new_occupied.add(nxt)
            new_h = set(h_positions)
            if seq[i] == "H":
                new_h.add(nxt)
            stack.append((i + 1, new_coords, new_occupied, new_h, new_weight))

        # Adapt thresholds every 100 tours based on running geometric mean.
        if adapt_weights and total_tours % 100 == 0 and weight_samples:
            log_weights = [math.log(w) for w in weight_samples[-500:]]
            gm = math.exp(sum(log_weights) / len(log_weights))
            weight_lower = gm / 3.0
            weight_upper = gm * 3.0

        if verbose and total_tours % 1000 == 0 and total_tours > 0:
            elapsed = time.monotonic() - start
            print(f"  {total_tours:>5d} tours "
                  f"(complete: {complete_tours}, best: {best_contacts}, "
                  f"wall: {elapsed:.1f}s)")

    walltime = time.monotonic() - start
    if best_coords is None:
        # No tour completed — fall back to straight-line fold.
        best_coords = [(i, 0) for i in range(n)]
        best_contacts = count_contacts(seq, best_coords)

    return PERMResult(
        fold=Fold(sequence=seq, coords=best_coords),
        contacts=best_contacts,
        total_tours=total_tours,
        complete_tours=complete_tours,
        walltime_seconds=walltime,
        best_history=best_history,
    )


def _sample_index(probs: list[float], rng: random.Random) -> int:
    """Sample an index from a discrete distribution given by ``probs``."""
    u = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if u < cum:
            return i
    return len(probs) - 1
