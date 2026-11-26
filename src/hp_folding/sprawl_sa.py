"""Sprawl-biased simulated annealing — explicitly counter the bias diagnosed
by bias_analyzer.

Standard SA optimizes -contacts. Methods minimizing -contacts converge to
*compact* hcores. The exclusion analysis showed our entire toolkit
systematically misses optima with SPRAWLED hcores (high concave-corner count,
high perimeter, less compact).

This SA adds a REWARD for sprawl during search, then tracks the best fold
by TRUE contact count. Effectively: explore the sprawl-biased landscape, but
keep folds scored by the real objective.

Auxiliary energy: E_aux = -contacts - α · h_cc - β · h_perim
                                       (← rewards sprawl)

Tuned α, β > 0 push the search toward sprawled hcores. Set α=β=0 to recover
standard SA. Set them too high and the sampler ignores contacts. The sweet
spot lets the chain explore sprawled topologies while still preferring contacts
within those sprawled regions.
"""
from __future__ import annotations
import math, random, time
from dataclasses import dataclass
from typing import Callable

from hp_folding.fold import Fold
from hp_folding.verifier import count_contacts
from hp_folding.moves import random_move

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _hcore_features(coords: list[Coord], sequence: str) -> tuple[int, int]:
    """Return (h_cc, h_perim) — sprawl proxies."""
    h_cells = frozenset((c[0], c[1]) for c, r in zip(coords, sequence) if r == 'H')
    if not h_cells:
        return 0, 0
    n_h = len(h_cells)
    h_internal = 0
    for x, y in h_cells:
        for dx, dy in NBRS:
            if (x + dx, y + dy) in h_cells:
                h_internal += 1
    h_internal //= 2
    h_perim = 4 * n_h - 2 * h_internal
    # Concave corners
    h_boundary = set()
    for x, y in h_cells:
        for dx, dy in NBRS:
            n_cell = (x + dx, y + dy)
            if n_cell not in h_cells:
                h_boundary.add(n_cell)
    h_cc = 0
    for c in h_boundary:
        n_h_neighbors = sum(1 for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in h_cells)
        if n_h_neighbors >= 2:
            h_cc += 1
    return h_cc, h_perim


@dataclass
class SprawlSAResult:
    fold: Fold
    contacts: int
    iterations: int
    accepted: int
    walltime_seconds: float
    history: list[tuple[float, int]]


def sprawl_sa(
    sequence: str,
    *,
    iterations: int = 500_000,
    t_start: float = 3.0,
    t_end: float = 0.02,
    alpha: float = 0.5,    # reward per concave corner
    beta: float = 0.2,     # reward per perimeter unit
    seed: int = 0,
    initial_coords: list[Coord] | None = None,
    move_fn: Callable = random_move,
    verbose: bool = False,
) -> SprawlSAResult:
    """Run sprawl-biased SA. Returns best fold by TRUE contact count."""
    rng = random.Random(seed)
    n = len(sequence)
    if initial_coords is None:
        coords = [(i, 0) for i in range(n)]
    else:
        coords = list(initial_coords)
    cc, perim = _hcore_features(coords, sequence)
    contacts = count_contacts(sequence, coords)
    e = -contacts - alpha * cc - beta * perim

    best_contacts = contacts
    best_coords = list(coords)
    history = []
    accepted = 0
    start = time.monotonic()
    log_ratio = math.log(t_end / t_start)

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        proposal = move_fn(list(coords), rng)
        if proposal is None:
            continue
        p_contacts = count_contacts(sequence, proposal)
        p_cc, p_perim = _hcore_features(proposal, sequence)
        p_e = -p_contacts - alpha * p_cc - beta * p_perim
        delta = p_e - e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            coords = proposal
            e = p_e
            contacts = p_contacts
            accepted += 1
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = list(coords)
                history.append((time.monotonic() - start, contacts))
                if verbose:
                    print(f'  step {step}: best_contacts={contacts}', flush=True)

    return SprawlSAResult(
        fold=Fold(sequence=sequence, coords=best_coords),
        contacts=best_contacts,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=time.monotonic() - start,
        history=history,
    )
