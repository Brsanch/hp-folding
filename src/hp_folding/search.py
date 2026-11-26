"""Simulated annealing baseline for HP-lattice folding.

The solver starts from a straight-line fold of the given sequence, proposes
random moves via :mod:`hp_folding.moves`, and accepts them under a
Metropolis criterion with a geometrically-decreasing temperature schedule.

This is a deliberately simple baseline: no pull-move biasing, no replica
exchange, no chain-growth initialisation. It exists to establish a
reproducible lower bound against which more elaborate solvers can be
compared. The top-level ``solver.py`` in the repo root is intended as the
file researchers iterate on; this module provides a stable library API.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from hp_folding.fold import Fold
from hp_folding.moves import random_move
from hp_folding.verifier import count_contacts


@dataclass
class AnnealResult:
    """Outcome of a single simulated-annealing run."""

    fold: Fold
    contacts: int
    iterations: int
    accepted: int
    walltime_seconds: float
    history: list[int] = field(default_factory=list)


def simulated_annealing(
    sequence: str,
    *,
    iterations: int = 100_000,
    t_start: float = 2.0,
    t_end: float = 0.05,
    seed: int | None = None,
    move_fn: Callable = random_move,
    initial_fold: Fold | None = None,
    log_every: int = 0,
) -> AnnealResult:
    """Run simulated annealing to maximise H-H contacts.

    Parameters
    ----------
    sequence
        HP sequence to fold.
    iterations
        Number of MC steps.
    t_start, t_end
        Temperatures at the start and end of the run; a geometric schedule
        interpolates between them.
    seed
        PRNG seed; if ``None``, a system-entropy seed is used.
    move_fn
        Callable ``(coords, rng) -> new_coords | None`` proposing a move.
    initial_fold
        Optional starting configuration; defaults to the straight-line fold.
    log_every
        If positive, append best-contacts-so-far to ``history`` every this
        many iterations.

    Returns
    -------
    AnnealResult
        Best fold discovered, contact count, and run statistics.
    """
    rng = random.Random(seed)
    current = initial_fold.copy() if initial_fold is not None else Fold.linear(sequence)
    if current.sequence != sequence:
        raise ValueError("initial_fold has a different sequence")

    current_e = -count_contacts(sequence, current.coords)
    best_coords = list(current.coords)
    best_e = current_e

    accepted = 0
    history: list[int] = []

    start = time.monotonic()
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        proposal = move_fn(current.coords, rng)
        if proposal is None:
            if log_every and step % log_every == 0:
                history.append(-best_e)
            continue

        proposal_e = -count_contacts(sequence, proposal)
        delta = proposal_e - current_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            current = Fold(sequence=sequence, coords=proposal)
            current_e = proposal_e
            accepted += 1
            if current_e < best_e:
                best_e = current_e
                best_coords = list(current.coords)

        if log_every and step % log_every == 0:
            history.append(-best_e)

    walltime = time.monotonic() - start
    best_fold = Fold(sequence=sequence, coords=best_coords)
    return AnnealResult(
        fold=best_fold,
        contacts=-best_e,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        history=history,
    )


def multi_start(
    sequence: str,
    *,
    n_starts: int = 4,
    iterations: int = 100_000,
    t_start: float = 2.0,
    t_end: float = 0.05,
    base_seed: int = 0,
    log_every: int = 0,
) -> AnnealResult:
    """Run ``n_starts`` independent annealing runs and return the best."""
    best: AnnealResult | None = None
    for k in range(n_starts):
        r = simulated_annealing(
            sequence,
            iterations=iterations,
            t_start=t_start,
            t_end=t_end,
            seed=base_seed + k,
            log_every=log_every,
        )
        if best is None or r.contacts > best.contacts:
            best = r
    assert best is not None
    return best


def soft_shape_sa(
    sequence: str,
    shape: frozenset[tuple[int, int]],
    *,
    iterations: int = 500_000,
    t_start: float = 3.0,
    t_end: float = 0.02,
    h_out_penalty: float = 1.0,
    p_in_penalty: float = 0.2,
    seed: int | None = None,
    move_fn: Callable = random_move,
) -> AnnealResult:
    """Simulated annealing with a soft shape bias.

    Energy = -contacts + h_out_penalty * (H residues outside the shape)
                       + p_in_penalty  * (P residues inside  the shape)

    The shape is advisory, not mandatory. The optimiser can violate either
    constraint but pays an energy cost. Tuned penalties steer the chain
    toward folds whose H-subset clusters into the target shape while still
    letting the sequence structurally fit.
    """
    rng = random.Random(seed)
    seq = sequence.upper()
    h_mask = [c == "H" for c in seq]

    def shape_penalty(coords: list[tuple[int, int]]) -> float:
        h_out = sum(
            1 for i, c in enumerate(coords)
            if h_mask[i] and c not in shape
        )
        p_in = sum(
            1 for i, c in enumerate(coords)
            if not h_mask[i] and c in shape
        )
        return h_out_penalty * h_out + p_in_penalty * p_in

    def energy(coords: list[tuple[int, int]]) -> float:
        return -count_contacts(seq, coords) + shape_penalty(coords)

    current = Fold.linear(seq)
    current_energy = energy(current.coords)
    current_contacts = count_contacts(seq, current.coords)
    best_contacts = current_contacts
    best_coords = list(current.coords)

    accepted = 0
    start = time.monotonic()
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        proposal = move_fn(current.coords, rng)
        if proposal is None:
            continue
        proposal_energy = energy(proposal)
        delta = proposal_energy - current_energy
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            current = Fold(sequence=seq, coords=proposal)
            current_energy = proposal_energy
            proposal_contacts = count_contacts(seq, proposal)
            accepted += 1
            # Track best on CONTACTS, not energy — shape penalty is advisory.
            if proposal_contacts > best_contacts:
                best_contacts = proposal_contacts
                best_coords = list(proposal)

    walltime = time.monotonic() - start
    return AnnealResult(
        fold=Fold(sequence=seq, coords=best_coords),
        contacts=best_contacts,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
    )


def basin_hopping(
    sequence: str,
    *,
    iterations_per_basin: int = 100_000,
    n_basins: int = 10,
    reheat_t_start: float = 3.0,
    reheat_t_end: float = 0.02,
    seed: int | None = None,
    move_fn: Callable = random_move,
) -> AnnealResult:
    """Iterated simulated annealing: each "basin" is one SA run that starts
    from the best configuration found so far (not from a straight line),
    with a fresh high-temperature reheat followed by cooling.

    This escapes local minima that a single long SA run gets stuck in, by
    re-warming the trajectory around the best-known-so-far and letting it
    explore before cooling back down. A simple form of iterated local
    search / basin hopping.

    Returns the global best fold across all basins.
    """
    rng = random.Random(seed)
    start = time.monotonic()

    # Bootstrap: first basin starts from a straight-line fold.
    current_fold = Fold.linear(sequence)
    current_result = simulated_annealing(
        sequence,
        iterations=iterations_per_basin,
        t_start=reheat_t_start,
        t_end=reheat_t_end,
        seed=rng.randrange(1 << 30),
        move_fn=move_fn,
        initial_fold=current_fold,
    )
    best_fold = current_result.fold.copy()
    best_contacts = current_result.contacts
    total_accepted = current_result.accepted

    for _ in range(n_basins - 1):
        # Reheat from the best-known fold.
        r = simulated_annealing(
            sequence,
            iterations=iterations_per_basin,
            t_start=reheat_t_start,
            t_end=reheat_t_end,
            seed=rng.randrange(1 << 30),
            move_fn=move_fn,
            initial_fold=best_fold,
        )
        total_accepted += r.accepted
        if r.contacts > best_contacts:
            best_fold = r.fold.copy()
            best_contacts = r.contacts

    walltime = time.monotonic() - start
    return AnnealResult(
        fold=best_fold,
        contacts=best_contacts,
        iterations=iterations_per_basin * n_basins,
        accepted=total_accepted,
        walltime_seconds=walltime,
    )


def replica_exchange(
    sequence: str,
    *,
    temperatures: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 4.0),
    iterations: int = 100_000,
    exchange_every: int = 1000,
    seed: int | None = None,
    move_fn: Callable = random_move,
    initial_folds: list[Fold] | None = None,
) -> AnnealResult:
    """Replica-exchange Monte Carlo (parallel tempering) for HP folding.

    Runs ``len(temperatures)`` replicas of the chain at different temperatures
    in parallel. Every ``exchange_every`` iterations, proposes swaps between
    adjacent-temperature replicas with the standard Metropolis-Hastings
    acceptance criterion

        p_accept = min(1, exp((E_i - E_j) * (1 / T_i - 1 / T_j)))

    where E_i is the energy (-contacts) of replica i. Because contact counts
    are non-negative integers, the cold replica accumulates configurations
    from the hot replicas that explore broadly, producing substantially
    better ground-state searches than a single cooling schedule.

    Returns the best fold discovered across all replicas throughout the run.
    """
    rng = random.Random(seed)
    n_replicas = len(temperatures)
    if n_replicas < 2:
        raise ValueError("replica_exchange requires at least 2 temperatures")

    # Initialise each replica from the straight-line fold (or provided starts).
    folds: list[Fold]
    if initial_folds is not None:
        if len(initial_folds) != n_replicas:
            raise ValueError(
                f"expected {n_replicas} initial folds, got {len(initial_folds)}"
            )
        folds = [f.copy() for f in initial_folds]
    else:
        folds = [Fold.linear(sequence) for _ in range(n_replicas)]
    for f in folds:
        if f.sequence != sequence:
            raise ValueError("initial fold has different sequence")

    energies = [-count_contacts(sequence, f.coords) for f in folds]
    best_e = min(energies)
    best_idx = energies.index(best_e)
    best_coords = list(folds[best_idx].coords)

    accepted = 0
    attempted = 0
    exchanges_accepted = 0
    exchanges_attempted = 0
    start = time.monotonic()

    for step in range(iterations):
        # One MC step per replica at its own temperature.
        for i in range(n_replicas):
            proposal = move_fn(folds[i].coords, rng)
            attempted += 1
            if proposal is None:
                continue
            proposal_e = -count_contacts(sequence, proposal)
            delta = proposal_e - energies[i]
            if delta <= 0 or rng.random() < math.exp(-delta / max(temperatures[i], 1e-12)):
                folds[i] = Fold(sequence=sequence, coords=proposal)
                energies[i] = proposal_e
                accepted += 1
                if proposal_e < best_e:
                    best_e = proposal_e
                    best_coords = list(proposal)

        # Exchange adjacent replicas periodically.
        if exchange_every > 0 and step > 0 and step % exchange_every == 0:
            for i in range(n_replicas - 1):
                exchanges_attempted += 1
                t_i, t_j = temperatures[i], temperatures[i + 1]
                e_i, e_j = energies[i], energies[i + 1]
                # Delta for swap: (E_i - E_j) * (1/T_i - 1/T_j)
                # If E_i > E_j (worse at cold replica) and T_i < T_j, 1/T_i - 1/T_j > 0,
                # delta > 0 so swap is favoured.
                log_acc = (e_i - e_j) * (1.0 / t_i - 1.0 / t_j)
                if log_acc >= 0 or rng.random() < math.exp(log_acc):
                    folds[i], folds[i + 1] = folds[i + 1], folds[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]
                    exchanges_accepted += 1

    walltime = time.monotonic() - start
    return AnnealResult(
        fold=Fold(sequence=sequence, coords=best_coords),
        contacts=-best_e,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        history=[
            exchanges_accepted,
            exchanges_attempted,
            attempted,
        ],
    )
