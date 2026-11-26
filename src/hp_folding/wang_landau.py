"""Wang-Landau sampling for 2D HP lattice protein folding.

Density-of-states MCMC that estimates g(E) — the number of microstates at
energy E — by doing a random walk in energy space with acceptance
min(1, g(E_old) / g(E_new)), updating g on every visit by a modification
factor f that shrinks over successive sweeps as the visit histogram
flattens.

Energy is taken as the negative of H-H contact count. We track the lowest
energy (highest contact count) ever visited and return the corresponding
fold as the best estimate of the ground-state configuration.

Reference: Wang & Landau (2001) PRL 86, 2050. HP-specific tuning follows
Wüst & Landau (2012) J. Chem. Phys. 137, 064903 (the paper whose records
we are trying to beat).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from hp_folding.fold import Fold
from hp_folding.moves import (
    end_flip, pivot, pull_move, pull_move_v2, corner_flip, crankshaft,
    segment_rotate, local_rebridge, bond_rebridge, end_rebridge,
    lmw_pull_full,
)
from hp_folding.verifier import count_contacts

Coord = tuple[int, int]


@dataclass
class WangLandauResult:
    best_contacts: int
    best_coords: list[Coord] | None
    log_g: dict[int, float]  # log density of states by energy (=-contacts)
    hist: dict[int, int]
    log_f_final: float
    sweeps: int
    wall_seconds: float
    # per-energy best fold snapshots (for diagnostic / restart)
    best_fold_at_energy: dict[int, list[Coord]] = field(default_factory=dict)
    # time series: list of (step, wall_s, current_contacts, best_contacts, log_f,
    #                       flatness_ratio, accept_rate_since_last)
    trace: list[tuple[int, float, int, int, float, float, float]] = field(default_factory=list)
    # move-class counts: (proposed, accepted) for (end_flip, pivot, pull, null)
    move_stats: dict[str, tuple[int, int]] = field(default_factory=dict)


def _propose_move(
    coords: list[Coord], rng: random.Random
) -> tuple[list[Coord] | None, str]:
    """Wüst-Landau-aligned proposal: pull + bond-rebridging + pivot.

    The Wust & Landau (2012) paper specifically reports that ground states
    of some HP sequences "would not have been found at all without the
    concerted interplay" of pull + bond-rebridging. The earlier mix in this
    file was corner_flip-heavy because the original pull_move was broken;
    we now use pull_move_v2 (working LMW with cascade) and the canonical
    bond_rebridge (segment-reversal Wüst-Landau move).

    Distribution (Wüst-Landau-inspired, similar to launch_record_attempt's
    macro_plus mix that produces records elsewhere in this repo):
      - 25% pull_move_v2     (working LMW pull)
      - 15% lmw_pull_full    (full LMW with multi-step cascade)
      - 20% bond_rebridge    (Wüst-Landau topology rewiring)
      -  8% end_rebridge     (chain-terminus rebridging)
      -  8% local_rebridge   (small-segment local rewiring)
      - 12% pivot
      -  5% crankshaft
      -  4% corner_flip
      -  3% end_flip
    """
    u = rng.random()
    if u < 0.25:
        return pull_move_v2(coords, rng), "pull_v2"
    if u < 0.40:
        return lmw_pull_full(coords, rng), "lmw_pull"
    if u < 0.60:
        return bond_rebridge(coords, rng), "bond_rebridge"
    if u < 0.68:
        return end_rebridge(coords, rng), "end_rebridge"
    if u < 0.76:
        seg = rng.choice([3, 4, 5, 6])
        return local_rebridge(coords, rng, seg_len=seg), "local_rebridge"
    if u < 0.88:
        return pivot(coords, rng), "pivot"
    if u < 0.93:
        return crankshaft(coords, rng), "crankshaft"
    if u < 0.97:
        return corner_flip(coords, rng), "corner_flip"
    return end_flip(coords, rng), "end_flip"


def wang_landau(
    sequence: str,
    *,
    initial_coords: list[Coord] | None = None,
    time_budget_seconds: float = 60.0,
    log_f_init: float = 1.0,          # initial modification factor (log scale)
    log_f_final: float = 1e-6,        # stop when log f falls below this
    flatness_threshold: float = 0.80,  # H flat when min >= flatness * mean
    flatness_check_interval: int = 10_000,  # steps between flatness checks
    trace_interval: int = 50_000,  # steps between time-series snapshots
    rng_seed: int = 0,
) -> WangLandauResult:
    """Run Wang-Landau on `sequence` until time runs out or log_f converges.

    Returns a WangLandauResult with the best fold found, the estimated log
    density of states, the visit histogram, and the final modification
    factor.
    """
    n = len(sequence)
    rng = random.Random(rng_seed)

    if initial_coords is None:
        fold = Fold.random_saw(sequence, rng=rng)
        coords = list(fold.coords)
    else:
        coords = list(initial_coords)

    # Energy = -contacts (we minimise).
    contacts = count_contacts(sequence, coords)
    energy = -contacts

    log_g: dict[int, float] = {energy: 0.0}
    hist: dict[int, int] = {energy: 0}

    best_contacts = contacts
    best_coords = list(coords)
    best_by_energy: dict[int, list[Coord]] = {energy: list(coords)}

    log_f = log_f_init
    step = 0
    sweeps = 0
    t0 = time.monotonic()
    deadline = t0 + time_budget_seconds

    # Telemetry accumulators.
    trace: list[tuple[int, float, int, int, float, float, float]] = []
    _move_classes = ("pull_v2", "lmw_pull", "bond_rebridge", "end_rebridge",
                     "local_rebridge", "pivot", "crankshaft", "corner_flip",
                     "end_flip", "null")
    move_proposed: dict[str, int] = {k: 0 for k in _move_classes}
    move_accepted: dict[str, int] = {k: 0 for k in _move_classes}
    accepts_since_last_trace = 0
    steps_since_last_trace = 0

    while time.monotonic() < deadline and log_f > log_f_final:
        step += 1
        steps_since_last_trace += 1
        new_coords, move_class = _propose_move(coords, rng)
        move_proposed[move_class] = move_proposed.get(move_class, 0) + 1
        if new_coords is None:
            # Null move: still update histogram & log_g for current state.
            move_proposed["null"] = move_proposed.get("null", 0) + 1
            log_g[energy] = log_g.get(energy, 0.0) + log_f
            hist[energy] = hist.get(energy, 0) + 1
        else:
            new_contacts = count_contacts(sequence, new_coords)
            new_energy = -new_contacts
            # Acceptance: min(1, g(old)/g(new)) = min(1, exp(log_g[old] - log_g[new])).
            lg_old = log_g.get(energy, 0.0)
            lg_new = log_g.get(new_energy, 0.0)
            log_acc = lg_old - lg_new
            accept = log_acc >= 0 or math.log(rng.random() + 1e-300) < log_acc
            if accept:
                coords = new_coords
                contacts = new_contacts
                energy = new_energy
                move_accepted[move_class] = move_accepted.get(move_class, 0) + 1
                accepts_since_last_trace += 1
                if contacts > best_contacts:
                    best_contacts = contacts
                    best_coords = list(coords)
                if energy not in best_by_energy:
                    best_by_energy[energy] = list(coords)
            log_g[energy] = log_g.get(energy, 0.0) + log_f
            hist[energy] = hist.get(energy, 0) + 1

        # Flatness check.
        if step % flatness_check_interval == 0 and hist:
            values = list(hist.values())
            mean_h = sum(values) / len(values)
            min_h = min(values)
            if mean_h > 0 and min_h >= flatness_threshold * mean_h:
                log_f *= 0.5
                sweeps += 1
                hist = {e: 0 for e in hist}

        # Time-series snapshot.
        if step % trace_interval == 0 and hist:
            values = list(hist.values())
            mean_h = sum(values) / max(1, len(values))
            min_h = min(values) if values else 0
            flatness_ratio = (min_h / mean_h) if mean_h > 0 else 0.0
            acc_rate = (accepts_since_last_trace / max(1, steps_since_last_trace))
            trace.append((
                step,
                time.monotonic() - t0,
                contacts,
                best_contacts,
                log_f,
                flatness_ratio,
                acc_rate,
            ))
            accepts_since_last_trace = 0
            steps_since_last_trace = 0

    wall = time.monotonic() - t0
    move_stats = {
        k: (move_proposed.get(k, 0), move_accepted.get(k, 0))
        for k in _move_classes
    }
    return WangLandauResult(
        best_contacts=best_contacts,
        best_coords=best_coords,
        log_g=log_g,
        hist=hist,
        log_f_final=log_f,
        sweeps=sweeps,
        wall_seconds=wall,
        best_fold_at_energy=best_by_energy,
        trace=trace,
        move_stats=move_stats,
    )
