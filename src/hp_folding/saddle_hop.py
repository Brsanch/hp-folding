"""Saddle-crossing basin hop — principled escape from local minima.

Standard basin hopping uses random kicks: from a local min, apply some random
perturbation, then polish back down. The hope is the perturbation lands you in
a different basin. The reality is that on hard landscapes (like S6 / S9 / S10
in our experiments) random kicks tend to fall back into the same basin.

This module replaces the random kick with a **saddle-crossing climb**:

1. From the current basin floor, enumerate all single-move neighbors.
2. Pick the one with the *smallest* positive energy increase. That's the
   "easiest" uphill direction — the gentlest slope toward a ridge.
3. Repeat: keep climbing the smallest +ΔE direction until no neighbor has
   higher energy (we're at a local max ≈ saddle point).
4. From the saddle, polish back down (greedy / SA at low T) into a new basin.
5. Compare: same basin (rejected) or new basin?
6. Track visited basins to drive adaptive behavior.

This is the discrete-optimization analog of the Dimer/Eigenvector-Following
methods used in chemistry to find transition states between molecular wells.

The "neighbors" of a fold here are all distinct folds reachable in one move
from the existing move library (corner_flip, pivot, pull, local_rebridge,
end_flip). Enumeration is by repeated random sampling + dedup — exhaustive
enumeration of all moves would be O(N · k) but the random version is
sufficient for finding a saddle.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from hp_folding import moves
from hp_folding.fold import Fold
from hp_folding.verifier import count_contacts

Coord = tuple[int, int]


@dataclass
class SaddleHopResult:
    fold: Fold
    contacts: int
    n_basins_visited: int
    n_climbs: int
    n_improvements: int
    history: list[tuple[float, int]]   # (wall_s, contacts) at each new best
    walltime_seconds: float


def _basin_id(coords: list[Coord], sequence: str) -> tuple:
    """Cheap structural fingerprint of a fold's basin.

    Uses the canonical-translated set of H-residue positions (the H-core
    polyomino). Two folds with the same hcore in the same orientation count
    as the same basin. This is a coarser equivalence than fold-equality but
    captures what determines the energy.
    """
    h_pos = [coords[i] for i, r in enumerate(sequence) if r == "H"]
    if not h_pos:
        return tuple(coords)
    mx_, my = min(p[0] for p in h_pos), min(p[1] for p in h_pos)
    return tuple(sorted((p[0] - mx_, p[1] - my) for p in h_pos))


def _enumerate_neighbors(
    coords: list[Coord],
    sequence: str,
    rng: random.Random,
    n_samples: int = 400,
) -> list[tuple[list[Coord], int]]:
    """Sample distinct one-move neighbors of `coords` and return
    (neighbor_coords, contact_count) pairs.

    Not exhaustive — uses random sampling from each move function. With
    n_samples=400 and 5 move types, we typically discover ~50–200 distinct
    neighbors for an N=50 fold, which is plenty to identify a saddle.
    """
    seen: set[tuple[Coord, ...]] = {tuple(coords)}
    out: list[tuple[list[Coord], int]] = []

    move_fns: list[Callable] = [
        moves.corner_flip,
        moves.pivot,
        moves.pull_move_v2,
        moves.local_rebridge,
        moves.end_flip,
        moves.crankshaft,
    ]

    for _ in range(n_samples):
        for fn in move_fns:
            try:
                m = fn(list(coords), rng)
            except Exception:
                m = None
            if m is None:
                continue
            t = tuple(m)
            if t in seen:
                continue
            seen.add(t)
            c = count_contacts(sequence, m)
            out.append((m, c))
    return out


def _polish_to_floor(
    coords: list[Coord],
    sequence: str,
    rng: random.Random,
    max_passes: int = 50,
    n_samples_per_pass: int = 200,
) -> tuple[list[Coord], int]:
    """Greedy descent: repeatedly take the best contact-increasing neighbor
    until none exists. Stops when at a local maximum of contacts."""
    cur = list(coords)
    cur_c = count_contacts(sequence, cur)
    for _ in range(max_passes):
        nbrs = _enumerate_neighbors(cur, sequence, rng, n_samples=n_samples_per_pass)
        # Pick best strict improvement.
        better = [(m, c) for m, c in nbrs if c > cur_c]
        if not better:
            return cur, cur_c
        # Take the largest improvement.
        m, c = max(better, key=lambda x: x[1])
        cur = m
        cur_c = c
    return cur, cur_c


def _climb_to_saddle(
    coords: list[Coord],
    sequence: str,
    rng: random.Random,
    max_climb_steps: int = 25,
    n_samples_per_step: int = 300,
) -> tuple[list[Coord], int, int]:
    """Climb gentlest +ΔE direction until at a local max (saddle).
    Returns (saddle_coords, saddle_contacts, steps_climbed)."""
    cur = list(coords)
    cur_c = count_contacts(sequence, cur)
    for step in range(max_climb_steps):
        nbrs = _enumerate_neighbors(cur, sequence, rng, n_samples=n_samples_per_step)
        # +ΔE in energy = decrease in contacts → take smallest decrease.
        # Equivalently: among neighbors with c < cur_c, pick the largest c.
        worse = [(m, c) for m, c in nbrs if c < cur_c]
        if not worse:
            # Local max — but we want at least one step uphill, so accept
            # equal-energy moves on the first step.
            if step == 0:
                equal = [(m, c) for m, c in nbrs if c == cur_c]
                if equal:
                    cur, cur_c = rng.choice(equal)
                    continue
            return cur, cur_c, step
        # Smallest energy increase = largest contact count among the worse set.
        m, c = max(worse, key=lambda x: x[1])
        cur = m
        cur_c = c
    return cur, cur_c, max_climb_steps


def saddle_hop(
    sequence: str,
    *,
    initial_coords: list[Coord] | None = None,
    time_budget_seconds: float = 600.0,
    seed: int = 0,
    on_improve: Callable | None = None,
    verbose: bool = False,
) -> SaddleHopResult:
    """Saddle-crossing basin hop.

    Repeatedly: polish to a basin floor, fingerprint it, climb to a saddle,
    polish down to the new basin, compare. Track best-known fold throughout.

    Returns SaddleHopResult with the best fold found and statistics.
    """
    rng = random.Random(seed)
    if initial_coords is None:
        start = Fold.linear(sequence)
        coords = list(start.coords)
    else:
        coords = list(initial_coords)

    # Initial polish to the seed basin's floor.
    floor, floor_c = _polish_to_floor(coords, sequence, rng)
    visited_basins: set[tuple] = {_basin_id(floor, sequence)}
    best_coords = list(floor)
    best_c = floor_c
    history: list[tuple[float, int]] = []
    t0 = time.monotonic()
    if verbose:
        print(f"  initial floor: {floor_c} contacts", flush=True)

    n_basins = 1
    n_climbs = 0
    n_improvements = 0

    while time.monotonic() - t0 < time_budget_seconds:
        # Climb the gentlest slope from the current floor.
        saddle, saddle_c, climb_steps = _climb_to_saddle(floor, sequence, rng)
        n_climbs += 1

        # Descend the other side.
        new_floor, new_c = _polish_to_floor(saddle, sequence, rng)
        new_id = _basin_id(new_floor, sequence)

        if new_id not in visited_basins:
            visited_basins.add(new_id)
            n_basins += 1
            # Move to new basin; accept improvements OR small steps backward
            # (anneal-like, prevents getting stuck).
            if new_c >= floor_c:
                floor = new_floor
                floor_c = new_c
            elif rng.random() < 0.3:
                # Occasional acceptance of worse new basins for exploration.
                floor = new_floor
                floor_c = new_c

            if new_c > best_c:
                best_c = new_c
                best_coords = list(new_floor)
                n_improvements += 1
                wall = time.monotonic() - t0
                history.append((wall, best_c))
                if on_improve is not None:
                    on_improve(best_c, best_coords)
                if verbose:
                    print(f"  basin {n_basins}: NEW BEST {best_c} contacts "
                          f"(climbed {climb_steps}, wall {wall:.0f}s)", flush=True)
        else:
            # Same basin — climb didn't escape. Take a bigger random kick.
            kicker = rng.choice([
                moves.local_rebridge,
                moves.bond_rebridge,
                moves.fragment_random_walk,
            ])
            try:
                kick = kicker(list(floor), rng)
                if kick is not None:
                    new_floor, new_c = _polish_to_floor(kick, sequence, rng)
                    new_id = _basin_id(new_floor, sequence)
                    if new_id not in visited_basins:
                        visited_basins.add(new_id)
                        n_basins += 1
                        if new_c >= floor_c:
                            floor, floor_c = new_floor, new_c
                        if new_c > best_c:
                            best_c = new_c
                            best_coords = list(new_floor)
                            n_improvements += 1
                            wall = time.monotonic() - t0
                            history.append((wall, best_c))
                            if on_improve is not None:
                                on_improve(best_c, best_coords)
                            if verbose:
                                print(f"  basin {n_basins} (via kick): NEW BEST {best_c} "
                                      f"(wall {wall:.0f}s)", flush=True)
            except Exception:
                pass

    return SaddleHopResult(
        fold=Fold(sequence=sequence, coords=best_coords),
        contacts=best_c,
        n_basins_visited=n_basins,
        n_climbs=n_climbs,
        n_improvements=n_improvements,
        history=history,
        walltime_seconds=time.monotonic() - t0,
    )


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json, os
    from hp_folding import sequences as hp_sequences

    ap = argparse.ArgumentParser(description="Saddle-crossing basin hop for HP folding")
    ap.add_argument("--protein-key", required=True)
    ap.add_argument("--time-budget", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-fold", default=None,
                    help="JSON file with starting fold (uses initial_coords)")
    ap.add_argument("--out-dir", default="./saddle_hop_results")
    args = ap.parse_args()

    s = hp_sequences.get(args.protein_key)
    sid = args.protein_key

    init_coords = None
    if args.seed_fold:
        init_coords = [tuple(c) for c in json.load(open(args.seed_fold))["coords"]]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    best_path = f"{out_dir}/{sid}_seed{args.seed}_best.json"

    def on_improve(contacts, coords):
        with open(best_path, "w") as f:
            json.dump({
                "sequence_id": sid, "sequence": s.sequence,
                "n_residues": s.length, "n_h_residues": s.sequence.count("H"),
                "contacts": int(contacts),
                "coords": [list(c) for c in coords],
                "note": f"saddle_hop seed={args.seed} budget={args.time_budget}s",
            }, f, indent=2)

    print(f"=== saddle_hop {sid} target=-{s.best_known} ===", flush=True)
    res = saddle_hop(
        s.sequence, initial_coords=init_coords,
        time_budget_seconds=args.time_budget, seed=args.seed,
        on_improve=on_improve, verbose=True,
    )
    print(f"DONE  contacts={res.contacts}/-{s.best_known}  "
          f"basins={res.n_basins_visited}  climbs={res.n_climbs}  "
          f"improvements={res.n_improvements}  wall={res.walltime_seconds:.0f}s")
