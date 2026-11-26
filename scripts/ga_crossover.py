"""Genetic algorithm with crossover for HP folding.

Crossover here means: take residues 0..k from parent A and residues
k+1..n-1 from parent B, applying a rigid transform (translation +
rotation/reflection) to B's tail so that residue k+1 lands lattice-
adjacent to residue k. Validate the result is a SAW.

Most crossovers fail (collisions). The few that succeed produce children
whose H-cell pattern is GENUINELY different from either parent — this is
the "teleport across basins" move that local MCMC cannot do.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.fold import Fold as FoldCls
from hp_folding.io import write_submission
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip, local_rebridge,
)
from hp_folding.verifier import count_contacts, verify

Coord = tuple[int, int]
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _transformed_tail(tail, transform):
    """Apply rigid 2D transform: (translation, rotation_index in 0..3, reflect_x).
    Returns transformed list."""
    tx, ty, rot, refl = transform
    out = []
    for x, y in tail:
        if refl:
            x = -x
        # Rotate by 90*rot
        for _ in range(rot):
            x, y = -y, x
        out.append((x + tx, y + ty))
    return out


def crossover(parent_a, parent_b, rng):
    """Try to splice A[0..k] with transformed B[k+1..n-1]. Returns child
    coords or None."""
    n = len(parent_a)
    occupied_attempts = 0
    # Try several split points and transforms
    for trial in range(100):
        k = rng.randrange(2, n - 2)
        anchor = parent_a[k]
        # b[k+1] needs to map to a lattice neighbor of anchor.
        # Try all 4 neighbor positions × 4 rotations × 2 reflections.
        b_orig = list(parent_b[k + 1:])
        if not b_orig:
            continue
        b0 = b_orig[0]
        for dx, dy in _NEIGHBOURS:
            target = (anchor[0] + dx, anchor[1] + dy)
            for rot in range(4):
                for refl in (False, True):
                    # Compute what transformation places b0 at target.
                    # Apply rotation + reflection to b0 first, then translate.
                    bx, by = b0
                    if refl:
                        bx = -bx
                    for _ in range(rot):
                        bx, by = -by, bx
                    tx = target[0] - bx
                    ty = target[1] - by
                    transform = (tx, ty, rot, refl)
                    new_tail = _transformed_tail(b_orig, transform)
                    child = list(parent_a[:k + 1]) + new_tail
                    # Validate SAW
                    if len(set(child)) != n:
                        occupied_attempts += 1
                        continue
                    # Bonds (already preserved within tail and prefix; only
                    # need to check the splice bond at k → k+1)
                    if abs(child[k][0] - child[k + 1][0]) + abs(child[k][1] - child[k + 1][1]) != 1:
                        continue
                    return child
    return None


def _propose(coords, rng):
    u = rng.random()
    if u < 0.40:
        return corner_flip(coords, rng)
    if u < 0.65:
        return crankshaft(coords, rng)
    if u < 0.85:
        return pivot(coords, rng)
    if u < 0.95:
        return segment_rotate(coords, rng)
    return end_flip(coords, rng)


def quick_anneal(seq, coords, n_steps, T_start, T_end, rng):
    cur = list(coords)
    cur_e = -count_contacts(seq, cur)
    best_e = cur_e
    best_coords = list(cur)
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        T = T_start * (1 - frac) + T_end * frac
        new = _propose(cur, rng)
        if new is None:
            continue
        new_e = -count_contacts(seq, new)
        dE = new_e - cur_e
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new
            cur_e = new_e
            if cur_e < best_e:
                best_e = cur_e
                best_coords = list(cur)
    return best_coords, -best_e


def diversify(seq, coords, rng, n_lateral=15):
    """Use local_rebridge to drift to a different same-or-equal contact fold."""
    cur = list(coords)
    cur_c = count_contacts(seq, cur)
    for _ in range(n_lateral * 30):
        seg = rng.choice([3, 4, 5])
        r = local_rebridge(cur, rng, seg_len=seg)
        if r is None:
            continue
        rc = count_contacts(seq, r)
        if rc >= cur_c:
            cur = r
            cur_c = rc
            n_lateral -= 1
            if n_lateral <= 0:
                break
    return cur


def ga_worker(args):
    seq, init_coords, n_generations, pop_size, anneal_steps, seed = args
    rng = random.Random(seed)

    # Initial population: diversified mutants of the seed
    population = []
    for i in range(pop_size):
        d = diversify(seq, init_coords, random.Random(seed * 1000 + i), n_lateral=10)
        c = count_contacts(seq, d)
        population.append((c, d))
    population.sort(key=lambda x: -x[0])

    best_c = population[0][0]
    best_coords = population[0][1]

    for gen in range(n_generations):
        # Take top half as parents, generate children via crossover + anneal
        parents = population[:max(2, pop_size // 2)]
        children = []
        for _ in range(pop_size):
            a = rng.choice(parents)[1]
            b = rng.choice(parents)[1]
            child = crossover(a, b, rng)
            if child is None:
                # Mutation fallback: lateral hop on parent a
                child = diversify(seq, a, rng, n_lateral=5)
            # Quick anneal
            polished, polished_c = quick_anneal(
                seq, child, anneal_steps, T_start=1.5, T_end=0.05, rng=rng,
            )
            children.append((polished_c, polished))
        # Combine and select top
        combined = parents + children
        combined.sort(key=lambda x: -x[0])
        population = combined[:pop_size]
        if population[0][0] > best_c:
            best_c = population[0][0]
            best_coords = population[0][1]
    return seed, best_c, best_coords


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=16)
    p.add_argument("--islands", type=int, default=16,
                   help="Number of independent GA islands (each runs in its own process)")
    p.add_argument("--pop-size", type=int, default=12)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--anneal-steps", type=int, default=200_000)
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_coords = [tuple(c) for c in sub["coords"]]
    init_contacts = verify(bench.sequence, init_coords).contacts
    print(f"=== GA crossover on {args.sequence} ===")
    print(f"  starting from submission: {init_contacts} contacts (record {bench.best_known})")
    print(f"  islands: {args.islands}, pop_size: {args.pop_size}, "
          f"generations: {args.generations}, anneal_steps: {args.anneal_steps}")
    sys.stdout.flush()

    work = [(bench.sequence, init_coords, args.generations, args.pop_size,
             args.anneal_steps, seed) for seed in range(args.islands)]

    t0 = time.monotonic()
    best_contacts = init_contacts
    best_coords = init_coords
    with mp.Pool(args.processes) as pool:
        for i, (seed, contacts, coords) in enumerate(
            pool.imap_unordered(ga_worker, work)
        ):
            tag = " [BEST]" if contacts > best_contacts else ""
            print(f"  [{i+1:>3d}/{len(work)}] island_seed={seed} "
                  f"-> contacts={contacts}{tag}",
                  flush=True)
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = coords
                if args.target is not None and best_contacts >= args.target:
                    pool.terminate()
                    break

    wall = time.monotonic() - t0
    print(f"\nTotal walltime: {wall:.1f}s")
    print(f"Best: {best_contacts} (record {bench.best_known})")

    if best_contacts > init_contacts and best_coords:
        out = Path(f"results/{args.sequence}_ga_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=bench.sequence, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: "
                  f"{init_contacts} -> {best_contacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
