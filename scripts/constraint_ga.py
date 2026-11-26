"""Constraint-set genetic algorithm.

Apply GA to TARGET CONTACT-PAIR SPECIFICATIONS, not lattice structures.
Each "genome" is a set of H-H pairs we WANT in contact. Fitness is
measured by running biased SA and counting actual contacts achieved.

Mutation: swap pairs in/out of target set
Crossover: combine target sets from two parents
Selection: elitist + tournament

Hypothesis: code-level evolution can escape structure-level basin
attractors that have trapped MCMC, crossover, etc.
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
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.fold import Fold as FoldCls
from hp_folding.io import write_submission
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip, local_rebridge,
)
from hp_folding.verifier import count_contacts, verify

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class Genome:
    pairs: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    fitness_cache: tuple[int, list] | None = None  # (contacts, fold_coords)


def candidate_pairs(seq: str) -> list[tuple[int, int]]:
    """All possible H-H pairs (i, j), i<j, both H, |i-j|>1, parity odd."""
    out = []
    for i in range(len(seq)):
        if seq[i] != "H":
            continue
        for j in range(i + 2, len(seq)):
            if seq[j] != "H":
                continue
            if (j - i) % 2 == 1:
                out.append((i, j))
    return out


def random_genome(all_pairs, target_size: int, rng: random.Random) -> Genome:
    """Random subset of size target_size."""
    n = min(target_size, len(all_pairs))
    return Genome(pairs=frozenset(rng.sample(all_pairs, n)))


def mutate(g: Genome, all_pairs, rng: random.Random,
           p_add=0.4, p_remove=0.2, p_swap=0.4) -> Genome:
    """Mutate a genome: add/remove/swap pair(s)."""
    pairs = set(g.pairs)
    op = rng.random()
    if op < p_add and len(pairs) < len(all_pairs):
        # Add a random pair not already in genome
        avail = [p for p in all_pairs if p not in pairs]
        if avail:
            pairs.add(rng.choice(avail))
    elif op < p_add + p_remove and pairs:
        # Remove a random pair
        pairs.discard(rng.choice(list(pairs)))
    else:
        # Swap: remove one, add one
        if pairs:
            pairs.discard(rng.choice(list(pairs)))
        avail = [p for p in all_pairs if p not in pairs]
        if avail:
            pairs.add(rng.choice(avail))
    return Genome(pairs=frozenset(pairs))


def crossover(g1: Genome, g2: Genome, rng: random.Random) -> Genome:
    """Combine two genomes: take ~half from each."""
    p1, p2 = list(g1.pairs), list(g2.pairs)
    rng.shuffle(p1)
    rng.shuffle(p2)
    n_take_each = (len(p1) + len(p2)) // 4
    child = set(p1[:n_take_each]) | set(p2[:n_take_each])
    return Genome(pairs=frozenset(child))


def initial_random_saw(seq, rng):
    """Try to construct a random valid SAW for the sequence."""
    n = len(seq)
    for _ in range(200):
        coords = [(0, 0)]
        occupied = {(0, 0)}
        ok = True
        for _ in range(n - 1):
            last = coords[-1]
            avail = []
            for dx, dy in NB:
                nb = (last[0] + dx, last[1] + dy)
                if nb not in occupied:
                    avail.append(nb)
            if not avail:
                ok = False
                break
            choice = rng.choice(avail)
            coords.append(choice)
            occupied.add(choice)
        if ok and len(coords) == n:
            return coords
    # Fallback: straight line
    return [(i, 0) for i in range(n)]


def _propose(coords, rng):
    u = rng.random()
    if u < 0.30:
        return local_rebridge(coords, rng, seg_len=rng.choice([3, 4, 5, 6]))
    if u < 0.55:
        return corner_flip(coords, rng)
    if u < 0.75:
        return crankshaft(coords, rng)
    if u < 0.88:
        return segment_rotate(coords, rng)
    if u < 0.96:
        return pivot(coords, rng)
    return end_flip(coords, rng)


def count_target_realized(seq, coords, target_pairs):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    n = 0
    for (i, j) in target_pairs:
        ci = coords[i]
        cj = coords[j]
        if abs(ci[0] - cj[0]) + abs(ci[1] - cj[1]) == 1:
            n += 1
    return n


def biased_sa(seq, target_pairs, n_steps, T_start, T_end, seed,
              lambda_target=0.5, init_coords=None):
    """SA with energy = -contacts - lambda_target * (# target pairs realized)."""
    rng = random.Random(seed)
    if init_coords is None:
        cur = initial_random_saw(seq, rng)
    else:
        cur = list(init_coords)
    cur_c = count_contacts(seq, cur)
    cur_t = count_target_realized(seq, cur, target_pairs)
    cur_e = -cur_c - lambda_target * cur_t
    best_c = cur_c
    best_coords = list(cur)
    for step in range(n_steps):
        T = T_start * (1 - step / n_steps) + T_end * (step / n_steps)
        new = _propose(cur, rng)
        if new is None:
            continue
        new_c = count_contacts(seq, new)
        new_t = count_target_realized(seq, new, target_pairs)
        new_e = -new_c - lambda_target * new_t
        dE = new_e - cur_e
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new
            cur_c = new_c
            cur_t = new_t
            cur_e = new_e
            if cur_c > best_c:
                best_c = cur_c
                best_coords = list(cur)
    return best_coords, best_c


def fitness_worker(args):
    seq, target_pairs_list, n_steps, seed, init_coords = args
    target = frozenset(tuple(p) for p in target_pairs_list)
    coords, contacts = biased_sa(
        seq, target, n_steps, T_start=1.0, T_end=0.05, seed=seed,
        lambda_target=0.5, init_coords=init_coords,
    )
    return contacts, coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--genome-size", type=int, default=None,
                   help="Initial genome size (default: current contacts + 2)")
    p.add_argument("--n-steps-fitness", type=int, default=30000)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--mutation-rate", type=float, default=0.4)
    p.add_argument("--seed-from-current", action="store_true",
                   help="Initial population seeded from current submission's contacts")
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    seq = bench.sequence
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_contacts = verify(seq, [tuple(c) for c in sub["coords"]]).contacts
    record = bench.best_known
    target = args.target or record

    all_pairs = candidate_pairs(seq)
    print(f"=== Constraint-set GA on {args.sequence} ===")
    print(f"  current: {init_contacts} (record {record}, target {target})")
    print(f"  candidate pair pool: {len(all_pairs)}")
    print(f"  population: {args.population}, generations: {args.generations}")
    sys.stdout.flush()

    rng = random.Random(0)
    genome_size = args.genome_size or (init_contacts + 2)

    # Initialize population
    population: list[Genome] = []
    if args.seed_from_current:
        # Seed first genome from current contacts
        sub_coords = [tuple(c) for c in sub["coords"]]
        from hp_folding.verifier import IncrementalContactCounter
        # extract contact pairs
        pos = {c: i for i, c in enumerate(sub_coords)}
        sub_pairs = set()
        for i, (x, y) in enumerate(sub_coords):
            if seq[i] != "H": continue
            for dx, dy in NB:
                j = pos.get((x + dx, y + dy))
                if j is None or abs(j - i) <= 1: continue
                if seq[j] == "H":
                    a, b = (i, j) if i < j else (j, i)
                    sub_pairs.add((a, b))
        population.append(Genome(pairs=frozenset(sub_pairs)))
        for _ in range(args.population - 1):
            # Mutated copies of seed
            g = Genome(pairs=frozenset(sub_pairs))
            for _ in range(rng.randint(2, 8)):
                g = mutate(g, all_pairs, rng)
            population.append(g)
    else:
        for _ in range(args.population):
            population.append(random_genome(all_pairs, genome_size, rng))

    best_overall = init_contacts
    best_coords = None

    for gen in range(args.generations):
        t0 = time.monotonic()
        # Evaluate fitness in parallel
        work = [(seq, list(g.pairs), args.n_steps_fitness,
                 gen * 1000 + i, None)
                for i, g in enumerate(population)]
        fitness: list[tuple[int, int]] = []  # (contacts, idx)
        with mp.Pool(args.processes) as pool:
            for i, (contacts, coords) in enumerate(pool.imap(fitness_worker, work)):
                fitness.append((contacts, i, coords))
                if contacts > best_overall:
                    best_overall = contacts
                    best_coords = coords
                    print(f"  gen {gen} idx {i}: contacts={contacts} [BEST]", flush=True)
                    if best_overall >= target:
                        break

        fitness.sort(key=lambda x: -x[0])
        # Cache fitness on genomes
        for c, i, _ in fitness:
            population[i].fitness_cache = (c, None)

        wall = time.monotonic() - t0
        avg = sum(c for c, _, _ in fitness) / len(fitness)
        max_c = fitness[0][0]
        print(f"  gen {gen}: max={max_c} avg={avg:.1f} wall={wall:.0f}s", flush=True)

        if best_overall >= target:
            print(f"  TARGET REACHED at gen {gen}", flush=True)
            break

        # Selection: top-K elitism + crossover + mutation
        n_elite = args.population // 5
        elite = [population[i] for c, i, _ in fitness[:n_elite]]
        next_pop = list(elite)
        while len(next_pop) < args.population:
            pa = rng.choice(elite)
            pb = rng.choice(elite)
            child = crossover(pa, pb, rng)
            if rng.random() < args.mutation_rate:
                # Apply 1-3 mutations
                for _ in range(rng.randint(1, 3)):
                    child = mutate(child, all_pairs, rng)
            next_pop.append(child)
        population = next_pop

    print(f"\nFinal best: {best_overall} (record {record}, was {init_contacts})")
    if best_overall > init_contacts and best_coords:
        out = Path(f"results/{args.sequence}_constraint_ga_c{best_overall}.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=seq, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: {init_contacts} -> {best_overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
