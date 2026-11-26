"""Crossover genetic algorithm for HP folding.

Population: existing valid folds (loaded from saved JSONs).
Each generation:
  1. Select two parents weighted by contact count (= -energy).
  2. Crossover: pick split point r. Take parent A's prefix [0:r], parent B's
     suffix [r:N] transformed by a dihedral isometry + translation so suffix[0]
     is lattice-adjacent to prefix[-1]. Try all 8 isometries × 4 attachment
     points until one yields a valid SAW (or give up).
  3. Polish the child via greedy local search (existing pull/pivot/rebridge moves).
  4. Mutation: with small probability apply 1-3 macro moves.
  5. Score by contact count; keep top-K across generations as new elite pool.

Key invariant: every accepted child is a verified-valid SAW. Crossover children
that can't be made valid are discarded; we never carry broken chains forward.

This generative approach is FUNDAMENTALLY DIFFERENT from mutation-from-best:
children inherit fragments from MULTIPLE parents simultaneously, so they can
visit basins that no single parent's neighborhood reaches. The 19-basin trap
on S6 might be escaped if pieces of two different 19-folds combine into a
20+ contact fold.
"""
from __future__ import annotations
import random, time, json, os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from hp_folding import moves
from hp_folding.fold import Fold
from hp_folding.verifier import count_contacts, verify

Coord = tuple[int, int]
NBRS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ════════════════════════════════════════════════════════════
# Dihedral isometries on (x, y) lattice
# ════════════════════════════════════════════════════════════

ISOMETRIES = [
    lambda p: p,                      # identity
    lambda p: (-p[0], p[1]),          # reflect x
    lambda p: (p[0], -p[1]),          # reflect y
    lambda p: (-p[0], -p[1]),         # 180° rotation
    lambda p: (p[1], p[0]),           # transpose (reflect across diagonal)
    lambda p: (-p[1], p[0]),          # 90° rotation
    lambda p: (p[1], -p[0]),          # -90° rotation
    lambda p: (-p[1], -p[0]),         # transpose + 180°
]


def _is_saw(coords: list[Coord]) -> bool:
    if len(set(coords)) != len(coords):
        return False
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


def crossover(
    parent_a: list[Coord],
    parent_b: list[Coord],
    sequence: str,
    rng: random.Random,
    *,
    n_split_attempts: int = 5,
) -> list[Coord] | None:
    """Try to combine parent_a's prefix with parent_b's transformed suffix into a valid SAW."""
    N = len(parent_a)
    if N != len(parent_b) or N != len(sequence):
        return None

    for _ in range(n_split_attempts):
        r = rng.randint(2, N - 2)
        prefix = list(parent_a[:r])
        suffix_orig = list(parent_b[r:])
        prefix_set = set(prefix)
        last_prefix = prefix[-1]

        # Try all 8 isometries
        for iso in ISOMETRIES:
            # Apply isometry; result preserves SAW property internally.
            iso_suffix = [iso(c) for c in suffix_orig]
            iso_first = iso_suffix[0]
            # The suffix's first cell must be a lattice neighbor of last_prefix.
            for dx, dy in NBRS:
                target = (last_prefix[0] + dx, last_prefix[1] + dy)
                # Translate so iso_first = target
                tx, ty = target[0] - iso_first[0], target[1] - iso_first[1]
                trans_suffix = [(c[0] + tx, c[1] + ty) for c in iso_suffix]
                # Check no overlap with prefix and no internal overlap (which is preserved).
                if any(c in prefix_set for c in trans_suffix):
                    continue
                # The trans_suffix is internally a SAW because isometries preserve SAW.
                # We just need to verify joint = prefix + trans_suffix is SAW.
                child = prefix + trans_suffix
                if _is_saw(child):
                    return child
    return None


def _polish_local(
    coords: list[Coord],
    sequence: str,
    rng: random.Random,
    n_steps: int = 100,
) -> tuple[list[Coord], int]:
    """Greedy: repeatedly apply random moves, accept any contact-improving."""
    cur = list(coords)
    cur_c = count_contacts(sequence, cur)
    move_set = [moves.corner_flip, moves.pivot, moves.pull_move_v2,
                moves.local_rebridge, moves.end_flip, moves.crankshaft]
    for _ in range(n_steps):
        m_fn = rng.choice(move_set)
        try:
            m = m_fn(list(cur), rng)
        except Exception:
            m = None
        if m is None:
            continue
        c = count_contacts(sequence, m)
        if c > cur_c:
            cur, cur_c = m, c
    return cur, cur_c


def _mutate(coords: list[Coord], sequence: str, rng: random.Random) -> list[Coord]:
    """Apply 1-3 random macro moves; return new coords (or original if all fail)."""
    macro = [moves.local_rebridge, moves.bond_rebridge, moves.fragment_random_walk,
             moves.compact_pivot, moves.subchain_reflect]
    cur = list(coords)
    for _ in range(rng.randint(1, 3)):
        fn = rng.choice(macro)
        try:
            if fn in (moves.compact_pivot, moves.subchain_reflect):
                m = fn(cur, rng, sequence)
            else:
                m = fn(cur, rng)
            if m is not None and _is_saw(m):
                cur = m
        except Exception:
            pass
    return cur


@dataclass
class GAResult:
    best_fold: Fold
    best_contacts: int
    n_generations: int
    n_children: int
    n_valid_crossovers: int
    n_improvements: int
    history: list[tuple[float, int]]


def crossover_ga(
    sequence: str,
    initial_population: list[list[Coord]],
    *,
    elite_size: int = 30,
    children_per_gen: int = 50,
    mutation_rate: float = 0.2,
    polish_steps: int = 80,
    time_budget_seconds: float = 600.0,
    seed: int = 0,
    on_improve: Callable | None = None,
    verbose: bool = False,
) -> GAResult:
    """Run crossover GA from the seed population."""
    rng = random.Random(seed)
    # Filter and score initial population.
    pop: list[tuple[int, list[Coord]]] = []
    for c in initial_population:
        if _is_saw(c) and len(c) == len(sequence):
            pop.append((count_contacts(sequence, c), list(c)))
    pop.sort(key=lambda x: -x[0])
    pop = pop[:elite_size]
    if not pop:
        raise ValueError("no valid folds in initial population")

    best_c = pop[0][0]
    best_f = list(pop[0][1])
    history: list[tuple[float, int]] = []
    t0 = time.monotonic()

    n_gen = 0
    n_children = 0
    n_valid = 0
    n_improvements = 0

    while time.monotonic() - t0 < time_budget_seconds:
        n_gen += 1
        new_children = []
        for _ in range(children_per_gen):
            n_children += 1
            # Tournament selection: pick 2 parents weighted by contacts (squared)
            weights = [max(1, c) ** 2 for c, _ in pop]
            parents = rng.choices(range(len(pop)), weights=weights, k=2)
            pa = pop[parents[0]][1]
            pb = pop[parents[1]][1]
            child = crossover(pa, pb, sequence, rng)
            if child is None:
                continue
            n_valid += 1
            # Mutation
            if rng.random() < mutation_rate:
                m = _mutate(child, sequence, rng)
                if m and _is_saw(m):
                    child = m
            # Polish
            polished, polished_c = _polish_local(child, sequence, rng, n_steps=polish_steps)
            new_children.append((polished_c, polished))
            if polished_c > best_c:
                best_c = polished_c
                best_f = list(polished)
                n_improvements += 1
                wall = time.monotonic() - t0
                history.append((wall, best_c))
                if on_improve is not None:
                    on_improve(best_c, best_f)
                if verbose:
                    print(f'  gen {n_gen}: NEW BEST {best_c} contacts (wall {wall:.0f}s)', flush=True)

        # Replacement: keep top elite_size from pop ∪ new_children
        merged = pop + new_children
        # Dedup by exact coord tuple (trivial structural dedup)
        seen = set(); deduped = []
        for c, f in sorted(merged, key=lambda x: -x[0]):
            key = tuple(f)
            if key in seen: continue
            seen.add(key); deduped.append((c, f))
        pop = deduped[:elite_size]

    return GAResult(
        best_fold=Fold(sequence=sequence, coords=best_f),
        best_contacts=best_c,
        n_generations=n_gen,
        n_children=n_children,
        n_valid_crossovers=n_valid,
        n_improvements=n_improvements,
        history=history,
    )


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from hp_folding import sequences as hp_sequences

    ap = argparse.ArgumentParser()
    ap.add_argument('--protein-key', required=True)
    ap.add_argument('--time-budget', type=float, default=3600)
    ap.add_argument('--elite-size', type=int, default=30)
    ap.add_argument('--children-per-gen', type=int, default=50)
    ap.add_argument('--mutation-rate', type=float, default=0.2)
    ap.add_argument('--polish-steps', type=int, default=80)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default='./crossover_ga_results')
    args = ap.parse_args()

    s = hp_sequences.get(args.protein_key)
    sid = args.protein_key
    os.makedirs(args.out_dir, exist_ok=True)

    # Load initial population from all results dirs
    base = Path(__file__).resolve().parents[2]
    init_pop = []
    for d in ['perm_results', 'basin_hop_results', 'replica_exchange_results',
              'record_attempt_results', 'saddle_hop_results',
              'scout_results/folds', 'constructive_results',
              'hcore_constructor_results']:
        full = base / d
        if not full.is_dir(): continue
        for f in full.iterdir():
            if not f.suffix == '.json' or not f.name.startswith(sid + '_'): continue
            try:
                data = json.load(open(f))
                if data.get('sequence_id') != sid: continue
                coords = [tuple(c) for c in data.get('coords', [])]
                if len(coords) == s.length:
                    init_pop.append(coords)
            except Exception: continue
    print(f'=== crossover_ga {sid} target=-{s.best_known} ===', flush=True)
    print(f'loaded {len(init_pop)} initial folds', flush=True)

    best_path = f'{args.out_dir}/{sid}_best.json'
    def on_imp(c, coords):
        with open(best_path, 'w') as f:
            json.dump({'sequence_id': sid, 'sequence': s.sequence,
                       'n_residues': s.length, 'n_h_residues': s.sequence.count('H'),
                       'contacts': int(c),
                       'coords': [list(x) for x in coords],
                       'note': f'crossover_ga seed={args.seed}'}, f, indent=2)
        print(f'  *** NEW BEST: {c} contacts (saved {best_path}) ***', flush=True)

    res = crossover_ga(
        s.sequence, init_pop,
        elite_size=args.elite_size, children_per_gen=args.children_per_gen,
        mutation_rate=args.mutation_rate, polish_steps=args.polish_steps,
        time_budget_seconds=args.time_budget, seed=args.seed,
        on_improve=on_imp, verbose=True,
    )
    print(f'\nDONE  best={res.best_contacts}/-{s.best_known}  '
          f'gens={res.n_generations}  children={res.n_children}  '
          f'valid_xover={res.n_valid_crossovers}  improvements={res.n_improvements}', flush=True)
