"""Exclusion analyzer: empirically test whether our methods systematically miss
specific optimal folds even when those optima exist.

Methodology:
  1. Pick a synthetic sequence with brute-forceable optimum.
  2. Enumerate ALL folds with the optimum contact count, canonicalize via the
     full dihedral group (8 transforms + translation).
  3. Run each method N times with different seeds. Each run's best fold is
     canonicalized. Record which canonical optima each method-seed combination
     reaches.
  4. Coverage = |reached_canonical_optima| / |all_canonical_optima|. Low
     coverage means the method is biased toward a subset of optima — i.e.,
     EXCLUDING others.

The key test: if multiple distinct canonical optima exist for a sequence
but a method's 20 seeds all converge to the same 1-2 of them, that's
empirical proof of exclusion. If exclusion gets WORSE as N grows, that's
the smoking gun for why S6 plateaus at 19.
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, random
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding import (
    perm, search, sequences as seqs_mod, fold as fold_mod,
    moves, saddle_hop, crossover_ga, verifier,
)

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ════════════════════════════════════════════════════════════
# Canonicalization (dihedral + translation)
# ════════════════════════════════════════════════════════════

DIHEDRAL_TRANSFORMS = [
    lambda p: p,                      # identity
    lambda p: (-p[0], p[1]),          # reflect x
    lambda p: (p[0], -p[1]),          # reflect y
    lambda p: (-p[0], -p[1]),         # 180° rotation
    lambda p: (p[1], p[0]),           # diagonal reflection
    lambda p: (-p[1], p[0]),          # 90° rotation
    lambda p: (p[1], -p[0]),          # 270° rotation
    lambda p: (-p[1], -p[0]),         # anti-diagonal reflection
]


def canonicalize_fold(coords: list[Coord]) -> tuple[Coord, ...]:
    """Canonical form invariant to dihedral symmetry + translation.

    Returns the lex-min of all 8 dihedral images, each translated so the
    first cell is at origin.
    """
    candidates = []
    for transform in DIHEDRAL_TRANSFORMS:
        transformed = [transform(c) for c in coords]
        x0, y0 = transformed[0]
        shifted = tuple((c[0] - x0, c[1] - y0) for c in transformed)
        candidates.append(shifted)
    # Also reverse the chain (a fold and its reverse have the same energy)
    rev = list(reversed(coords))
    for transform in DIHEDRAL_TRANSFORMS:
        transformed = [transform(c) for c in rev]
        x0, y0 = transformed[0]
        shifted = tuple((c[0] - x0, c[1] - y0) for c in transformed)
        candidates.append(shifted)
    return min(candidates)


# ════════════════════════════════════════════════════════════
# Brute-force enumeration of ALL optimal folds
# ════════════════════════════════════════════════════════════

def enumerate_all_optima(sequence: str, time_limit: float = 60.0):
    """Find all canonical-distinct folds achieving the optimum contact count."""
    n = len(sequence)
    best_contacts = 0
    optimal_canon = set()
    deadline = time.monotonic() + time_limit

    coords = [(0, 0), (1, 0)]
    occupied = {(0, 0), (1, 0)}
    pos_to_idx = {(0, 0): 0, (1, 0): 1}
    n_total = 0

    def dfs(depth, contacts):
        nonlocal best_contacts, n_total
        if time.monotonic() > deadline: return
        if depth == n:
            n_total += 1
            if contacts > best_contacts:
                best_contacts = contacts
                optimal_canon.clear()
                optimal_canon.add(canonicalize_fold(coords))
            elif contacts == best_contacts:
                optimal_canon.add(canonicalize_fold(coords))
            return
        last = coords[-1]
        for dx, dy in NBRS:
            cand = (last[0] + dx, last[1] + dy)
            if cand in occupied: continue
            d_c = 0
            if sequence[depth] == 'H':
                for ddx, ddy in NBRS:
                    nc = (cand[0] + ddx, cand[1] + ddy)
                    j = pos_to_idx.get(nc)
                    if j is None: continue
                    if sequence[j] != 'H': continue
                    if abs(j - depth) <= 1: continue
                    d_c += 1
            coords.append(cand)
            occupied.add(cand)
            pos_to_idx[cand] = depth
            dfs(depth + 1, contacts + d_c)
            coords.pop()
            occupied.discard(cand)
            del pos_to_idx[cand]

    if n >= 2:
        dfs(2, 0)
    return best_contacts, optimal_canon, n_total


# ════════════════════════════════════════════════════════════
# Method runners (each returns best fold's canonical form)
# ════════════════════════════════════════════════════════════

def run_perm(sequence, seed, time_budget):
    res = perm.perm(sequence, temperature=0.3, n_tours=10**9, weight_lower=0.5,
                     weight_upper=4.0, adapt_weights=True, seed=seed,
                     timeout_seconds=time_budget)
    return res.contacts, canonicalize_fold(res.fold.coords)


def run_basin_hop(sequence, seed, time_budget):
    res = search.basin_hopping(sequence, iterations_per_basin=20_000, n_basins=20,
                                seed=seed)
    return res.contacts, canonicalize_fold(res.fold.coords)


def run_replica_exchange(sequence, seed, time_budget):
    res = search.replica_exchange(sequence,
                                    temperatures=(0.2, 0.5, 1.0, 2.0, 4.0),
                                    iterations=50_000, seed=seed)
    return res.contacts, canonicalize_fold(res.fold.coords)


def run_saddle_hop(sequence, seed, time_budget):
    res = saddle_hop.saddle_hop(sequence, time_budget_seconds=time_budget,
                                  seed=seed, verbose=False)
    return res.contacts, canonicalize_fold(res.fold.coords)


def run_crossover_ga(sequence, seed, time_budget):
    rng = random.Random(seed)
    initial = []
    for attempt in range(200):
        coords = [(0, 0)]; occupied = {(0, 0)}
        ok = True
        for _ in range(len(sequence) - 1):
            last = coords[-1]
            valid = [(last[0]+dx, last[1]+dy) for dx, dy in NBRS
                       if (last[0]+dx, last[1]+dy) not in occupied]
            if not valid: ok = False; break
            n = rng.choice(valid); coords.append(n); occupied.add(n)
        if ok and len(coords) == len(sequence):
            initial.append(coords)
            if len(initial) >= 30: break
    if not initial:
        return 0, canonicalize_fold([(i, 0) for i in range(len(sequence))])
    res = crossover_ga.crossover_ga(sequence, initial,
                                       elite_size=15, children_per_gen=20,
                                       mutation_rate=0.2, polish_steps=30,
                                       time_budget_seconds=time_budget, seed=seed)
    return res.best_contacts, canonicalize_fold(res.best_fold.coords)


METHODS = {
    'PERM':       run_perm,
    'basin_hop':  run_basin_hop,
    'rep_exch':   run_replica_exchange,
    'saddle_hop': run_saddle_hop,
    'crossover':  run_crossover_ga,
}


# ════════════════════════════════════════════════════════════
# Main analysis
# ════════════════════════════════════════════════════════════

def analyze(sequence: str, name: str, n_seeds: int = 20, time_per_run: float = 8.0):
    print(f'\n{"=" * 72}')
    print(f'== {name}  (sequence: {sequence})')
    print(f'{"=" * 72}')

    print(f'enumerating all canonical-distinct optima...')
    t0 = time.time()
    best, all_optima, n_total = enumerate_all_optima(sequence, time_limit=120.0)
    enum_wall = time.time() - t0
    print(f'  optimum: {best} contacts')
    print(f'  {n_total} total folds enumerated in {enum_wall:.1f}s')
    print(f'  {len(all_optima)} canonically-distinct optima')
    print()

    # Run each method × n_seeds; record canonical reached
    print(f'running each method × {n_seeds} seeds (budget {time_per_run}s each):')
    method_results = {}
    for mname, mfn in METHODS.items():
        reached = set()
        n_optimal_seeds = 0
        for seed in range(n_seeds):
            try:
                contacts, canon = mfn(sequence, seed, time_per_run)
                if contacts == best:
                    n_optimal_seeds += 1
                    reached.add(canon)
            except Exception as e:
                pass
        coverage = len(reached & all_optima) / max(1, len(all_optima))
        method_results[mname] = {
            'reached': reached,
            'coverage_pct': 100 * coverage,
            'n_optimal_seeds': n_optimal_seeds,
            'distinct_optima_reached': len(reached & all_optima),
            'total_optima': len(all_optima),
        }
        print(f'  {mname:12s}  hit_optimum={n_optimal_seeds:>2d}/{n_seeds}  '
              f'distinct_optima_reached={len(reached & all_optima):>3d}/{len(all_optima):<3d}  '
              f'coverage={100*coverage:5.1f}%')

    # Identify optima that NO method ever reached (hardest to find)
    union_reached = set()
    for r in method_results.values():
        union_reached |= r['reached']
    never_reached = all_optima - union_reached
    print(f'\n  optima never reached by ANY method: {len(never_reached)} / {len(all_optima)}')
    if never_reached and len(never_reached) <= 5:
        for o in list(never_reached)[:3]:
            print(f'    sample missed: {o[:8]}...')

    return {
        'name': name, 'sequence': sequence, 'optimum': best,
        'total_folds': n_total, 'total_canonical_optima': len(all_optima),
        'methods': {m: {k: (list(v) if isinstance(v, set) else v)
                          for k, v in r.items() if k != 'reached'}
                     for m, r in method_results.items()},
        'never_reached': len(never_reached),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--time-per-run', type=float, default=8.0)
    ap.add_argument('--out', default='./exclusion_results.json')
    args = ap.parse_args()

    # Test on a graduated set: low N (easy), middle N (where methods still work),
    # higher N (where the action is). Uses synthetic-bench sequences we already
    # know exact optima for via brute force.
    test_set = [
        ('L14_pseudo_S6',  'PPHPPHPHPHHHPP'),
        ('L14_mostly_isolated', 'HPPHPHPPHPHPPH'),
        ('L16_pseudo_S6',  'PPHPPHPHPHHPPHPH'),
        ('L18_pseudo_S6',  'PPHPPHPHPHHHHPHPPP'),
        ('L18_mostly_isolated', 'HPPHPHPPHPHPHPPHPH'),
    ]

    results = []
    for name, seq in test_set:
        try:
            results.append(analyze(seq, name,
                                     n_seeds=args.n_seeds,
                                     time_per_run=args.time_per_run))
        except Exception as e:
            print(f'  ERROR on {name}: {e}')

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved to {args.out}')

    # Summary: coverage trend
    print(f'\n{"="*72}')
    print(f'COVERAGE SUMMARY (% of canonically-distinct optima reached)')
    print(f'{"="*72}')
    print(f'{"sequence":22s}  {"N":>3s} {"opt":>4s} {"#optima":>8s}  ', end='')
    for m in METHODS: print(f'{m:>11s}  ', end='')
    print()
    for r in results:
        n = len(r['sequence'])
        print(f'{r["name"]:22s}  {n:>3d} {r["optimum"]:>4d} {r["total_canonical_optima"]:>8d}  ', end='')
        for m in METHODS:
            cov = r['methods'][m]['coverage_pct']
            print(f'{cov:>10.1f}%  ', end='')
        print()


if __name__ == "__main__":
    main()
