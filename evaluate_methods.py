"""Evaluate all our HP-folding methods against the synthetic ground-truth corpus.

For each synthetic sequence (with known true optimum), run each method with a
small time budget and report whether it hit the optimum, how close, and time.

Output: a method × sequence matrix showing where each method fails. The
hypothesis to confirm: methods fail specifically on high-scatteredness
sequences (where most H residues are isolated singletons).
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, random
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding import (
    perm, search, sequences as seqs_mod, fold as fold_mod,
    moves, saddle_hop, crossover_ga, verifier
)


def run_perm(sequence, time_budget):
    res = perm.perm(sequence, temperature=0.3, n_tours=10**9, weight_lower=0.5,
                     weight_upper=4.0, adapt_weights=True, seed=42,
                     timeout_seconds=time_budget)
    return res.contacts


def run_basin_hop(sequence, time_budget):
    # Basin hop has its own internal scaling; for small N we make it short.
    res = search.basin_hopping(sequence, iterations_per_basin=20_000, n_basins=20,
                                seed=42)
    return res.contacts


def run_replica_exchange(sequence, time_budget):
    res = search.replica_exchange(sequence,
                                    temperatures=(0.2, 0.5, 1.0, 2.0, 4.0),
                                    iterations=50_000, seed=42)
    return res.contacts


def run_saddle_hop(sequence, time_budget):
    res = saddle_hop.saddle_hop(sequence, time_budget_seconds=time_budget,
                                  seed=42, verbose=False)
    return res.contacts


def run_crossover_ga(sequence, time_budget):
    # Need an initial population — generate random valid folds.
    NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
    rng = random.Random(0)
    initial = []
    for attempt in range(200):
        coords = [(0, 0)]
        occupied = {(0, 0)}
        ok = True
        for _ in range(len(sequence) - 1):
            last = coords[-1]
            valid_nbrs = [(last[0]+dx, last[1]+dy) for dx, dy in NBRS
                           if (last[0]+dx, last[1]+dy) not in occupied]
            if not valid_nbrs:
                ok = False; break
            n = rng.choice(valid_nbrs)
            coords.append(n); occupied.add(n)
        if ok and len(coords) == len(sequence):
            initial.append(coords)
            if len(initial) >= 30: break
    if not initial:
        # Fallback: straight-line fold
        initial = [[(i, 0) for i in range(len(sequence))]]
    res = crossover_ga.crossover_ga(sequence, initial,
                                       elite_size=20, children_per_gen=30,
                                       mutation_rate=0.2, polish_steps=30,
                                       time_budget_seconds=time_budget, seed=42)
    return res.best_contacts


METHODS = {
    'PERM':       run_perm,
    'basin_hop':  run_basin_hop,
    'rep_exch':   run_replica_exchange,
    'saddle_hop': run_saddle_hop,
    'crossover':  run_crossover_ga,
}


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='./synthetic_corpus.json')
    ap.add_argument('--time-per-method', type=float, default=10.0,
                    help="Time budget per (method, sequence) pair, seconds")
    ap.add_argument('--out', default='./eval_results.json')
    args = ap.parse_args()

    corpus = json.load(open(args.corpus))
    print(f'Evaluating {len(METHODS)} methods on {len(corpus)} synthetics, '
          f'budget={args.time_per_method}s each.\n')

    print(f'{"sequence":22s}  {"opt":>4s}  {"sc":>5s}  ', end='')
    for mname in METHODS:
        print(f'{mname:>11s}  ', end='')
    print()
    print('-' * (35 + 13 * len(METHODS)))

    all_results = []
    for entry in corpus:
        seq = entry['sequence']
        opt = entry['optimum']
        sc = entry['scatteredness']
        print(f'{entry["name"]:22s}  {opt:>4d}  {sc:>5.2f}  ', end='', flush=True)
        row = {'name': entry['name'], 'optimum': opt, 'scatteredness': sc, 'methods': {}}
        for mname, mfn in METHODS.items():
            try:
                t0 = time.time()
                got = mfn(seq, args.time_per_method)
                wall = time.time() - t0
                hit = got == opt
                row['methods'][mname] = {'best': got, 'hit_opt': hit, 'wall': round(wall, 2)}
                marker = '✓' if hit else f'-{opt-got}'
                print(f'{got:>5d}{marker:<5s} ', end='', flush=True)
            except Exception as e:
                row['methods'][mname] = {'best': None, 'error': str(e)[:50]}
                print(f'{"ERR":>11s}  ', end='', flush=True)
        print()
        all_results.append(row)

    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nsaved per-(method,seq) results to {args.out}')

    # Summary by method
    print('\n=== success rate by method ===')
    for mname in METHODS:
        hits = sum(1 for r in all_results if r['methods'].get(mname, {}).get('hit_opt'))
        total = len(all_results)
        print(f'  {mname:12s}  {hits}/{total} ({100*hits/total:.0f}%)')

    # Hypothesis test: do methods fail more on high-scatteredness sequences?
    print('\n=== success rate by scatteredness ===')
    buckets = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo, hi in buckets:
        in_bucket = [r for r in all_results if lo <= r['scatteredness'] < hi]
        if not in_bucket: continue
        method_hits = []
        for mname in METHODS:
            hits = sum(1 for r in in_bucket if r['methods'].get(mname, {}).get('hit_opt'))
            method_hits.append(f'{mname}={hits}/{len(in_bucket)}')
        print(f'  scatter [{lo:.2f}, {hi:.2f}) ({len(in_bucket)} seqs):  '
              f'{"  ".join(method_hits)}')


if __name__ == "__main__":
    main()
