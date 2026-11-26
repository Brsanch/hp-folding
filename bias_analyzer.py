"""Identify what features distinguish FOUND optima from MISSED optima.

Workflow:
  1. Brute-force enumerate all canonical optima for a small synthetic where
     coverage was low (L16_pseudo_S6: 182 optima, methods reach ~12).
  2. Run each method many times with diverse seeds; record every canonical
     optimum reached.
  3. Classify each canonical optimum as FOUND or MISSED based on whether any
     method-seed combination reached it.
  4. Compute per-optimum structural features and compare distributions.
     Output the feature(s) where found and missed differ most strongly —
     that's the bias.
"""
from __future__ import annotations
from pathlib import Path
import sys, time, json, random
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding import (
    perm, search, sequences as seqs_mod, fold as fold_mod,
    moves, saddle_hop, crossover_ga, verifier,
)
from exclusion_analyzer import (
    canonicalize_fold, enumerate_all_optima, NBRS,
    run_perm, run_basin_hop, run_replica_exchange, run_saddle_hop, run_crossover_ga,
)

METHODS = {
    'PERM': run_perm,
    'basin_hop': run_basin_hop,
    'rep_exch': run_replica_exchange,
    'saddle_hop': run_saddle_hop,
    'crossover': run_crossover_ga,
}


# ════════════════════════════════════════════════════════════
# Per-fold structural features
# ════════════════════════════════════════════════════════════

def features(coords, sequence):
    """Compute a feature dict for one fold."""
    n = len(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bbox_w = max(xs) - min(xs) + 1
    bbox_h = max(ys) - min(ys) + 1
    bbox_area = bbox_w * bbox_h
    bbox_aspect = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)

    # Chain bends: how many times consecutive moves change direction
    moves_seq = []
    for i in range(n - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        moves_seq.append((dx, dy))
    bends = sum(1 for i in range(len(moves_seq)-1) if moves_seq[i] != moves_seq[i+1])

    # Longest straight run in chain
    longest_straight = 1
    cur = 1
    for i in range(1, len(moves_seq)):
        if moves_seq[i] == moves_seq[i-1]:
            cur += 1
            longest_straight = max(longest_straight, cur)
        else:
            cur = 1

    # Compactness: chain length / bbox area
    compactness = n / bbox_area

    # End-to-end distance
    e2e = abs(coords[0][0] - coords[n-1][0]) + abs(coords[0][1] - coords[n-1][1])

    # Hcore properties
    h_cells = frozenset((c[0], c[1]) for c, r in zip(coords, sequence) if r == 'H')
    h_internal = 0
    for x, y in h_cells:
        for dx, dy in NBRS:
            if (x+dx, y+dy) in h_cells:
                h_internal += 1
    h_internal //= 2  # each edge counted twice
    n_h = len(h_cells)
    if n_h == 0:
        h_perim = 0
        h_cc = 0
    else:
        h_perim = 4 * n_h - 2 * h_internal
        # Concave corners: H boundary cells with ≥2 H neighbors
        h_cc = 0
        h_boundary = set()
        for x, y in h_cells:
            for dx, dy in NBRS:
                n_cell = (x+dx, y+dy)
                if n_cell not in h_cells:
                    h_boundary.add(n_cell)
        for c in h_boundary:
            n_h_neighbors = sum(1 for dx, dy in NBRS if (c[0]+dx, c[1]+dy) in h_cells)
            if n_h_neighbors >= 2:
                h_cc += 1

    return {
        'bbox_w': bbox_w, 'bbox_h': bbox_h,
        'bbox_area': bbox_area, 'bbox_aspect': round(bbox_aspect, 3),
        'bends': bends, 'longest_straight': longest_straight,
        'compactness': round(compactness, 3),
        'end_to_end': e2e,
        'h_internal': h_internal, 'h_perim': h_perim, 'h_cc': h_cc,
    }


# ════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════

def collect_found_optima(sequence, time_budget, n_seeds_per_method):
    """Run each method × n_seeds, return set of canonical optima reached."""
    found = set()
    for mname, mfn in METHODS.items():
        for seed in range(n_seeds_per_method):
            try:
                contacts, canon = mfn(sequence, seed, time_budget)
                # We only care about TRULY optimal results
                # Caller will filter to optimum
                found.add((canon, contacts))
            except Exception:
                pass
    return found


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sequence', default='PPHPPHPHPHHPPHPH')   # L16_pseudo_S6
    ap.add_argument('--name', default='L16_pseudo_S6')
    ap.add_argument('--n-seeds', type=int, default=30)
    ap.add_argument('--time-per-run', type=float, default=4.0)
    ap.add_argument('--out', default='./bias_analysis.json')
    args = ap.parse_args()

    print(f'== Analyzing {args.name}: {args.sequence} ==', flush=True)
    print(f'enumerating all canonical optima...', flush=True)
    t0 = time.time()
    best, all_optima, n_total = enumerate_all_optima(args.sequence, time_limit=120)
    print(f'  optimum: {best} contacts  ({n_total} total folds, {len(all_optima)} optima)  '
          f'in {time.time()-t0:.1f}s', flush=True)

    print(f'running {len(METHODS)} methods × {args.n_seeds} seeds × {args.time_per_run}s...',
          flush=True)
    t0 = time.time()
    found_with_contacts = collect_found_optima(args.sequence, args.time_per_run, args.n_seeds)
    found_optimal_canonical = {canon for canon, c in found_with_contacts if c == best}
    found_in_optima = found_optimal_canonical & all_optima
    missed = all_optima - found_optimal_canonical
    print(f'  found {len(found_in_optima)}/{len(all_optima)} optima '
          f'(coverage {100*len(found_in_optima)/len(all_optima):.1f}%)  '
          f'in {time.time()-t0:.0f}s', flush=True)

    print(f'\ncomputing features for all {len(all_optima)} optima...', flush=True)
    found_features = []
    missed_features = []
    for canon in all_optima:
        feats = features(list(canon), args.sequence)
        if canon in found_in_optima:
            found_features.append(feats)
        else:
            missed_features.append(feats)
    print(f'  found_n={len(found_features)}  missed_n={len(missed_features)}', flush=True)

    # Compute mean and stdev per feature for found vs missed; report effect sizes
    print(f'\n{"="*78}')
    print(f'FEATURE DISTRIBUTIONS — found vs missed optima')
    print(f'{"="*78}')

    if not found_features or not missed_features:
        print('  Cannot compare: one set is empty')
        return

    feat_keys = list(found_features[0].keys())
    print(f'{"feature":18s}  {"found mean ± std":>22s}  {"missed mean ± std":>22s}  {"effect":>8s}  hint')
    print('-' * 100)
    insights = []
    import statistics
    for fk in feat_keys:
        fv = [f[fk] for f in found_features]
        mv = [f[fk] for f in missed_features]
        f_mean = statistics.mean(fv); f_std = statistics.pstdev(fv) if len(fv) > 1 else 0
        m_mean = statistics.mean(mv); m_std = statistics.pstdev(mv) if len(mv) > 1 else 0
        # Pooled stdev for effect size (Cohen's d)
        pooled_std = ((f_std**2 + m_std**2) / 2) ** 0.5 if (f_std or m_std) else 0
        d = (f_mean - m_mean) / pooled_std if pooled_std > 0 else 0.0
        hint = ''
        if abs(d) >= 0.8: hint = '*** STRONG bias'
        elif abs(d) >= 0.5: hint = '** moderate bias'
        elif abs(d) >= 0.3: hint = '* weak bias'
        if d > 0:
            direction = 'found > missed'
        elif d < 0:
            direction = 'found < missed'
        else:
            direction = '—'
        print(f'{fk:18s}  {f_mean:>10.2f} ± {f_std:>6.2f}  '
              f'{m_mean:>10.2f} ± {m_std:>6.2f}  {d:>+8.2f}  {hint} ({direction})')
        if abs(d) >= 0.5:
            insights.append((abs(d), fk, d, f_mean, m_mean))

    print(f'\n{"="*78}')
    if insights:
        insights.sort(reverse=True)
        print('TOP BIAS FEATURES (sorted by effect size):')
        for ad, fk, d, fm, mm in insights[:5]:
            direction = 'higher' if d > 0 else 'lower'
            print(f'  {fk}: methods bias toward {direction} values  '
                  f'(found={fm:.2f}, missed={mm:.2f}, d={d:+.2f})')
    else:
        print('No strong bias features found. The methods either miss randomly OR')
        print('the structural differentiator is not in our feature set.')

    # Save
    with open(args.out, 'w') as f:
        json.dump({
            'sequence': args.sequence,
            'name': args.name,
            'optimum': best,
            'total_optima': len(all_optima),
            'found_count': len(found_in_optima),
            'missed_count': len(missed),
            'found_features': found_features,
            'missed_features': missed_features,
        }, f, indent=2)
    print(f'\nsaved to {args.out}')


if __name__ == "__main__":
    main()
