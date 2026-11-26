"""Iterative bias diagnosis: identify what's special about residual missed optima.

After sprawl_sa, we still miss ~25% of optima. This script:
  1. Reproduces the standard + sprawl_sa baseline (find what's reached)
  2. Identifies the residual missed optima
  3. Computes EXTENDED features (more than first round) and reports the
     features distinguishing missed from reached
  4. Suggests an intervention based on the dominant bias
"""
from __future__ import annotations
from pathlib import Path
import sys, time, random, statistics
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from hp_folding import sprawl_sa, perm, search
from exclusion_analyzer import (
    canonicalize_fold, enumerate_all_optima, NBRS,
    run_perm, run_basin_hop, run_replica_exchange, run_saddle_hop,
)

SEQ = 'PPHPPHPHPHHPPHPH'   # L16_pseudo_S6


# ════════════════════════════════════════════════════════════
# Extended structural features (more than first round)
# ════════════════════════════════════════════════════════════

def features_extended(coords, sequence):
    n = len(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bbox_w = max(xs) - min(xs) + 1
    bbox_h = max(ys) - min(ys) + 1

    # Move sequence
    moves_seq = []
    for i in range(n - 1):
        moves_seq.append((coords[i+1][0] - coords[i][0],
                            coords[i+1][1] - coords[i][1]))
    bends = sum(1 for i in range(len(moves_seq)-1) if moves_seq[i] != moves_seq[i+1])

    # Direction transitions: how often each (prev, next) pair occurs
    # This catches "screw direction" — clockwise vs counterclockwise turns
    cw_turns = 0; ccw_turns = 0
    for i in range(len(moves_seq) - 1):
        a, b = moves_seq[i], moves_seq[i+1]
        if a == b: continue
        # Cross product
        cross = a[0]*b[1] - a[1]*b[0]
        if cross > 0: ccw_turns += 1
        elif cross < 0: cw_turns += 1
    total_turns = cw_turns + ccw_turns
    chirality = (cw_turns - ccw_turns) / max(1, total_turns)  # -1..+1

    # End-to-end vector and its angle
    ee_dx = coords[-1][0] - coords[0][0]
    ee_dy = coords[-1][1] - coords[0][1]
    ee_distance = abs(ee_dx) + abs(ee_dy)

    # Hcore properties
    h_cells = frozenset((c[0], c[1]) for c, r in zip(coords, sequence) if r == 'H')
    n_h = len(h_cells)
    h_internal = 0
    for x, y in h_cells:
        for dx, dy in NBRS:
            if (x + dx, y + dy) in h_cells:
                h_internal += 1
    h_internal //= 2
    h_perim = 4 * n_h - 2 * h_internal

    # Chain "compactness vs hcore compactness"
    chain_bbox_area = bbox_w * bbox_h
    chain_density = n / chain_bbox_area  # closer to 1 = chain fills bbox densely

    # Position of FIRST and LAST residue in the bbox (relative)
    rx0 = (coords[0][0] - min(xs)) / max(1, bbox_w - 1)
    ry0 = (coords[0][1] - min(ys)) / max(1, bbox_h - 1)
    rxN = (coords[-1][0] - min(xs)) / max(1, bbox_w - 1)
    ryN = (coords[-1][1] - min(ys)) / max(1, bbox_h - 1)

    # Endpoint at edge?
    edge_dist_0 = min(rx0, 1-rx0, ry0, 1-ry0)  # 0 = on edge, 0.5 = center
    edge_dist_N = min(rxN, 1-rxN, ryN, 1-ryN)

    # H residues at chain ends? (often determines reachability)
    h_at_start = 1 if sequence[0] == 'H' else 0
    h_at_end = 1 if sequence[-1] == 'H' else 0

    # Hcore "sprawl ratio": h_perim / (4*sqrt(n_h)) -- normalized perimeter
    import math
    min_perim_for_n = 4 * int(math.ceil(2 * math.sqrt(n_h)))
    sprawl_ratio = h_perim / max(1, min_perim_for_n)

    # First-move direction (canonicalized: should be (1, 0) but check anyway)
    first_dir = moves_seq[0] if moves_seq else (0, 0)

    return {
        'bbox_w': bbox_w, 'bbox_h': bbox_h,
        'bbox_aspect': max(bbox_w, bbox_h) / min(bbox_w, bbox_h),
        'bends': bends,
        'chirality': round(chirality, 3),
        'cw_turns': cw_turns, 'ccw_turns': ccw_turns,
        'ee_distance': ee_distance,
        'chain_density': round(chain_density, 3),
        'h_internal': h_internal, 'h_perim': h_perim,
        'sprawl_ratio': round(sprawl_ratio, 3),
        'edge_dist_0': round(edge_dist_0, 3),
        'edge_dist_N': round(edge_dist_N, 3),
    }


def main():
    print('Step 1: enumerate optima', flush=True)
    best, all_optima, n_total = enumerate_all_optima(SEQ, time_limit=60)
    print(f'  optimum={best}, total={len(all_optima)} ({n_total} folds)', flush=True)

    print('\nStep 2: collect reached set (standard methods + sprawl_sa)', flush=True)
    reached = set()
    # Standard methods × 30 seeds each
    for runner in (run_perm, run_basin_hop, run_replica_exchange, run_saddle_hop):
        for seed in range(30):
            try:
                c, canon = runner(SEQ, seed, 3.0)
                if c == best: reached.add(canon)
            except Exception:
                pass
    # sprawl_sa across configs and seeds
    for alpha, beta in [(0.3, 0.1), (0.5, 0.2), (0.8, 0.3), (1.0, 0.4)]:
        for seed in range(20):
            try:
                res = sprawl_sa.sprawl_sa(SEQ, iterations=80_000, t_start=3.0, t_end=0.02,
                                            alpha=alpha, beta=beta, seed=seed)
                if res.contacts == best:
                    reached.add(canonicalize_fold(res.fold.coords))
            except Exception:
                pass
    reached &= all_optima
    missed = all_optima - reached
    print(f'  reached: {len(reached)}/{len(all_optima)} ({100*len(reached)/len(all_optima):.1f}%)', flush=True)
    print(f'  STILL MISSED: {len(missed)}', flush=True)

    if not missed:
        print('  No remaining missed optima — bias fully addressed!')
        return

    print(f'\nStep 3: extended-feature comparison', flush=True)
    reached_features = [features_extended(list(c), SEQ) for c in reached]
    missed_features = [features_extended(list(c), SEQ) for c in missed]
    feat_keys = list(reached_features[0].keys())

    print(f'{"feature":18s}  {"reached mean ± std":>22s}  {"missed mean ± std":>22s}  {"effect":>8s}  hint')
    print('-' * 100)
    insights = []
    for fk in feat_keys:
        rv = [f[fk] for f in reached_features]
        mv = [f[fk] for f in missed_features]
        # Try to coerce to float for stats
        try:
            r_mean = statistics.mean(rv); r_std = statistics.pstdev(rv) if len(rv) > 1 else 0
            m_mean = statistics.mean(mv); m_std = statistics.pstdev(mv) if len(mv) > 1 else 0
        except Exception:
            continue
        pooled_std = ((r_std**2 + m_std**2) / 2) ** 0.5 if (r_std or m_std) else 0
        d = (r_mean - m_mean) / pooled_std if pooled_std > 0 else 0.0
        hint = ''
        if abs(d) >= 0.8: hint = '*** STRONG'
        elif abs(d) >= 0.5: hint = '** moderate'
        elif abs(d) >= 0.3: hint = '* weak'
        print(f'{fk:18s}  {r_mean:>10.3f} ± {r_std:>6.3f}  {m_mean:>10.3f} ± {m_std:>6.3f}  '
              f'{d:>+8.2f}  {hint}')
        if abs(d) >= 0.4:
            insights.append((abs(d), fk, d, r_mean, m_mean))

    print(f'\nTOP residual biases:')
    insights.sort(reverse=True)
    for ad, fk, d, rm, mm in insights[:5]:
        direction = 'higher' if d > 0 else 'lower'
        print(f'  {fk}: reached has {direction} values  (reached={rm:.3f}, missed={mm:.3f}, d={d:+.2f})')

    # Count exact bias counts
    print(f'\nDistribution counts of {len(missed)} missed optima:')
    # sprawl_ratio buckets
    sr = [f['sprawl_ratio'] for f in missed_features]
    print(f'  sprawl_ratio: min={min(sr):.2f} max={max(sr):.2f} mean={statistics.mean(sr):.2f}')
    bd = [f['bends'] for f in missed_features]
    print(f'  bends: min={min(bd)} max={max(bd)} mean={statistics.mean(bd):.1f}')
    ch = [f['chirality'] for f in missed_features]
    print(f'  chirality: min={min(ch):.2f} max={max(ch):.2f} mean={statistics.mean(ch):.2f}')


if __name__ == "__main__":
    main()
