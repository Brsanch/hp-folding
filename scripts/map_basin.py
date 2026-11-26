"""Map the basin structure of best-known folds.

(1) Generate N distinct top-contact folds for a sequence using lateral_hop
    + diversification.
(2) For each pair, compute Jaccard distance on H-H contact sets.
(3) Single-linkage cluster at distance threshold 0.3.
(4) Print one representative fold from each cluster as ASCII art.
(5) Report cluster sizes — tells us how diverse the basin is.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip, local_rebridge,
)
from hp_folding.verifier import count_contacts

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _h_contact_set(sequence, coords):
    pos = {c: i for i, c in enumerate(coords)}
    pairs = set()
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue
        for dx, dy in _NEIGHBOURS:
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1:
                continue
            if sequence[j] != "H":
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return frozenset(pairs)


def _propose(coords, rng):
    u = rng.random()
    if u < 0.4:
        return corner_flip(coords, rng)
    if u < 0.65:
        return crankshaft(coords, rng)
    if u < 0.85:
        return pivot(coords, rng)
    if u < 0.95:
        return segment_rotate(coords, rng)
    return end_flip(coords, rng)


def find_best_fold(seq, init_coords, rng, n_steps=30000):
    """SA from init, return best fold."""
    cur = list(init_coords)
    cur_c = count_contacts(seq, cur)
    best_c = cur_c
    best = list(cur)
    T_start = 0.5
    T_end = 0.05
    for step in range(n_steps):
        T = T_start * (1 - step / n_steps) + T_end * (step / n_steps)
        r = _propose(cur, rng)
        if r is None:
            continue
        c = count_contacts(seq, r)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = r
            cur_c = c
            if c > best_c:
                best_c = c
                best = list(cur)
    return best, best_c


def diversify_then_anneal(seq, init_coords, seed, n_lateral=20, n_anneal=20000):
    rng = random.Random(seed)
    cur = list(init_coords)
    cur_c = count_contacts(seq, cur)
    # Lateral drift
    for _ in range(n_lateral * 30):
        r = local_rebridge(cur, rng, seg_len=rng.choice([3, 4, 5]))
        if r is None:
            continue
        rc = count_contacts(seq, r)
        if rc >= cur_c:
            cur = r
            cur_c = rc
    # Anneal
    return find_best_fold(seq, cur, rng, n_steps=n_anneal)


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


def render_ascii(coords, sequence):
    """Render a fold as ASCII art."""
    minx = min(x for x, _ in coords)
    miny = min(y for _, y in coords)
    coords = [(x - minx, y - miny) for x, y in coords]
    maxx = max(x for x, _ in coords)
    maxy = max(y for _, y in coords)
    grid = [[" " for _ in range(2 * (maxx + 1) + 1)] for _ in range(2 * (maxy + 1) + 1)]
    for i, (x, y) in enumerate(coords):
        ch = "H" if sequence[i] == "H" else "p"
        grid[2 * y][2 * x] = ch
    # bonds
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        if x1 == x2:
            mid_y = y1 + y2
            grid[mid_y][2 * x1] = "|"
        else:
            mid_x = x1 + x2
            grid[2 * y1][mid_x] = "-"
    return "\n".join("".join(row) for row in reversed(grid))


def _map_worker(args_tuple):
    seq, init, seed, n_anneal = args_tuple
    coords, c = diversify_then_anneal(seq, init, seed=seed, n_anneal=n_anneal)
    return seed, coords, c


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--n-folds", type=int, default=200)
    p.add_argument("--cluster-threshold", type=float, default=0.3)
    p.add_argument("--render-clusters", type=int, default=4,
                   help="Render top-N clusters as ASCII")
    p.add_argument("--random-init", action="store_true",
                   help="Use a fresh random SAW each iteration instead of submission")
    p.add_argument("--min-contacts", type=int, default=None,
                   help="Override the contact threshold (default: current submission's count)")
    p.add_argument("--n-anneal", type=int, default=80000)
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub = json.loads(open(f"submissions/{args.sequence}.json").read())
    sub_coords = [tuple(c) for c in sub["coords"]]
    sub_contacts = count_contacts(bench.sequence, sub_coords)
    threshold = args.min_contacts if args.min_contacts is not None else sub_contacts - 2
    print(f"Mapping basin for {args.sequence} (submission {sub_contacts}, record {bench.best_known}, threshold ≥ {threshold})")

    from hp_folding.fold import Fold
    folds: list[tuple[int, list, frozenset]] = []
    t0 = time.monotonic()

    work = []
    for k in range(args.n_folds):
        if args.random_init:
            r0 = random.Random(k * 9941 + 7)
            f0 = Fold.random_saw(bench.sequence, rng=r0)
            init = list(f0.coords)
        else:
            init = sub_coords
        work.append((bench.sequence, init, k, args.n_anneal))

    n_proc = max(1, min(16, mp.cpu_count() - 1))
    with mp.Pool(n_proc) as pool:
        for k, (seed, coords, c) in enumerate(pool.imap_unordered(_map_worker, work)):
            if c >= threshold:
                cs = _h_contact_set(bench.sequence, coords)
                folds.append((c, coords, cs))
            if (k + 1) % 16 == 0:
                print(f"  generated {k+1}/{args.n_folds}, {len(folds)} at peak, "
                      f"max_c={max(f[0] for f in folds) if folds else None}, "
                      f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
    print(f"  total: {len(folds)} folds at contacts >= {init_contacts}, "
          f"contact-set diversity:")
    distinct_sets = {f[2] for f in folds}
    print(f"    distinct contact sets: {len(distinct_sets)}")

    # Cluster by Jaccard distance (single linkage)
    n = len(folds)
    if n == 0:
        return 0
    parents = list(range(n))
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parents[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(folds[i][2], folds[j][2]) <= args.cluster_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    print(f"\n  {len(sorted_clusters)} clusters at jaccard ≤ {args.cluster_threshold}:")
    for k, cluster in enumerate(sorted_clusters[:10]):
        rep = folds[cluster[0]]
        print(f"    cluster {k}: size={len(cluster)}, contacts={rep[0]}")

    # Render top-N cluster representatives
    print()
    for k, cluster in enumerate(sorted_clusters[: args.render_clusters]):
        rep = folds[cluster[0]]
        print(f"=== Cluster {k} (size {len(cluster)}, contacts {rep[0]}) ===")
        print(render_ascii(rep[1], bench.sequence))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
