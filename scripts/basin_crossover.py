"""Pairwise basin crossover.

Given N distinct fold-basins for a sequence, take all C(N, 2) pairs.
For each pair (A, B), try multiple chain-splice points k. The child
fold has chain[0..k] from A and chain[k+1..n] from B (with rigid
transform applied to B's tail to align the splice bond).

If the spliced child is a valid SAW, it's in a topologically NEW basin
(not necessarily a new basin from EITHER parent). Score by contact count,
SA-polish to local maximum.

The combination of basin diversity + crossover generates fold-space
regions that no single algorithm has explored.
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
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip, local_rebridge,
)
from hp_folding.verifier import count_contacts, verify

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def contact_set(seq, coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    pairs = set()
    for i, (x, y) in enumerate(coords):
        if seq[i] != "H":
            continue
        for dx, dy in NB:
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1:
                continue
            if seq[j] == "H":
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    return frozenset(pairs)


def _transform_tail(tail, rot, refl, tx, ty):
    out = []
    for x, y in tail:
        if refl:
            y = -y
        for _ in range(rot % 4):
            x, y = -y, x
        out.append((x + tx, y + ty))
    return out


def crossover(parent_a, parent_b, k, rng):
    """Splice A[0..k] + transformed B[k+1..n-1]. Try all 8 D4 transforms,
    each placing B[k+1] at one of 4 lattice neighbors of A[k]. Return
    valid SAW or None.
    """
    n = len(parent_a)
    if k < 1 or k > n - 2:
        return None
    head = list(parent_a[: k + 1])
    anchor = head[-1]
    head_set = set(head)
    tail_orig = list(parent_b[k + 1:])
    if not tail_orig:
        return None
    b_first_orig = tail_orig[0]

    # Try all combinations: 4 placements × 4 rotations × 2 reflections = 32
    for dx, dy in NB:
        target = (anchor[0] + dx, anchor[1] + dy)
        for rot in range(4):
            for refl in (False, True):
                bx, by = b_first_orig
                if refl:
                    by = -by
                for _ in range(rot):
                    bx, by = -by, bx
                tx = target[0] - bx
                ty = target[1] - by
                tail = _transform_tail(tail_orig, rot, refl, tx, ty)
                # Check no overlap with head
                tail_set = set(tail)
                if len(tail_set) != len(tail):
                    continue
                if head_set & tail_set:
                    continue
                # All bonds in tail are preserved by rigid transform.
                # Splice bond head[-1]-tail[0] manhattan must be 1
                if abs(target[0] - anchor[0]) + abs(target[1] - anchor[1]) != 1:
                    continue
                child = head + tail
                return child
    return None


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


def quick_polish(seq, coords, n_steps, T_start, T_end, seed):
    rng = random.Random(seed)
    cur = list(coords)
    cur_c = count_contacts(seq, cur)
    best_c = cur_c
    best = list(cur)
    for step in range(n_steps):
        T = T_start * (1 - step / n_steps) + T_end * (step / n_steps)
        new = _propose(cur, rng)
        if new is None:
            continue
        c = count_contacts(seq, new)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new
            cur_c = c
            if c > best_c:
                best_c = c
                best = list(cur)
    return best, best_c


def worker(args):
    (seq, parent_a, parent_b, k, n_polish_steps, seed) = args
    rng = random.Random(seed)
    child = crossover(parent_a, parent_b, k, rng)
    if child is None:
        return seed, k, None, -1, None
    init_c = count_contacts(seq, child)
    polished, polished_c = quick_polish(
        seq, child, n_polish_steps, T_start=1.0, T_end=0.05, seed=seed,
    )
    return seed, k, init_c, polished_c, polished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--source-level", type=int, required=True,
                   help="Stepping-stone level to draw parent folds from")
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--n-polish-steps", type=int, default=200_000)
    p.add_argument("--max-pairs", type=int, default=2000,
                   help="Cap on number of crossover pairs (chosen randomly)")
    p.add_argument("--splice-points-per-pair", type=int, default=3,
                   help="Number of splice positions to try per pair")
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--update-submission", action="store_true")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for pair shuffling and splice points")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    seq = bench.sequence
    sub = json.loads(open(f"submissions/{args.sequence}.json").read())
    init_contacts = verify(seq, [tuple(c) for c in sub["coords"]]).contacts

    ss = json.loads(open(ROOT / "diagnostics/stepping_stones.json").read())
    sd = ss.get(args.sequence, {})
    pool = sd.get(str(args.source_level), [])
    if not pool:
        print(f"no stepping-stones at level {args.source_level}")
        return 1

    distinct_folds = []
    seen_sets = set()
    for f in pool:
        coords = [tuple(c) for c in f]
        cs = contact_set(seq, coords)
        if cs not in seen_sets:
            seen_sets.add(cs)
            distinct_folds.append(coords)
    print(f"=== Basin crossover on {args.sequence} ===")
    print(f"  current: {init_contacts} (record {bench.best_known})")
    print(f"  parents: {len(distinct_folds)} distinct level-{args.source_level} folds")

    n_pairs_max = len(distinct_folds) * (len(distinct_folds) - 1) // 2
    print(f"  total possible pairs: {n_pairs_max}")
    rng = random.Random(args.seed)

    pairs = list(combinations(range(len(distinct_folds)), 2))
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    print(f"  using {len(pairs)} pairs × {args.splice_points_per_pair} splice points "
          f"= {len(pairs) * args.splice_points_per_pair} crossover trials")
    sys.stdout.flush()

    n = len(seq)
    work = []
    for trial_id, (i, j) in enumerate(pairs):
        for sp_idx in range(args.splice_points_per_pair):
            k = rng.randrange(n // 4, n - n // 4)
            work.append((seq, distinct_folds[i], distinct_folds[j], k,
                         args.n_polish_steps,
                         args.seed * 1000003 + trial_id * 100 + sp_idx))
    print(f"  total trials: {len(work)}")
    sys.stdout.flush()

    t0 = time.monotonic()
    best_overall = init_contacts
    best_coords = None
    n_done = 0
    n_valid = 0
    n_above_init = 0
    contact_dist = defaultdict(int)

    with mp.Pool(args.processes) as pool_proc:
        for seed, k, init_c, contacts, coords in pool_proc.imap_unordered(
            worker, work, chunksize=4
        ):
            n_done += 1
            if coords is not None:
                n_valid += 1
                contact_dist[contacts] += 1
                if contacts > init_contacts:
                    n_above_init += 1
                if contacts > best_overall:
                    best_overall = contacts
                    best_coords = coords
                    print(f"  trial {seed}: init={init_c} polished={contacts} "
                          f"[BEST overall, target={args.target}]", flush=True)
                    if best_overall >= args.target:
                        print(f"  RECORD CANDIDATE — stopping", flush=True)
                        pool_proc.terminate()
                        break
            if n_done % 100 == 0:
                print(f"  [{n_done}/{len(work)}] valid={n_valid} above_init={n_above_init} "
                      f"best={best_overall} elapsed={time.monotonic()-t0:.0f}s", flush=True)

    wall = time.monotonic() - t0
    print(f"\nWall: {wall:.0f}s, valid_crossovers={n_valid}/{n_done} "
          f"({n_valid/max(n_done,1)*100:.1f}%)")
    print(f"contact distribution: {dict(sorted(contact_dist.items(), reverse=True)[:10])}")
    print(f"Best: {best_overall} (record {bench.best_known}, was {init_contacts})")

    if best_coords and best_overall > init_contacts:
        out = Path(f"results/{args.sequence}_crossover_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=seq, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, f"submissions/{args.sequence}.json")
            print(f"PROMOTED submissions/{args.sequence}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
