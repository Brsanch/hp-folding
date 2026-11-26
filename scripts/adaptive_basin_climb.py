"""Adaptive-temperature SA with basin-shape-aware kicks.

For each starting basin from the stepping-stone library:
  - Run SA with dynamically adjusted T (target acceptance rate)
  - When acceptance crashes (basin too deep at current T): heat up
  - When stuck without improvement for N steps: apply random perturbation
    proportional to recently-measured basin depth (barrier estimate)

Per-basin adaptive scheduling is the addition over multibasin_climb.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import random
import shutil
import sys
import time
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


def measure_barrier(seq, coords, rng, n_samples=20):
    """Sample N neighboring states; return the median energy DROP (i.e.,
    median contact loss). This estimates 'how high the wall is'."""
    cur_c = count_contacts(seq, coords)
    drops = []
    for _ in range(n_samples):
        new = _propose(coords, rng)
        if new is None:
            continue
        c = count_contacts(seq, new)
        if c < cur_c:
            drops.append(cur_c - c)
    if not drops:
        return 1
    drops.sort()
    return drops[len(drops) // 2]  # median drop


def adaptive_sa(seq, init_coords, n_steps, T_init, seed,
                target_accept=0.30, kick_threshold=4000):
    rng = random.Random(seed)
    cur = list(init_coords)
    cur_c = count_contacts(seq, cur)
    best_c = cur_c
    best = list(cur)
    T = T_init
    accept_window = collections.deque(maxlen=500)
    last_best_step = 0
    n_kicks = 0
    n_t_increases = 0
    n_t_decreases = 0

    for step in range(n_steps):
        new = _propose(cur, rng)
        if new is None:
            continue
        c = count_contacts(seq, new)
        dE = -(c - cur_c)
        accepted = dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6))
        accept_window.append(1 if accepted else 0)
        if accepted:
            cur = new
            cur_c = c
            if c > best_c:
                best_c = c
                best = list(cur)
                last_best_step = step

        # Adapt T every 200 steps once warmed up
        if step % 200 == 0 and len(accept_window) >= 100:
            rate = sum(accept_window) / len(accept_window)
            if rate < target_accept * 0.5:
                T *= 1.05
                n_t_increases += 1
            elif rate > target_accept * 1.7:
                T *= 0.97
                n_t_decreases += 1
            T = min(T, 5.0)
            T = max(T, 0.01)

        # Detect stuck — apply a kick proportional to basin barrier height
        if step - last_best_step > kick_threshold:
            barrier = measure_barrier(seq, cur, rng, n_samples=10)
            n_kick_moves = max(3, barrier * 3)
            for _ in range(n_kick_moves):
                k = _propose(cur, rng)
                if k is not None:
                    cur = k
                    cur_c = count_contacts(seq, cur)
            n_kicks += 1
            last_best_step = step
            T = max(T * 1.3, 0.5)  # heat up after kick to allow recovery

    return best, best_c, dict(
        kicks=n_kicks, t_up=n_t_increases, t_down=n_t_decreases, final_T=T,
    )


def worker(args):
    seq, init_coords, n_steps, T_init, seed, basin_id = args
    coords, contacts, stats = adaptive_sa(seq, init_coords, n_steps, T_init, seed)
    return basin_id, contacts, coords, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--start-level", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=800_000)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--T-inits", default="0.5,1.0,1.5")
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    seq = bench.sequence
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_contacts = verify(seq, [tuple(c) for c in sub["coords"]]).contacts

    ss = json.loads(open(ROOT / "diagnostics/stepping_stones.json").read())
    sd = ss.get(args.sequence, {})
    if args.start_level is None:
        start_level = init_contacts - 2
    else:
        start_level = args.start_level
    pool = sd.get(str(start_level), [])
    if not pool:
        print(f"no stepping-stones at level {start_level}")
        return 1
    print(f"=== Adaptive-SA basin climb on {args.sequence} ===")
    print(f"  current submission: {init_contacts} (record {bench.best_known})")
    print(f"  using {len(pool)} stones at level {start_level}")

    distinct = {}
    for f in pool:
        coords = [tuple(c) for c in f]
        cs = contact_set(seq, coords)
        if cs not in distinct:
            distinct[cs] = coords
    print(f"  {len(distinct)} distinct basins")

    T_inits = [float(t) for t in args.T_inits.split(",")]
    work = []
    for i, (cs, coords) in enumerate(distinct.items()):
        for T0 in T_inits:
            work.append((seq, coords, args.n_steps, T0,
                         i * 1000 + int(T0 * 100), i))
    print(f"  {len(work)} adaptive trials ({len(distinct)} basins × {len(T_inits)} T's)")
    sys.stdout.flush()

    out_dir = Path(f"results/adaptive_{args.sequence}_lvl{start_level}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    best_overall = init_contacts
    best_coords = None
    by_basin: dict[int, list] = {}
    by_basin_best: dict[int, tuple] = {}
    cumulative_stats = {"kicks": 0, "t_up": 0, "t_down": 0}
    n_done = 0

    with mp.Pool(args.processes) as pool_proc:
        for basin_id, contacts, coords, stats in pool_proc.imap_unordered(
            worker, work, chunksize=1
        ):
            n_done += 1
            by_basin.setdefault(basin_id, []).append(contacts)
            cumulative_stats["kicks"] += stats["kicks"]
            cumulative_stats["t_up"] += stats["t_up"]
            cumulative_stats["t_down"] += stats["t_down"]
            prev = by_basin_best.get(basin_id)
            if prev is None or contacts > prev[0]:
                by_basin_best[basin_id] = (contacts, coords)
                if contacts >= init_contacts:
                    fold_path = out_dir / f"basin_{basin_id:03d}_c{contacts}.json"
                    write_submission(
                        Fold(sequence=seq, coords=coords),
                        args.sequence, fold_path,
                    )
            if contacts > best_overall:
                best_overall = contacts
                best_coords = coords
                print(f"  basin {basin_id}: contacts={contacts} kicks={stats['kicks']} "
                      f"T_final={stats['final_T']:.2f} [BEST overall, target={args.target}]",
                      flush=True)
                if best_overall >= args.target:
                    print(f"  RECORD CANDIDATE — stopping pool", flush=True)
                    pool_proc.terminate()
                    break
            if n_done % 32 == 0:
                print(f"  [{n_done}/{len(work)}] best={best_overall} "
                      f"kicks={cumulative_stats['kicks']} t_up={cumulative_stats['t_up']} "
                      f"elapsed={time.monotonic()-t0:.0f}s", flush=True)

    wall = time.monotonic() - t0
    print(f"\nWall: {wall:.0f}s")
    print(f"Stats: {cumulative_stats}")
    pb = sorted([(max(v), bid) for bid, v in by_basin.items()], reverse=True)
    print(f"Per-basin top-15:")
    for c, bid in pb[:15]:
        print(f"  basin {bid}: max={c}")
    print(f"\nOverall best: {best_overall} (record {bench.best_known}, was {init_contacts})")

    if best_coords and best_overall > init_contacts:
        out = Path(f"results/{args.sequence}_adaptive_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=seq, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: {init_contacts} -> {best_overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
