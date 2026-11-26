"""Stepping-stone attack.

Use the 44k stepping-stone folds (from `mine_diagnostics.py`) as DIVERSE
starting points. We've been seeding everything from the current submission;
now we leverage the population of intermediate-energy folds saved across
all trials.

Strategy:
  1. Load stepping_stones.json
  2. For each sequence, sample top-K stepping-stones at HIGH contact levels
  3. Run SA from each in parallel
  4. Auto-promote on improvement
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
from hp_folding.io import write_submission
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip,
)
from hp_folding.verifier import count_contacts, verify

ROOT = Path(__file__).resolve().parents[1]


def _propose(coords, rng):
    u = rng.random()
    if u < 0.40: return corner_flip(coords, rng)
    if u < 0.65: return crankshaft(coords, rng)
    if u < 0.85: return pivot(coords, rng)
    if u < 0.95: return segment_rotate(coords, rng)
    return end_flip(coords, rng)


def sa_polish(seq, init_coords, n_steps, T_start, T_end, seed):
    rng = random.Random(seed)
    cur = list(init_coords)
    cur_c = count_contacts(seq, cur)
    best_c = cur_c
    best = list(cur)
    for step in range(n_steps):
        T = T_start * (1 - step/n_steps) + T_end * (step/n_steps)
        new = _propose(cur, rng)
        if new is None: continue
        c = count_contacts(seq, new)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new; cur_c = c
            if c > best_c:
                best_c = c
                best = list(cur)
    return best, best_c


def worker(args):
    seq, init_coords, n_steps, T_start, T_end, seed, source_contacts = args
    coords, contacts = sa_polish(seq, init_coords, n_steps, T_start, T_end, seed)
    return seed, source_contacts, contacts, coords


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--n-steps", type=int, default=400_000)
    p.add_argument("--T-starts", default="0.5,1.0,1.5")
    p.add_argument("--T-end", type=float, default=0.05)
    p.add_argument("--top-k-stones", type=int, default=200,
                   help="Use top-K stepping-stones (highest contact levels)")
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub = json.loads(open(f"submissions/{args.sequence}.json").read())
    init_contacts = verify(bench.sequence, [tuple(c) for c in sub["coords"]]).contacts
    print(f"=== Stepping-stone attack on {args.sequence} ===")
    print(f"  current submission: {init_contacts} (record {bench.best_known})")

    # Load stepping stones
    stones_data = json.loads(open(ROOT / "diagnostics" / "stepping_stones.json").read())
    sd = stones_data.get(args.sequence, {})
    print(f"  stepping-stone library: {sum(len(v) for v in sd.values())} folds across {len(sd)} levels")

    # Pick top-K stones from highest contact levels
    levels_desc = sorted(sd.keys(), key=lambda x: -int(x))
    stones: list[tuple[int, list]] = []
    for level_str in levels_desc:
        level = int(level_str)
        for fold in sd[level_str]:
            stones.append((level, [tuple(c) for c in fold]))
            if len(stones) >= args.top_k_stones:
                break
        if len(stones) >= args.top_k_stones:
            break
    print(f"  using {len(stones)} stepping-stones (contact range "
          f"{min(s[0] for s in stones)}-{max(s[0] for s in stones)})")
    sys.stdout.flush()

    T_starts = [float(t) for t in args.T_starts.split(",")]
    work = []
    for i, (level, coords) in enumerate(stones):
        T0 = T_starts[i % len(T_starts)]
        work.append((bench.sequence, coords, args.n_steps, T0,
                     args.T_end, i, level))

    t0 = time.monotonic()
    best_contacts = init_contacts
    best_coords = None
    with mp.Pool(args.processes) as pool:
        for i, (seed, src, contacts, coords) in enumerate(
            pool.imap_unordered(worker, work)
        ):
            tag = " [BEST]" if contacts > best_contacts else ""
            if contacts > init_contacts or (i + 1) % 16 == 0:
                print(f"  [{i+1}/{len(work)}] seed={seed} src_contacts={src} "
                      f"-> polished={contacts}{tag}",
                      flush=True)
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = coords
                if best_contacts >= bench.best_known + 1:
                    print(f"  RECORD BEATEN! Stopping.", flush=True)
                    pool.terminate()
                    break

    wall = time.monotonic() - t0
    print(f"\nTotal wall: {wall:.1f}s")
    print(f"Best: {best_contacts} (record {bench.best_known}, was {init_contacts})")

    if best_contacts > init_contacts and best_coords:
        out = Path(f"results/{args.sequence}_stepping_stone_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=bench.sequence, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, f"submissions/{args.sequence}.json")
            print(f"PROMOTED submissions/{args.sequence}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
