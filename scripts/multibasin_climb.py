"""Multi-basin climb: run SA from EACH of our distinct stepping-stone
folds at high contact levels. Each is a potentially distinct basin
Wüst's published reference doesn't sample. Some basins might climb to
unprecedented contact levels.
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


def sa_climb(seq, init_coords, n_steps, T_start, T_end, seed):
    rng = random.Random(seed)
    cur = list(init_coords)
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
    seq, init_coords, n_steps, T_start, seed, basin_id = args
    coords, contacts = sa_climb(seq, init_coords, n_steps, T_start, 0.05, seed)
    return basin_id, contacts, coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--start-level", type=int, default=38,
                   help="Pick stepping-stones at this contact level")
    p.add_argument("--n-steps", type=int, default=600_000)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    seq = bench.sequence
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_contacts = verify(seq, [tuple(c) for c in sub["coords"]]).contacts

    # Load stepping-stones at requested level. Filter to distinct contact sets.
    ss = json.loads(open(ROOT / "diagnostics/stepping_stones.json").read())
    sd = ss.get(args.sequence, {})
    pool = sd.get(str(args.start_level), [])
    if not pool:
        print(f"no stepping-stones at level {args.start_level}")
        return 1
    print(f"=== Multi-basin climb on {args.sequence} ===")
    print(f"  current submission: {init_contacts} (record {bench.best_known})")
    print(f"  using {len(pool)} stepping-stones at level {args.start_level}")

    # Dedupe by contact set
    distinct = {}
    for f in pool:
        coords = [tuple(c) for c in f]
        cs = contact_set(seq, coords)
        if cs not in distinct:
            distinct[cs] = coords
    print(f"  {len(distinct)} distinct basins")
    sys.stdout.flush()

    work = []
    for i, (cs, coords) in enumerate(distinct.items()):
        for T0 in (0.5, 1.0, 1.5):
            work.append((seq, coords, args.n_steps, T0, i * 1000 + int(T0 * 100), i))
    print(f"  {len(work)} climb trials ({len(distinct)} basins × 3 temps)")
    sys.stdout.flush()

    t0 = time.monotonic()
    best_overall = init_contacts
    best_coords = None
    by_basin = {}
    by_basin_best_fold: dict[int, tuple[int, list]] = {}
    out_dir = Path(f"results/multibasin_{args.sequence}_lvl{args.start_level}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.processes) as pool_proc:
        for basin_id, contacts, coords in pool_proc.imap_unordered(worker, work, chunksize=1):
            by_basin.setdefault(basin_id, []).append(contacts)
            # Track per-basin best fold (save all distinct top folds)
            prev = by_basin_best_fold.get(basin_id)
            if prev is None or contacts > prev[0]:
                by_basin_best_fold[basin_id] = (contacts, coords)
                # Save fold per-basin if it reaches a meaningful threshold
                if contacts >= init_contacts:
                    fold_path = out_dir / f"basin_{basin_id:03d}_c{contacts}.json"
                    write_submission(
                        Fold(sequence=seq, coords=coords),
                        args.sequence, fold_path,
                    )
            if contacts > best_overall:
                best_overall = contacts
                best_coords = coords
                print(f"  basin {basin_id}: contacts={contacts} [BEST overall, target={args.target}]", flush=True)
                if best_overall >= args.target:
                    print(f"  RECORD CANDIDATE — stopping pool", flush=True)
                    pool_proc.terminate()
                    break

    wall = time.monotonic() - t0
    # Per-basin max
    print(f"\nWall: {wall:.0f}s")
    print(f"Per-basin best contacts (top-15):")
    pb = sorted([(max(v), bid) for bid, v in by_basin.items()], reverse=True)
    for c, bid in pb[:15]:
        print(f"  basin {bid}: max={c}")

    print(f"\nOverall best: {best_overall} (record {bench.best_known}, was {init_contacts})")
    if best_coords and best_overall > init_contacts:
        out = Path(f"results/{args.sequence}_multibasin.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=seq, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: {init_contacts} -> {best_overall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
