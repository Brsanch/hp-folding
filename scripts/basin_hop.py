"""Basin hopping: perturb the current best fold then re-optimize.

Strategy: take the current submitted fold, apply N random moves (the
"shock") to push it out of its basin, then run Wang-Landau from the
shocked state. Iterate.

Many seeds in parallel — each samples a different perturbation. The
hope: one of the perturbations lands in the basin of a record-class
fold.
"""

from __future__ import annotations

import argparse
import json
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
from hp_folding.wang_landau import wang_landau


def shock(coords, n_shocks: int, rng: random.Random) -> list:
    """Apply n_shocks random moves to perturb the fold."""
    cur = list(coords)
    moves = (corner_flip, crankshaft, pivot, segment_rotate, end_flip)
    for _ in range(n_shocks):
        rng.shuffle(list(moves))
        for fn in moves:
            r = fn(cur, rng)
            if r is not None:
                cur = r
                break
    return cur


def worker(args):
    seq, init_coords, n_shocks, time_budget, seed, sequence_id, target = args
    rng = random.Random(seed)
    shocked = shock(init_coords, n_shocks, rng)
    shocked_contacts = count_contacts(seq, shocked)
    t0 = time.monotonic()
    res = wang_landau(
        seq,
        initial_coords=shocked,
        time_budget_seconds=time_budget,
        rng_seed=seed,
    )
    return (
        seed,
        n_shocks,
        shocked_contacts,
        res.best_contacts,
        res.best_coords,
        time.monotonic() - t0,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=16)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--time-budget", type=float, default=120.0)
    p.add_argument("--shock-counts", default="20,40,60,80",
                   help="Comma-separated n_shocks values (cycled across trials)")
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_coords = [tuple(c) for c in sub["coords"]]
    init_contacts = verify(bench.sequence, init_coords).contacts
    print(f"=== Basin hop on {args.sequence} ===")
    print(f"  current submission: {init_contacts} contacts (record {bench.best_known})")
    print(f"  trials: {args.trials} × {args.time_budget}s on {args.processes} procs")
    sys.stdout.flush()

    shock_counts = [int(s) for s in args.shock_counts.split(",")]
    work = []
    for seed in range(args.trials):
        n_shocks = shock_counts[seed % len(shock_counts)]
        work.append((bench.sequence, init_coords, n_shocks, args.time_budget,
                     seed, args.sequence, args.target))

    t0 = time.monotonic()
    best_contacts = init_contacts
    best_coords = init_coords
    with mp.Pool(args.processes) as pool:
        for i, (seed, n_shocks, sc, contacts, coords, wall) in enumerate(
            pool.imap_unordered(worker, work)
        ):
            tag = " [BEST]" if contacts > best_contacts else ""
            print(f"  [{i+1:>3d}/{len(work)}] seed={seed} shocks={n_shocks} "
                  f"shocked_contacts={sc} best={contacts}{tag} wall={wall:.1f}s",
                  flush=True)
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = coords
                if args.target is not None and best_contacts >= args.target:
                    pool.terminate()
                    break

    wall = time.monotonic() - t0
    print(f"\nTotal walltime: {wall:.1f}s")
    print(f"Best: {best_contacts} (started at {init_contacts}, record {bench.best_known})")

    if best_contacts > init_contacts and best_coords:
        out = Path(f"results/{args.sequence}_basin_hop_best.json")
        out.parent.mkdir(exist_ok=True)
        fold = Fold(sequence=bench.sequence, coords=best_coords)
        write_submission(fold, args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: "
                  f"{init_contacts} -> {best_contacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
