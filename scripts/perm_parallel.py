"""Parallel PERM hunt: N workers at different (T, seed) combinations."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.perm import perm


def worker(args):
    seq, temperature, seed, n_tours, timeout = args
    t0 = time.monotonic()
    res = perm(
        seq,
        temperature=temperature,
        n_tours=n_tours,
        seed=seed,
        timeout_seconds=timeout,
        verbose=False,
    )
    return (
        temperature,
        seed,
        res.contacts,
        list(res.fold.coords) if res.contacts > 0 else None,
        res.complete_tours,
        time.monotonic() - t0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="S8")
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--time-budget", type=float, default=240.0)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--n-tours", type=int, default=100_000)
    parser.add_argument("--target", type=int, default=None)
    args = parser.parse_args()

    bench = BENCHMARKS[args.sequence]

    # Temperature sweep: cold to warm. Records usually land at T ∈ [0.3, 0.7].
    temperatures = [0.25, 0.35, 0.45, 0.55, 0.70, 0.90, 1.20, 1.50]
    work = []
    for t in range(args.trials):
        T = temperatures[t % len(temperatures)]
        seed = t
        work.append((bench.sequence, T, seed, args.n_tours, args.time_budget))

    print(f"PERM on {args.sequence} (n={len(bench.sequence)}):")
    print(f"  {len(work)} trials × {args.time_budget}s on {args.processes} procs")
    print(f"  temperatures: {sorted(set(temperatures))}")
    print(f"  expected max wall: ~{args.time_budget * len(work) / args.processes:.0f}s")
    sys.stdout.flush()

    t0 = time.monotonic()
    best_contacts = -1
    best_coords = None
    best_cfg = None

    with mp.Pool(args.processes) as pool:
        for i, (T, seed, contacts, coords, tours, wall) in enumerate(
            pool.imap_unordered(worker, work)
        ):
            msg = (
                f"  [{i+1:>3d}/{len(work)}] T={T:.2f} seed={seed} "
                f"contacts={contacts} tours={tours} wall={wall:.1f}s"
            )
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = coords
                best_cfg = (T, seed)
                msg += " [BEST]"
            print(msg, flush=True)
            if args.target is not None and best_contacts >= args.target:
                pool.terminate()
                break

    wall = time.monotonic() - t0
    print(f"\nTotal walltime: {wall:.1f}s")
    print(f"Best: {best_contacts} contacts (T={best_cfg[0] if best_cfg else None}, seed={best_cfg[1] if best_cfg else None})")
    if best_coords is not None:
        fold = Fold(sequence=bench.sequence, coords=best_coords)
        path = Path(f"results/{args.sequence}_perm_parallel_best.json")
        path.parent.mkdir(exist_ok=True)
        write_submission(fold, args.sequence, path)
        print(f"Wrote: {path}")
    return 0 if best_contacts >= (args.target or 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
