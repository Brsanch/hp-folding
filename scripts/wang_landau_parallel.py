"""Parallel Wang-Landau hunt for HP ground states.

Spawns N workers (one per core) running Wang-Landau with different seeds,
collects the best fold found across all workers, and writes a submission.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.wang_landau import wang_landau


def worker(args):
    seq, seed, time_budget, flatness, diag_dir, sequence_id, init_coords = args
    t0 = time.monotonic()
    res = wang_landau(
        seq,
        time_budget_seconds=time_budget,
        rng_seed=seed,
        flatness_threshold=flatness,
        initial_coords=init_coords,
    )
    # Persist full telemetry for this trial.
    if diag_dir is not None:
        diag_path = Path(diag_dir) / f"{sequence_id}_wl_seed{seed:03d}.json"
        payload = {
            "sequence_id": sequence_id,
            "seed": seed,
            "best_contacts": res.best_contacts,
            "best_coords": res.best_coords,
            "log_g": {str(k): v for k, v in res.log_g.items()},
            "hist": {str(k): v for k, v in res.hist.items()},
            "log_f_final": res.log_f_final,
            "sweeps": res.sweeps,
            "wall_seconds": res.wall_seconds,
            "best_fold_at_energy": {
                str(k): v for k, v in res.best_fold_at_energy.items()
            },
            "trace": res.trace,
            "move_stats": res.move_stats,
            "trace_columns": [
                "step", "wall_s", "current_contacts", "best_contacts",
                "log_f", "flatness_ratio", "accept_rate_since_last",
            ],
        }
        diag_path.write_text(json.dumps(payload))
    return (
        seed,
        res.best_contacts,
        res.best_coords,
        res.sweeps,
        res.log_f_final,
        time.monotonic() - t0,
        res.move_stats,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S8")
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--time-budget", type=float, default=120.0)
    parser.add_argument("--trials", type=int, default=16,
                        help="Total WL trials (default = processes)")
    parser.add_argument("--flatness", type=float, default=0.80)
    parser.add_argument("--target", type=int, default=None,
                        help="Early stop if best >= target")
    parser.add_argument("--diag-dir", default="diagnostics/wl",
                        help="Directory for per-trial telemetry JSONs")
    parser.add_argument("--init-from", default=None,
                        help="Path to a fold JSON to seed every trial from")
    args = parser.parse_args()

    bench = BENCHMARKS[args.sequence]
    n_trials = args.trials if args.trials else args.processes

    diag_dir = Path(args.diag_dir) if args.diag_dir else None
    if diag_dir is not None:
        diag_dir.mkdir(parents=True, exist_ok=True)

    init_coords = None
    if args.init_from:
        d = json.loads(Path(args.init_from).read_text())
        init_coords = [tuple(c) for c in d["coords"]]
        print(f"  init_from: {args.init_from} ({len(init_coords)} residues)")

    work = [
        (bench.sequence, seed, args.time_budget, args.flatness,
         str(diag_dir) if diag_dir else None, args.sequence,
         init_coords)
        for seed in range(n_trials)
    ]

    print(f"Wang-Landau on {args.sequence} (n={len(bench.sequence)}):")
    print(f"  {n_trials} trials × {args.time_budget}s on {args.processes} procs")
    print(f"  flatness={args.flatness}")
    print(f"  expected max wall: ~{args.time_budget * n_trials / args.processes:.0f}s")
    sys.stdout.flush()

    t0 = time.monotonic()
    best_contacts = -1
    best_coords = None
    best_seed = None

    all_move_stats: list[dict] = []
    with mp.Pool(args.processes) as pool:
        for i, (seed, contacts, coords, sweeps, log_f, wall, move_stats) in enumerate(
            pool.imap_unordered(worker, work)
        ):
            all_move_stats.append(move_stats)
            # Per-class accept rate
            def _rate(ms, k):
                p, a = ms.get(k, (0, 0))
                return (a / p) if p else 0.0
            acc_pull = _rate(move_stats, "pull")
            acc_pivot = _rate(move_stats, "pivot")
            acc_end = _rate(move_stats, "end_flip")
            msg = (
                f"  [{i+1:>3d}/{n_trials}] seed={seed:>3d} "
                f"contacts={contacts} sweeps={sweeps} "
                f"log_f={log_f:.2e} wall={wall:.1f}s "
                f"acc[pull={acc_pull:.2f} pivot={acc_pivot:.2f} end={acc_end:.2f}]"
            )
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = coords
                best_seed = seed
                msg += " [BEST]"
            print(msg, flush=True)
            if args.target is not None and best_contacts >= args.target:
                print(f"  target {args.target} reached — terminating pool", flush=True)
                pool.terminate()
                break

    wall = time.monotonic() - t0
    print(f"\nTotal walltime: {wall:.1f}s")
    print(f"Best: {best_contacts} contacts (seed={best_seed})")

    if best_coords is not None and best_contacts > 0:
        fold = Fold(sequence=bench.sequence, coords=best_coords)
        path = Path(f"results/{args.sequence}_wang_landau_best.json")
        path.parent.mkdir(exist_ok=True)
        write_submission(fold, args.sequence, path)
        print(f"Wrote: {path}")
    return 0 if best_contacts >= (args.target or 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
