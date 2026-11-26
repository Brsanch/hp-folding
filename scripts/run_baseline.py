"""Run the baseline simulated-annealing solver on a benchmark sequence.

Example:
    python scripts/run_baseline.py --sequence S5 --iterations 200000 --starts 4

The script prints the best contact count found, writes a submission JSON,
and appends a row to ``results/runs.tsv`` for later analysis.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, SUBMITTABLE_IDS
from hp_folding.io import write_submission
from hp_folding.search import multi_start


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--t-start", type=float, default=2.0)
    parser.add_argument("--t-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="directory for submission and log output (default: results/)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    if args.sequence not in SUBMITTABLE_IDS:
        print(
            f"NOTE: {args.sequence} is a reference instance (proven-optimal "
            f"= {bench.best_known}); solving anyway for validation."
        )

    print(f"Sequence {bench.id} (length {bench.length})")
    print(f"Best known: {bench.best_known} ({bench.note})")
    print(
        f"Running {args.starts} starts of {args.iterations} iters, "
        f"T: {args.t_start} -> {args.t_end}"
    )

    t0 = time.monotonic()
    result = multi_start(
        bench.sequence,
        n_starts=args.starts,
        iterations=args.iterations,
        t_start=args.t_start,
        t_end=args.t_end,
        base_seed=args.seed,
        log_every=args.log_every,
    )
    wall = time.monotonic() - t0

    print(f"\nResult: {result.contacts} contacts (best known: {bench.best_known})")
    print(f"Walltime: {wall:.2f}s, accepted: {result.accepted}")

    args.output.mkdir(parents=True, exist_ok=True)
    sub_path = args.output / f"{bench.id}_sa_{args.iterations}x{args.starts}_seed{args.seed}.json"
    write_submission(result.fold, bench.id, sub_path)
    print(f"Wrote submission: {sub_path}")

    log_path = args.output / "runs.tsv"
    header = "timestamp\tsequence_id\titerations\tstarts\tt_start\tt_end\tseed\tcontacts\tbest_known\twalltime_s\n"
    if not log_path.exists():
        log_path.write_text(header)
    with log_path.open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{bench.id}\t"
            f"{args.iterations}\t{args.starts}\t{args.t_start}\t{args.t_end}\t"
            f"{args.seed}\t{result.contacts}\t{bench.best_known}\t{wall:.3f}\n"
        )

    return 0 if result.contacts >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
