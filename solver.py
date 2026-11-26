"""Top-level baseline solver entry point.

Provides a plain multi-start simulated-annealing baseline built on the core
modules under ``src/hp_folding/`` (verifier, benchmark sequences, move
operators, and submission I/O). Extend or replace :func:`solve` to experiment
with alternative search strategies, move sets, or annealing schedules.

Run:
    python solver.py --sequence S5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hp_folding import BENCHMARKS
from hp_folding.io import write_submission
from hp_folding.search import AnnealResult, multi_start


def solve(
    sequence_id: str,
    *,
    iterations: int = 200_000,
    starts: int = 8,
    t_start: float = 2.5,
    t_end: float = 0.05,
    base_seed: int = 0,
) -> AnnealResult:
    """Baseline solver: multi-start simulated annealing.

    Rewrite or replace this function to try new search strategies.
    """
    bench = BENCHMARKS[sequence_id]
    return multi_start(
        bench.sequence,
        n_starts=starts,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        base_seed=base_seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S5", choices=list(BENCHMARKS))
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    bench = BENCHMARKS[args.sequence]
    print(f"Solving {bench.id} (length {bench.length}).")
    print(f"Best known: {bench.best_known} H-H contacts.")

    t0 = time.monotonic()
    result = solve(
        args.sequence,
        iterations=args.iterations,
        starts=args.starts,
        base_seed=args.seed,
    )
    wall = time.monotonic() - t0

    print(f"Best found: {result.contacts} contacts in {wall:.2f}s.")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{bench.id}_best.json"
    write_submission(result.fold, bench.id, path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
