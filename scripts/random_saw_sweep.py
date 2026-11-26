"""Test random-SAW initialisation against straight-line start.

If SA converges to 21 contacts from a straight line independent of other
settings, the initial configuration is the prime suspect for that ceiling.
This sweep starts each SA run from a fresh random SAW instead.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.moves import MOVE_STRATEGIES, make_move_fn
from hp_folding.search import simulated_annealing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S5", choices=list(BENCHMARKS))
    parser.add_argument("--iterations", type=int, default=500_000)
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--move-strategy", default="pivot_heavy",
                        choices=list(MOVE_STRATEGIES))
    parser.add_argument("--t-start", type=float, default=1.5)
    parser.add_argument("--t-end", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    weights = MOVE_STRATEGIES[args.move_strategy]
    move_fn = make_move_fn(*weights)

    print(
        f"Random-SAW-init SA on {args.sequence} (length {bench.length}, "
        f"best-known {bench.best_known})"
    )
    print(
        f"{args.starts} starts × {args.iterations:,} iterations, "
        f"{args.move_strategy}, T: {args.t_start} -> {args.t_end}"
    )

    t0 = time.monotonic()
    best = None
    for k in range(args.starts):
        rng_init = random.Random(args.seed + k)
        init = Fold.random_saw(bench.sequence, rng_init)
        init_v = init.verify()
        if not init_v.valid:
            # Growth failed; fall back to straight line inside this start
            init = Fold.linear(bench.sequence)
        r = simulated_annealing(
            bench.sequence,
            iterations=args.iterations,
            t_start=args.t_start,
            t_end=args.t_end,
            seed=args.seed + k,
            move_fn=move_fn,
            initial_fold=init,
        )
        if best is None or r.contacts > best.contacts:
            best = r
        if (k + 1) % max(1, args.starts // 8) == 0:
            print(f"  start {k+1:>3d}/{args.starts}: best so far = {best.contacts}")

    assert best is not None
    wall = time.monotonic() - t0
    print(f"\nBest: {best.contacts} contacts in {wall:.2f}s")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{bench.id}_random_saw_init.json"
    write_submission(best.fold, bench.id, path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
