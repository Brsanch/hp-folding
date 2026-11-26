"""Basin-hopping (iterated SA with reheat) runner for HP folding."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS
from hp_folding.io import write_submission
from hp_folding.moves import MOVE_STRATEGIES, make_move_fn
from hp_folding.search import basin_hopping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S5", choices=list(BENCHMARKS))
    parser.add_argument("--iterations-per-basin", type=int, default=200_000)
    parser.add_argument("--n-basins", type=int, default=20)
    parser.add_argument("--t-start", type=float, default=3.0)
    parser.add_argument("--t-end", type=float, default=0.02)
    parser.add_argument("--move-strategy", default="pivot_heavy",
                        choices=list(MOVE_STRATEGIES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starts", type=int, default=4,
                        help="Independent basin-hopping chains")
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    weights = MOVE_STRATEGIES[args.move_strategy]
    move_fn = make_move_fn(*weights)

    print(f"Basin hopping on {args.sequence} (length {bench.length}, best-known {bench.best_known})")
    print(f"{args.starts} chains × {args.n_basins} basins × {args.iterations_per_basin:,} iters")
    print(f"Reheat T: {args.t_start} -> {args.t_end}, moves = {args.move_strategy}")

    t0 = time.monotonic()
    best = None
    for k in range(args.starts):
        r = basin_hopping(
            bench.sequence,
            iterations_per_basin=args.iterations_per_basin,
            n_basins=args.n_basins,
            reheat_t_start=args.t_start,
            reheat_t_end=args.t_end,
            seed=args.seed + k,
            move_fn=move_fn,
        )
        print(f"  chain {k}: {r.contacts} contacts ({r.walltime_seconds:.1f}s)")
        if best is None or r.contacts > best.contacts:
            best = r
    assert best is not None
    wall = time.monotonic() - t0
    print(f"\nBest: {best.contacts} contacts in {wall:.2f}s")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{bench.id}_basin_hopping.json"
    write_submission(best.fold, bench.id, path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
