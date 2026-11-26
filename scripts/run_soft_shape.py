"""Soft-shape SA runner: shape is a bias, not a hard constraint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS
from hp_folding.io import write_submission
from hp_folding.moves import MOVE_STRATEGIES, make_move_fn
from hp_folding.search import soft_shape_sa
from hp_folding.shape_search import rectangle_shape

SHAPES = {
    "5x5": (5, 5), "6x7": (6, 7), "7x6": (7, 6), "7x7": (7, 7),
    "7x8": (7, 8), "8x7": (8, 7), "8x8": (8, 8), "8x9": (8, 9),
    "9x8": (9, 8), "6x8": (6, 8), "3x8": (3, 8), "4x7": (4, 7),
    "9x7": (9, 7), "7x9": (7, 9),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--shape", required=True, choices=list(SHAPES))
    parser.add_argument("--iterations", type=int, default=500_000)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--h-out-penalty", type=float, default=1.0)
    parser.add_argument("--p-in-penalty", type=float, default=0.2)
    parser.add_argument("--t-start", type=float, default=3.0)
    parser.add_argument("--t-end", type=float, default=0.02)
    parser.add_argument("--move-strategy", default="pivot_heavy",
                        choices=list(MOVE_STRATEGIES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    bench = BENCHMARKS[args.sequence]
    shape_r, shape_c = SHAPES[args.shape]
    shape = rectangle_shape(shape_r, shape_c)
    move_fn = make_move_fn(*MOVE_STRATEGIES[args.move_strategy])

    print(f"Soft-shape SA on {args.sequence} (len {bench.length})")
    print(f"Shape {args.shape}: {len(shape)} cells, record = {bench.best_known}")
    print(f"Penalties: H-out={args.h_out_penalty}, P-in={args.p_in_penalty}")
    print(f"{args.starts} starts × {args.iterations:,} iterations")

    t0 = time.monotonic()
    best = None
    for k in range(args.starts):
        r = soft_shape_sa(
            bench.sequence, shape,
            iterations=args.iterations,
            t_start=args.t_start, t_end=args.t_end,
            h_out_penalty=args.h_out_penalty,
            p_in_penalty=args.p_in_penalty,
            seed=args.seed + k,
            move_fn=move_fn,
        )
        print(f"  start {k}: {r.contacts} contacts ({r.walltime_seconds:.1f}s)")
        if best is None or r.contacts > best.contacts:
            best = r
    assert best is not None
    wall = time.monotonic() - t0

    print(f"\nBest: {best.contacts} contacts in {wall:.2f}s (record {bench.best_known})")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{bench.id}_soft_{args.shape}.json"
    write_submission(best.fold, bench.id, path)
    print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
