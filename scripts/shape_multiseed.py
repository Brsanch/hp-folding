"""Multi-seed shape-constrained DFS: tries many randomised neighbour orderings
until it finds a valid fold matching the shape constraint.

Useful when strict packing is theoretically feasible but a single fixed DFS
ordering happens to get stuck in a dead region of the tree first."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.shape_search import (
    rectangle_shape,
    search_shape_constrained,
    shape_6x7_plus_1,
    shape_7x7_minus_6,
    shape_7x8_plus_3,
)

SHAPES = {
    "5x5": lambda: rectangle_shape(5, 5),
    "6x7": lambda: rectangle_shape(6, 7),
    "7x6": lambda: rectangle_shape(7, 6),
    "7x8": lambda: rectangle_shape(7, 8),
    "8x7": lambda: rectangle_shape(8, 7),
    "7x7": lambda: rectangle_shape(7, 7),
    "6x8": lambda: rectangle_shape(6, 8),
    "8x8": lambda: rectangle_shape(8, 8),
    "3x8": lambda: rectangle_shape(3, 8),
    "2x11": lambda: rectangle_shape(2, 11),
    "6x7+1_right": lambda: shape_6x7_plus_1("right"),
    "6x7+1_top": lambda: shape_6x7_plus_1("top"),
    "6x7+1_left": lambda: shape_6x7_plus_1("left"),
    "6x7+1_bottom": lambda: shape_6x7_plus_1("bottom"),
    "7x7-6": lambda: shape_7x7_minus_6(),
    "7x8+3": lambda: shape_7x8_plus_3(),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--shape", required=True, choices=list(SHAPES))
    parser.add_argument("--seeds", type=int, default=10, help="Number of DFS restarts")
    parser.add_argument("--timeout-per-seed", type=float, default=30.0)
    parser.add_argument("--min-contacts-to-stop", type=int, default=None)
    parser.add_argument("--allow-p-in-shape", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    bench = BENCHMARKS[args.sequence]
    shape = SHAPES[args.shape]()

    print(f"Multi-seed shape-constrained DFS on {args.sequence} (len {bench.length})")
    print(f"Shape {args.shape}: {len(shape)} cells, best known = {bench.best_known}")
    print(f"{args.seeds} seeds × {args.timeout_per_seed}s each")

    best_coords = None
    best_contacts = -1
    total_nodes = 0
    t0 = time.monotonic()

    for seed in range(args.seeds):
        coords, contacts, nodes = search_shape_constrained(
            bench.sequence,
            shape,
            timeout_seconds=args.timeout_per_seed,
            min_contacts_to_stop=args.min_contacts_to_stop,
            p_may_enter_shape=args.allow_p_in_shape,
            seed=seed,
        )
        total_nodes += nodes
        if coords is not None and contacts > best_contacts:
            best_coords = list(coords)
            best_contacts = contacts
            print(f"  seed {seed}: {contacts} contacts (nodes={nodes:,})")
            if args.min_contacts_to_stop is not None and contacts >= args.min_contacts_to_stop:
                break
        else:
            print(f"  seed {seed}: no fold (nodes={nodes:,})")

    wall = time.monotonic() - t0
    print(f"\nTotal nodes: {total_nodes:,}, walltime: {wall:.2f}s")

    if best_coords is None:
        print("No valid fold found across any seed.")
        return 1

    fold = Fold(sequence=bench.sequence, coords=best_coords)
    print(f"Best: {best_contacts} contacts (gap to record: {bench.best_known - best_contacts})")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{bench.id}_shape_{args.shape}_multiseed.json"
    write_submission(fold, bench.id, path)
    print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
