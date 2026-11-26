"""Run shape-constrained SAW search on a benchmark sequence.

Example:
    python scripts/run_shape_search.py --sequence S5 --shape 5x5 --timeout 120
"""

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
)

SHAPES = {
    "5x5": lambda: rectangle_shape(5, 5),    # 25 cells (S5)
    "4x6": lambda: rectangle_shape(4, 6),    # 24 cells
    "6x5": lambda: rectangle_shape(6, 5),    # 30 cells
    "3x9": lambda: rectangle_shape(3, 9),    # 27 cells
    "2x11": lambda: rectangle_shape(2, 11),  # 22 cells (S6)
    "3x8": lambda: rectangle_shape(3, 8),    # 24 cells (S6 + slack)
    "4x7": lambda: rectangle_shape(4, 7),    # 28 cells (S6 + slack)
    "6x7": lambda: rectangle_shape(6, 7),    # 42 cells (S8)
    "7x6": lambda: rectangle_shape(7, 6),    # 42 cells (S8 alt orientation)
    "6x8": lambda: rectangle_shape(6, 8),    # 48 cells (S8 + slack)
    "7x7": lambda: rectangle_shape(7, 7),    # 49 cells (S8 + slack)
    "7x8": lambda: rectangle_shape(7, 8),    # 56 cells (S10)
    "8x7": lambda: rectangle_shape(8, 7),    # 56 cells
    "8x8": lambda: rectangle_shape(8, 8),    # 64 cells (S10 + slack)
    "8x9": lambda: rectangle_shape(8, 9),    # 72 cells
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S5", choices=list(BENCHMARKS))
    parser.add_argument("--shape", default="5x5", choices=list(SHAPES))
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Wall-clock seconds")
    parser.add_argument("--min-contacts-to-stop", type=int, default=None,
                        help="Stop as soon as a fold with at least this many contacts is found")
    parser.add_argument("--allow-p-in-shape", action="store_true",
                        help="Allow P residues inside the target shape (non-strict mode)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    shape = SHAPES[args.shape]()
    h_count = sum(1 for c in bench.sequence if c == "H")

    print(f"Shape-constrained search on {args.sequence} (length {bench.length})")
    print(f"H residues in sequence: {h_count}")
    print(f"Target shape '{args.shape}': {len(shape)} cells")
    print(f"Best known (open): {bench.best_known} contacts")
    print(f"Timeout: {args.timeout}s")

    if len(shape) != h_count and not args.allow_p_in_shape:
        print(f"\nNOTE: shape size ({len(shape)}) != H count ({h_count}) —")
        print("strict 1:1 packing isn't possible. Use --allow-p-in-shape for non-strict.")
        return 2

    t0 = time.monotonic()
    coords, best_contacts, nodes = search_shape_constrained(
        bench.sequence,
        shape,
        timeout_seconds=args.timeout,
        min_contacts_to_stop=args.min_contacts_to_stop,
        p_may_enter_shape=args.allow_p_in_shape,
    )
    wall = time.monotonic() - t0

    print(f"\nDFS walltime:  {wall:.2f}s")
    print(f"Nodes expanded: {nodes:,}")
    print(f"Best found:    {best_contacts} contacts")

    if coords is None:
        print("No valid full fold found within timeout.")
        return 1

    fold = Fold(sequence=bench.sequence, coords=coords)
    verdict = fold.verify()
    print(f"Verdict:       {'VALID' if verdict.valid else 'INVALID'}")
    if not verdict.valid:
        print(f"  Error: {verdict.error}")
        return 1
    print(f"Contacts:      {verdict.contacts} (gap to record: {bench.best_known - verdict.contacts})")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{bench.id}_shape_{args.shape}.json"
    write_submission(fold, bench.id, path)
    print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
