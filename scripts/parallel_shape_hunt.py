"""Parallel shape-hunt: search MANY 42-cell shape variants for S8
across all CPU cores simultaneously, each with a longer per-shape budget.
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
from hp_folding.shape_search import (
    generate_42cell_corner_shift_shapes,
    generate_42cell_two_corner_shapes,
    search_shape_constrained,
)


def worker(args):
    """Worker: try one shape across multiple seeds."""
    seq, shape_items, timeout_per_seed, n_seeds, p_may_enter, target = args
    name, shape_cells = shape_items
    shape = frozenset(shape_cells)
    best_contacts = -1
    best_coords = None
    for seed in range(n_seeds):
        coords, contacts, _nodes = search_shape_constrained(
            seq, shape,
            timeout_seconds=timeout_per_seed,
            seed=seed, forward_check=False,
            p_may_enter_shape=p_may_enter,
            min_contacts_to_stop=target,
        )
        if coords is not None and contacts > best_contacts:
            best_contacts = contacts
            best_coords = coords
            if target is not None and best_contacts >= target:
                break
    return (name, best_contacts, best_coords)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-per-seed", type=float, default=20.0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--limit", type=int, default=200,
                        help="Max number of shapes to test")
    parser.add_argument("--double", action="store_true",
                        help="Use two-corner shifts (larger catalogue)")
    parser.add_argument("--nonstrict-rectangles", action="store_true",
                        help="Use non-strict large rectangles (6x8, 7x7, 8x6, 7x8, 8x7)")
    parser.add_argument("--target", type=int, default=42,
                        help="Min contacts to stop early")
    args = parser.parse_args()

    bench = BENCHMARKS["S8"]
    if args.nonstrict_rectangles:
        from hp_folding.shape_search import rectangle_shape
        shapes = {
            "7x7": rectangle_shape(7, 7),
            "6x8": rectangle_shape(6, 8),
            "8x6": rectangle_shape(8, 6),
            "7x8": rectangle_shape(7, 8),
            "8x7": rectangle_shape(8, 7),
        }
    elif args.double:
        shapes = generate_42cell_two_corner_shapes()
    else:
        shapes = generate_42cell_corner_shift_shapes()

    # Truncate to limit
    items = list(shapes.items())[: args.limit]
    # Convert frozenset to list for pickling
    items_picklable = [(n, list(s)) for n, s in items]

    print(f"Testing {len(items)} shapes across {args.processes} processes", flush=True)
    print(f"  timeout_per_seed={args.timeout_per_seed}s, n_seeds={args.n_seeds}", flush=True)
    print(f"  expected max wall: ~{args.timeout_per_seed * args.n_seeds * len(items) / args.processes:.0f}s", flush=True)

    p_may_enter = args.nonstrict_rectangles
    work_args = [
        (bench.sequence, item, args.timeout_per_seed, args.n_seeds,
         p_may_enter, args.target)
        for item in items_picklable
    ]
    # For non-strict rectangles, each "shape" gets many more seeds (one per process)
    if args.nonstrict_rectangles:
        expanded = []
        for item in items_picklable:
            for seed_offset in range(8):  # 8 separate "trials" per shape
                expanded.append(
                    (bench.sequence, item, args.timeout_per_seed, 1,
                     p_may_enter, args.target)
                )
        work_args = expanded

    t0 = time.monotonic()
    best_overall = -1
    best_coords = None
    best_name = None
    with mp.Pool(args.processes) as pool:
        for i, (name, contacts, coords) in enumerate(
            pool.imap_unordered(worker, work_args)
        ):
            if coords is not None:
                print(f"  [{i+1:>3d}/{len(items)}] {name}: {contacts} contacts",
                      flush=True)
                if contacts > best_overall:
                    best_overall = contacts
                    best_coords = coords
                    best_name = name
                    if contacts >= 42:
                        pool.terminate()
                        break

    wall = time.monotonic() - t0
    print(f"\nTotal walltime: {wall:.1f}s", flush=True)
    print(f"Best: {best_overall} ({best_name})", flush=True)
    if best_coords is not None:
        fold = Fold(sequence=bench.sequence, coords=best_coords)
        path = Path("results/S8_parallel_hunt_best.json")
        write_submission(fold, "S8", path)
        print(f"Wrote: {path}", flush=True)
        return 0 if best_overall >= 42 else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
