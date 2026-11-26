"""Sweep concave 42-cell shape variants for S8.

Generates a catalogue of 42-cell concave shapes (6×7 rectangle with one
edge cell removed and one external cell added, creating an inlet) and
runs shape-constrained DFS on each. Reports contact counts and where
each search bottoms out, so we can see whether any inlet position
unlocks the chain for S8.
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
    generate_42cell_concave_shapes,
    generate_42cell_double_concave_shapes,
    search_shape_constrained,
)
from hp_folding.telemetry import SearchTelemetry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S8", choices=list(BENCHMARKS))
    parser.add_argument("--timeout-per-shape", type=float, default=30.0)
    parser.add_argument("--min-contacts-to-stop", type=int, default=None)
    parser.add_argument("--double", action="store_true",
                        help="Also include 2-concavity variants")
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    shapes = dict(generate_42cell_concave_shapes())
    if args.double:
        shapes.update(generate_42cell_double_concave_shapes())

    print(f"Concave-shape sweep on {args.sequence} (len {bench.length}, rec {bench.best_known})")
    print(f"{len(shapes)} shape variants, {args.timeout_per_shape}s each\n")

    results = []
    best_overall = None
    for name, shape in shapes.items():
        if sum(1 for c in bench.sequence if c == "H") != len(shape):
            print(f"  {name:24s}: shape size {len(shape)} != H count, skip")
            continue

        tel = SearchTelemetry()
        t0 = time.monotonic()
        coords, contacts, nodes = search_shape_constrained(
            bench.sequence, shape,
            timeout_seconds=args.timeout_per_shape,
            min_contacts_to_stop=args.min_contacts_to_stop,
            telemetry=tel,
        )
        wall = time.monotonic() - t0
        max_depth = max(tel.anchor_max_depth.values()) if tel.anchor_max_depth else -1
        status = f"{contacts:>3d} contacts" if coords is not None else f"no fold (max depth {max_depth})"
        print(f"  {name:24s}: {status}  ({nodes:>10,} nodes, {wall:.1f}s)")
        results.append((name, contacts, nodes, wall, coords))

        if coords is not None and (best_overall is None or contacts > best_overall[1]):
            best_overall = (name, contacts, coords)

    print()
    if best_overall is not None:
        name, contacts, coords = best_overall
        gap = bench.best_known - contacts
        print(f"Best: {contacts} contacts on {name} (gap to record: {gap})")
        fold = Fold(sequence=bench.sequence, coords=coords)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"{bench.id}_concave_{name}.json"
        write_submission(fold, bench.id, path)
        print(f"Wrote: {path}")
        return 0
    else:
        print("No concave shape admitted a valid fold.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
