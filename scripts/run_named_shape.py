"""Run shape-constrained DFS on a sequence with a named non-rectangular shape.

Example:
    python scripts/run_named_shape.py --sequence S9 --shape 7x9_minus_4corners --timeout 600 --min-contacts-to-stop 53
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
    search_shape_constrained,
    shape_6x8_minus_5,
    shape_7x9_minus_4corners,
)
from hp_folding.telemetry import SearchTelemetry, format_report

SHAPES = {
    "7x9_minus_4corners": shape_7x9_minus_4corners,  # S9, 59 cells
    "6x8_minus_5": shape_6x8_minus_5,                # S7, 43 cells
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--shape", required=True, choices=list(SHAPES))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--min-contacts-to-stop", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-p-in-shape", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="Print telemetry report after the run")
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    shape = SHAPES[args.shape]()
    h_count = sum(1 for c in bench.sequence if c == "H")

    print(f"Shape-constrained DFS on {args.sequence} (len {bench.length})")
    print(f"Shape '{args.shape}': {len(shape)} cells, H count: {h_count}, record: {bench.best_known}")
    print(f"Timeout: {args.timeout}s")
    print()

    tel = SearchTelemetry() if args.report else None
    t0 = time.monotonic()
    coords, contacts, nodes = search_shape_constrained(
        bench.sequence,
        shape,
        timeout_seconds=args.timeout,
        min_contacts_to_stop=args.min_contacts_to_stop,
        seed=args.seed,
        p_may_enter_shape=args.allow_p_in_shape,
        telemetry=tel,
    )
    wall = time.monotonic() - t0

    print(f"Walltime: {wall:.2f}s, nodes: {nodes:,}")
    if coords is None:
        print("No valid fold found.")
    else:
        fold = Fold(sequence=bench.sequence, coords=coords)
        v = fold.verify()
        print(f"Verdict: {'VALID' if v.valid else 'INVALID'}, contacts: {v.contacts}")
        print(f"Gap to record: {bench.best_known - v.contacts}")
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"{bench.id}_shape_{args.shape}.json"
        write_submission(fold, bench.id, path)
        print(f"Wrote: {path}")

    if tel is not None:
        print()
        print(format_report(tel, max_depth=len(bench.sequence)))
    return 0 if coords is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
