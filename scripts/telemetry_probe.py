"""Run a shape-constrained DFS with telemetry enabled and print a diagnostic
report. Useful for understanding where the search spends its compute and
which anchors/depths dominate the exhaustion pattern.

Example:
    # Success case (S5 on 5x5) — fast, should find 23 contacts
    python scripts/telemetry_probe.py --sequence S5 --shape 5x5 --timeout 5

    # Failure case (S8 on 6x7) — exhausts without finding a fold
    python scripts/telemetry_probe.py --sequence S8 --shape 6x7 --timeout 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS
from hp_folding.shape_search import rectangle_shape, search_shape_constrained
from hp_folding.telemetry import SearchTelemetry, format_report

SHAPES = {
    "5x5": (5, 5), "6x7": (6, 7), "7x7": (7, 7),
    "7x8": (7, 8), "8x8": (8, 8), "3x8": (3, 8), "4x7": (4, 7),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S5", choices=list(BENCHMARKS))
    parser.add_argument("--shape", default="5x5", choices=list(SHAPES))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--allow-p-in-shape", action="store_true")
    parser.add_argument(
        "--min-contacts-to-stop", type=int, default=None,
        help="If set, stop when a fold with at least this many contacts is found",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    shape = rectangle_shape(*SHAPES[args.shape])
    h_count = sum(1 for c in bench.sequence if c == "H")

    tel = SearchTelemetry()
    print(f"Probe on {args.sequence} (length {bench.length})")
    print(f"Shape {args.shape}: {len(shape)} cells")
    print(f"H count: {h_count}")
    print(f"Record: {bench.best_known}")
    print(f"Timeout: {args.timeout}s")
    print()

    t0 = time.monotonic()
    coords, contacts, nodes = search_shape_constrained(
        bench.sequence, shape,
        timeout_seconds=args.timeout,
        min_contacts_to_stop=args.min_contacts_to_stop,
        p_may_enter_shape=args.allow_p_in_shape,
        telemetry=tel,
    )
    wall = time.monotonic() - t0

    print(f"Walltime: {wall:.2f}s")
    print(f"Nodes expanded: {nodes:,}")
    print(f"Best contacts: {contacts}")
    print(f"Fold found: {'YES' if coords is not None else 'NO'}")
    print()

    print(format_report(tel, max_depth=len(bench.sequence)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
