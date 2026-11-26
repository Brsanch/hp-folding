"""Seeded shape DFS on S9 with prescribed first-run placements.

S9 starts with a 4-H run (residues 0-3), then 4-P bridge, then a 12-H run.
Rather than letting DFS try to find a good starting placement for these
from scratch, we enumerate a handful of prescribed placements for the
first 20-odd residues and let DFS complete each.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.shape_search import (
    search_shape_constrained,
    shape_7x9_minus_4corners,
)


def prescribed_seeds() -> list[tuple[str, list[tuple[int, int]]]]:
    """Return (name, seed-coords) pairs for the first 4 residues of S9.

    S9 starts HHHHPPPP... — 4 H's then 4 P's. The 4 H's must be in the
    7x9-minus-4-corners shape; the 4 P's must be outside.

    We try several prescribed orientations for the first 4 H-residues —
    each is a length-3 SAW path inside the shape — so DFS starts its
    search with the initial run already placed."""
    seeds = []
    # Top edge, row 0, cols 1-4: (0,1)->(0,2)->(0,3)->(0,4)
    seeds.append(("top_row_h",
                  [(0, 1), (0, 2), (0, 3), (0, 4)]))
    # Top edge, cols 2-5
    seeds.append(("top_row_h_shift",
                  [(0, 2), (0, 3), (0, 4), (0, 5)]))
    # First column (col 1, rows 0-3)
    seeds.append(("left_col_v",
                  [(1, 0), (2, 0), (3, 0), (4, 0)]))
    # L-shape at top-left
    seeds.append(("L_top_left",
                  [(0, 1), (1, 1), (2, 1), (2, 2)]))
    # Descending snake top-right
    seeds.append(("snake_TR",
                  [(0, 7), (1, 7), (2, 7), (2, 6)]))
    # Snake bottom-left
    seeds.append(("snake_BL",
                  [(5, 1), (5, 2), (5, 3), (5, 4)]))
    # Middle-row horizontal
    seeds.append(("mid_row",
                  [(3, 2), (3, 3), (3, 4), (3, 5)]))
    return seeds


def main() -> int:
    bench = BENCHMARKS["S9"]
    shape = shape_7x9_minus_4corners()
    target = 53  # record-matching

    print(f"Seeded DFS on S9 (len {bench.length})")
    print(f"Shape 7x9_minus_4corners: {len(shape)} cells")
    print(f"Record: {bench.best_known}  Target: {target}\n")

    best_fold = None
    best_contacts = -1
    total_nodes = 0

    for name, seed_coords in prescribed_seeds():
        print(f"[{name}] seed = {seed_coords}")
        t0 = time.monotonic()
        try:
            coords, contacts, nodes = search_shape_constrained(
                bench.sequence, shape,
                timeout_seconds=60.0,
                min_contacts_to_stop=target,
                initial_coords=seed_coords,
            )
        except ValueError as e:
            print(f"  seed rejected: {e}")
            continue
        wall = time.monotonic() - t0
        total_nodes += nodes
        if coords is None:
            print(f"  no fold (nodes={nodes:,}, {wall:.1f}s)")
        else:
            print(f"  {contacts} contacts (nodes={nodes:,}, {wall:.1f}s)")
            if contacts > best_contacts:
                best_contacts = contacts
                best_fold = Fold(sequence=bench.sequence, coords=coords)
            if contacts >= target:
                break

    print(f"\nBest overall: {best_contacts}, total nodes: {total_nodes:,}")
    if best_fold is not None:
        Path("results").mkdir(exist_ok=True)
        path = Path("results/S9_seeded.json")
        write_submission(best_fold, "S9", path)
        print(f"Wrote {path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
