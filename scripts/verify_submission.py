"""Verify a submission JSON file against the CAISc rules.

Example:
    python scripts/verify_submission.py results/S5_sa.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, verify
from hp_folding.io import read_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    sequence_id, coords = read_submission(args.path)
    if sequence_id not in BENCHMARKS:
        print(f"ERROR: unknown sequence_id {sequence_id!r}")
        return 2
    bench = BENCHMARKS[sequence_id]
    verdict = verify(bench.sequence, coords)

    print(f"File:       {args.path}")
    print(f"Sequence:   {bench.id} (length {bench.length})")
    print(f"Valid:      {verdict.valid}")
    if not verdict.valid:
        print(f"Error:      {verdict.error}")
        return 1
    print(f"Contacts:   {verdict.contacts}")
    print(f"Best known: {bench.best_known} ({bench.note})")
    print(f"Rg^2:       {verdict.radius_of_gyration_squared:.4f}")
    if verdict.contacts > bench.best_known:
        print(f"\n*** NEW RECORD: {verdict.contacts} > {bench.best_known} ***")
    elif verdict.contacts == bench.best_known:
        print("\nMatches current best known.")
    else:
        print(
            f"\nBelow best known by {bench.best_known - verdict.contacts} contacts."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
