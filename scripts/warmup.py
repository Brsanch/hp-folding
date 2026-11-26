"""Warm-up sanity check: HHHHHH on a 2x3 snake achieves 2 contacts.

Run:  python scripts/warmup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import verify


def main() -> int:
    sequence = "HHHHHH"
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    verdict = verify(sequence, coords)
    print(f"Sequence: {sequence}")
    print(f"Coords:   {coords}")
    print(f"Valid:    {verdict.valid}")
    if not verdict.valid:
        print(f"Error:    {verdict.error}")
        return 1
    print(f"Contacts: {verdict.contacts}")
    print(f"Rg^2:     {verdict.radius_of_gyration_squared:.4f}")
    assert verdict.contacts == 2, "warm-up expects exactly 2 H-H contacts"
    print("\nOK: warm-up verifier agrees with the CAISc problem statement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
