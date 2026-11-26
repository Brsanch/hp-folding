"""Prepare the final CAISc Verifiable Track submission bundle.

Selects the best valid fold found for each submittable sequence
(S5--S10) and copies it to ``submissions/S<id>.json`` with a consistent
filename. Prints a summary table suitable for pasting into a submission
note.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, SUBMITTABLE_IDS, verify
from hp_folding.io import read_submission


def _best_per_sequence(results_dir: Path) -> dict[str, tuple[int, Path]]:
    best: dict[str, tuple[int, Path]] = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            sid, coords = read_submission(p)
        except Exception:
            continue
        if sid not in SUBMITTABLE_IDS:
            continue
        v = verify(BENCHMARKS[sid].sequence, coords)
        if not v.valid:
            continue
        if sid not in best or v.contacts > best[sid][0]:
            best[sid] = (v.contacts, p)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results"),
        help="Directory containing candidate submission JSON files",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("submissions"),
        help="Output directory for packaged submissions",
    )
    args = parser.parse_args()

    best = _best_per_sequence(args.results)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{'seq':4s} {'contacts':>8s}  {'record':>6s}  {'gap':>4s}  status")
    print("-" * 55)
    for sid in SUBMITTABLE_IDS:
        bench = BENCHMARKS[sid]
        if sid not in best:
            print(f"{sid:4s}  {'—':>8s}  {bench.best_known:>6d}  {'—':>4s}  MISSING")
            continue
        c, src = best[sid]
        gap = bench.best_known - c
        if c > bench.best_known:
            status = f"NEW RECORD (+{c - bench.best_known})"
        elif c == bench.best_known:
            status = "tied with record"
        else:
            status = f"{gap} below record"
        dst = args.out / f"{sid}.json"
        shutil.copy2(src, dst)
        print(f"{sid:4s}  {c:>8d}  {bench.best_known:>6d}  {gap:>4d}  {status}")

    print(f"\nBundle written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
