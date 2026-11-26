"""Produce diagnostic pictures for a saved submission JSON.

Example:
    python scripts/visualize.py results/S5_best.json
    python scripts/visualize.py results/*.json --grid figures/s5_grid.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold, verify
from hp_folding.io import read_submission
from hp_folding.viz import diagnostic_panel, plot_fold, plot_fold_grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("figures"),
        help="Directory for per-fold diagnostic figures",
    )
    parser.add_argument(
        "--grid", type=Path, default=None,
        help="If given, also write a grid comparison figure here",
    )
    return parser.parse_args()


def load(path: Path) -> tuple[Fold, str, int, str]:
    sid, coords = read_submission(path)
    bench = BENCHMARKS[sid]
    fold = Fold(sequence=bench.sequence, coords=coords)
    v = verify(bench.sequence, coords)
    if not v.valid:
        raise RuntimeError(f"{path}: invalid fold — {v.error}")
    return fold, sid, v.contacts, str(path.stem)


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds: list[tuple[Fold, str]] = []
    for p in args.paths:
        fold, sid, contacts, stem = load(p)
        bench = BENCHMARKS[sid]
        label = f"{stem}\n{contacts} contacts / {bench.best_known} known"
        folds.append((fold, label))
        out = args.out_dir / f"{stem}.pdf"
        diagnostic_panel(
            fold, best_known=bench.best_known, history=[], path=out,
            subtitle=f"{sid}  —  {stem}",
        )
        plot_fold(
            fold, path=args.out_dir / f"{stem}_fold.png",
            title=f"{sid} — {stem} — {contacts} contacts",
        )
        print(f"  wrote {out}")

    if args.grid and folds:
        plot_fold_grid(folds, path=args.grid)
        print(f"  wrote grid {args.grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
