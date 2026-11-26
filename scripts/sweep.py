"""Small-grid sweep over compute budget, temperature schedule, and move set.

Runs ~6-8 experiments on a chosen benchmark sequence and writes the result
of each to ``results/sweep.tsv``. Each experiment is a multi-start simulated
annealing run with one knob changed relative to the baseline.

Example:
    python scripts/sweep.py --sequence S5
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS
from hp_folding.io import write_submission
from hp_folding.moves import MOVE_STRATEGIES, make_move_fn


@dataclass
class Experiment:
    name: str
    iterations: int
    starts: int
    t_start: float
    t_end: float
    move_strategy: str
    base_seed: int = 0


EXPERIMENTS: list[Experiment] = [
    # Scale compute
    Experiment("baseline",         200_000, 8,   2.5,  0.05,  "mixed"),
    Experiment("compute_10x",      2_000_000, 8, 2.5,  0.05,  "mixed"),
    Experiment("compute_starts_4x", 200_000, 32,  2.5,  0.05,  "mixed"),

    # Temperature schedule
    Experiment("temp_cold",        200_000, 8,   1.0,  0.01,  "mixed"),
    Experiment("temp_hot",         200_000, 8,   5.0,  0.05,  "mixed"),
    Experiment("temp_extreme_end", 200_000, 8,   2.5,  0.001, "mixed"),

    # Move set ablation
    Experiment("pull_only",        200_000, 8,   2.5,  0.05,  "pull_only"),
    Experiment("pull_heavy",       200_000, 8,   2.5,  0.05,  "pull_heavy"),
    Experiment("pivot_heavy",      200_000, 8,   2.5,  0.05,  "pivot_heavy"),

    # Round 3 — combine best levers
    Experiment("combined_best",    2_000_000, 32, 1.0,  0.01,  "pivot_heavy"),
    Experiment("massive_starts",     500_000, 128, 1.0, 0.01, "pivot_heavy"),
    Experiment("pivot_only",         200_000, 8,   2.5,  0.05, "pivot_only"),
    Experiment("pivot_dominant",     200_000, 8,   2.5,  0.05, "pivot_dominant"),
    Experiment("no_pull",            200_000, 8,   2.5,  0.05, "no_pull"),
    Experiment("long_pivot",       5_000_000, 16, 1.0,  0.01,  "pivot_heavy"),
]


def run_one(exp: Experiment, sequence_id: str, out_dir: Path) -> dict[str, object]:
    bench = BENCHMARKS[sequence_id]
    weights = MOVE_STRATEGIES[exp.move_strategy]
    move_fn = make_move_fn(*weights)

    t0 = time.monotonic()
    # Use solver's multi_start but swap in the custom move_fn via an inner call.
    # multi_start hardcodes random_move; we reimplement a tiny multi-start here
    # so we can thread move_fn through.
    from hp_folding.search import simulated_annealing
    best = None
    for k in range(exp.starts):
        r = simulated_annealing(
            bench.sequence,
            iterations=exp.iterations,
            t_start=exp.t_start,
            t_end=exp.t_end,
            seed=exp.base_seed + k,
            move_fn=move_fn,
        )
        if best is None or r.contacts > best.contacts:
            best = r
    wall = time.monotonic() - t0
    assert best is not None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sequence_id}_{exp.name}.json"
    write_submission(best.fold, sequence_id, path)

    return {
        "name": exp.name,
        "sequence_id": sequence_id,
        "contacts": best.contacts,
        "best_known": bench.best_known,
        "gap": bench.best_known - best.contacts,
        "iterations": exp.iterations,
        "starts": exp.starts,
        "t_start": exp.t_start,
        "t_end": exp.t_end,
        "move_strategy": exp.move_strategy,
        "walltime_s": round(wall, 2),
        "rg2": round(best.fold.verify().radius_of_gyration_squared, 4),
        "artifact": str(path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S5", choices=list(BENCHMARKS))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--subset", nargs="*", default=None,
                        help="Run only experiments whose name is in this list")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "sweep.tsv"

    selected = EXPERIMENTS
    if args.subset:
        selected = [e for e in EXPERIMENTS if e.name in args.subset]

    header = (
        "timestamp\tsequence_id\tname\tcontacts\tbest_known\tgap\t"
        "iterations\tstarts\tt_start\tt_end\tmove_strategy\twalltime_s\trg2\tartifact\n"
    )
    if not tsv_path.exists():
        tsv_path.write_text(header)

    bench = BENCHMARKS[args.sequence]
    print(f"Sweep on {args.sequence} (length {bench.length}, best-known {bench.best_known})")
    print(f"{'experiment':20s}  {'contacts':>8s}  {'gap':>5s}  {'walltime':>10s}")
    print("-" * 55)

    for exp in selected:
        r = run_one(exp, args.sequence, out_dir)
        print(f"{r['name']:20s}  {r['contacts']:>8d}  {r['gap']:>5d}  {r['walltime_s']:>9.2f}s")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with tsv_path.open("a") as f:
            f.write(
                f"{ts}\t{r['sequence_id']}\t{r['name']}\t"
                f"{r['contacts']}\t{r['best_known']}\t{r['gap']}\t"
                f"{r['iterations']}\t{r['starts']}\t{r['t_start']}\t{r['t_end']}\t"
                f"{r['move_strategy']}\t{r['walltime_s']}\t{r['rg2']}\t{r['artifact']}\n"
            )
    print(f"\nResults appended to {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
