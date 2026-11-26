"""Replica-exchange Monte Carlo runner for HP folding.

Example:
    python scripts/run_replica_exchange.py --sequence S5 \\
        --iterations 500000 --temperatures 0.2 0.5 1.0 2.0 4.0 \\
        --exchange-every 1000 --move-strategy pivot_heavy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS
from hp_folding.io import write_submission
from hp_folding.moves import MOVE_STRATEGIES, make_move_fn
from hp_folding.search import replica_exchange


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="S5", choices=list(BENCHMARKS))
    parser.add_argument("--iterations", type=int, default=500_000)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument("--exchange-every", type=int, default=1000)
    parser.add_argument(
        "--move-strategy",
        default="pivot_heavy",
        choices=list(MOVE_STRATEGIES),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starts", type=int, default=4,
                        help="independent PT runs to launch and take the best of")
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bench = BENCHMARKS[args.sequence]
    weights = MOVE_STRATEGIES[args.move_strategy]
    move_fn = make_move_fn(*weights)

    print(f"Replica-exchange MC on {args.sequence} (length {bench.length})")
    print(f"Best known:   {bench.best_known} contacts")
    print(f"Replicas:     {args.temperatures}")
    print(f"Iterations:   {args.iterations}")
    print(f"Exchange:     every {args.exchange_every} steps")
    print(f"Move weights: {args.move_strategy} = {weights}")
    print(f"Starts:       {args.starts}")

    t0 = time.monotonic()
    best = None
    for k in range(args.starts):
        r = replica_exchange(
            bench.sequence,
            temperatures=tuple(args.temperatures),
            iterations=args.iterations,
            exchange_every=args.exchange_every,
            seed=args.seed + k,
            move_fn=move_fn,
        )
        print(
            f"  start {k:2d}: {r.contacts:>3d} contacts  "
            f"(accepted {r.history[0]}/{r.history[1]} exchanges, "
            f"{r.walltime_seconds:.1f}s)"
        )
        if best is None or r.contacts > best.contacts:
            best = r
    assert best is not None
    wall = time.monotonic() - t0

    print(f"\nBest over {args.starts} starts: {best.contacts} contacts in {wall:.2f}s")

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{bench.id}_pt_{args.iterations}_{args.move_strategy}.json"
    write_submission(best.fold, bench.id, path)
    print(f"Wrote {path}")

    log_path = args.output / "replica_exchange.tsv"
    header = "timestamp\tsequence_id\titerations\tstarts\ttemperatures\texchange_every\tmove_strategy\tseed\tcontacts\tbest_known\twalltime_s\tartifact\n"
    if not log_path.exists():
        log_path.write_text(header)
    with log_path.open("a") as f:
        temps = ",".join(str(t) for t in args.temperatures)
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{bench.id}\t{args.iterations}\t"
            f"{args.starts}\t{temps}\t{args.exchange_every}\t{args.move_strategy}\t"
            f"{args.seed}\t{best.contacts}\t{bench.best_known}\t{wall:.2f}\t{path}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
