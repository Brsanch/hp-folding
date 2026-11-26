"""Post-run analyzer for Wang-Landau telemetry JSONs.

Reads diagnostics/wl/<SEQ>_wl_seed*.json and reports:
  - Per-seed best contacts, sweeps, final log_f
  - Fraction of seeds that reached each sweep count
  - Distribution of best contacts across seeds
  - Per-move-class acceptance rates (averaged across seeds)
  - Best-fold coverage: which energies were ever visited, count of seeds
    that found a fold at that energy (tight energies = likely bottleneck)
  - Candidate restart points: the single best fold seen at each energy across
    all seeds (can be used to seed a follow-up run with narrower energy windows)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_diagnostics(diag_dir: Path, seq_id: str) -> list[dict]:
    files = sorted(diag_dir.glob(f"{seq_id}_wl_seed*.json"))
    return [json.loads(p.read_text()) for p in files]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--diag-dir", default="diagnostics/wl")
    p.add_argument("--sequence", required=True)
    p.add_argument("--dump-restart", default=None,
                   help="If given, write consolidated best-fold-at-energy "
                        "map to this JSON path.")
    args = p.parse_args()

    diags = load_diagnostics(Path(args.diag_dir), args.sequence)
    if not diags:
        print(f"No diagnostics found for {args.sequence}")
        return 1

    print(f"=== WL diagnostics for {args.sequence} ({len(diags)} trials) ===\n")

    # Contact distribution
    bests = [d["best_contacts"] for d in diags]
    print("Best-contacts distribution:")
    for c, n in sorted(Counter(bests).items()):
        bar = "#" * n
        print(f"  {c:>3d}: {n:>3d} {bar}")
    print(f"  max={max(bests)} median={sorted(bests)[len(bests)//2]} "
          f"min={min(bests)}\n")

    # Sweep convergence
    sweeps = [d["sweeps"] for d in diags]
    logs = [d["log_f_final"] for d in diags]
    print("Sweep convergence:")
    print(f"  sweeps: min={min(sweeps)} median={sorted(sweeps)[len(sweeps)//2]} "
          f"max={max(sweeps)}")
    print(f"  log_f_final: min={min(logs):.2e} median={sorted(logs)[len(logs)//2]:.2e} "
          f"max={max(logs):.2e}\n")

    # Per-move-class acceptance rates (seed-averaged)
    print("Move acceptance rates (seed-averaged):")
    tot_p: dict[str, int] = defaultdict(int)
    tot_a: dict[str, int] = defaultdict(int)
    for d in diags:
        for k, (prop, acc) in d.get("move_stats", {}).items():
            tot_p[k] += prop
            tot_a[k] += acc
    for k in ("corner_flip", "crankshaft", "segment_rot", "pivot", "end_flip", "pull", "null"):
        p_, a_ = tot_p[k], tot_a[k]
        rate = a_ / p_ if p_ else 0.0
        print(f"  {k:>9s}: proposed={p_:>12d} accepted={a_:>12d} "
              f"rate={rate:.3f}")
    print()

    # Energy coverage: how many seeds found a fold at each energy (= -contacts)
    energy_to_seed_count: Counter = Counter()
    # consolidated best fold at each energy across all seeds
    best_fold_at_energy: dict[int, list[tuple[int, int]]] = {}
    for d in diags:
        for e_str, coords in d.get("best_fold_at_energy", {}).items():
            e = int(e_str)
            energy_to_seed_count[e] += 1
            # arbitrary: first seed's fold at this energy is fine as a restart
            if e not in best_fold_at_energy:
                best_fold_at_energy[e] = [tuple(c) for c in coords]

    print("Energy coverage (energy = -contacts; lower = better):")
    for e in sorted(energy_to_seed_count):
        n = energy_to_seed_count[e]
        print(f"  E={e:>4d} (contacts={-e:>3d}): {n:>3d} seeds")
    print()

    # Trace summary: for each seed, time at which best_contacts last improved
    print("Time-to-best (s) across seeds (quartiles):")
    times = []
    for d in diags:
        trace = d.get("trace", [])
        # trace rows: (step, wall_s, current, best, log_f, flatness, acc_rate)
        best = -1
        t_best = 0.0
        for row in trace:
            if row[3] > best:
                best = row[3]
                t_best = row[1]
        times.append(t_best)
    times.sort()
    if times:
        q = [times[int(len(times) * f)] for f in (0.0, 0.25, 0.5, 0.75, 0.99)]
        print(f"  min={q[0]:.1f}  Q1={q[1]:.1f}  med={q[2]:.1f}  "
              f"Q3={q[3]:.1f}  max~={q[4]:.1f}\n")

    # Dump restart points if requested
    if args.dump_restart and best_fold_at_energy:
        outpath = Path(args.dump_restart)
        payload = {
            "sequence_id": args.sequence,
            "energy_to_fold": {str(e): v for e, v in best_fold_at_energy.items()},
            "energy_to_seed_count": {str(e): n for e, n in energy_to_seed_count.items()},
        }
        outpath.write_text(json.dumps(payload))
        print(f"Wrote restart points to {outpath} "
              f"({len(best_fold_at_energy)} distinct energies)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
