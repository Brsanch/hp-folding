"""Hybrid GPU+CPU HP folder.

Phase 1 (GPU, MLX): K parallel SA chains with pivot+end-flip moves.
Phase 2 (CPU, multiprocessing): pick top folds from phase 1, polish each
  with pull-move Wang-Landau.
Phase 3: pick global best, update submissions/<seq>.json if improved.

Diagnostics from both phases are written to disk.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.verifier import verify
from hp_folding.wang_landau import wang_landau


def gpu_phase(sequence: str, *, K: int, n_steps: int, T_start: float,
              T_end: float, seed: int) -> tuple[list[int], list[list[tuple[int, int]]], float]:
    # Imported lazily so that CPU-only workers don't pull in MLX.
    from hp_folding.mlx_batch import batch_simulated_annealing
    res = batch_simulated_annealing(
        sequence, K=K, n_steps=n_steps,
        T_start=T_start, T_end=T_end, seed=seed,
    )
    return res.best_contacts_per_chain, res.best_coords_per_chain, res.wall_seconds


def cpu_polish_worker(args):
    seq, init_coords, time_budget, seed, sequence_id, diag_dir = args
    t0 = time.monotonic()
    res = wang_landau(
        seq,
        initial_coords=init_coords,
        time_budget_seconds=time_budget,
        rng_seed=seed,
    )
    if diag_dir:
        diag_path = Path(diag_dir) / f"{sequence_id}_polish_seed{seed:03d}.json"
        payload = {
            "sequence_id": sequence_id,
            "seed": seed,
            "init_contacts_estimate": -min(res.best_fold_at_energy.keys())
                                     if res.best_fold_at_energy else None,
            "best_contacts": res.best_contacts,
            "best_coords": res.best_coords,
            "log_g": {str(k): v for k, v in res.log_g.items()},
            "hist": {str(k): v for k, v in res.hist.items()},
            "log_f_final": res.log_f_final,
            "sweeps": res.sweeps,
            "wall_seconds": res.wall_seconds,
            "trace": res.trace,
            "move_stats": res.move_stats,
        }
        diag_path.write_text(json.dumps(payload))
    return seed, res.best_contacts, res.best_coords, time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--gpu-K", type=int, default=2048)
    p.add_argument("--gpu-steps", type=int, default=80_000)
    p.add_argument("--gpu-T-start", type=float, default=4.0)
    p.add_argument("--gpu-T-end", type=float, default=0.05)
    p.add_argument("--gpu-seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=32, help="How many GPU folds to polish on CPU")
    p.add_argument("--cpu-procs", type=int, default=16)
    p.add_argument("--cpu-time-per", type=float, default=60.0)
    p.add_argument("--diag-dir", default="diagnostics/hybrid")
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    diag_dir = Path(args.diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Hybrid run on {args.sequence} (n={bench.length}, record={bench.best_known}) ===")

    # GPU phase
    print(f"\n[GPU] K={args.gpu_K}, n_steps={args.gpu_steps}, "
          f"T={args.gpu_T_start}->{args.gpu_T_end}")
    sys.stdout.flush()
    bcs, coords_list, wall_g = gpu_phase(
        bench.sequence, K=args.gpu_K, n_steps=args.gpu_steps,
        T_start=args.gpu_T_start, T_end=args.gpu_T_end, seed=args.gpu_seed,
    )
    sorted_idx = sorted(range(len(bcs)), key=lambda i: bcs[i], reverse=True)
    print(f"[GPU] wall={wall_g:.1f}s "
          f"best={bcs[sorted_idx[0]]} "
          f"top10={[bcs[i] for i in sorted_idx[:10]]}")

    # Sanity-check the GPU folds before polishing.
    valid_gpu_folds: list[tuple[int, list[tuple[int, int]]]] = []
    for i in sorted_idx[: args.top_k]:
        coords = coords_list[i]
        v = verify(bench.sequence, coords)
        if v:
            valid_gpu_folds.append((v.contacts, list(coords)))
    print(f"[GPU] valid top-{args.top_k}: {len(valid_gpu_folds)} folds, "
          f"contacts={[c for c, _ in valid_gpu_folds[:10]]}")

    if not valid_gpu_folds:
        print("[GPU] no valid folds — abort")
        return 1

    # CPU polish phase
    work = [
        (bench.sequence, coords, args.cpu_time_per, seed, args.sequence,
         str(diag_dir))
        for seed, (_, coords) in enumerate(valid_gpu_folds)
    ]
    print(f"\n[CPU polish] {len(work)} starts × {args.cpu_time_per}s "
          f"on {args.cpu_procs} procs")
    sys.stdout.flush()
    t0 = time.monotonic()
    polished: list[tuple[int, int, list[tuple[int, int]] | None, float]] = []
    with mp.Pool(args.cpu_procs) as pool:
        for i, (seed, contacts, coords, wall) in enumerate(
            pool.imap_unordered(cpu_polish_worker, work)
        ):
            print(f"  [{i+1:>3d}/{len(work)}] seed={seed:>3d} init~={valid_gpu_folds[seed][0]} "
                  f"-> contacts={contacts} wall={wall:.1f}s",
                  flush=True)
            polished.append((seed, contacts, coords, wall))
    wall_c = time.monotonic() - t0
    print(f"[CPU polish] wall={wall_c:.1f}s")

    best = max(polished, key=lambda x: x[1])
    best_seed, best_contacts, best_coords, _ = best
    print(f"\n=== {args.sequence} best across hybrid: {best_contacts} "
          f"(record {bench.best_known}, gap {bench.best_known - best_contacts}) ===")

    if best_coords is None:
        return 1

    fold = Fold(sequence=bench.sequence, coords=best_coords)
    out = Path(f"results/{args.sequence}_hybrid_best.json")
    out.parent.mkdir(exist_ok=True)
    write_submission(fold, args.sequence, out)
    print(f"Wrote: {out}")

    if args.update_submission:
        sub_path = Path(f"submissions/{args.sequence}.json")
        sub = json.loads(sub_path.read_text()) if sub_path.exists() else None
        sub_contacts = (
            verify(bench.sequence, [tuple(c) for c in sub["coords"]]).contacts
            if sub else -1
        )
        if best_contacts > sub_contacts:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: "
                  f"{sub_contacts} -> {best_contacts}")
        else:
            print(f"Not promoted (submission has {sub_contacts}, hybrid {best_contacts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
