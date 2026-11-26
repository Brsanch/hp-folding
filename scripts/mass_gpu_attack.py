"""Massive GPU-parallel SA — explore the entire reachable basin space.

K=8192+ chains × millions of steps × multiple temperature schedules.
Each chain is one independent SA run. After SA, take top-K folds and
polish on CPU with corner-flip-heavy WL.

The hypothesis: with N_chains_total = K * n_seed_rounds we're sampling
an enormous fraction of the reachable basin space. The record-class folds
have nonzero probability, and probability × N_chains > 1 lands at least one.
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
from hp_folding.mlx_batch import batch_simulated_annealing
from hp_folding.verifier import verify, count_contacts


def cpu_polish_worker(args):
    seq, init_coords, time_budget, seed = args
    import math, random
    from hp_folding.moves import (
        corner_flip, crankshaft, pivot, segment_rotate, end_flip
    )
    rng = random.Random(seed)
    cur = list(init_coords)
    cur_c = count_contacts(seq, cur)
    best_c = cur_c
    best = list(cur)
    n_steps = int(time_budget * 5000)
    T_start = 1.0
    T_end = 0.05
    for step in range(n_steps):
        T = T_start * (1 - step/n_steps) + T_end * (step/n_steps)
        u = rng.random()
        if u < 0.4: r = corner_flip(cur, rng)
        elif u < 0.65: r = crankshaft(cur, rng)
        elif u < 0.85: r = pivot(cur, rng)
        elif u < 0.95: r = segment_rotate(cur, rng)
        else: r = end_flip(cur, rng)
        if r is None: continue
        c = count_contacts(seq, r)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = r; cur_c = c
            if c > best_c:
                best_c = c
                best = list(cur)
    return seed, best_c, best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--K-gpu", type=int, default=8192)
    p.add_argument("--gpu-steps", type=int, default=200_000)
    p.add_argument("--gpu-rounds", type=int, default=4,
                   help="Number of GPU SA rounds with different seeds")
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--cpu-procs", type=int, default=14)
    p.add_argument("--cpu-time-per", type=float, default=180.0)
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_contacts = verify(bench.sequence, [tuple(c) for c in sub["coords"]]).contacts
    print(f"=== Mass GPU attack on {args.sequence} ===")
    print(f"  current submission: {init_contacts} (record {bench.best_known})")
    print(f"  GPU: K={args.K_gpu} chains × {args.gpu_steps} steps × {args.gpu_rounds} rounds")
    print(f"  CPU polish: top-{args.top_k} × {args.cpu_time_per}s on {args.cpu_procs} procs")
    sys.stdout.flush()

    # Phase 1: many GPU SA rounds
    all_top: list[tuple[int, list]] = []
    t0 = time.monotonic()
    for round_idx in range(args.gpu_rounds):
        # Spread T_start across rounds for diversity
        T_start = 2.0 + round_idx * 1.5
        T_end = 0.05
        print(f"\n[GPU round {round_idx+1}/{args.gpu_rounds}] T={T_start}->{T_end}")
        sys.stdout.flush()
        res = batch_simulated_annealing(
            bench.sequence, K=args.K_gpu, n_steps=args.gpu_steps,
            T_start=T_start, T_end=T_end, seed=round_idx * 9941 + 13,
        )
        bcs = res.best_contacts_per_chain
        print(f"  wall={res.wall_seconds:.1f}s top10={sorted(bcs, reverse=True)[:10]}")
        sys.stdout.flush()
        # Keep best K from this round
        sorted_idx = sorted(range(len(bcs)), key=lambda i: bcs[i], reverse=True)
        for i in sorted_idx[:args.top_k]:
            v = verify(bench.sequence, res.best_coords_per_chain[i])
            if v:
                all_top.append((v.contacts, list(res.best_coords_per_chain[i])))

    # Sort all collected top folds, dedupe by contacts/coords
    all_top.sort(key=lambda x: -x[0])
    seen_keys = set()
    polish_pool = []
    for c, coords in all_top:
        key = (c, tuple(coords))
        if key in seen_keys: continue
        seen_keys.add(key)
        polish_pool.append((c, coords))
        if len(polish_pool) >= args.top_k:
            break
    print(f"\n[CPU polish] {len(polish_pool)} folds queued, contacts={[p[0] for p in polish_pool[:10]]}...")
    sys.stdout.flush()

    work = [(bench.sequence, coords, args.cpu_time_per, i)
            for i, (_, coords) in enumerate(polish_pool)]
    best_contacts = init_contacts
    best_coords = None
    with mp.Pool(args.cpu_procs) as pool:
        for i, (seed, contacts, coords) in enumerate(
            pool.imap_unordered(cpu_polish_worker, work)
        ):
            tag = " [BEST]" if contacts > best_contacts else ""
            print(f"  [{i+1}/{len(work)}] seed={seed} contacts={contacts}{tag}",
                  flush=True)
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = coords
                if args.target is not None and best_contacts >= args.target:
                    pool.terminate()
                    break

    wall = time.monotonic() - t0
    print(f"\nTotal wall: {wall:.1f}s")
    print(f"Best: {best_contacts} (record {bench.best_known}, was {init_contacts})")
    if best_contacts > init_contacts and best_coords:
        out = Path(f"results/{args.sequence}_mass_gpu_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=bench.sequence, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
