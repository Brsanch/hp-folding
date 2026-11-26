"""Lateral basin hopping: jump to ALTERNATIVE folds at the same contact
count via local_rebridge, then anneal each to its local optimum.

This is the answer to the contact-set attractor problem: instead of
random shocks (which destroy contacts and only refind the original
basin), we generate folds with DIFFERENT contact patterns but EQUAL
contact count, then climb. Each lateral hop lands in a different basin's
plateau, and the subsequent anneal converges to whatever local maximum
that basin contains.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip, local_rebridge,
)
from hp_folding.verifier import count_contacts, verify


def _propose(coords, rng):
    u = rng.random()
    if u < 0.40:
        return corner_flip(coords, rng)
    if u < 0.65:
        return crankshaft(coords, rng)
    if u < 0.85:
        return pivot(coords, rng)
    if u < 0.95:
        return segment_rotate(coords, rng)
    return end_flip(coords, rng)


def lateral_hop_then_anneal(
    sequence,
    init_coords,
    *,
    n_lateral_hops,
    n_anneal_steps,
    T_start,
    T_end,
    seed,
):
    rng = random.Random(seed)
    cur = list(init_coords)
    init_contacts = count_contacts(sequence, cur)

    # Phase 1: do n_lateral_hops local_rebridge attempts to drift to a
    # DIFFERENT same-contact fold.
    hops_done = 0
    for _ in range(n_lateral_hops * 30):
        seg = rng.choice([3, 4, 5])
        r = local_rebridge(cur, rng, seg_len=seg)
        if r is None:
            continue
        rc = count_contacts(sequence, r)
        if rc >= init_contacts:
            cur = r
            hops_done += 1
            if hops_done >= n_lateral_hops:
                break

    # Phase 2: SA from drifted fold to climb
    cur_contacts = count_contacts(sequence, cur)
    cur_e = -cur_contacts
    best_e = cur_e
    best_coords = list(cur)

    for step in range(n_anneal_steps):
        frac = step / max(1, n_anneal_steps - 1)
        T = T_start * (1 - frac) + T_end * frac
        new = _propose(cur, rng)
        if new is None:
            continue
        new_e = -count_contacts(sequence, new)
        dE = new_e - cur_e
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new
            cur_e = new_e
            if cur_e < best_e:
                best_e = cur_e
                best_coords = list(cur)
    return best_coords, -best_e, hops_done


def worker(args):
    seq, init_coords, n_hops, n_steps, T_start, T_end, seed = args
    t0 = time.monotonic()
    coords, contacts, hops = lateral_hop_then_anneal(
        seq, init_coords,
        n_lateral_hops=n_hops, n_anneal_steps=n_steps,
        T_start=T_start, T_end=T_end, seed=seed,
    )
    return seed, n_hops, T_start, hops, contacts, coords, time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=16)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--n-hops", type=int, default=20)
    p.add_argument("--n-anneal-steps", type=int, default=1_000_000)
    p.add_argument("--T-starts", default="0.5,1.0,2.0,4.0")
    p.add_argument("--T-end", type=float, default=0.05)
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_coords = [tuple(c) for c in sub["coords"]]
    init_contacts = verify(bench.sequence, init_coords).contacts
    print(f"=== Lateral hop on {args.sequence} ===")
    print(f"  starting from submission: {init_contacts} contacts (record {bench.best_known})")
    print(f"  trials: {args.trials} × ({args.n_hops} hops + {args.n_anneal_steps} anneal steps)")
    sys.stdout.flush()

    T_starts = [float(t) for t in args.T_starts.split(",")]
    work = []
    for seed in range(args.trials):
        T0 = T_starts[seed % len(T_starts)]
        work.append((bench.sequence, init_coords, args.n_hops,
                     args.n_anneal_steps, T0, args.T_end, seed))

    t0 = time.monotonic()
    best_contacts = init_contacts
    best_coords = init_coords
    with mp.Pool(args.processes) as pool:
        for i, (seed, n_hops, T_start, hops, contacts, coords, wall) in enumerate(
            pool.imap_unordered(worker, work)
        ):
            tag = " [BEST]" if contacts > best_contacts else ""
            print(f"  [{i+1:>3d}/{len(work)}] seed={seed} T0={T_start:.2f} "
                  f"hops_done={hops}/{n_hops} -> contacts={contacts}{tag} "
                  f"wall={wall:.1f}s",
                  flush=True)
            if contacts > best_contacts:
                best_contacts = contacts
                best_coords = coords
                if args.target is not None and best_contacts >= args.target:
                    pool.terminate()
                    break

    wall = time.monotonic() - t0
    print(f"\nTotal walltime: {wall:.1f}s")
    print(f"Best: {best_contacts} (record {bench.best_known})")

    if best_contacts > init_contacts and best_coords:
        out = Path(f"results/{args.sequence}_lateral_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=bench.sequence, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: "
                  f"{init_contacts} -> {best_contacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
