"""'Hose' simulated annealing — continuous directed pressure.

Instead of discrete random shocks (basin hop), apply continuous
high-temperature SA from the current best fold and cool slowly. Many
small moves under temperature accumulate to drift the chain into nearby
basins without scrambling its structure.

Move set: corner_flip + crankshaft + segment_rotate + pivot.
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
    corner_flip, crankshaft, pivot, segment_rotate, end_flip,
)
from hp_folding.verifier import count_contacts, verify


def _propose(coords, rng):
    u = rng.random()
    if u < 0.40:
        return corner_flip(coords, rng), "corner_flip"
    if u < 0.65:
        return crankshaft(coords, rng), "crankshaft"
    if u < 0.85:
        return pivot(coords, rng), "pivot"
    if u < 0.95:
        return segment_rotate(coords, rng), "segment_rot"
    return end_flip(coords, rng), "end_flip"


def hose_sa(
    sequence,
    init_coords,
    *,
    n_steps,
    T_start=2.0,
    T_end=0.05,
    seed=0,
):
    rng = random.Random(seed)
    cur = list(init_coords)
    cur_e = -count_contacts(sequence, cur)  # energy = -contacts (minimize)
    best_e = cur_e
    best_coords = list(cur)

    log_T0 = math.log(T_start)
    log_T1 = math.log(T_end)

    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        T = math.exp(log_T0 + (log_T1 - log_T0) * frac)
        new_coords, _cls = _propose(cur, rng)
        if new_coords is None:
            continue
        new_e = -count_contacts(sequence, new_coords)
        dE = new_e - cur_e
        if dE <= 0 or rng.random() < math.exp(-dE / T):
            cur = new_coords
            cur_e = new_e
            if cur_e < best_e:
                best_e = cur_e
                best_coords = list(cur)
    return best_coords, -best_e  # contacts


def worker(args):
    seq, init_coords, n_steps, T_start, T_end, seed = args
    t0 = time.monotonic()
    coords, contacts = hose_sa(
        seq, init_coords,
        n_steps=n_steps, T_start=T_start, T_end=T_end, seed=seed,
    )
    return seed, T_start, contacts, coords, time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=16)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=2_000_000)
    p.add_argument("--T-end", type=float, default=0.05)
    p.add_argument("--T-starts", default="0.5,1.0,1.5,2.5,4.0,8.0",
                   help="Diverse starting temperatures cycled across trials")
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    p.add_argument("--random-init", action="store_true",
                   help="Use a different random SAW per trial instead of "
                        "starting from the current submission")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_coords = [tuple(c) for c in sub["coords"]]
    init_contacts = verify(bench.sequence, init_coords).contacts
    print(f"=== Hose SA on {args.sequence} ===")
    print(f"  starting from submission: {init_contacts} contacts (record {bench.best_known})")
    print(f"  trials: {args.trials} × {args.n_steps} steps on {args.processes} procs")
    sys.stdout.flush()

    T_starts = [float(t) for t in args.T_starts.split(",")]
    work = []
    for seed in range(args.trials):
        T0 = T_starts[seed % len(T_starts)]
        if args.random_init:
            from hp_folding.fold import Fold
            r = random.Random(seed * 9176 + 13)
            f = Fold.random_saw(bench.sequence, rng=r)
            trial_init = list(f.coords)
        else:
            trial_init = init_coords
        work.append((bench.sequence, trial_init, args.n_steps, T0,
                     args.T_end, seed))

    t0 = time.monotonic()
    best_contacts = init_contacts
    best_coords = init_coords
    with mp.Pool(args.processes) as pool:
        for i, (seed, T_start, contacts, coords, wall) in enumerate(
            pool.imap_unordered(worker, work)
        ):
            tag = " [BEST]" if contacts > best_contacts else ""
            print(f"  [{i+1:>3d}/{len(work)}] seed={seed} T0={T_start:.2f} "
                  f"-> contacts={contacts}{tag} wall={wall:.1f}s",
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
        out = Path(f"results/{args.sequence}_hose_best.json")
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
