"""Tabu-aware SA: close off the current basin's contact pattern until late.

The current best fold's H-H contact set acts as a tabu pattern. During
the early "hose" phase, the SA energy includes a strong penalty
proportional to overlap with the tabu set — pushing the chain toward
DIFFERENT contact topologies. As the run cools, the tabu penalty fades
to zero, letting the search converge on whichever new basin it landed in.

If any state during the run reaches a higher total contact count than
the original best AND has substantially different contact pattern, it
becomes the new champion.
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

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _h_contact_set(sequence, coords):
    pos = {c: i for i, c in enumerate(coords)}
    pairs = set()
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue
        for dx, dy in _NEIGHBOURS:
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1:
                continue
            if sequence[j] != "H":
                continue
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return pairs


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


def tabu_sa(sequence, init_coords, tabu_set, *, n_steps,
            T_start=2.0, T_end=0.05,
            lambda_start=8.0, lambda_end=0.0,
            seed=0):
    """SA with tabu-overlap penalty in the energy.

    Energy = -contacts + λ * |contacts_pairs ∩ tabu_set|

    Both T and λ cool linearly. Final λ = 0 means standard SA at the end.
    """
    rng = random.Random(seed)
    cur = list(init_coords)
    cur_contacts = count_contacts(sequence, cur)
    cur_set = _h_contact_set(sequence, cur)

    best_coords = list(cur)
    best_contacts = cur_contacts
    best_set = set(cur_set)

    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        T = T_start * (1 - frac) + T_end * frac
        lam = lambda_start * (1 - frac) + lambda_end * frac

        new_coords = _propose(cur, rng)
        if new_coords is None:
            continue
        new_contacts = count_contacts(sequence, new_coords)
        new_set = _h_contact_set(sequence, new_coords)

        overlap_cur = len(cur_set & tabu_set)
        overlap_new = len(new_set & tabu_set)
        E_cur = -cur_contacts + lam * overlap_cur
        E_new = -new_contacts + lam * overlap_new
        dE = E_new - E_cur
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new_coords
            cur_contacts = new_contacts
            cur_set = new_set
            # Track best (by raw contacts only)
            if new_contacts > best_contacts:
                best_contacts = new_contacts
                best_coords = list(new_coords)
                best_set = set(new_set)
    return best_coords, best_contacts, best_set


def worker(args):
    (seq, init_coords, tabu_set, n_steps, T_start, lambda_start, seed) = args
    t0 = time.monotonic()
    coords, contacts, contact_set = tabu_sa(
        seq, init_coords, tabu_set,
        n_steps=n_steps, T_start=T_start, lambda_start=lambda_start, seed=seed,
    )
    overlap = len(contact_set & tabu_set)
    return seed, T_start, lambda_start, contacts, overlap, coords, time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=16)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=1_500_000)
    p.add_argument("--T-starts", default="1.0,2.0,3.0,5.0")
    p.add_argument("--lambda-starts", default="4.0,8.0,16.0,32.0",
                   help="Tabu-overlap penalty weights cycled across trials")
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_coords = [tuple(c) for c in sub["coords"]]
    init_contacts = verify(bench.sequence, init_coords).contacts
    tabu_set = _h_contact_set(bench.sequence, init_coords)
    print(f"=== Tabu SA on {args.sequence} ===")
    print(f"  current submission: {init_contacts} contacts (record {bench.best_known})")
    print(f"  tabu set size: {len(tabu_set)} contact pairs")
    sys.stdout.flush()

    T_starts = [float(t) for t in args.T_starts.split(",")]
    lam_starts = [float(t) for t in args.lambda_starts.split(",")]
    work = []
    for seed in range(args.trials):
        T0 = T_starts[seed % len(T_starts)]
        L0 = lam_starts[(seed // len(T_starts)) % len(lam_starts)]
        work.append((bench.sequence, init_coords, tabu_set, args.n_steps,
                     T0, L0, seed))

    t0 = time.monotonic()
    best_contacts = init_contacts
    best_coords = init_coords
    with mp.Pool(args.processes) as pool:
        for i, (seed, T0, L0, contacts, overlap, coords, wall) in enumerate(
            pool.imap_unordered(worker, work)
        ):
            tag = " [BEST]" if contacts > best_contacts else ""
            print(f"  [{i+1:>3d}/{len(work)}] seed={seed} T0={T0:.1f} L0={L0:.0f} "
                  f"-> contacts={contacts} overlap={overlap}/{len(tabu_set)}{tag} "
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
        out = Path(f"results/{args.sequence}_tabu_best.json")
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
