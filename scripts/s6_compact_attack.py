"""S6 attack with bbox-compactness-biased SA.

Energy = -contacts + alpha * bbox_area / nH.
Run from each of the 4 distinct 19-fold basins, slow anneal.
Hypothesis: temporary area-cost gradient + standard moves can find
the compact 5x5-ish H-core that escapes the elongated 10x5 trap.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip, local_rebridge,
)
from hp_folding.verifier import count_contacts, verify

SEQ = BENCHMARKS["S6"].sequence
NH = SEQ.count("H")


def h_bbox_area(coords):
    h = [coords[i] for i in range(len(coords)) if SEQ[i] == "H"]
    xs = [c[0] for c in h]
    ys = [c[1] for c in h]
    return (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)


def propose(coords, rng):
    u = rng.random()
    if u < 0.30:
        return local_rebridge(coords, rng, seg_len=rng.choice([3, 4, 5, 6]))
    if u < 0.55:
        return corner_flip(coords, rng)
    if u < 0.75:
        return crankshaft(coords, rng)
    if u < 0.88:
        return segment_rotate(coords, rng)
    if u < 0.96:
        return pivot(coords, rng)
    return end_flip(coords, rng)


def compact_sa(args):
    init_coords, n_steps, T_start, T_end, alpha, seed = args
    rng = random.Random(seed)
    cur = list(init_coords)
    cur_c = count_contacts(SEQ, cur)
    cur_a = h_bbox_area(cur)
    cur_e = -cur_c + alpha * cur_a / NH
    best_c = cur_c
    best_coords = list(cur)
    for step in range(n_steps):
        T = T_start * (1 - step / n_steps) + T_end * (step / n_steps)
        # Anneal alpha down too (start with bias, end at pure contacts)
        a_now = alpha * (1 - step / n_steps)
        new = propose(cur, rng)
        if new is None:
            continue
        new_c = count_contacts(SEQ, new)
        new_a = h_bbox_area(new)
        new_e = -new_c + a_now * new_a / NH
        dE = new_e - (-cur_c + a_now * cur_a / NH)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new
            cur_c = new_c
            cur_a = new_a
            if cur_c > best_c:
                best_c = cur_c
                best_coords = list(cur)
    return best_c, best_coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-steps", type=int, default=2_000_000)
    p.add_argument("--alpha", type=float, default=2.0,
                   help="bbox-compactness weight in energy")
    p.add_argument("--n-trials-per-basin", type=int, default=20)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--target", type=int, default=21)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    ss = json.loads((ROOT / "diagnostics/stepping_stones.json").read_text())
    pool = ss["S6"]["19"]
    distinct = []
    seen = set()
    for f in pool:
        coords = [tuple(c) for c in f]
        pos = {c: i for i, c in enumerate(coords)}
        cs = []
        for i, (x, y) in enumerate(coords):
            if SEQ[i] != "H":
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = pos.get((x + dx, y + dy))
                if j is not None and j > i + 1 and SEQ[j] == "H":
                    cs.append((i, j))
        key = frozenset(cs)
        if key not in seen:
            seen.add(key)
            distinct.append(coords)

    sub_path = ROOT / "submissions/S6.json"
    sub = json.loads(sub_path.read_text())
    init_contacts = verify(SEQ, [tuple(c) for c in sub["coords"]]).contacts

    print(f"=== S6 compact-biased SA attack ===")
    print(f"  current: {init_contacts}, target: {args.target}")
    print(f"  basins: {len(distinct)}, trials/basin: {args.n_trials_per_basin}")
    print(f"  alpha={args.alpha}, n_steps={args.n_steps:,}")

    work = []
    seed = 0
    for b_idx, coords in enumerate(distinct):
        for t in range(args.n_trials_per_basin):
            seed += 1
            T_start = 1.0 + 0.5 * (t % 3)  # vary start temp
            work.append((coords, args.n_steps, T_start, 0.05, args.alpha, seed))

    t0 = time.monotonic()
    best_overall = init_contacts
    best_coords = None
    n_done = 0
    with mp.Pool(args.processes) as pool_proc:
        for c, coords in pool_proc.imap_unordered(compact_sa, work):
            n_done += 1
            if c > best_overall:
                best_overall = c
                best_coords = coords
                print(f"  [{n_done}/{len(work)}] NEW BEST = {c}", flush=True)
                if best_overall >= args.target:
                    print(f"  RECORD MATCH at trial {n_done}", flush=True)
                    pool_proc.terminate()
                    break
            if n_done % 5 == 0:
                elapsed = time.monotonic() - t0
                print(f"  [{n_done}/{len(work)}] best={best_overall} "
                      f"elapsed={elapsed:.0f}s", flush=True)

    print(f"\nFinal best: {best_overall} (was {init_contacts})")
    if best_coords and best_overall > init_contacts:
        out = ROOT / f"results/S6_compact_attack_c{best_overall}.json"
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=SEQ, coords=best_coords), "S6", out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/S6.json: {init_contacts} -> {best_overall}")


if __name__ == "__main__":
    main()
