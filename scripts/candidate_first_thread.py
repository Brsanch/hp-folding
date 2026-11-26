"""Candidate-set-first chain construction.

Take the UNION of contact pairs from multiple known good folds (e.g., 4
distinct 42-folds for S8). Build the chain incrementally with beam search,
where the score at each step is "# of candidate pairs already realized
by partial chain". The best chain found realizes maximum candidate pairs.

This separates the contact-set decision from per-pair-CSP: instead of
testing one specific 43-subset, we ask "find a chain that realizes as
many of these 93 candidate pairs as possible." If max realized ≥ 43, we
have a record-break.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.verifier import count_contacts, verify

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def fold_pairs(seq, coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    pairs = set()
    for i, (x, y) in enumerate(coords):
        if seq[i] != "H":
            continue
        for dx, dy in NB:
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1:
                continue
            if seq[j] == "H":
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    return pairs


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def beam_build(seq, candidate_pairs, beam_width, n_starts, seed):
    """Beam search chain construction. Returns (best_coords, best_realized)."""
    rng = random.Random(seed)
    n = len(seq)
    # Adjacency for fast lookup
    pair_partners = defaultdict(set)
    for i, j in candidate_pairs:
        pair_partners[i].add(j)
        pair_partners[j].add(i)
    candidate_pairs_set = set(candidate_pairs)

    best_coords_global = None
    best_realized_global = 0

    for start in range(n_starts):
        # Beam state: list of (coords_tuple, occupied_set, realized_count)
        beam = [(((0, 0),), frozenset({(0, 0)}), 0)]
        for i in range(1, n):
            new_beam = []
            for coords, occupied, realized in beam:
                last = coords[-1]
                # Try 4 lattice neighbors
                for dx, dy in NB:
                    nb = (last[0] + dx, last[1] + dy)
                    if nb in occupied:
                        continue
                    # Compute incremental realized contacts at this step
                    new_realized = realized
                    for j in pair_partners.get(i, ()):
                        if j < i:
                            cj = coords[j]
                            if abs(nb[0] - cj[0]) + abs(nb[1] - cj[1]) == 1:
                                new_realized += 1
                    new_coords = coords + (nb,)
                    new_occupied = occupied | {nb}
                    new_beam.append((new_coords, new_occupied, new_realized))
            if not new_beam:
                break
            # Keep top beam_width by realized count, with diversity sampling
            new_beam.sort(key=lambda x: -x[2])
            # Diverse selection: take top half by score, sample bottom half
            cutoff = beam_width // 2
            top = new_beam[:cutoff]
            rest = new_beam[cutoff:]
            rng.shuffle(rest)
            beam = top + rest[:beam_width - cutoff]

        if not beam:
            continue

        # Evaluate full chains
        for coords_t, occupied, realized in beam:
            coords = list(coords_t)
            if len(coords) == n:
                actual_contacts = count_contacts(seq, coords)
                # Use actual contacts for score (some pairs may be realized
                # outside our candidate set too)
                if actual_contacts > best_realized_global:
                    best_realized_global = actual_contacts
                    best_coords_global = coords

    return best_coords_global, best_realized_global


def worker(args):
    seq, candidate_pairs, beam_width, n_starts, seed = args
    coords, score = beam_build(seq, candidate_pairs, beam_width, n_starts, seed)
    return seed, score, coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", default="S8")
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--beam-width", type=int, default=2000)
    p.add_argument("--starts-per-worker", type=int, default=1)
    p.add_argument("--n-workers", type=int, default=14)
    p.add_argument("--update-submission", action="store_true")
    p.add_argument("--include-near-contacts", action="store_true",
                   help="Also include manhattan-2 H pairs from each parent fold")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    seq = bench.sequence
    n = len(seq)
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    init_contacts = verify(seq, [tuple(c) for c in sub["coords"]]).contacts

    # Build candidate pair union from all distinct top-tier folds
    # For S8: load the 4 distinct 42-folds we have
    parents = []
    if args.sequence == "S8":
        for fp in sorted(Path("results/multibasin_S8_lvl38").glob("*c42.json")):
            d = json.loads(fp.read_text())
            parents.append([tuple(c) for c in d["coords"]])
    # Fallback: use submission
    if not parents:
        parents = [[tuple(c) for c in sub["coords"]]]

    pair_union: set[tuple[int, int]] = set()
    for p_coords in parents:
        for pair in fold_pairs(seq, p_coords):
            pair_union.add(pair)
    if args.include_near_contacts:
        # Add H-H pairs at manhattan ≤ 2 in any parent
        for p_coords in parents:
            for i in range(n):
                if seq[i] != "H":
                    continue
                for j in range(i + 2, n):
                    if seq[j] != "H":
                        continue
                    if manhattan(p_coords[i], p_coords[j]) <= 2:
                        pair_union.add((i, j))
    # Filter by parity (i, j with j-i ODD only — chain must reach distance 1)
    pair_union = {p for p in pair_union if (p[1] - p[0]) % 2 == 1}

    print(f"=== Candidate-first thread on {args.sequence} ===")
    print(f"  current submission: {init_contacts} (record {bench.best_known})")
    print(f"  parent folds: {len(parents)}")
    print(f"  candidate pair union (parity-filtered): {len(pair_union)}")
    print(f"  beam width: {args.beam_width}, n_workers: {args.n_workers}, starts/worker: {args.starts_per_worker}")
    sys.stdout.flush()

    pair_list = sorted(pair_union)
    work = [(seq, pair_list, args.beam_width, args.starts_per_worker, seed)
            for seed in range(args.n_workers)]

    t0 = time.monotonic()
    best_score = init_contacts
    best_coords = None

    with mp.Pool(args.processes) as pool:
        for seed, score, coords in pool.imap_unordered(worker, work):
            tag = " [BEST]" if score > best_score else ""
            print(f"  worker seed={seed}: realized contacts={score}{tag}",
                  flush=True)
            if score > best_score:
                best_score = score
                best_coords = coords

    wall = time.monotonic() - t0
    print(f"\nWall: {wall:.1f}s")
    print(f"Best realized contacts: {best_score} (was {init_contacts}, record {bench.best_known})")

    if best_coords and best_score > init_contacts:
        out = Path(f"results/{args.sequence}_candidate_first.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=seq, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, sub_path)
            print(f"PROMOTED submissions/{args.sequence}.json: {init_contacts} -> {best_score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
