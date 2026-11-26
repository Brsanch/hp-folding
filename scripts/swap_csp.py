"""Swap-CSP: search for record-breaking folds by 1-out, 2-in swaps.

Given a known fold with C contacts, for each existing pair p:
  base = (existing pairs) - {p}     # C - 1 pairs
  for each candidate (q1, q2) of new pairs:
    if SAW realizable(base ∪ {q1, q2}):
      return C + 1 contact fold

The candidate (q1, q2) pairs are H-H pairs at manhattan distance 2 in the
known fold (geometrically near-contact). Specific to the H residues that
participate in the existing pair p, we restrict to pairs that involve
residues geometrically near p's location to keep the search tractable.

Net effect: one contact removed, two added → +1 net = potential record break.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.verifier import verify, count_contacts
from scripts.targeted_csp import solve_target_pairs, _h_contact_pairs  # type: ignore


def find_candidate_pairs(seq, coords, max_distance=3):
    """H-H pairs (i, j), |i-j|>1, both H, currently NOT a contact, at
    manhattan distance ≤ max_distance in current fold."""
    out = []
    n = len(seq)
    NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
    pos = {tuple(c): i for i, c in enumerate(coords)}
    for i in range(n):
        if seq[i] != "H":
            continue
        for j in range(i + 2, n):
            if seq[j] != "H":
                continue
            d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            if d == 1:  # already a contact
                continue
            if d <= max_distance:
                out.append((i, j))
    return out


def worker(args):
    seq, target_pairs_list, time_budget = args
    target_pairs = frozenset(tuple(p) for p in target_pairs_list)
    result, nodes = solve_target_pairs(seq, target_pairs, time_budget=time_budget)
    return target_pairs, result, nodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--starting-fold", required=True,
                   help="Path to JSON fold to use as base")
    p.add_argument("--target-contacts", type=int, required=True,
                   help="Contact count we're aiming for (current + 1)")
    p.add_argument("--time-per-test", type=float, default=2.0)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--max-q-distance", type=int, default=2,
                   help="Manhattan distance threshold for candidate pairs")
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    seq = bench.sequence
    fold = json.loads(open(args.starting_fold).read())
    coords = [tuple(c) for c in fold["coords"]]
    init_contacts = verify(seq, coords).contacts
    base_pairs = _h_contact_pairs(seq, coords)
    print(f"=== Swap-CSP on {args.sequence} ===")
    print(f"  starting fold: {init_contacts} contacts ({len(base_pairs)} pairs)")
    print(f"  target: {args.target_contacts}")

    candidates = find_candidate_pairs(seq, coords, max_distance=args.max_q_distance)
    print(f"  candidate near-contact pairs (manhattan ≤ {args.max_q_distance}): {len(candidates)}")
    sys.stdout.flush()

    # Build all (p, q1, q2) targets where p ∈ base_pairs, q1, q2 ∈ candidates,
    # all distinct. Net swap: -1 base + 2 new = +1 contact.
    targets = []
    for p in sorted(base_pairs):
        # base' = base - {p}
        for i, q1 in enumerate(candidates):
            for q2 in candidates[i + 1:]:
                if q1 == q2:
                    continue
                tp = (base_pairs - {p}) | {q1, q2}
                if len(tp) != args.target_contacts:
                    continue
                targets.append(list(tp))
    print(f"  swap targets to test: {len(targets)}")
    sys.stdout.flush()

    if not targets:
        print("no targets to test")
        return 0

    work = [(seq, t, args.time_per_test) for t in targets]

    t0 = time.monotonic()
    best_total = init_contacts
    best_coords = None
    n_tested = 0
    n_solved = 0

    with mp.Pool(args.processes) as pool:
        for tp, result, nodes in pool.imap_unordered(worker, work, chunksize=8):
            n_tested += 1
            if result is not None:
                n_solved += 1
                v = verify(seq, result)
                actual_c = count_contacts(seq, result)
                if v and actual_c > best_total:
                    best_total = actual_c
                    best_coords = result
                    print(f"  *** found {actual_c}-contact fold (test {n_tested}/{len(work)}, "
                          f"target had {len(tp)} pairs, nodes={nodes}) ***", flush=True)
                    if best_total >= args.target_contacts:
                        print(f"  RECORD CANDIDATE — stopping pool", flush=True)
                        pool.terminate()
                        break
            if n_tested % 1000 == 0:
                print(f"  [{n_tested}/{len(work)}] solved={n_solved} best={best_total} "
                      f"elapsed={time.monotonic()-t0:.0f}s", flush=True)

    wall = time.monotonic() - t0
    print(f"\nWall: {wall:.0f}s, tested={n_tested}, solved={n_solved}")
    print(f"Best: {best_total} (was {init_contacts})")

    if best_total > init_contacts and best_coords:
        out = Path(f"results/{args.sequence}_swap_csp.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=seq, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            sub_path = Path(f"submissions/{args.sequence}.json")
            sub = json.loads(sub_path.read_text())
            sub_c = verify(seq, [tuple(c) for c in sub["coords"]]).contacts
            if best_total > sub_c:
                import shutil
                shutil.copy(out, sub_path)
                print(f"PROMOTED submissions/{args.sequence}.json: {sub_c} -> {best_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
