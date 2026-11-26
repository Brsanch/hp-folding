"""S6 variable-subset CSP. 12 essential contacts + each k-subset of 15
variable contacts = test for SAW-realizability. If any k=8+ subset is
realizable, that's 20+ contacts on S6 = +1 above current submission and
testing the path to record (21).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.verifier import verify, count_contacts
from scripts.targeted_csp import solve_target_pairs  # type: ignore

ESSENTIAL = [
    (2, 5), (2, 11), (5, 10), (7, 10), (9, 14), (9, 18),
    (11, 14), (22, 27), (35, 38), (39, 42), (44, 49), (46, 49),
]

VARIABLE = [
    (7, 22), (7, 46), (27, 42), (27, 46), (31, 38), (31, 40),
    (31, 48), (35, 40), (35, 48), (37, 40), (37, 48), (38, 49),
    (39, 44), (39, 48), (42, 49),
]


def worker(args):
    seq, target_pairs_list, time_budget = args
    target = frozenset(tuple(p) for p in target_pairs_list)
    result, nodes = solve_target_pairs(seq, target, time_budget=time_budget)
    return target, result, nodes


def main():
    seq = BENCHMARKS["S6"].sequence
    print(f"S6 variable-CSP: 12 essential + k variable, target 20+")
    print(f"  variable pool: {len(VARIABLE)} pairs")

    # Build all targets: each is essential + a k-subset of VARIABLE for k=8
    # That gives 12+8=20 contacts. Try k=8, then 9, 10, ...
    procs = 4
    for k in (8, 9, 10):
        n_subsets = 1
        from math import comb
        n_subsets = comb(len(VARIABLE), k)
        print(f"\n=== k={k} ({12+k} target contacts), {n_subsets} subsets ===")
        targets = []
        for sub in combinations(VARIABLE, k):
            target_pairs = list(ESSENTIAL) + list(sub)
            targets.append(target_pairs)
        print(f"  {len(targets)} targets queued")
        sys.stdout.flush()
        work = [(seq, t, 1.0) for t in targets]
        t0 = time.monotonic()
        n_done = 0
        n_solved = 0
        best_total = 19
        best_coords = None
        with mp.Pool(procs) as pool:
            for target, result, nodes in pool.imap_unordered(worker, work, chunksize=8):
                n_done += 1
                if result is not None:
                    actual_c = count_contacts(seq, result)
                    v = verify(seq, result)
                    if v and actual_c > best_total:
                        best_total = actual_c
                        best_coords = result
                        print(f"  *** {actual_c}-contact fold! "
                              f"({n_done}/{len(work)} nodes={nodes}) ***",
                              flush=True)
                        # Write to results
                        out = Path(f"results/S6_variable_csp_c{actual_c}.json")
                        out.parent.mkdir(exist_ok=True)
                        write_submission(Fold(sequence=seq, coords=result),
                                         "S6", out)
                if n_done % 200 == 0:
                    print(f"  [{n_done}/{len(work)}] solved={n_solved} best={best_total} "
                          f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
        print(f"k={k} done: best={best_total} wall={time.monotonic()-t0:.0f}s")
        if best_total >= 21:
            print("RECORD MATCH/BREAK!")
            break


if __name__ == "__main__":
    main()
