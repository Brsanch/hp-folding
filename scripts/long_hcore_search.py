"""Long-running H-core enumeration + CSP search.

Designed for overnight/multi-day runs. Iteratively:
  1. Generate a candidate suite-block polyomino (mutation-BFS from a pool
     of high-internal-edge shapes).
  2. Run H-core CSP for a per-block time budget.
  3. If a fold is found, write it to results/. If it beats current best,
     update submissions/.
  4. Checkpoint progress periodically.

Tracks every distinct block tried, every fold found, and writes a
checkpoint file so the run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.hcore import (
    enumerate_suite_blocks,
    _internal_edges,
    _concave_corner_count,
    _canon_translate,
    _is_connected,
    _single_mutations,
)
from hp_folding.hcore_csp import solve_csp
from hp_folding.io import write_submission
from hp_folding.verifier import verify


def block_worker(args):
    seq_id, seq, block_cells, time_per_block, h_run_max_placements = args
    block = frozenset(block_cells)
    sol = solve_csp(seq, block,
                    time_budget_seconds=time_per_block,
                    h_run_max_placements=h_run_max_placements)
    if sol is None:
        return (block_cells, _internal_edges(block), None, 0)
    return (block_cells, _internal_edges(block), sol.coords, sol.contacts)


def generate_candidates(n_h: int, n_target: int, seed_blocks=None):
    """Mix of enumerator output + mutations of seed blocks."""
    candidates: list[frozenset] = []
    seen: set[frozenset] = set()

    def add(block):
        canon = _canon_translate(block)
        if canon not in seen:
            seen.add(canon)
            candidates.append(canon)

    # Seed with enumerator's top-N
    for max_extra_p in (4, 6, 8, 10):
        for b in enumerate_suite_blocks(n_h, max_extra_perimeter=max_extra_p,
                                        max_blocks=200):
            add(b)
            if len(candidates) >= n_target:
                return candidates

    # Augment with multi-mutation expansions of seed blocks
    if seed_blocks:
        frontier = list(seed_blocks)
        depth = 0
        while frontier and len(candidates) < n_target and depth < 4:
            new_frontier = []
            for s in frontier:
                for m in _single_mutations(s):
                    add(m)
                    if len(candidates) >= n_target:
                        return candidates
                    new_frontier.append(m)
            frontier = new_frontier[:200]  # cap exploration breadth
            depth += 1
    return candidates


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--time-per-block", type=float, default=20.0)
    p.add_argument("--h-run-max-placements", type=int, default=50000)
    p.add_argument("--n-blocks", type=int, default=10000)
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint JSON (default: state/<seq>_long_hcore.json)")
    p.add_argument("--update-submission", action="store_true")
    p.add_argument("--max-wall-hours", type=float, default=12.0,
                   help="Hard wall-time limit (h)")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    n_h = sum(1 for c in bench.sequence if c == "H")
    sub_path = Path(f"submissions/{args.sequence}.json")
    sub = json.loads(sub_path.read_text())
    sub_coords = [tuple(c) for c in sub["coords"]]
    init_contacts = verify(bench.sequence, sub_coords).contacts

    cp_path = Path(args.checkpoint) if args.checkpoint else \
        Path(f"state/{args.sequence}_long_hcore.json")
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"tested_blocks": [], "best_contacts": init_contacts,
             "best_coords": sub_coords, "started": time.time()}
    if cp_path.exists():
        state = json.loads(cp_path.read_text())
        print(f"Resumed from checkpoint: tested={len(state['tested_blocks'])}, "
              f"best={state['best_contacts']}")

    print(f"=== Long H-core search on {args.sequence} ===")
    print(f"  n_h={n_h} record={bench.best_known} starting_best={state['best_contacts']}")
    print(f"  budget: {args.time_per_block}s/block × {args.n_blocks} blocks max, "
          f"{args.processes} procs, max wall {args.max_wall_hours}h")
    sys.stdout.flush()

    # Build candidate pool (with submission's H-block as a seed)
    seed_blocks = []
    h_cells = frozenset(c for i, c in enumerate(sub_coords)
                        if bench.sequence[i] == "H")
    if _is_connected(h_cells):
        seed_blocks.append(_canon_translate(h_cells))
    candidates = generate_candidates(n_h, args.n_blocks, seed_blocks=seed_blocks)
    print(f"  candidate pool: {len(candidates)} unique blocks")
    # Filter out already-tested
    tested = {tuple(t) for t in state["tested_blocks"]}
    candidates = [c for c in candidates if tuple(sorted(c)) not in tested]
    print(f"  to test (after dedupe with checkpoint): {len(candidates)}")
    sys.stdout.flush()

    work = [(args.sequence, bench.sequence, list(c),
             args.time_per_block, args.h_run_max_placements)
            for c in candidates]

    deadline = time.time() + args.max_wall_hours * 3600
    t0 = time.monotonic()
    n_solved = 0
    n_tried = 0
    last_checkpoint = time.monotonic()

    with mp.Pool(args.processes) as pool:
        for i, (block_cells, ie, coords, contacts) in enumerate(
            pool.imap_unordered(block_worker, work)
        ):
            n_tried += 1
            if coords is not None and contacts > 0:
                n_solved += 1
                if contacts > state["best_contacts"]:
                    state["best_contacts"] = contacts
                    state["best_coords"] = coords
                    state["best_block"] = block_cells
                    print(f"  [{n_tried}/{len(work)}] ie={ie}: "
                          f"contacts={contacts} [BEST]", flush=True)
                    if args.update_submission:
                        out = Path(f"results/{args.sequence}_long_hcore_best.json")
                        out.parent.mkdir(exist_ok=True)
                        write_submission(
                            Fold(sequence=bench.sequence, coords=coords),
                            args.sequence, out,
                        )
                        shutil.copy(out, sub_path)
                        print(f"  PROMOTED submissions/{args.sequence}.json", flush=True)
                    if state["best_contacts"] >= bench.best_known + 1:
                        print(f"  RECORD BEATEN! Stopping.", flush=True)
                        pool.terminate()
                        break
            state["tested_blocks"].append(sorted(block_cells))
            # Checkpoint every 60s
            if time.monotonic() - last_checkpoint > 60:
                cp_path.write_text(json.dumps(state))
                last_checkpoint = time.monotonic()
                print(f"  [checkpoint] tested={n_tried} solved={n_solved} "
                      f"best={state['best_contacts']} "
                      f"elapsed={time.monotonic()-t0:.0f}s", flush=True)
            if time.time() > deadline:
                print(f"  [deadline reached at {time.monotonic()-t0:.0f}s]")
                pool.terminate()
                break

    # Final checkpoint
    cp_path.write_text(json.dumps(state))
    wall = time.monotonic() - t0
    print(f"\nDone. wall={wall:.0f}s tested={n_tried} solved={n_solved} "
          f"best={state['best_contacts']} (record {bench.best_known})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
