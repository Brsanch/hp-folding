"""Targeted contact-set CSP.

Given a sequence and a target set of (i, j) H-H contact pairs that MUST be
lattice-adjacent, search for a SAW realization. Backtracking residue
placement with forward checking on contact constraints.

Strategy: take the current submission's 35-contact pattern, enumerate
each candidate "missing contact" pair (degree-1 H residues at manhattan
distance 2 in the current fold), add it to the constraint set, and try
to find a 36-contact realization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.verifier import count_contacts, verify

Coord = tuple[int, int]
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _h_contact_pairs(sequence, coords):
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


def solve_target_pairs(sequence, target_pairs, *, time_budget=30.0):
    """Find a SAW where every (i, j) in target_pairs is at lattice manhattan
    distance 1, AND no pair (i, j) NOT in target_pairs gets to manhattan 1
    UNLESS one or both is P (we don't constrain P-P or H-P adjacencies).

    Actually simpler: just enforce target_pairs are adjacent. The total
    contact count will be ≥ |target_pairs| (could be more if other pairs
    happen to also be adjacent — that's fine, more contacts).
    """
    n = len(sequence)
    # Per-residue list of constraint partners
    must_adj = {i: set() for i in range(n)}
    for i, j in target_pairs:
        must_adj[i].add(j)
        must_adj[j].add(i)

    coords: list[Coord | None] = [None] * n
    occupied: dict[Coord, int] = {}
    deadline = time.monotonic() + time_budget

    # Place residue 0 at origin.
    coords[0] = (0, 0)
    occupied[(0, 0)] = 0

    nodes = [0]

    def dfs(i: int) -> list[Coord] | None:
        nodes[0] += 1
        if time.monotonic() > deadline:
            return None
        if i == n:
            return [c for c in coords]  # type: ignore
        prev = coords[i - 1]
        # Candidate cells: lattice neighbours of prev
        cands = []
        for dx, dy in _NEIGHBOURS:
            cell = (prev[0] + dx, prev[1] + dy)
            if cell in occupied:
                continue
            # Check: for every j < i with j in must_adj[i], cell must be at
            # manhattan distance 1 from coords[j].
            ok = True
            for j in must_adj[i]:
                if j < i and coords[j] is not None:
                    cj = coords[j]
                    if abs(cell[0] - cj[0]) + abs(cell[1] - cj[1]) != 1:
                        ok = False
                        break
            if not ok:
                continue
            # Forward check: for every j > i with j in must_adj[k] for some
            # k <= i already placed, the future residue j must still be
            # reachable. Skip detailed FC for simplicity.
            cands.append(cell)
        # Order candidates by # of remaining required contact partners
        # in target — prefer cells that satisfy more upcoming constraints.
        cands.sort(key=lambda c: -sum(
            1 for j in must_adj[i]
            if j > i and coords[j] is None and any(
                abs(c[0] - (prev[0] + dx)) + abs(c[1] - (prev[1] + dy)) <= 1
                for dx, dy in _NEIGHBOURS
            )
        ))
        for cell in cands:
            coords[i] = cell
            occupied[cell] = i
            r = dfs(i + 1)
            if r is not None:
                return r
            del occupied[cell]
            coords[i] = None
        return None

    result = dfs(1)
    return result, nodes[0]


def candidate_missing_pairs(sequence, coords, *, max_distance=2):
    """H-H pairs (i, j), |i-j|>1, both H, currently at manhattan distance
    EXACTLY max_distance — they're 'almost contacts'."""
    out = []
    for i in range(len(sequence)):
        if sequence[i] != "H":
            continue
        for j in range(i + 2, len(sequence)):
            if sequence[j] != "H":
                continue
            d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            if d == max_distance:
                out.append((i, j))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--time-per-pair", type=float, default=30.0)
    p.add_argument("--target-extra", type=int, default=1,
                   help="How many extra contacts to seek (default 1: 35 -> 36)")
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub = json.loads(open(f"submissions/{args.sequence}.json").read())
    coords = [tuple(c) for c in sub["coords"]]
    init_contacts = count_contacts(bench.sequence, coords)
    base_pairs = _h_contact_pairs(bench.sequence, coords)
    print(f"Targeted CSP on {args.sequence}")
    print(f"  current submission: {init_contacts} contacts (record {bench.best_known})")
    print(f"  base pair set: {len(base_pairs)} pairs")
    print(f"  seeking +{args.target_extra} via candidate pairs")
    sys.stdout.flush()

    candidates = candidate_missing_pairs(bench.sequence, coords, max_distance=2)
    # Filter: candidates not already in base
    candidates = [p for p in candidates if p not in base_pairs]
    print(f"  candidate missing pairs (manhattan-2 H-H): {len(candidates)}")
    sys.stdout.flush()

    best_total = init_contacts
    best_coords = coords
    t0 = time.monotonic()
    for k, candidate in enumerate(candidates):
        target = base_pairs | {candidate}
        if len(target) != len(base_pairs) + args.target_extra:
            continue
        result, nodes = solve_target_pairs(
            bench.sequence, target, time_budget=args.time_per_pair,
        )
        if result is not None:
            total = count_contacts(bench.sequence, result)
            v = verify(bench.sequence, result)
            tag = " [BEST]" if total > best_total else ""
            print(f"  [{k+1}/{len(candidates)}] candidate {candidate}: "
                  f"valid={bool(v)} total_contacts={total} nodes={nodes}{tag}",
                  flush=True)
            if v and total > best_total:
                best_total = total
                best_coords = result
        else:
            print(f"  [{k+1}/{len(candidates)}] candidate {candidate}: "
                  f"no SAW realization (nodes={nodes})", flush=True)
        if time.monotonic() - t0 > args.time_per_pair * len(candidates) * 1.2:
            break

    wall = time.monotonic() - t0
    print(f"\nTotal walltime: {wall:.1f}s")
    print(f"Best: {best_total} (record {bench.best_known}, was {init_contacts})")
    if best_total > init_contacts:
        out = Path(f"results/{args.sequence}_targeted_csp_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=bench.sequence, coords=best_coords),
                         args.sequence, out)
        if args.update_submission:
            import shutil
            shutil.copy(out, f"submissions/{args.sequence}.json")
            print(f"PROMOTED submissions/{args.sequence}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
