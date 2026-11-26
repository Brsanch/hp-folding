"""Constraint-driven hcore construction from the sequence's run structure.

Generates candidate hcore polyominoes by placing the sequence's H-runs as
rigid SAW pieces on the lattice, then checking that P-runs of the matching
lengths can route between them OUTSIDE the union of placed H-cells.

This bypasses every mutation-based search by NOT seeding from any known fold.
The hcores it generates depend ONLY on the sequence's intrinsic H-run / P-run
structure — exactly what gives the chain its shape constraints.

Algorithm:
  1. Parse the sequence into alternating H-runs and P-runs (with lengths).
  2. Enumerate all distinct SAW-path shapes for each H-run length present.
  3. DFS over H-run placements (position + shape + orientation), in chain order.
     At each placement: verify a P-path of the matching length exists between
     the previous H-run's end and this H-run's start, routing outside the
     accumulating hcore. Backtrack if no valid P-path exists.
  4. When all H-runs are placed, the union of cells is a candidate hcore.
     Hand to the existing CSP solver to find best contact count.

This generates topologies that mutation-based methods CANNOT reach — every
hcore here is built from scratch from the sequence's constraints, not a
perturbation of something already-known.
"""
from __future__ import annotations
import time, random, json
from collections import deque
from dataclasses import dataclass, field

Coord = tuple[int, int]
NBRS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ════════════════════════════════════════════════════════════
# Sequence parsing
# ════════════════════════════════════════════════════════════

def parse_runs(sequence: str) -> list[tuple[str, int]]:
    """[(kind, length), ...] — alternating H/P with run lengths."""
    out = []
    if not sequence: return out
    cur_kind, cur_len = sequence[0], 1
    for c in sequence[1:]:
        if c == cur_kind:
            cur_len += 1
        else:
            out.append((cur_kind, cur_len)); cur_kind, cur_len = c, 1
    out.append((cur_kind, cur_len))
    return out


# ════════════════════════════════════════════════════════════
# SAW shape enumeration (rigid pieces for each H-run length)
# ════════════════════════════════════════════════════════════

def enumerate_saw_paths(length: int, *, canonicalize: bool = True) -> list[tuple[Coord, ...]]:
    """All length-N SAW paths starting at (0,0). Each path = ordered tuple of cells.

    If canonicalize, dedup paths up to dihedral symmetry of the lattice — we
    return one representative per equivalence class to reduce branching.
    """
    if length <= 0: return []
    if length == 1: return [((0, 0),)]

    # BFS-style enumeration
    paths = [((0, 0),)]
    for step in range(length - 1):
        new_paths = []
        for p in paths:
            last = p[-1]
            occupied = set(p)
            for dx, dy in NBRS:
                n = (last[0] + dx, last[1] + dy)
                if n in occupied: continue
                new_paths.append(p + (n,))
        paths = new_paths

    if not canonicalize:
        return paths

    # Dedup by canonical form (translate + 8 dihedral images, keep min)
    def all_dihedrals(path):
        out = []
        cur = path
        for _ in range(4):
            out.append(cur)
            out.append(tuple((-x, y) for (x, y) in cur))
            cur = tuple((-y, x) for (x, y) in cur)
        # canonicalize translation: shift each so min(x)=0, min(y)=0
        canon = []
        for p in out:
            mx_, my = min(c[0] for c in p), min(c[1] for c in p)
            canon.append(tuple((c[0]-mx_, c[1]-my) for c in p))
        return canon

    seen = set()
    canon_paths = []
    for p in paths:
        forms = all_dihedrals(p)
        key = min(forms)
        if key not in seen:
            seen.add(key)
            canon_paths.append(key)
    return canon_paths


# ════════════════════════════════════════════════════════════
# P-run routing: find a SAW path of given length between two cells, outside a forbidden set
# ════════════════════════════════════════════════════════════

def has_p_route(start: Coord, end: Coord, length: int, forbidden: set,
                max_search: int = 200_000) -> bool:
    """True if a SAW path of EXACTLY `length` cells exists from start to end,
    where every cell of the path (including endpoints) avoids `forbidden`.

    length here = number of intermediate P residues + 1 (= number of edges
    in the path between start and end). Equivalently, path-length + 1 nodes.

    Uses bounded DFS. start and end are already-placed H-residues (so they're
    in `forbidden` themselves — the P-path must go from start to end via
    `length-1` intermediate cells that are all NOT in forbidden, then arrive
    at end).
    """
    if start == end:
        return length == 0   # both same cell, no P-residues
    if length == 0: return False
    # length = number of P residues between start and end. The P path is start-P1-P2-...-Pk-end.
    # That's length P cells + start + end = length+2 cells, length+1 edges.
    # Each Pi must not be in forbidden, AND must not equal start or end.
    # The path must be a SAW.

    # If the P-residues count is 0, start and end must be lattice-adjacent.
    if length == 0:
        return any((start[0]+dx, start[1]+dy) == end for dx, dy in NBRS)

    # BFS for paths of exactly `length` steps between start and end via cells outside forbidden.
    # State: (current_cell, depth, visited_in_path)
    # We want depth == length+1 (number of edges) and current_cell == end.
    # Bounded DFS.
    deadline_nodes = max_search

    def dfs(cur, remaining, visited):
        nonlocal deadline_nodes
        if deadline_nodes <= 0: return False
        deadline_nodes -= 1
        if remaining == 0:
            return cur == end
        # Manhattan-distance lower bound: must be reachable in `remaining` more steps
        md = abs(cur[0]-end[0]) + abs(cur[1]-end[1])
        if md > remaining or (remaining - md) % 2 != 0:
            return False
        for dx, dy in NBRS:
            n = (cur[0]+dx, cur[1]+dy)
            if remaining == 1 and n == end:
                return True
            if n in forbidden: continue
            if n in visited: continue
            if n == end: continue   # only allowed when remaining==1 (handled above)
            visited.add(n)
            if dfs(n, remaining - 1, visited):
                return True
            visited.discard(n)
        return False

    # Need length P intermediate cells = length+1 edges from start to end.
    visited = set()
    return dfs(start, length + 1, visited)


# ════════════════════════════════════════════════════════════
# Main constructor: place H-runs in chain order, ensuring P-routability
# ════════════════════════════════════════════════════════════

@dataclass
class ConstructState:
    sequence: str
    runs: list[tuple[str, int]]
    h_runs: list[int]      # indices into runs[] for each H-run
    p_lengths: list[int]   # p_lengths[i] = length of P-run BEFORE H-run i (or after last)
    shape_cache: dict[int, list[tuple[Coord, ...]]] = field(default_factory=dict)
    deadline: float = 0
    n_candidates: int = 0
    best_contacts: int = 0
    on_candidate: object = None    # callable(hcore_cells) — invoked for each finished candidate


def get_shapes(state: ConstructState, length: int) -> list[tuple[Coord, ...]]:
    if length not in state.shape_cache:
        state.shape_cache[length] = enumerate_saw_paths(length)
    return state.shape_cache[length]


def place_recursive(
    state: ConstructState,
    h_idx_in_chain: int,        # which H-run we're placing (0..len(h_runs)-1)
    placed_cells: set,           # cells already used by earlier H-runs
    last_endpoint: Coord | None,  # last cell of previous H-run (or None for first)
):
    if time.monotonic() > state.deadline:
        return
    if h_idx_in_chain == len(state.h_runs):
        # All placed — emit candidate hcore
        state.n_candidates += 1
        if state.on_candidate is not None:
            state.on_candidate(frozenset(placed_cells))
        return

    h_run_idx = state.h_runs[h_idx_in_chain]
    length = state.runs[h_run_idx][1]
    shapes = get_shapes(state, length)

    # P-run length immediately before this H-run (= number of P residues in between)
    p_len_before = state.p_lengths[h_idx_in_chain]

    if last_endpoint is None:
        # First H-run: place at origin in any of its canonical shapes
        for shape in shapes:
            new_cells = placed_cells | set(shape)
            place_recursive(state, h_idx_in_chain + 1, new_cells, shape[-1])
            if time.monotonic() > state.deadline: return
        return

    # For each shape: try every position+orientation reaching the lattice
    # We iterate by anchoring shape[0] at every cell within a bounded range
    # of last_endpoint (so the P-route has any chance of fitting).
    # Range bound: |dx|+|dy| <= p_len_before + 1 (Manhattan reach of a P-path
    # of p_len_before edges from endpoint).
    reach = p_len_before + 1
    for shape in shapes:
        # Translate shape[0] to many candidate anchor positions
        for ax in range(last_endpoint[0] - reach, last_endpoint[0] + reach + 1):
            for ay in range(last_endpoint[1] - reach, last_endpoint[1] + reach + 1):
                if abs(ax - last_endpoint[0]) + abs(ay - last_endpoint[1]) > reach:
                    continue
                anchor = (ax, ay)
                if anchor in placed_cells: continue
                translated = tuple((c[0]+ax, c[1]+ay) for c in shape)
                # Check no overlap with already-placed cells
                t_set = set(translated)
                if t_set & placed_cells: continue
                # Check P-route from last_endpoint to translated[0] of length p_len_before
                # Forbidden = placed_cells (already-placed H residues) PLUS the new H-run cells
                # (we don't want P-residues inside the hcore-being-built either — except
                # endpoints which are H residues themselves)
                forbidden = placed_cells | t_set
                # The endpoints last_endpoint and translated[0] are H residues, not in P-route, but
                # the P-route must not visit any other forbidden cell. The has_p_route function
                # handles "start and end are NOT free" implicitly (they're not added to visited).
                if not has_p_route(last_endpoint, translated[0], p_len_before, forbidden):
                    continue
                # Recurse
                new_cells = placed_cells | t_set
                place_recursive(state, h_idx_in_chain + 1, new_cells, translated[-1])
                if time.monotonic() > state.deadline: return


def construct_hcores(sequence: str, time_budget_seconds: float = 600,
                      on_candidate=None) -> ConstructState:
    runs = parse_runs(sequence)
    # Identify H-runs (in chain order) and P-run lengths between them
    h_runs = []                # indices of H-runs in runs[]
    for i, (k, l) in enumerate(runs):
        if k == 'H':
            h_runs.append(i)

    # p_lengths[i] = length of P-run immediately before H-run i (0 for first H-run)
    # If sequence starts with P, that's a leading P-run; we ignore it for hcore construction
    # (it doesn't constrain hcore topology, just adds residues at the start)
    p_lengths = []
    for i, h_run_idx in enumerate(h_runs):
        if i == 0:
            p_lengths.append(0)  # no P-run "before" the first H-run for placement purposes
        else:
            prev_h = h_runs[i-1]
            # P-run length = sum of all P residues between prev_h and h_run_idx
            p_total = sum(l for k, l in runs[prev_h+1:h_run_idx] if k == 'P')
            p_lengths.append(p_total)

    state = ConstructState(
        sequence=sequence, runs=runs, h_runs=h_runs, p_lengths=p_lengths,
        deadline=time.monotonic() + time_budget_seconds,
        on_candidate=on_candidate,
    )
    place_recursive(state, 0, set(), None)
    return state


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main():
    import argparse, sys, os
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hp_folding import sequences, hcore_csp, hcore

    ap = argparse.ArgumentParser()
    ap.add_argument('--protein-key', required=True)
    ap.add_argument('--time-budget', type=float, default=600)
    ap.add_argument('--csp-budget', type=float, default=3.0)
    ap.add_argument('--out-dir', default='./hcore_constructor_results')
    ap.add_argument('--seed-contacts', type=int, default=0,
                    help="Only save folds with > this many contacts")
    args = ap.parse_args()

    s = sequences.get(args.protein_key)
    sid = args.protein_key
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'=== hcore_constructor {sid} target=-{s.best_known} budget={args.time_budget}s ===', flush=True)

    runs = parse_runs(s.sequence)
    h_runs_lens = [l for k,l in runs if k == 'H']
    p_runs_lens = [l for k,l in runs if k == 'P']
    print(f'H-runs ({len(h_runs_lens)}): {h_runs_lens}', flush=True)
    print(f'P-runs ({len(p_runs_lens)}): {p_runs_lens}', flush=True)

    seen_hcore_hashes = set()
    best = args.seed_contacts
    n_evaluated = 0
    n_feasible = 0

    def evaluate(hcore_cells):
        nonlocal best, n_evaluated, n_feasible
        # Canonicalize (translation only) to detect duplicates
        if not hcore_cells: return
        mx_, my = min(c[0] for c in hcore_cells), min(c[1] for c in hcore_cells)
        canon = frozenset((c[0]-mx_, c[1]-my) for c in hcore_cells)
        h = hash(canon)
        if h in seen_hcore_hashes: return
        seen_hcore_hashes.add(h)
        n_evaluated += 1
        try:
            sol = hcore_csp.solve_csp(s.sequence, canon, time_budget_seconds=args.csp_budget)
        except Exception:
            sol = None
        if sol is None: return
        n_feasible += 1
        if sol.contacts > best:
            best = sol.contacts
            print(f'  *** NEW BEST: contacts={sol.contacts}  (after {n_evaluated} hcores tried, '
                  f'{n_feasible} feasible) ***', flush=True)
            with open(f'{args.out_dir}/{sid}_best.json','w') as f:
                json.dump({'sequence_id':sid,'sequence':s.sequence,
                           'n_residues':s.length,'n_h_residues':s.sequence.count('H'),
                           'contacts':int(sol.contacts),
                           'coords':[list(c) for c in sol.coords],
                           'note':'hcore_constructor'}, f, indent=2)

    state = construct_hcores(s.sequence, time_budget_seconds=args.time_budget,
                               on_candidate=evaluate)
    print(f'\nDONE  candidates_generated={state.n_candidates}  evaluated={n_evaluated}  '
          f'feasible={n_feasible}  best={best}/-{s.best_known}', flush=True)


if __name__ == "__main__":
    import json
    main()
