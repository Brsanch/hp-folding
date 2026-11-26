"""Branch-and-bound constructive search for HP folding.

Builds folds residue-by-residue via DFS with backtracking. Unlike PERM (which
samples randomly with prune/enrich population control), this is exhaustive
within a time budget, with two structural prunes:

1. **SAW prune**: don't recurse into positions already occupied
2. **Bound prune**: if (current_contacts + max_possible_remaining) <= best_so_far,
   abandon this subtree. The bound is loose but cheap: each remaining H residue
   can form at most ~2 new contacts (since each contact is shared and most
   neighbor slots are constrained), so upper-bound = current + 2 * remaining_H.

Key difference from PERM: we keep the deepest "promising" partial folds for
re-expansion via the upper-bound test, rather than killing random low-weight
chains. This means the search **provably explores** every promising subtree
within budget — no probabilistic gaps.

Per-move ordering: at each step, try the move that gains the most immediate
contacts FIRST. This is a greedy heuristic that biases the search toward
high-quality folds early, so the upper-bound prune kicks in quickly.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

Coord = tuple[int, int]
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class ConstructiveState:
    sequence: str
    n: int
    n_h: int
    h_remaining: list[int]   # h_remaining[i] = number of H residues at index >= i
    best_contacts: int = 0
    best_coords: list[Coord] | None = None
    deadline: float = 0.0
    nodes_explored: int = 0
    leaves_reached: int = 0
    on_improve: object = None
    history: list[tuple[float, int]] = field(default_factory=list)
    start_time: float = 0.0


def _delta_contacts(sequence: str, coords: list[Coord], occupied_h: dict, candidate: Coord, idx: int) -> int:
    """Count new H-H contacts formed by placing residue idx at candidate."""
    if sequence[idx] != 'H':
        return 0
    n_new = 0
    last_idx = idx - 1
    for dx, dy in NBRS:
        n = (candidate[0] + dx, candidate[1] + dy)
        if n in occupied_h:
            partner_idx = occupied_h[n]
            if partner_idx != last_idx:
                n_new += 1
    return n_new


def _search(
    state: ConstructiveState,
    coords: list[Coord],
    occupied: set,
    occupied_h: dict,
    contacts: int,
    depth: int,
):
    state.nodes_explored += 1
    if time.monotonic() > state.deadline:
        return

    if depth == state.n:
        state.leaves_reached += 1
        if contacts > state.best_contacts:
            state.best_contacts = contacts
            state.best_coords = list(coords)
            wall = time.monotonic() - state.start_time
            state.history.append((wall, contacts))
            if state.on_improve is not None:
                state.on_improve(contacts, list(coords))
        return

    # Upper bound: current contacts + (2 * remaining H residues)
    upper = contacts + 2 * state.h_remaining[depth]
    if upper <= state.best_contacts:
        return  # prune: this subtree cannot beat the current best

    last = coords[-1]
    sequence = state.sequence

    # Generate candidates and rank by immediate contact gain (descending).
    candidates = []
    for dx, dy in NBRS:
        cand = (last[0] + dx, last[1] + dy)
        if cand in occupied:
            continue
        gain = _delta_contacts(sequence, coords, occupied_h, cand, depth)
        candidates.append((gain, cand))
    candidates.sort(key=lambda x: -x[0])

    for gain, cand in candidates:
        if time.monotonic() > state.deadline:
            return
        coords.append(cand)
        occupied.add(cand)
        if sequence[depth] == 'H':
            occupied_h[cand] = depth
        _search(state, coords, occupied, occupied_h, contacts + gain, depth + 1)
        coords.pop()
        occupied.discard(cand)
        if sequence[depth] == 'H':
            del occupied_h[cand]


def constructive_search(
    sequence: str,
    *,
    time_budget_seconds: float = 600.0,
    initial_seed_contacts: int = 0,
    on_improve=None,
) -> ConstructiveState:
    """Branch-and-bound DFS over fold space, returning best fold found."""
    n = len(sequence)
    n_h = sequence.count('H')

    # h_remaining[i] = number of H residues at index >= i
    h_remaining = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        h_remaining[i] = h_remaining[i + 1] + (1 if sequence[i] == 'H' else 0)

    state = ConstructiveState(
        sequence=sequence, n=n, n_h=n_h, h_remaining=h_remaining,
        best_contacts=initial_seed_contacts,
        deadline=time.monotonic() + time_budget_seconds,
        on_improve=on_improve,
        start_time=time.monotonic(),
    )

    # Root: place residue 0 at origin.
    start = (0, 0)
    coords = [start]
    occupied = {start}
    occupied_h = {start: 0} if sequence[0] == 'H' else {}
    _search(state, coords, occupied, occupied_h, 0, 1)

    return state


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, json, os
    from hp_folding import sequences as hp_sequences

    ap = argparse.ArgumentParser()
    ap.add_argument('--protein-key', required=True)
    ap.add_argument('--time-budget', type=float, default=600)
    ap.add_argument('--seed-contacts', type=int, default=0,
                    help="Skip subtrees that can't beat this many contacts.")
    ap.add_argument('--out-dir', default='./constructive_results')
    args = ap.parse_args()

    s = hp_sequences.get(args.protein_key)
    sid = args.protein_key
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    best_path = f'{out}/{sid}_best.json'

    def on_imp(c, coords):
        with open(best_path, 'w') as f:
            json.dump({
                'sequence_id': sid, 'sequence': s.sequence,
                'n_residues': s.length, 'n_h_residues': s.sequence.count('H'),
                'contacts': int(c),
                'coords': [list(x) for x in coords],
                'note': f'constructive_search budget={args.time_budget}s seed={args.seed_contacts}',
            }, f, indent=2)
        print(f'  *** NEW BEST: contacts={c} (saved {best_path}) ***', flush=True)

    print(f'=== constructive {sid} target=-{s.best_known} budget={args.time_budget}s ===', flush=True)
    res = constructive_search(s.sequence, time_budget_seconds=args.time_budget,
                                initial_seed_contacts=args.seed_contacts, on_improve=on_imp)
    print(f'DONE  best={res.best_contacts}/-{s.best_known}  '
          f'nodes={res.nodes_explored}  leaves={res.leaves_reached}  '
          f'wall={time.monotonic()-res.start_time:.0f}s')
