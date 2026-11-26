"""Sweep all compact 22-cell shapes for S6 to bypass basin trap."""
from __future__ import annotations
import sys, time, json, multiprocessing as mp
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.shape_search import search_shape_constrained
from hp_folding.io import write_submission
from hp_folding.verifier import count_contacts

SEQ = BENCHMARKS["S6"].sequence
NH = SEQ.count("H")  # 22


def is_connected(cells):
    s = set(cells)
    if not s:
        return False
    start = next(iter(s))
    visited = {start}
    stack = [start]
    while stack:
        p = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (p[0] + dx, p[1] + dy)
            if n in s and n not in visited:
                visited.add(n)
                stack.append(n)
    return len(visited) == len(s)


def sweep_rectangle(rows, cols, n_remove, timeout_per_shape, p_may_enter):
    base = [(r, c) for r in range(rows) for c in range(cols)]
    if len(base) - n_remove != NH:
        return []
    out = []
    for removed in combinations(base, n_remove):
        cells = [c for c in base if c not in removed]
        if not is_connected(cells):
            continue
        out.append((rows, cols, removed, frozenset(cells)))
    return out


def worker(args):
    label, shape, timeout, p_may_enter = args
    coords, contacts, nodes = search_shape_constrained(
        SEQ, shape, timeout_seconds=timeout,
        seed=0, forward_check=True,
        p_may_enter_shape=p_may_enter,
    )
    return label, contacts, coords


def main():
    # All compact-rectangle shapes (rows*cols cells, remove just enough to land at 22)
    candidates = []
    for rows, cols in [(4, 6), (6, 4), (5, 5), (4, 7), (7, 4),
                        (5, 6), (6, 5), (3, 8), (8, 3),
                        (4, 8), (8, 4), (5, 7), (7, 5),
                        (6, 6), (3, 9), (9, 3)]:
        n_remove = rows * cols - NH
        if n_remove < 0 or n_remove > 6:
            continue
        shapes = sweep_rectangle(rows, cols, n_remove, 2, True)
        for s in shapes:
            r, c, rem, shape_set = s
            label = f"{r}x{c}_minus_{n_remove}_at_{tuple(sorted(rem))}"
            candidates.append((label, shape_set))
    print(f"Total compact shapes to try: {len(candidates)}", flush=True)
    if not candidates:
        return

    work = [(lab, sh, 3, True) for lab, sh in candidates]
    t0 = time.monotonic()
    best_c = 0
    best_coords = None
    best_label = None
    n_done = 0
    n_feasible = 0
    with mp.Pool(14) as pool:
        for label, contacts, coords in pool.imap_unordered(worker, work, chunksize=4):
            n_done += 1
            if coords is not None:
                n_feasible += 1
                c = count_contacts(SEQ, coords)
                if c > best_c:
                    best_c = c
                    best_coords = coords
                    best_label = label
                    print(f"  [{n_done}/{len(work)}] NEW BEST: {label} → {c} contacts",
                          flush=True)
            if n_done % 100 == 0:
                elapsed = time.monotonic() - t0
                print(f"  [{n_done}/{len(work)}] feasible={n_feasible} "
                      f"best={best_c} elapsed={elapsed:.0f}s", flush=True)

    print(f"\nDone: {n_done} shapes tried, {n_feasible} feasible, best={best_c}")
    if best_coords is not None and best_c >= 20:
        print(f"  *** {best_c}-FOLD VIA SHAPE SEARCH ***")
        print(f"  shape: {best_label}")
        out = ROOT / f"results/S6_shape_sweep_c{best_c}.json"
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=SEQ, coords=best_coords), "S6", out)
        if best_c > 19:
            import shutil
            shutil.copy(out, ROOT / "submissions/S6.json")
            print(f"  PROMOTED submissions/S6.json: 19 -> {best_c}")


if __name__ == "__main__":
    main()
