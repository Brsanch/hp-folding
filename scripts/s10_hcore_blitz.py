"""S10 (2D100b, len=100) parallel CP-SAT routability blitz at ie in [90, 95] (>= 51 contacts).

Combines two complementary H-core generators (all parity-balanced 28E/28O, connected,
and satisfying the exact HP-transition bound P(H) >= 34 <=> ie <= 95):
  1. Multi-basin 1/2/3-cell H-core modifications from ALL distinct >=48-contact S10
     basins on disk (11 distinct basins).
  2. Notched compact rectangles (7x8, 7x9, 6x10, 8x8, 8x9) perturbed by boundary
     notches & corner transfers to match S10's 8 single-P bridges (H-P-H) at ie=90..93.

At ~40ms/check across 8 parallel processes, tests ~2,500+ distinct H-cores in ~60-90s.
"""
from __future__ import annotations

import glob
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges
from hp_folding.sequences import BENCHMARKS
from hp_folding.verifier import verify

REPO = Path(__file__).resolve().parents[1]
SEQ = BENCHMARKS["S10"].sequence
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return True
    st = [next(iter(cells))]
    seen = {st[0]}
    while st:
        x, y = st.pop()
        for dx, dy in NBRS:
            nb = (x + dx, y + dy)
            if nb in cells and nb not in seen:
                seen.add(nb)
                st.append(nb)
    return len(seen) == len(cells)


def parity_ok(cells: set[tuple[int, int]]) -> bool:
    return 2 * sum(1 for x, y in cells if (x + y) % 2 == 0) == len(cells)


def random_walk_mods(
    seed_hc: set[tuple[int, int]],
    rng: random.Random,
    *,
    n_walks: int = 200,
    steps: int = 4,
    min_ie: int = 90,
    max_ie: int = 95,
) -> dict[tuple[tuple[int, int], ...], tuple[int, str]]:
    found: dict[tuple[tuple[int, int], ...], tuple[int, str]] = {}
    for _ in range(n_walks):
        cur = set(seed_hc)
        for _ in range(steps):
            # Pick a removable cell (prefer low-degree boundary cells)
            cands_rem = []
            for c in cur:
                deg = sum(1 for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in cur)
                if deg <= 3:
                    cands_rem.append((deg, c))
            if not cands_rem:
                break
            rng.shuffle(cands_rem)
            rem = None
            for _, c in cands_rem[:10]:
                sm = cur - {c}
                if is_connected(sm):
                    rem = c
                    break
            if rem is None:
                break
            sm = cur - {rem}
            boundary = list({(x + dx, y + dy) for x, y in sm for dx, dy in NBRS if (x + dx, y + dy) not in sm} - {rem})
            if not boundary:
                break
            # Bias toward adding cells with >= 2 neighbors to keep ie high
            scored = [(sum(1 for dx, dy in NBRS if (b[0] + dx, b[1] + dy) in sm), b) for b in boundary]
            scored.sort(key=lambda x: (-x[0], rng.random()))
            add = rng.choice(scored[: min(8, len(scored))])[1]
            cur = sm | {add}
            ie = internal_edges(cur)
            if min_ie <= ie <= max_ie and parity_ok(cur):
                canon = canonicalize(cur)
                if canon not in found:
                    found[canon] = (ie, "basin_walk")
    return found


def notched_rectangles(
    rng: random.Random,
    *,
    target_count: int = 800,
    min_ie: int = 90,
    max_ie: int = 93,
) -> dict[tuple[tuple[int, int], ...], tuple[int, str]]:
    """Build 56-cell H-cores by perturbing compact rectangles (7x8, 6x10, 8x8, 7x9, 5x12)."""
    found: dict[tuple[tuple[int, int], ...], tuple[int, str]] = {}
    bases = [
        {(x, y) for x in range(7) for y in range(8)},  # 56, ie=97
        {(x, y) for x in range(6) for y in range(10)} - {(0, 0), (5, 0), (0, 9), (5, 9)},  # 56
        {(x, y) for x in range(8) for y in range(8)}
        - {(0, 0), (1, 0), (0, 1), (7, 0), (7, 7), (6, 7), (7, 6), (0, 7)},  # 56
        {(x, y) for x in range(7) for y in range(9)}
        - {(0, 0), (6, 0), (0, 8), (6, 8), (3, 0), (3, 8), (0, 4)},  # 56
    ]
    attempts = 0
    while len(found) < target_count and attempts < target_count * 25:
        attempts += 1
        cur = set(rng.choice(bases))
        n_moves = rng.randint(2, 7)
        for _ in range(n_moves):
            # Remove a boundary cell (creating a notch) and add on another boundary face
            b_rem = [c for c in cur if sum(1 for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in cur) <= 3]
            if not b_rem:
                break
            rem = rng.choice(b_rem)
            sm = cur - {rem}
            if not is_connected(sm):
                continue
            boundary = list({(x + dx, y + dy) for x, y in sm for dx, dy in NBRS if (x + dx, y + dy) not in sm} - {rem})
            if not boundary:
                continue
            add = rng.choice(boundary)
            cur = sm | {add}
        ie = internal_edges(cur)
        if min_ie <= ie <= max_ie and parity_ok(cur) and is_connected(cur):
            canon = canonicalize(cur)
            if canon not in found:
                found[canon] = (ie, "notched_rect")
    return found


def _check_one(args: tuple[int, str, tuple[tuple[int, int], ...], float, int]):
    ie, src, canon, budget, workers = args
    coords, st = is_routable_cpsat(
        SEQ, frozenset(canon), time_budget_seconds=budget, bbox_pad=5, n_workers=workers
    )
    if coords is not None:
        v = verify(SEQ, coords)
        return ie, src, st, coords, v.contacts if v.valid else -1
    return ie, src, st, None, 0


def main() -> None:
    rng = random.Random(20261123)
    # 1. Load all distinct S10 basins >= 48 contacts
    basins: list[tuple[int, str, set[tuple[int, int]]]] = []
    seen_hc: set[tuple[tuple[int, int], ...]] = set()
    for pat in ("submissions/S10*.json", "record_push_results/S10*.json", "record_attempt_results/S10*.json"):
        for f in sorted(glob.glob(str(REPO / pat))):
            try:
                d = json.load(open(f))
                if d.get("sequence") != SEQ or "coords" not in d:
                    continue
                coords = [tuple(c) for c in d["coords"]]
                v = verify(SEQ, coords)
                if v.valid and v.contacts >= 48:
                    hc = {coords[i] for i, r in enumerate(SEQ) if r == "H"}
                    c = canonicalize(hc)
                    if c not in seen_hc:
                        seen_hc.add(c)
                        basins.append((v.contacts, Path(f).name, hc))
            except Exception:
                pass
    basins.sort(key=lambda x: -x[0])
    print(f"Loaded {len(basins)} canonical distinct S10 basins (>=48 contacts)", flush=True)

    all_cands: dict[tuple[tuple[int, int], ...], tuple[int, str]] = {}
    for c_val, name, hc in basins:
        got = random_walk_mods(hc, rng, n_walks=160, steps=4, min_ie=90, max_ie=95)
        for k, v in got.items():
            all_cands.setdefault(k, (v[0], f"walk:{name}"))

    rect_cands = notched_rectangles(rng, target_count=800, min_ie=90, max_ie=93)
    for k, v in rect_cands.items():
        all_cands.setdefault(k, v)

    # Prioritize ie=90 and ie=91 (most routable!) with mix of 92-93
    items = [(ie, src, canon) for canon, (ie, src) in all_cands.items()]
    rng.shuffle(items)
    items.sort(key=lambda x: (x[0], rng.random()))  # ie=90 first, then 91, 92...
    items = items[:2000]
    ie_dist: dict[int, int] = {}
    for ie, _, _ in items:
        ie_dist[ie] = ie_dist.get(ie, 0) + 1
    print(f"Testing {len(items)} distinct parity-balanced S10 H-cores: ie_dist={ie_dist}", flush=True)

    t0 = time.time()
    n_inf = n_unk = n_done = 0
    jobs = [(ie, src, canon, 3.0, 2) for ie, src, canon in items]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_check_one, j) for j in jobs]
        for fut in as_completed(futs):
            ie, src, st, c_out, contacts = fut.result()
            n_done += 1
            if c_out is not None and contacts >= 51:
                out = REPO / f"s10_record_{contacts}.json"
                out.write_text(
                    json.dumps(
                        {
                            "sequence_id": "S10",
                            "sequence": SEQ,
                            "lattice": "2D",
                            "contacts": contacts,
                            "coords": [list(c) for c in c_out],
                            "note": f"S10 blitz hit ({src}) at ie={ie}",
                        },
                        indent=2,
                    )
                )
                print(
                    f"\n*** WORLD RECORD S10={contacts} (ie={ie}, {src}) FOUND at #{n_done} in {time.time()-t0:.1f}s!! ***",
                    flush=True,
                )
                ex.shutdown(wait=False, cancel_futures=True)
                return
            if st == "INFEASIBLE":
                n_inf += 1
            else:
                n_unk += 1
            if n_done % 250 == 0 or n_done == len(items):
                print(
                    f"  [{n_done}/{len(items)}] inf={n_inf}, unk={n_unk} ({time.time()-t0:.1f}s elapsed)",
                    flush=True,
                )

    print(f"\nS10 blitz complete in {time.time()-t0:.1f}s: 0/{len(items)} routable (inf={n_inf}, unk={n_unk})", flush=True)


if __name__ == "__main__":
    main()
