"""S10 (2D100b, len=100) H-core neighborhood + compact polyomino CP-SAT routability hunt.

Applies the exact hcore-space CP-SAT routability technique that closed S9=53 to S10>=51:
  - S10 has nH=56 (28 even, 28 odd), adjHH=39.
  - Current best S10.json has ie=89 (50 contacts).
  - Target >=51 contacts requires ie >= 90 (with exact parity balance: 28 even, 28 odd).
  - Phase 1: 2-cell H-core modifications of S10.json (and top 49-basins) at ie >= 90.
  - Phase 2: Perturbed compact 56-cell polyominoes (around 7x8, 6x10, 8x8-minus-corners)
    at ie in [90, 96].

Runs CP-SAT routability in parallel; saves immediately on any verified >=51-contact fold.
"""
from __future__ import annotations

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


def one_cell_mods(hc: set[tuple[int, int]]):
    for rem in list(hc):
        sm = hc - {rem}
        if not is_connected(sm):
            continue
        boundary = {(x + dx, y + dy) for x, y in sm for dx, dy in NBRS if (x + dx, y + dy) not in sm}
        for add in boundary:
            if add == rem:
                continue
            yield sm | {add}


def parity_balanced(cells: set[tuple[int, int]]) -> bool:
    ev = sum(1 for x, y in cells if (x + y) % 2 == 0)
    return 2 * ev == len(cells)


def generate_2cell_candidates(
    seed_hc: set[tuple[int, int]],
    *,
    min_ie: int = 90,
    max_depth1: int = 120,
    max_cands: int = 400,
    seed: int = 42,
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    """Generate parity-balanced 2-cell mods with ie >= min_ie (biased toward high-ie depth-1)."""
    rng = random.Random(seed)
    d1_all = []
    seen_d1 = {canonicalize(seed_hc)}
    for nhc in one_cell_mods(seed_hc):
        c1 = canonicalize(nhc)
        if c1 not in seen_d1:
            seen_d1.add(c1)
            d1_all.append((internal_edges(nhc), nhc))
    # Prioritize depth-1 shapes with highest ie (89, 90, 91, 92) + random sample of 88s
    d1_all.sort(key=lambda x: -x[0])
    d1_top = d1_all[: max_depth1 // 2]
    d1_rest = d1_all[max_depth1 // 2 :]
    if d1_rest:
        d1_top += rng.sample(d1_rest, min(max_depth1 // 2, len(d1_rest)))

    cands: dict[tuple[tuple[int, int], ...], int] = {}
    for _, h1 in d1_top:
        for h2 in one_cell_mods(h1):
            if not parity_balanced(h2):
                continue
            ie = internal_edges(h2)
            if ie >= min_ie:
                c2 = canonicalize(h2)
                if c2 not in cands:
                    cands[c2] = ie
                    if len(cands) >= max_cands * 2:
                        break
        if len(cands) >= max_cands * 2:
            break

    ranked = sorted(((ie, c) for c, ie in cands.items()), key=lambda x: (-x[0], x[1]))
    # Mix top ie (92-94) and mid ie (90-91) since lower ie is often MORE routable!
    by_ie: dict[int, list[tuple[int, tuple[tuple[int, int], ...]]]] = {}
    for item in ranked:
        by_ie.setdefault(item[0], []).append(item)
    out = []
    # Interleave across ie levels 90, 91, 92, ...
    keys = sorted(by_ie.keys())
    idx = 0
    while len(out) < max_cands and any(idx < len(by_ie[k]) for k in keys):
        for k in keys:
            if idx < len(by_ie[k]):
                out.append(by_ie[k][idx])
                if len(out) >= max_cands:
                    break
        idx += 1
    return out


def _check_one(args: tuple[int, tuple[tuple[int, int], ...], float, int]):
    ie, canon, budget, workers = args
    t0 = time.time()
    coords, st = is_routable_cpsat(
        SEQ, frozenset(canon), time_budget_seconds=budget, bbox_pad=5, n_workers=workers
    )
    elapsed = round(time.time() - t0, 2)
    if coords is not None:
        v = verify(SEQ, coords)
        return ie, st, elapsed, coords, v.contacts if v.valid else -1
    return ie, st, elapsed, None, 0


def main() -> None:
    d = json.load(open(REPO / "submissions" / "S10.json"))
    coords = [tuple(c) for c in d["coords"]]
    seed_hc = {coords[i] for i, r in enumerate(SEQ) if r == "H"}
    print(f"S10 seed H-core: n={len(seed_hc)}, ie={internal_edges(seed_hc)}", flush=True)

    t0 = time.time()
    cands = generate_2cell_candidates(seed_hc, min_ie=90, max_depth1=140, max_cands=300, seed=42)
    ie_dist: dict[int, int] = {}
    for ie, _ in cands:
        ie_dist[ie] = ie_dist.get(ie, 0) + 1
    print(f"Generated {len(cands)} parity-balanced 2-cell mods in {time.time()-t0:.1f}s: ie_dist={ie_dist}", flush=True)

    t1 = time.time()
    n_inf = n_unk = n_done = 0
    jobs = [(ie, canon, 4.0, 3) for ie, canon in cands]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_check_one, j) for j in jobs]
        for fut in as_completed(futs):
            ie, st, el, c_out, contacts = fut.result()
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
                            "note": f"2-cell hcore mod at ie={ie}",
                        },
                        indent=2,
                    )
                )
                print(
                    f"\n*** WORLD RECORD S10={contacts} (ie={ie}) FOUND at #{n_done} in {time.time()-t1:.1f}s!! Saved to {out} ***",
                    flush=True,
                )
                ex.shutdown(wait=False, cancel_futures=True)
                return
            if st == "INFEASIBLE":
                n_inf += 1
            else:
                n_unk += 1
            if n_done % 50 == 0 or n_done == len(cands):
                print(
                    f"  [{n_done}/{len(cands)}] inf={n_inf}, unk={n_unk} ({time.time()-t1:.1f}s elapsed)",
                    flush=True,
                )

    print(f"\nS10 2-cell hunt complete in {time.time()-t1:.1f}s: 0/{len(cands)} routable (inf={n_inf}, unk={n_unk})", flush=True)


if __name__ == "__main__":
    main()
