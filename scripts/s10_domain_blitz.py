"""Parallel 10,000-shape domain-level CP-SAT routability blitz for S10 >= 51.

Fixes the top+central-sheet (0..48) of S10.json and hunts for ANY 28-cell bottom-domain
H-core (49..99, including both the 9-P linker 49..57 and the 6-P linker 91..96) achieving
bot_score >= 46 (current = 45 => +1 contact => S10 = 51 world record!).
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ortools.sat.python import cp_model

from hp_folding.polyomino_enum import internal_edges
from hp_folding.sequences import BENCHMARKS
from hp_folding.verifier import verify

REPO = Path(__file__).resolve().parents[1]
SEQ = BENCHMARKS["S10"].sequence
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

_d = json.load(open(REPO / "submissions" / "S10.json"))
_raw = [tuple(c) for c in _d["coords"]]
_x0, _y0 = _raw[48]
COORDS = [(x - _x0, y - _y0) for x, y in _raw]
FIXED_CELLS = {COORDS[i] for i in range(0, 49)}
WALL_H = {COORDS[i] for i in range(0, 49) if SEQ[i] == "H"}
SUB_IDX = list(range(49, 100))
BONDS = [(i, i + 1) for i in range(49, 99)]
PAR_OFF = (COORDS[49][0] + COORDS[49][1] - 49) % 2
NEED_EV = sum(1 for i in SUB_IDX if SEQ[i] == "H" and (i + PAR_OFF) % 2 == 0)
HC0 = {COORDS[i] for i in SUB_IDX if SEQ[i] == "H"}


def score(hc: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> int:
    return internal_edges(hc) + sum(
        1 for c in hc for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in WALL_H
    )


def route_full_bot(hcore: frozenset[tuple[int, int]], budget: float = 0.6):
    xs = [c[0] for c in hcore]
    ys = [c[1] for c in hcore]
    cells = [
        (x, y)
        for x in range(min(xs) - 4, max(xs) + 5)
        for y in range(min(ys) - 4, max(ys) + 5)
        if (x, y) not in FIXED_CELLS
    ]
    cid = {c: i for i, c in enumerate(cells)}
    h_ids = frozenset(cid[c] for c in hcore if c in cid)
    if len(h_ids) != len(hcore):
        return None, "INFEASIBLE"
    p_ids = sorted(set(range(len(cells))) - h_ids)
    h_by_p = {
        0: sorted(i for i in h_ids if (cells[i][0] + cells[i][1]) % 2 == 0),
        1: sorted(i for i in h_ids if (cells[i][0] + cells[i][1]) % 2 == 1),
    }
    p_by_p = {
        0: sorted(i for i in p_ids if (cells[i][0] + cells[i][1]) % 2 == 0),
        1: sorted(i for i in p_ids if (cells[i][0] + cells[i][1]) % 2 == 1),
    }
    adj_pairs = [
        (i, cid[(x + dx, y + dy)])
        for i, (x, y) in enumerate(cells)
        for dx, dy in NBRS
        if (x + dx, y + dy) in cid
    ]
    model = cp_model.CpModel()
    xv = {}
    p49_dom = [cid[(dx, dy)] for dx, dy in NBRS if (dx, dy) in cid and cid[(dx, dy)] not in h_ids]
    if not p49_dom:
        return None, "INFEASIBLE"
    xv[49] = model.NewIntVarFromDomain(cp_model.Domain.FromValues(sorted(p49_dom)), "x_49")
    for i in SUB_IDX[1:]:
        par = (i + PAR_OFF) % 2
        dom = h_by_p[par] if SEQ[i] == "H" else p_by_p[par]
        if not dom:
            return None, "INFEASIBLE"
        xv[i] = model.NewIntVarFromDomain(cp_model.Domain.FromValues(dom), f"x_{i}")
    model.AddAllDifferent(list(xv.values()))
    for a, b in BONDS:
        model.AddAllowedAssignments([xv[a], xv[b]], adj_pairs)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = 2
    st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {i: cells[solver.Value(xv[i])] for i in SUB_IDX}, solver.StatusName(st)
    return None, solver.StatusName(st)


def check_batch(batch: list[tuple[frozenset[tuple[int, int]], int]]):
    inf = unk = 0
    for hc, sc in batch:
        sol, st = route_full_bot(hc, budget=0.6)
        if sol is not None:
            return ("HIT", sc, sol, inf, unk)
        if st == "INFEASIBLE":
            inf += 1
        else:
            unk += 1
    return ("DONE", 0, None, inf, unk)


def main() -> None:
    rng = random.Random(9999)
    bot_rects = [
        HC0,
        {(x, y) for x in range(0, 6) for y in range(-5, 0)} - {(0, -5), (5, -5)},
        {(x, y) for x in range(0, 7) for y in range(-4, 0)},
        {(x, y) for x in range(-1, 6) for y in range(-4, 0)},
        {(x, y) for x in range(0, 5) for y in range(-6, 0)} - {(0, -6), (4, -6)},
        {(x, y) for x in range(1, 7) for y in range(-5, 0)} - {(1, -5), (6, -5)},
        {(x, y) for x in range(-1, 7) for y in range(-4, 0)}
        - {(-1, -4), (6, -4), (-1, -3), (6, -3)},
        {(x, y) for x in range(0, 8) for y in range(-4, 0)}
        - {(0, -4), (7, -4), (6, -4), (7, -3)},
    ]
    cands: dict[frozenset[tuple[int, int]], int] = {}
    for s in bot_rects:
        for _ in range(1500):
            cur = set(s)
            for _ in range(rng.randint(1, 10)):
                b_rem = [
                    c for c in cur if sum(1 for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in cur) <= 3
                ]
                if not b_rem:
                    break
                rem = rng.choice(b_rem)
                sm = cur - {rem}
                b_add = list(
                    {
                        (x + dx, y + dy)
                        for x, y in sm
                        for dx, dy in NBRS
                        if (x + dx, y + dy) not in sm
                        and (x + dx, y + dy) not in FIXED_CELLS
                        and (y + dy) <= -1
                    }
                    - {rem}
                )
                if not b_add:
                    break
                scored = [
                    (
                        sum(
                            1
                            for dx, dy in NBRS
                            if (a[0] + dx, a[1] + dy) in sm or (a[0] + dx, a[1] + dy) in WALL_H
                        ),
                        a,
                    )
                    for a in b_add
                ]
                scored.sort(key=lambda x: (-x[0], rng.random()))
                add = rng.choice(scored[: min(7, len(scored))])[1]
                cur = sm | {add}
                sc = score(cur)
                if sc >= 46 and sum(1 for x, y in cur if (x + y) % 2 == 0) == NEED_EV:
                    key = frozenset(cur)
                    if key not in cands:
                        cands[key] = sc
                        if len(cands) >= 12000:
                            break
            if len(cands) >= 12000:
                break

    items = list(cands.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: (kv[1], rng.random()))
    items = items[:10000]
    print(f"Testing {len(items)} FULL-BOT (49..99) H-cores across 8 processes (score >= 46)...", flush=True)
    t0 = time.time()
    batches = [items[i : i + 250] for i in range(0, len(items), 250)]
    tot_inf = tot_unk = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(check_batch, b) for b in batches]
        for fut in as_completed(futs):
            kind, sc, sol, inf, unk = fut.result()
            tot_inf += inf
            tot_unk += unk
            if kind == "HIT":
                full_coords = list(COORDS[:49]) + [sol[i] for i in range(49, 100)]
                v = verify(SEQ, full_coords)
                print(
                    f"\n*** WORLD RECORD!! S10 contacts={v.contacts} valid={v.valid} "
                    f"(bot_score={sc}, {time.time()-t0:.1f}s)!! ***",
                    flush=True,
                )
                if v.valid and v.contacts >= 51:
                    out = REPO / f"s10_record_{v.contacts}.json"
                    out.write_text(
                        json.dumps(
                            {
                                "sequence_id": "S10",
                                "sequence": SEQ,
                                "lattice": "2D",
                                "contacts": v.contacts,
                                "coords": [list(c) for c in full_coords],
                            },
                            indent=2,
                        )
                    )
                    print(f"SAVED VERIFIED WORLD RECORD TO {out}!!", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    return
            if (tot_inf + tot_unk) % 2000 == 0 or (tot_inf + tot_unk) == len(items):
                print(
                    f"  [{tot_inf+tot_unk}/{len(items)}] inf={tot_inf}, unk={tot_unk} ({time.time()-t0:.1f}s)",
                    flush=True,
                )
    print(f"Done in {time.time()-t0:.1f}s: 0/{len(items)} routable (inf={tot_inf}, unk={tot_unk})", flush=True)


if __name__ == "__main__":
    main()
