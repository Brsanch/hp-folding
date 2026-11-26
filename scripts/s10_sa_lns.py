"""Multi-basin Simulated-Annealing Spatial/Multi-Strand CP-SAT LNS on S10 (len=100).

Starts from all distinct 49- and 50-contact S10 basins on disk and alternates:
  - Exact 2-strand / spatial-window CP-SAT LNS (non-decreasing + H-core diversity)
  - Controlled 1-contact dip escapes (accepting a maximally displaced -1 fold to
    break out of locked 50-contact basins, then re-climbing via overlapping windows)
  - 1-cell parity-balanced hcore-neighborhood CP-SAT check on every new 50-basin.

Goal: either discover a verified >=51-contact fold on S10 (beating Wust-Landau 2012)
or document thorough multi-basin LNS exhaustion across distinct 50-contact basins.
"""
from __future__ import annotations

import glob
import json
import random
import time
from pathlib import Path

from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges
from hp_folding.sequences import BENCHMARKS
from hp_folding.spatial_lns import (
    NBRS,
    lns_step_cpsat,
    select_spatial_window,
    select_two_strand_window,
)
from hp_folding.verifier import verify

REPO = Path(__file__).resolve().parents[1]
SEQ = BENCHMARKS["S10"].sequence


def lns_dip_step(
    coords: list[tuple[int, int]],
    free: set[int],
    *,
    pad: int = 3,
    budget: float = 3.0,
    n_workers: int = 6,
) -> tuple[list[tuple[int, int]] | None, int]:
    """Allow a 1-contact dip (-1) to escape a locked basin while maximizing H-core displacement."""
    from ortools.sat.python import cp_model

    n = len(SEQ)
    fixed_indices = set(range(n)) - free
    fixed_cells = {coords[i] for i in fixed_indices}
    fixed_h_cells = {coords[i] for i in fixed_indices if SEQ[i] == "H"}

    fxs = [coords[i][0] for i in free]
    fys = [coords[i][1] for i in free]
    free_grid = [
        (x, y)
        for x in range(min(fxs) - pad, max(fxs) + pad + 1)
        for y in range(min(fys) - pad, max(fys) + pad + 1)
        if (x, y) not in fixed_cells
    ]
    free_grid_set = set(free_grid)
    nbrs_of = {
        c: [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in free_grid_set]
        for c in free_grid
    }

    sorted_free = sorted(free)
    runs: list[tuple[int, int]] = []
    start = prev = sorted_free[0]
    for idx in sorted_free[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))

    valid_cells: dict[int, list[tuple[int, int]]] = {}
    for a, b in runs:
        la = coords[a - 1] if a > 0 else None
        ra = coords[b + 1] if b < n - 1 else None
        for i in range(a, b + 1):
            par = (coords[i][0] + coords[i][1]) % 2
            vc = []
            for c in free_grid:
                if (c[0] + c[1]) % 2 != par:
                    continue
                if la is not None and abs(c[0] - la[0]) + abs(c[1] - la[1]) > i - (a - 1):
                    continue
                if ra is not None and abs(c[0] - ra[0]) + abs(c[1] - ra[1]) > (b + 1) - i:
                    continue
                vc.append(c)
            valid_cells[i] = vc

    model = cp_model.CpModel()
    p = {}
    cell_to_res: dict[tuple[int, int], list[int]] = {c: [] for c in free_grid}
    for i in sorted_free:
        for c in valid_cells[i]:
            p[(i, c)] = model.NewBoolVar(f"p_{i}_{c[0]}_{c[1]}")
            cell_to_res[c].append(i)
    for i in sorted_free:
        model.AddExactlyOne([p[(i, c)] for c in valid_cells[i]])
    for c, rl in cell_to_res.items():
        if len(rl) > 1:
            model.AddAtMostOne([p[(i, c)] for i in rl])
    for a, b in runs:
        if a > 0:
            anc = coords[a - 1]
            allowed = {(anc[0] + dx, anc[1] + dy) for dx, dy in NBRS}
            for c in valid_cells[a]:
                if c not in allowed:
                    model.Add(p[(a, c)] == 0)
        if b < n - 1:
            anc = coords[b + 1]
            allowed = {(anc[0] + dx, anc[1] + dy) for dx, dy in NBRS}
            for c in valid_cells[b]:
                if c not in allowed:
                    model.Add(p[(b, c)] == 0)
        for i in range(a, b):
            vc_i, vc_n = set(valid_cells[i]), set(valid_cells[i + 1])
            for c in valid_cells[i]:
                nbs = [nb for nb in nbrs_of[c] if nb in vc_n]
                if not nbs:
                    model.Add(p[(i, c)] == 0)
                else:
                    model.Add(p[(i, c)] <= sum(p[(i + 1, nb)] for nb in nbs))
            for c in valid_cells[i + 1]:
                nbs = [nb for nb in nbrs_of[c] if nb in vc_i]
                if not nbs:
                    model.Add(p[(i + 1, c)] == 0)
                else:
                    model.Add(p[(i + 1, c)] <= sum(p[(i, nb)] for nb in nbs))

    h_var = {}
    for c in free_grid:
        hl = [i for i in cell_to_res[c] if SEQ[i] == "H"]
        if hl:
            hv = model.NewBoolVar(f"h_{c[0]}_{c[1]}")
            model.Add(hv == sum(p[(i, c)] for i in hl))
            h_var[c] = hv

    edge_terms = []
    for c, hv in h_var.items():
        w = sum(1 for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in fixed_h_cells)
        if w > 0:
            edge_terms.append(w * hv)
    for u, hu in h_var.items():
        if (u[0] + u[1]) % 2 != 0:
            continue
        for v in nbrs_of[u]:
            if v not in h_var:
                continue
            ze = model.NewBoolVar(f"z_{u}_{v}")
            model.Add(ze <= hu)
            model.Add(ze <= h_var[v])
            edge_terms.append(ze)

    cur_free_h = {coords[i] for i in free if SEQ[i] == "H"}
    cur_ff = sum(
        1 for c in cur_free_h for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in fixed_h_cells
    )
    cur_in = sum(
        1
        for u in cur_free_h
        if (u[0] + u[1]) % 2 == 0
        for dx, dy in NBRS
        if (u[0] + dx, u[1] + dy) in cur_free_h
    )
    base_sub = cur_ff + cur_in

    sub_edges = sum(edge_terms)
    # Allow up to 1-contact dip, but require moving at least 2 H-cells
    model.Add(sub_edges >= base_sub - 1)
    unchanged_h = sum(h_var[c] for c in cur_free_h if c in h_var)
    moved_h = len(cur_free_h) - unchanged_h
    model.Add(moved_h >= 2)
    model.Maximize(100 * (sub_edges - base_sub) + moved_h)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = n_workers
    st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        nc = list(coords)
        for i in sorted_free:
            nc[i] = next(c for c in valid_cells[i] if solver.Value(p[(i, c)]) == 1)
        v = verify(SEQ, nc)
        if v.valid:
            return nc, v.contacts
    return None, verify(SEQ, coords).contacts


def check_1cell_mods_51(coords: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Check all parity-balanced 1-cell hcore mods at ie >= 90 (>= 51 contacts)."""
    hc = {coords[i] for i, r in enumerate(SEQ) if r == "H"}

    def is_conn(cells):
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

    cands = []
    seen = set()
    for rem in list(hc):
        sm = hc - {rem}
        if not is_conn(sm):
            continue
        boundary = {(x + dx, y + dy) for x, y in sm for dx, dy in NBRS if (x + dx, y + dy) not in sm}
        for add in boundary:
            if add == rem:
                continue
            nhc = sm | {add}
            ev_c = sum(1 for x, y in nhc if (x + y) % 2 == 0)
            if 2 * ev_c != len(nhc):  # S10 has 28 even, 28 odd
                continue
            ie = internal_edges(nhc)
            if ie >= 90:
                canon = canonicalize(nhc)
                if canon not in seen:
                    seen.add(canon)
                    cands.append((ie, canon))
    cands.sort(key=lambda x: -x[0])
    for ie, canon in cands:
        c_out, st = is_routable_cpsat(SEQ, frozenset(canon), time_budget_seconds=5, bbox_pad=5, n_workers=6)
        if c_out is not None:
            v = verify(SEQ, c_out)
            if v.valid and v.contacts >= 51:
                return c_out
    return None


def main() -> None:
    # Load all unique S10 folds with contacts >= 49
    seeds: list[tuple[int, str, list[tuple[int, int]]]] = []
    seen_hc: set[frozenset[tuple[int, int]]] = set()
    for pat in ("submissions/S10*.json", "record_push_results/S10*.json", "record_attempt_results/S10*.json"):
        for f in sorted(glob.glob(str(REPO / pat))):
            try:
                d = json.load(open(f))
                if d.get("sequence") != SEQ or "coords" not in d:
                    continue
                coords = [tuple(c) for c in d["coords"]]
                v = verify(SEQ, coords)
                if v.valid and v.contacts >= 49:
                    hc = frozenset(coords[i] for i, r in enumerate(SEQ) if r == "H")
                    if hc not in seen_hc:
                        seen_hc.add(hc)
                        seeds.append((v.contacts, Path(f).name, coords))
            except Exception:
                pass
    seeds.sort(key=lambda x: -x[0])
    print(f"Loaded {len(seeds)} distinct S10 seeds (>=49 contacts):", flush=True)
    for c, name, _ in seeds:
        print(f"  {c}: {name}", flush=True)

    rng = random.Random(2026)
    best_c = seeds[0][0]
    best_coords = seeds[0][2]
    distinct_50: set[tuple[tuple[int, int], ...]] = set()
    t0 = time.time()

    for s_idx, (c0, name, start_coords) in enumerate(seeds):
        cur = list(start_coords)
        cur_c = c0
        if cur_c == 50:
            hc_c = canonicalize({cur[i] for i, r in enumerate(SEQ) if r == "H"})
            if hc_c not in distinct_50:
                distinct_50.add(hc_c)
                hit = check_1cell_mods_51(cur)
                if hit is not None:
                    best_c = verify(SEQ, hit).contacts
                    best_coords = hit
                    print(f"*** RECORD S10={best_c} via 1-cell mod of {name}!! ***", flush=True)
                    break

        print(f"\n--- Seed {s_idx+1}/{len(seeds)}: {name} (start={cur_c}) ---", flush=True)
        for step in range(1, 21):
            # Every 4th step, try a controlled -1 dip escape to jump basins, then climb back
            if step % 4 == 0 and cur_c >= 49:
                free = select_two_strand_window(SEQ, cur, rng, strand_len=16, n_strands=2)
                dip_c, dip_val = lns_dip_step(cur, free, pad=3, budget=3.0, n_workers=6)
                if dip_c is not None:
                    cur, cur_c = dip_c, dip_val
                    # Immediately re-optimize a shifted overlapping window
                    free2 = select_two_strand_window(SEQ, cur, rng, strand_len=18, n_strands=2)
                    nc, nv, st, _ = lns_step_cpsat(
                        SEQ, cur, free2, pad=3, time_budget_seconds=4.0, n_workers=6, maximize_diversity=True
                    )
                    if nc is not None and nv >= cur_c:
                        cur, cur_c = nc, nv
            else:
                if step % 2 == 0:
                    free = select_two_strand_window(SEQ, cur, rng, strand_len=17, n_strands=2)
                else:
                    free = select_spatial_window(SEQ, cur, rng, min_free=28, max_free=40)
                nc, nv, st, _ = lns_step_cpsat(
                    SEQ, cur, free, pad=3, time_budget_seconds=3.5, n_workers=6, maximize_diversity=True
                )
                if nc is not None and nv >= cur_c:
                    cur, cur_c = nc, nv

            if cur_c > best_c:
                best_c = cur_c
                best_coords = list(cur)
                out = REPO / f"s10_record_{best_c}.json"
                out.write_text(
                    json.dumps(
                        {"sequence_id": "S10", "sequence": SEQ, "lattice": "2D", "contacts": best_c, "coords": [list(c) for c in best_coords]},
                        indent=2,
                    )
                )
                print(f"  *** NEW S10 RECORD contacts={best_c} at step {step}!! Saved to {out} ***", flush=True)
                return

            if cur_c == 50:
                hc_c = canonicalize({cur[i] for i, r in enumerate(SEQ) if r == "H"})
                if hc_c not in distinct_50:
                    distinct_50.add(hc_c)
                    print(f"  [step {step:2d}] NEW 50-contact H-core basin #{len(distinct_50)}! Checking 1-cell mods...", flush=True)
                    hit = check_1cell_mods_51(cur)
                    if hit is not None:
                        best_c = verify(SEQ, hit).contacts
                        best_coords = hit
                        out = REPO / f"s10_record_{best_c}.json"
                        out.write_text(
                            json.dumps(
                                {"sequence_id": "S10", "sequence": SEQ, "lattice": "2D", "contacts": best_c, "coords": [list(c) for c in best_coords]},
                                indent=2,
                            )
                        )
                        print(f"  *** NEW S10 RECORD contacts={best_c} via 1-cell mod!! Saved to {out} ***", flush=True)
                        return

        print(f"  Seed {name} done: end={cur_c}, best={best_c}, unique_50_basins={len(distinct_50)}", flush=True)

    print(
        f"\n=== S10 SA-LNS COMPLETE in {time.time()-t0:.1f}s: best={best_c}, "
        f"unique_50_basins_checked={len(distinct_50)} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
