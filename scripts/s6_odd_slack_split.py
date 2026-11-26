"""10-way exact odd-H slack case-split CP-SAT attack on S6 (PP-prefix) >= 20 contacts.

S6 (PP-prefix) has 10 odd H residues with max H-core degree sum = 28 (adjHH = 7,
so parity UB = 21 contacts). Since slack=0 (21 contacts) is already proven INFEASIBLE
in 15x15 (~25s), any 20-contact fold has total odd-H slack == 1:
  - Exactly ONE odd H residue k in {5, 7, 9, 11, 27, 31, 35, 37, 39, 49} loses 1 contact,
  - ALL OTHER 9 odd H residues satisfy EXACT degree equality `==`!

Splitting into 10 independent zero-slack-variable subproblems eliminates the slack
variable entirely and allows parallel/case-by-case exact verification.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ortools.sat.python import cp_model

from hp_folding.grid_cpsat import NBRS, _max_hcore_deg
from hp_folding.sequences import BENCHMARKS

FULL_SEQ = BENCHMARKS["S6"].sequence
SEQ = FULL_SEQ[2:]  # drop leading P0, P1 (exact relaxation; use FULL_SEQ degrees)
N = len(SEQ)
ADJ_HH = sum(1 for i in range(N - 1) if SEQ[i] == "H" and SEQ[i + 1] == "H")
ODD_H_FULL = [i for i, c in enumerate(FULL_SEQ) if c == "H" and i % 2 == 1]


def solve_slack_case(
    slack_res_full: int,
    *,
    grid_w: int = 13,
    grid_h: int = 13,
    budget: float = 30.0,
    n_workers: int = 8,
) -> tuple[str, float, int, int, list[tuple[int, int]] | None]:
    """Solve S6[2:] >= 20 on the bond-centered box {0..grid_w-1} x {0..grid_h-1}
    around pinned bond H38-H39 (requires odd grid_h for reflection symmetry soundness),
    assuming odd H residue `slack_res_full` takes the 1 unit of slack.
    """
    assert grid_h % 2 == 1, f"Reflection break y >= cy requires odd grid_h; got {grid_h}"
    t0 = time.time()
    pin_index = 36  # H38-H39 in FULL_SEQ
    cx, cy = grid_w // 2, grid_h // 2
    if (cx + cy) % 2 != pin_index % 2:
        cx += 1
    p0_cell, p1_cell = (cx, cy), (cx + 1, cy)

    cells = [(x, y) for x in range(grid_w) for y in range(grid_h)]
    cell_set = set(cells)
    nbrs_of = {
        c: [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in cell_set]
        for c in cells
    }

    valid_cells: list[list[tuple[int, int]]] = []
    for i in range(N):
        if i == pin_index:
            valid_cells.append([p0_cell])
        elif i == pin_index + 1:
            valid_cells.append([p1_cell])
        else:
            d0, d1 = abs(i - pin_index), abs(i - (pin_index + 1))
            par = i % 2
            vc = [
                c
                for c in cells
                if (c[0] + c[1]) % 2 == par
                and abs(c[0] - p0_cell[0]) + abs(c[1] - p0_cell[1]) <= d0
                and abs(c[0] - p1_cell[0]) + abs(c[1] - p1_cell[1]) <= d1
            ]
            if i == pin_index + 2:
                vc = [c for c in vc if c[1] >= cy]  # WLOG reflection symmetry break
            valid_cells.append(vc)

    model = cp_model.CpModel()
    p: dict[tuple[int, tuple[int, int]], cp_model.IntVar] = {}
    cell_to_res: dict[tuple[int, int], list[int]] = {c: [] for c in cells}
    for i in range(N):
        for c in valid_cells[i]:
            p[(i, c)] = model.NewBoolVar(f"p_{i}_{c[0]}_{c[1]}")
            cell_to_res[c].append(i)

    for i in range(N):
        model.AddExactlyOne([p[(i, c)] for c in valid_cells[i]])
    for c, rlist in cell_to_res.items():
        if len(rlist) > 1:
            model.AddAtMostOne([p[(i, c)] for i in rlist])

    for i in range(N - 1):
        vc_i, vc_next = set(valid_cells[i]), set(valid_cells[i + 1])
        for c in valid_cells[i]:
            nbs = [nb for nb in nbrs_of[c] if nb in vc_next]
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

    h_var: dict[tuple[int, int], cp_model.IntVar] = {}
    h_res_at: dict[tuple[int, int], list[int]] = {}
    for c in cells:
        hl = [i for i in cell_to_res[c] if SEQ[i] == "H"]
        if hl:
            hv = model.NewBoolVar(f"h_{c[0]}_{c[1]}")
            model.Add(hv == sum(p[(i, c)] for i in hl))
            h_var[c] = hv
            h_res_at[c] = hl

    z_vars = []
    inc_edges: dict[tuple[int, int], list[cp_model.IntVar]] = {c: [] for c in h_var}
    for u, hu in h_var.items():
        if (u[0] + u[1]) % 2 != 0:
            continue
        for v in nbrs_of[u]:
            if v not in h_var:
                continue
            ze = model.NewBoolVar(f"z_{u}_{v}")
            model.Add(ze <= hu)
            model.Add(ze <= h_var[v])
            z_vars.append(ze)
            inc_edges[u].append(ze)
            inc_edges[v].append(ze)

    # Exact degree equality on EVERY odd cell (zero slack variables!)
    for c, edges in inc_edges.items():
        if (c[0] + c[1]) % 2 == 1:
            terms = []
            for i in h_res_at[c]:
                req_deg = _max_hcore_deg(FULL_SEQ, i + 2) - (1 if (i + 2) == slack_res_full else 0)
                if len(edges) < req_deg:
                    model.Add(p[(i, c)] == 0)
                terms.append(req_deg * p[(i, c)])
            model.Add(sum(edges) == sum(terms))
        else:
            rhs = sum(_max_hcore_deg(FULL_SEQ, i + 2) * p[(i, c)] for i in h_res_at[c])
            model.Add(sum(edges) <= rhs)

    model.Add(sum(z_vars) == 20 + ADJ_HH)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = n_workers
    st = solver.Solve(model)
    elapsed = round(time.time() - t0, 2)
    coords = None
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        coords = [next(c for c in valid_cells[i] if solver.Value(p[(i, c)]) == 1) for i in range(N)]
    return solver.StatusName(st), elapsed, solver.NumBranches(), solver.NumConflicts(), coords


def _worker(args: tuple[int, int, float, int]):
    k, grid, budget, n_workers = args
    st, elapsed, br, cf, coords = solve_slack_case(
        k, grid_w=grid, grid_h=grid, budget=budget, n_workers=n_workers
    )
    return k, st, elapsed, br, cf, coords


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    parallel = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    workers_per_case = max(2, 12 // parallel) if parallel > 1 else 8
    print(
        f"S6[2:] 10-way odd-slack case split (grid={grid}x{grid}, budget={budget}s/case, "
        f"parallel={parallel}x{workers_per_case}w):",
        flush=True,
    )
    t_all = time.time()
    n_inf = n_unk = n_hit = 0
    jobs = [(k, grid, budget, workers_per_case) for k in ODD_H_FULL]
    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            results = list(ex.map(_worker, jobs))
    else:
        results = [_worker(j) for j in jobs]
    for k, st, elapsed, br, cf, coords in results:
        print(
            f"  slack@H{k:2d} (max_deg={_max_hcore_deg(FULL_SEQ, k)}): "
            f"{st:10s} in {elapsed:6.2f}s (br={br}, cf={cf})",
            flush=True,
        )
        if coords is not None:
            n_hit += 1
            out = Path(__file__).resolve().parents[1] / f"s6_20_candidate_slackH{k}.json"
            out.write_text(json.dumps({"seq_suffix": SEQ, "slack_res": k, "coords": coords}, indent=2))
            print(f"    *** HIT!! Saved suffix fold to {out} ***", flush=True)
        elif st == "INFEASIBLE":
            n_inf += 1
        else:
            n_unk += 1
    print(
        f"\nSummary ({grid}x{grid}, wall={time.time()-t_all:.1f}s): "
        f"hits={n_hit}, infeasible={n_inf}/10, unknown={n_unk}/10",
        flush=True,
    )


if __name__ == "__main__":
    main()
