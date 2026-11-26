"""Unconditional Z^2 CP-SAT proof of C*(S6) = 19 for the PP-prefix 50-mer S6
(Chen & Huang 2005, Table 1), with ZERO bounding-box or center-pinning restriction.

Proves two unconditional theorems over all of Z^2:
  1. C(S6) <= 20 (refuting the literature's cited 21 for the PP-prefix sequence):
     Zero odd-H slack; spanning-tree bound (W_H - 1) + (H_H - 1) <= 19; single CP-SAT UNSAT.
  2. C(S6) <= 19 (matching our verified 19-contact fold submissions/S6.json => C*(S6) = 19):
     Single odd-H slack (10-way case split over k in {5, 7, 9, 11, 27, 31, 35, 37, 39, 49});
     spanning-tree bound (W_H - 1) + (H_H - 1) <= 22; all 10/10 CP-SAT UNSAT.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from ortools.sat.python import cp_model

from hp_folding.grid_cpsat import NBRS, _max_hcore_deg
from hp_folding.sequences import BENCHMARKS

REPO = Path(__file__).resolve().parents[1]
FULL_SEQ = BENCHMARKS["S6"].sequence
SEQ = FULL_SEQ[2:]  # Exact relaxation: drop P0, P1; keep FULL_SEQ degrees for H2..H49
N = len(SEQ)
PIN = 36  # H38 in FULL_SEQ (pinned bond H38-H39)
ODD_H_FULL = [i for i, c in enumerate(FULL_SEQ) if c == "H" and i % 2 == 1]


def verify_exact_z2_diameter_bound(target_c: int) -> int:
    """Exhaustively verify the exact unconditional Z^2 upper bound on (W_H - 1) + (H_H - 1)
    for ANY fold of the full 50-mer S6 achieving target_c in {20, 21} contacts.
    """
    assert target_c in (20, 21)
    ie_req = target_c + 7  # adj_HH = 7
    p2_max = 44 - ie_req   # P(H)/2 = 2*|H| - ie(H)
    odd_slack = 28 - ie_req
    even_slack = 31 - ie_req

    h_pos = [i for i, c in enumerate(FULL_SEQ) if c == "H"]
    gaps = sorted([h_pos[i + 1] - h_pos[i] for i in range(len(h_pos) - 1)], reverse=True)
    # Every even H has d_max >= 2
    assert all(_max_hcore_deg(FULL_SEQ, i) >= 2 for i in h_pos if i % 2 == 0)

    def _parts(rem: int, max_v: int, cur: list[int]):
        if rem == 0:
            yield cur
            return
        for v in range(min(rem, max_v), 0, -1):
            twos = cur.count(2)
            ones = cur.count(1)
            # Each size-2 component requires an odd H of degree 1 (consumes 1 odd slack AND >= 1 even slack)
            if v == 2 and twos >= odd_slack:
                continue
            # Each size-1 component is an isolated even H (consumes >= 2 even slack)
            if v == 1 and (2 * (ones + 1) + twos > even_slack):
                continue
            yield from _parts(rem - v, v, cur + [v])

    best_wh = 0
    for part in _parts(22, 22, []):
        if sum(math.ceil(2 * math.sqrt(v)) for v in part) > p2_max:
            continue
        r = len(part)
        wh = p2_max - 2 * r + sum(gaps[: r - 1])
        if wh > best_wh:
            best_wh = wh
    return best_wh


def solve_s6_unconditional_case(
    *,
    target_c: int,
    slack_set: set[int],
    max_wh_minus_2: int,
    budget: float = 300.0,
    n_workers: int = 6,
    return_coords: bool = False,
) -> tuple[str, float, int] | tuple[str, float, int, list[tuple[int, int]] | None]:
    """Solve S6[2..49] >= target_c unconditionally over Z^2 using the exact
    spanning-tree diameter bound (W_H - 1) + (H_H - 1) <= max_wh_minus_2,
    assuming each odd H residue in `slack_set` (FULL_SEQ indices) loses 1 contact.
    """
    t0 = time.time()
    p0, p1 = (0, 0), (1, 0)
    h_indices = [i for i, c in enumerate(SEQ) if c == "H"]
    dist_to_h = [min(abs(i - hi) for hi in h_indices) for i in range(N)]

    valid_cells: list[list[tuple[int, int]]] = []
    for i in range(N):
        if i == PIN:
            valid_cells.append([p0])
            continue
        if i == PIN + 1:
            valid_cells.append([p1])
            continue
        par = i % 2  # PIN = 36 is even, placed at (0, 0)
        d0, d1 = abs(i - PIN), abs(i - (PIN + 1))
        rad = min(d0, max_wh_minus_2 + dist_to_h[i])
        rad1 = min(d1, max_wh_minus_2 + dist_to_h[i])
        vc = []
        for x in range(-rad, rad + 1):
            rem = rad - abs(x)
            for y in range(-rem, rem + 1):
                if (x + y) % 2 != par:
                    continue
                if abs(x - 1) + abs(y) > rad1:
                    continue
                if i == PIN + 2 and y < 0:
                    continue  # WLOG reflection across y = 0 (preserves (0,0)->(1,0))
                vc.append((x, y))
        valid_cells.append(vc)

    all_cells = sorted({c for vc in valid_cells for c in vc})
    h_cells = sorted({c for i in h_indices for c in valid_cells[i]})

    model = cp_model.CpModel()
    p: dict[tuple[int, tuple[int, int]], cp_model.IntVar] = {}
    cell_to_res: dict[tuple[int, int], list[int]] = {c: [] for c in all_cells}
    for i in range(N):
        for c in valid_cells[i]:
            p[(i, c)] = model.NewBoolVar(f"p_{i}_{c[0]}_{c[1]}")
            cell_to_res[c].append(i)

    # 1. Placement & self-avoidance
    for i in range(N):
        model.AddExactlyOne([p[(i, c)] for c in valid_cells[i]])
    for c, rlist in cell_to_res.items():
        if len(rlist) > 1:
            model.AddAtMostOne([p[(i, c)] for i in rlist])

    # 2. Backbone adjacency
    for i in range(N - 1):
        vc_i, vc_next = set(valid_cells[i]), set(valid_cells[i + 1])
        for c in valid_cells[i]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_next]
            if not nbs:
                model.Add(p[(i, c)] == 0)
            else:
                model.Add(p[(i, c)] <= sum(p[(i + 1, nb)] for nb in nbs))
        for c in valid_cells[i + 1]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_i]
            if not nbs:
                model.Add(p[(i + 1, c)] == 0)
            else:
                model.Add(p[(i + 1, c)] <= sum(p[(i, nb)] for nb in nbs))

    # 3. H occupancy on h_cells
    h_var: dict[tuple[int, int], cp_model.IntVar] = {}
    h_res_at: dict[tuple[int, int], list[int]] = {}
    for c in h_cells:
        hl = [i for i in cell_to_res[c] if SEQ[i] == "H"]
        if hl:
            hv = model.NewBoolVar(f"h_{c[0]}_{c[1]}")
            model.Add(hv == sum(p[(i, c)] for i in hl))
            h_var[c] = hv
            h_res_at[c] = hl

    # 4. Exact spanning-tree bounding-box cut: (max_x(H) - min_x(H)) + (max_y(H) - min_y(H)) <= max_wh_minus_2
    min_xh = model.NewIntVar(-max_wh_minus_2, 0, "min_xh")
    max_xh = model.NewIntVar(1, max_wh_minus_2, "max_xh")
    min_yh = model.NewIntVar(-max_wh_minus_2, 0, "min_yh")
    max_yh = model.NewIntVar(0, max_wh_minus_2, "max_yh")
    model.Add((max_xh - min_xh) + (max_yh - min_yh) <= max_wh_minus_2)
    for (x, y), hv in h_var.items():
        if x < 0:
            model.Add(min_xh <= x).OnlyEnforceIf(hv)
        if x > 1:
            model.Add(max_xh >= x).OnlyEnforceIf(hv)
        if y < 0:
            model.Add(min_yh <= y).OnlyEnforceIf(hv)
        if y > 0:
            model.Add(max_yh >= y).OnlyEnforceIf(hv)

    # 5. Exact odd-H degree equations
    z_vars = []
    inc_edges: dict[tuple[int, int], list[cp_model.IntVar]] = {c: [] for c in h_var}
    for u, hu in h_var.items():
        if (u[0] + u[1]) % 2 != 1:
            continue
        for dx, dy in NBRS:
            v = (u[0] + dx, u[1] + dy)
            if v in h_var:
                z = model.NewBoolVar(f"z_{u[0]}_{u[1]}_{v[0]}_{v[1]}")
                model.Add(z <= hu)
                model.Add(z <= h_var[v])
                z_vars.append(z)
                inc_edges[u].append(z)
                inc_edges[v].append(z)

    for u, rlist in h_res_at.items():
        if (u[0] + u[1]) % 2 == 1:
            rhs = sum(
                (_max_hcore_deg(FULL_SEQ, i + 2) - (1 if (i + 2) in slack_set else 0))
                * p[(i, u)]
                for i in rlist
            )
            model.Add(sum(inc_edges[u]) == rhs)
        else:
            rhs = sum(_max_hcore_deg(FULL_SEQ, i + 2) * p[(i, u)] for i in rlist)
            model.Add(sum(inc_edges[u]) <= rhs)

    # 6. Direct neighbor-forcing cuts for high-degree odd H residues (exact implication of step 5)
    for idx in (9, 37, 47):  # H11, H39, H49 (d_max = 4)
        req = 4 - (1 if (idx + 2) in slack_set else 0)
        for c in valid_cells[idx]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in h_var]
            if len(nbs) < req:
                model.Add(p[(idx, c)] == 0)
            elif req == 4:
                for nb in nbs:
                    model.Add(p[(idx, c)] <= h_var[nb])
            else:
                model.Add(sum(h_var[nb] for nb in nbs) >= req * p[(idx, c)])

    for idx in (7, 35):  # H9, H37 (d_max = 3)
        req = 3 - (1 if (idx + 2) in slack_set else 0)
        for c in valid_cells[idx]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in h_var]
            if len(nbs) < req:
                model.Add(p[(idx, c)] == 0)
            else:
                model.Add(sum(h_var[nb] for nb in nbs) >= req * p[(idx, c)])

    model.Add(sum(z_vars) == target_c + 7)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = budget
    solver.parameters.num_search_workers = n_workers
    status = solver.Solve(model)
    st_name = solver.StatusName(status)
    dt = time.time() - t0
    if not return_coords:
        return st_name, dt, solver.NumConflicts()
    coords = None
    if st_name in ("OPTIMAL", "FEASIBLE"):
        coords = [next(c for c in valid_cells[i] if solver.Value(p[(i, c)])) for i in range(N)]
    return st_name, dt, solver.NumConflicts(), coords


def main() -> None:
    resume = "--resume" in sys.argv
    d21 = verify_exact_z2_diameter_bound(21)
    d20 = verify_exact_z2_diameter_bound(20)
    assert d21 == 19 and d20 == 22, f"Unexpected bounds: {d21}, {d20}"
    print(f"Verified exact unconditional Z^2 diameter bounds: C=21 -> {d21}, C=20 -> {d20}", flush=True)

    # Built-in sanity check: verify the EXACT SAME encoding finds a valid 19-contact SAW when slack = {31, 37}
    st19, dt19, conf19, coords19 = solve_s6_unconditional_case(
        target_c=19, slack_set={31, 37}, max_wh_minus_2=d21, budget=60.0, n_workers=4, return_coords=True
    )
    assert st19 in ("OPTIMAL", "FEASIBLE") and coords19 is not None
    assert len(set(coords19)) == N
    assert all(abs(coords19[i][0] - coords19[i + 1][0]) + abs(coords19[i][1] - coords19[i + 1][1]) == 1 for i in range(N - 1))
    h_set19 = {coords19[i] for i, c in enumerate(SEQ) if c == "H"}
    ie19 = sum(1 for u in h_set19 if (u[0] + u[1]) % 2 == 1 for dx, dy in NBRS if (u[0] + dx, u[1] + dy) in h_set19)
    assert ie19 == 26, f"Expected 26 H-edges (19 contacts), got {ie19}"
    print(f"[Sanity check C=19, slack={{31,37}}] {st19} in {dt19:.2f}s (verified 19-contact SAW, conflicts={conf19})", flush=True)

    out_path = REPO / "results" / "s6_unconditional_z2_proof.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and out_path.exists():
        report = json.loads(out_path.read_text())
    else:
        report = {"d21_max_wh_minus_2": d21, "d20_max_wh_minus_2": d20, "c21": None, "c20_cases": {}}
    report["sanity_c19"] = {"status": st19, "elapsed_s": round(dt19, 3), "conflicts": conf19, "verified_contacts": 19}

    # Stage 1: C = 21 unconditional over all of Z^2
    if not (resume and report.get("c21") and report["c21"].get("status") == "INFEASIBLE"):
        st21, dt21, conf21 = solve_s6_unconditional_case(
            target_c=21, slack_set=set(), max_wh_minus_2=d21, budget=300.0, n_workers=6
        )
        print(f"[C=21 unconditional Z^2] {st21} in {dt21:.2f}s (conflicts={conf21})", flush=True)
        report["c21"] = {"status": st21, "elapsed_s": round(dt21, 3), "conflicts": conf21}
        out_path.write_text(json.dumps(report, indent=2))
        assert st21 == "INFEASIBLE", f"C=21 failed: {st21}"
    else:
        print(f"[C=21 unconditional Z^2] RESUMED INFEASIBLE ({report['c21']['elapsed_s']}s)", flush=True)

    # Stage 2: C = 20 unconditional over all of Z^2 (10-way odd-H slack split)
    all_c20_unsat = True
    for k in ODD_H_FULL:
        key = f"H{k}"
        if resume and report["c20_cases"].get(key, {}).get("status") == "INFEASIBLE":
            print(f"  [C=20 unconditional Z^2, slack=H{k:2d}] RESUMED INFEASIBLE ({report['c20_cases'][key]['elapsed_s']}s)", flush=True)
            continue
        st, dt, conf = solve_s6_unconditional_case(
            target_c=20, slack_set={k}, max_wh_minus_2=d20, budget=600.0, n_workers=6
        )
        print(f"  [C=20 unconditional Z^2, slack=H{k:2d}] {st} in {dt:.2f}s (conflicts={conf})", flush=True)
        report["c20_cases"][key] = {"status": st, "elapsed_s": round(dt, 3), "conflicts": conf}
        out_path.write_text(json.dumps(report, indent=2))
        if st != "INFEASIBLE":
            all_c20_unsat = False

    report["c20_total_elapsed_s"] = round(sum(v["elapsed_s"] for v in report["c20_cases"].values()), 3)
    report["c20_all_infeasible"] = all_c20_unsat
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSUMMARY: C<=20 UNCONDITIONAL Z^2 = PASS | C<=19 (10/10) UNCONDITIONAL Z^2 = {all_c20_unsat}", flush=True)
    if not all_c20_unsat:
        sys.exit(1)


if __name__ == "__main__":
    main()
