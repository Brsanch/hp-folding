"""Independent solver cross-verification (HiGHS MILP via highspy, ZERO OR-Tools)
for the unconditional Z^2 optimality proofs of S8 (= 42, unique 6x7 rectangle)
and S7 (= 36, all 4/4 parity-compatible minimum-perimeter 43-ominoes), plus an
exact MILP primal feasibility check verifying that the transformed known
19-contact fold submissions/S6.json satisfies every constraint of the S6
unconditional Z^2 formulation (including the bounding-box cut).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import highspy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prove_optimality import enumerate_min_perimeter_polyominoes, max_ie_harary_harborth
from prove_s6_unconditional_z2 import verify_exact_z2_diameter_bound

from hp_folding.grid_cpsat import NBRS, _max_hcore_deg
from hp_folding.sequences import BENCHMARKS
from hp_folding.verifier import verify

REPO = Path(__file__).resolve().parents[1]


def solve_hcore_routability_highs(
    seq: str,
    h_cells: set[tuple[int, int]],
    pad: int,
    *,
    time_limit: float = 60.0,
) -> tuple[str, float]:
    """Independent HiGHS MILP check of exact H-core routability on Z^2."""
    t0 = time.time()
    n = len(seq)
    xs = [c[0] for c in h_cells]
    ys = [c[1] for c in h_cells]
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    all_cells = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
    p_cells = [c for c in all_cells if c not in h_cells]

    # Check both parities (H-core can match even-index H on parity 0 or parity 1)
    h_even_seq = sum(1 for i, c in enumerate(seq) if c == "H" and i % 2 == 0)
    h_odd_seq = sum(1 for i, c in enumerate(seq) if c == "H" and i % 2 == 1)
    c0 = sum(1 for u in h_cells if (u[0] + u[1]) % 2 == 0)
    c1 = len(h_cells) - c0

    parities = []
    if c0 == h_even_seq and c1 == h_odd_seq:
        parities.append(0)
    if c1 == h_even_seq and c0 == h_odd_seq:
        parities.append(1)

    for par_off in parities:
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("time_limit", time_limit)
        h.setOptionValue("threads", 4)

        valid_cells: list[list[tuple[int, int]]] = []
        for i, ch in enumerate(seq):
            want = (i + par_off) % 2
            pool = h_cells if ch == "H" else p_cells
            valid_cells.append([c for c in pool if (c[0] + c[1]) % 2 == want])

        var_idx: dict[tuple[int, tuple[int, int]], int] = {}
        cell_vars: dict[tuple[int, int], list[int]] = {c: [] for c in all_cells}
        col = 0
        for i in range(n):
            for c in valid_cells[i]:
                h.addBinary(0.0)
                var_idx[(i, c)] = col
                cell_vars[c].append(col)
                col += 1

        # Exactly one cell per residue
        for i in range(n):
            cols = [var_idx[(i, c)] for c in valid_cells[i]]
            h.addRow(1.0, 1.0, len(cols), cols, [1.0] * len(cols))

        # At most one residue per cell (exactly one for H-cells)
        for c, cols in cell_vars.items():
            if not cols:
                continue
            lb = 1.0 if c in h_cells else 0.0
            h.addRow(lb, 1.0, len(cols), cols, [1.0] * len(cols))

        # Backbone adjacency
        for i in range(n - 1):
            vc_next = set(valid_cells[i + 1])
            for c in valid_cells[i]:
                nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_next]
                cols = [var_idx[(i, c)]] + [var_idx[(i + 1, nb)] for nb in nbs]
                vals = [1.0] + [-1.0] * len(nbs)
                h.addRow(-highspy.kHighsInf, 0.0, len(cols), cols, vals)

        h.run()
        st = str(h.getModelStatus())
        if "Infeasible" not in st:
            return st, time.time() - t0
    return "INFEASIBLE", time.time() - t0


def solve_s6_highs(
    *,
    target_c: int,
    slack_set: set[int],
    max_wh_minus_2: int,
    time_limit: float = 180.0,
    fixed_coords: list[tuple[int, int]] | None = None,
) -> tuple[str, float, list[tuple[int, int]] | None]:
    """Independent HiGHS MILP re-verification of the exact unconditional Z^2 S6 model."""
    t0 = time.time()
    full_seq = BENCHMARKS["S6"].sequence
    seq = full_seq[2:]
    n = len(seq)
    pin = 36
    p0, p1 = (0, 0), (1, 0)
    h_indices = [i for i, c in enumerate(seq) if c == "H"]
    dist_to_h = [min(abs(i - hi) for hi in h_indices) for i in range(n)]

    valid_cells: list[list[tuple[int, int]]] = []
    for i in range(n):
        if i == pin:
            valid_cells.append([p0]); continue
        if i == pin + 1:
            valid_cells.append([p1]); continue
        par = i % 2
        rad = min(abs(i - pin), max_wh_minus_2 + dist_to_h[i])
        rad1 = min(abs(i - (pin + 1)), max_wh_minus_2 + dist_to_h[i])
        vc = []
        for x in range(-rad, rad + 1):
            rem = rad - abs(x)
            for y in range(-rem, rem + 1):
                if (x + y) % 2 != par:
                    continue
                if abs(x - 1) + abs(y) > rad1:
                    continue
                if i == pin + 2 and y < 0:
                    continue
                vc.append((x, y))
        valid_cells.append(vc)

    all_cells = sorted({c for vc in valid_cells for c in vc})
    h_cells = sorted({c for i in h_indices for c in valid_cells[i]})

    h = highspy.Highs()
    h.silent()
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("threads", 4)

    p_var: dict[tuple[int, tuple[int, int]], int] = {}
    cell_to_res: dict[tuple[int, int], list[int]] = {c: [] for c in all_cells}
    col = 0
    for i in range(n):
        for c in valid_cells[i]:
            h.addBinary(0.0)
            p_var[(i, c)] = col
            cell_to_res[c].append(i)
            col += 1

    # Exactly one per residue & at most one per cell
    for i in range(n):
        cols = [p_var[(i, c)] for c in valid_cells[i]]
        h.addRow(1.0, 1.0, len(cols), cols, [1.0] * len(cols))
    for c, rlist in cell_to_res.items():
        if len(rlist) > 1:
            cols = [p_var[(i, c)] for i in rlist]
            h.addRow(0.0, 1.0, len(cols), cols, [1.0] * len(cols))

    # Backbone adjacency
    for i in range(n - 1):
        vc_i, vc_next = set(valid_cells[i]), set(valid_cells[i + 1])
        for c in valid_cells[i]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_next]
            cols = [p_var[(i, c)]] + [p_var[(i + 1, nb)] for nb in nbs]
            h.addRow(-highspy.kHighsInf, 0.0, len(cols), cols, [1.0] + [-1.0] * len(nbs))
        for c in valid_cells[i + 1]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_i]
            cols = [p_var[(i + 1, c)]] + [p_var[(i, nb)] for nb in nbs]
            h.addRow(-highspy.kHighsInf, 0.0, len(cols), cols, [1.0] + [-1.0] * len(nbs))

    # H occupancy vars
    h_var: dict[tuple[int, int], int] = {}
    h_res_at: dict[tuple[int, int], list[int]] = {}
    for c in h_cells:
        hl = [i for i in cell_to_res[c] if seq[i] == "H"]
        if hl:
            h.addBinary(0.0)
            hv = col
            col += 1
            h_var[c] = hv
            h_res_at[c] = hl
            cols = [hv] + [p_var[(i, c)] for i in hl]
            h.addRow(0.0, 0.0, len(cols), cols, [1.0] + [-1.0] * len(hl))

    # Bounding-box vars (big-M linearization, exact since x in [-max_wh_minus_2, max_wh_minus_2])
    M = float(max_wh_minus_2)
    h.addVar(-M, 0.0); min_xh = col; col += 1
    h.addVar(1.0, M); max_xh = col; col += 1
    h.addVar(-M, 0.0); min_yh = col; col += 1
    h.addVar(0.0, M); max_yh = col; col += 1
    h.addRow(-highspy.kHighsInf, M, 4, [max_xh, min_xh, max_yh, min_yh], [1.0, -1.0, 1.0, -1.0])
    for (x, y), hv in h_var.items():
        if x < 0:
            # min_xh <= x + (0 - x)*(1 - hv) => min_xh - x*hv <= 0
            h.addRow(-highspy.kHighsInf, 0.0, 2, [min_xh, hv], [1.0, float(-x)])
        if x > 1:
            # max_xh >= 1 + (x - 1)*hv => max_xh - (x - 1)*hv >= 1
            h.addRow(1.0, highspy.kHighsInf, 2, [max_xh, hv], [1.0, float(-(x - 1))])
        if y < 0:
            h.addRow(-highspy.kHighsInf, 0.0, 2, [min_yh, hv], [1.0, float(-y)])
        if y > 0:
            h.addRow(0.0, highspy.kHighsInf, 2, [max_yh, hv], [1.0, float(-y)])

    # H-H edge vars
    z_vars: list[int] = []
    inc_edges: dict[tuple[int, int], list[int]] = {c: [] for c in h_var}
    for u, hu in h_var.items():
        if (u[0] + u[1]) % 2 != 1:
            continue
        for dx, dy in NBRS:
            v = (u[0] + dx, u[1] + dy)
            if v in h_var:
                h.addBinary(0.0)
                zv = col
                col += 1
                h.addRow(-highspy.kHighsInf, 0.0, 2, [zv, hu], [1.0, -1.0])
                h.addRow(-highspy.kHighsInf, 0.0, 2, [zv, h_var[v]], [1.0, -1.0])
                h.addRow(-1.0, highspy.kHighsInf, 3, [zv, hu, h_var[v]], [1.0, -1.0, -1.0])
                z_vars.append(zv)
                inc_edges[u].append(zv)
                inc_edges[v].append(zv)

    # Odd-H degree equalities and even-H degree upper bounds
    for u, rlist in h_res_at.items():
        if (u[0] + u[1]) % 2 == 1:
            cols = inc_edges[u] + [p_var[(i, u)] for i in rlist]
            vals = [1.0] * len(inc_edges[u]) + [
                float(-(_max_hcore_deg(full_seq, i + 2) - (1 if (i + 2) in slack_set else 0)))
                for i in rlist
            ]
            h.addRow(0.0, 0.0, len(cols), cols, vals)
        else:
            cols = inc_edges[u] + [p_var[(i, u)] for i in rlist]
            vals = [1.0] * len(inc_edges[u]) + [float(-_max_hcore_deg(full_seq, i + 2)) for i in rlist]
            h.addRow(-highspy.kHighsInf, 0.0, len(cols), cols, vals)

    # Total H-edges == target_c + 7
    h.addRow(float(target_c + 7), float(target_c + 7), len(z_vars), z_vars, [1.0] * len(z_vars))

    # Optional primal feasibility check for a known fold (e.g. submissions/S6.json)
    if fixed_coords is not None:
        assert len(fixed_coords) == n
        for i, c in enumerate(fixed_coords):
            v = p_var[(i, c)]
            h.addRow(1.0, 1.0, 1, [v], [1.0])

    h.run()
    st = str(h.getModelStatus())
    coords = None
    if "Optimal" in st or "Feasible" in st:
        sol = h.getSolution().col_value
        sub_coords = []
        for i in range(n):
            chosen = [c for c in valid_cells[i] if sol[p_var[(i, c)]] > 0.5]
            assert len(chosen) == 1
            sub_coords.append(chosen[0])
        # Re-attach P0, P1 via simple greedy SAW extension
        occ = set(sub_coords)
        p2 = sub_coords[0]
        attached = False
        for dx1, dy1 in NBRS:
            c1 = (p2[0] + dx1, p2[1] + dy1)
            if c1 in occ:
                continue
            for dx0, dy0 in NBRS:
                c0 = (c1[0] + dx0, c1[1] + dy0)
                if c0 not in occ and c0 != c1:
                    coords = [c0, c1] + sub_coords
                    attached = True
                    break
            if attached:
                break
        assert coords is not None, "Could not attach P0, P1"
    return st, time.time() - t0, coords


def main() -> None:
    print("=== Independent HiGHS MILP Verification (ZERO OR-Tools) ===", flush=True)
    # 1. S8 = 42 (unique 6x7 rectangle)
    s8_seq = BENCHMARKS["S8"].sequence
    rect67 = {(x, y) for x in range(6) for y in range(7)}
    st8, dt8 = solve_hcore_routability_highs(s8_seq, rect67, pad=3, time_limit=60.0)
    print(f"  S8 (6x7 rectangle, pad=3): {st8} in {dt8:.2f}s", flush=True)
    assert st8 == "INFEASIBLE"

    # 2. S7 = 36 (all 4 parity-ok minimum-perimeter 43-ominoes)
    s7_seq = BENCHMARKS["S7"].sequence
    s7_all = enumerate_min_perimeter_polyominoes(43, max_ie_harary_harborth(43))
    h_even = sum(1 for i, c in enumerate(s7_seq) if c == "H" and i % 2 == 0)
    h_odd = sum(1 for i, c in enumerate(s7_seq) if c == "H" and i % 2 == 1)
    s7_ok = []
    for hc in sorted(s7_all):
        c0 = sum(1 for x, y in hc if (x + y) % 2 == 0)
        c1 = len(hc) - c0
        if (c0, c1) in ((h_even, h_odd), (h_odd, h_even)):
            s7_ok.append(set(hc))
    assert len(s7_all) == 147 and len(s7_ok) == 4
    t_s7 = time.time()
    for idx, hc in enumerate(s7_ok):
        st7, dt7 = solve_hcore_routability_highs(s7_seq, hc, pad=5, time_limit=60.0)
        print(f"  S7 candidate {idx+1}/4 (pad=5): {st7} in {dt7:.2f}s", flush=True)
        assert st7 == "INFEASIBLE"
    print(f"  S7 ALL 4/4 INFEASIBLE in HiGHS ({time.time() - t_s7:.2f}s total)", flush=True)

    # 3. S6 C=19 known-fold exact MILP feasibility check in HiGHS -> MUST BE OPTIMAL!
    d21 = verify_exact_z2_diameter_bound(21)
    s6_data = json.loads((REPO / "submissions" / "S6.json").read_text())
    raw_c = [tuple(c) for c in s6_data["coords"]]
    p0, p1 = raw_c[38], raw_c[39]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    tr = [((x - p0[0]) * dx + (y - p0[1]) * dy, -(x - p0[0]) * dy + (y - p0[1]) * dx) for x, y in raw_c]
    if tr[40][1] < 0:
        tr = [(x, -y) for x, y in tr]
    st19, dt19, coords19 = solve_s6_highs(
        target_c=19, slack_set={31, 37}, max_wh_minus_2=d21, time_limit=30.0, fixed_coords=tr[2:]
    )
    print(f"  S6 C=19 known-fold MILP feasibility check (HiGHS): {st19} in {dt19:.2f}s", flush=True)
    assert coords19 is not None
    vr = verify(BENCHMARKS["S6"].sequence, coords19)
    assert vr.valid and vr.contacts == 19, f"HiGHS C=19 verify failed: {vr}"
    print(f"    -> Verified valid SAW with contacts={vr.contacts}", flush=True)
    print("ALL HiGHS INDEPENDENT CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
