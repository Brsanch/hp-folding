"""Independent pure-CNF SAT (PicoSAT / pycosat) cross-verification (ZERO OR-Tools!)
for the unconditional Z^2 optimality proofs of S8 (= 42) and S7 (= 36), plus
known-fold CNF feasibility verification for S6 (C = 19) and optional DIMACS CNF
export (--export-dimacs <dir>) / S6 C = 21 solve (--solve-s6-c21).

Exported DIMACS CNF files allow any external SAT solver + DRAT/LRAT checker
(e.g., CaDiCaL + lrat-check) to independently emit and check formal clausal
refutation certificates.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prove_optimality import enumerate_min_perimeter_polyominoes, max_ie_harary_harborth
from prove_s6_unconditional_z2 import verify_exact_z2_diameter_bound

from hp_folding.grid_cpsat import NBRS, _max_hcore_deg
from hp_folding.sequences import BENCHMARKS
from hp_folding.verifier import verify

REPO = Path(__file__).resolve().parents[1]


class CNFBuilder:
    """Minimal DIMACS CNF builder with sequential counter (Sinz 2005) AtMostOne."""

    def __init__(self) -> None:
        self.nvars = 0
        self.clauses: list[list[int]] = []

    def new_var(self) -> int:
        self.nvars += 1
        return self.nvars

    def add_clause(self, lits: list[int]) -> None:
        self.clauses.append(lits)

    def add_at_most_one(self, lits: list[int]) -> None:
        k = len(lits)
        if k <= 1:
            return
        if k <= 4:
            for a, b in itertools.combinations(lits, 2):
                self.clauses.append([-a, -b])
            return
        # Sinz sequential counter: O(k) vars and clauses
        s = [self.new_var() for _ in range(k - 1)]
        self.clauses.append([-lits[0], s[0]])
        for i in range(1, k - 1):
            self.clauses.append([-lits[i], s[i]])
            self.clauses.append([-s[i - 1], s[i]])
            self.clauses.append([-lits[i], -s[i - 1]])
        self.clauses.append([-lits[k - 1], -s[k - 2]])

    def add_exactly_one(self, lits: list[int]) -> None:
        self.clauses.append(list(lits))
        self.add_at_most_one(lits)

    def write_dimacs(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(f"p cnf {self.nvars} {len(self.clauses)}\n")
            for cl in self.clauses:
                f.write(" ".join(str(x) for x in cl) + " 0\n")


def build_hcore_cnf(
    seq: str,
    h_cells: set[tuple[int, int]],
    pad: int,
    par_off: int,
) -> CNFBuilder:
    """Pure propositional CNF encoding of exact H-core routability on Z^2."""
    n = len(seq)
    xs = [c[0] for c in h_cells]
    ys = [c[1] for c in h_cells]
    all_cells = [
        (x, y)
        for x in range(min(xs) - pad, max(xs) + pad + 1)
        for y in range(min(ys) - pad, max(ys) + pad + 1)
    ]
    p_cells = [c for c in all_cells if c not in h_cells]

    cnf = CNFBuilder()
    valid_cells: list[list[tuple[int, int]]] = []
    vmap: dict[tuple[int, tuple[int, int]], int] = {}
    cell_vars: dict[tuple[int, int], list[int]] = {c: [] for c in all_cells}

    for i, ch in enumerate(seq):
        want = (i + par_off) % 2
        pool = h_cells if ch == "H" else p_cells
        vc = [c for c in pool if (c[0] + c[1]) % 2 == want]
        valid_cells.append(vc)
        for c in vc:
            v = cnf.new_var()
            vmap[(i, c)] = v
            cell_vars[c].append(v)

    for i in range(n):
        cnf.add_exactly_one([vmap[(i, c)] for c in valid_cells[i]])
    for c, vl in cell_vars.items():
        if not vl:
            continue
        if c in h_cells:
            cnf.add_exactly_one(vl)
        else:
            cnf.add_at_most_one(vl)

    for i in range(n - 1):
        vc_i, vc_next = set(valid_cells[i]), set(valid_cells[i + 1])
        for c in valid_cells[i]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_next]
            cnf.add_clause([-vmap[(i, c)]] + [vmap[(i + 1, nb)] for nb in nbs])
        for c in valid_cells[i + 1]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_i]
            cnf.add_clause([-vmap[(i + 1, c)]] + [vmap[(i, nb)] for nb in nbs])

    return cnf


def verify_hcore_picosat(
    seq: str,
    h_cells: set[tuple[int, int]],
    pad: int,
    dimacs_prefix: Path | None = None,
) -> tuple[str, float, int, int]:
    import pycosat

    t0 = time.time()
    h_even = sum(1 for i, c in enumerate(seq) if c == "H" and i % 2 == 0)
    h_odd = sum(1 for i, c in enumerate(seq) if c == "H" and i % 2 == 1)
    c0 = sum(1 for u in h_cells if (u[0] + u[1]) % 2 == 0)
    c1 = len(h_cells) - c0
    parities = []
    if c0 == h_even and c1 == h_odd:
        parities.append(0)
    if c1 == h_even and c0 == h_odd:
        parities.append(1)

    tot_vars = tot_clauses = 0
    for par_off in parities:
        cnf = build_hcore_cnf(seq, h_cells, pad, par_off)
        tot_vars += cnf.nvars
        tot_clauses += len(cnf.clauses)
        if dimacs_prefix is not None:
            cnf.write_dimacs(Path(f"{dimacs_prefix}_par{par_off}.cnf"))
        res = pycosat.solve(cnf.clauses)
        if res != "UNSAT":
            return "SAT", time.time() - t0, tot_vars, tot_clauses
    return "UNSAT", time.time() - t0, tot_vars, tot_clauses


def build_s6_cnf(
    *,
    slack_set: set[int],
    max_wh_minus_2: int,
) -> tuple[CNFBuilder, dict[tuple[int, tuple[int, int]], int], list[list[tuple[int, int]]]]:
    """Pure propositional CNF relaxation of S6[2..49] on Z^2 (ZERO arithmetic!).

    For every odd H residue i in FULL_SEQ (odd index in SEQ = FULL_SEQ[2:]):
      required H-core degree req(i) = d_max(i+2) - (1 if i+2 in slack_set else 0).
    This is encoded purely as k-of-4 neighbor clauses on H-occupancy bits h_u:
      req=4: p(i,c) => h(nb) for all 4 neighbors
      req=3: p(i,c) => h(nb_a) OR h(nb_b) for all 6 pairs
      req=2: p(i,c) => h(nb_a) OR h(nb_b) OR h(nb_c) for all 4 triples
      req=1: p(i,c) => OR of all neighbors
    Notice: this pure-CNF encoding enforces Cut 1's exact bounding-box inequality
    (max_x(H) - min_x(H)) + (max_y(H) - min_y(H)) <= max_wh_minus_2 via a unary
    order encoding, Cut 2's pinned-bond diamond and reflection symmetry, Cut 3's
    deg_H(H_2) <= 2, and Cuts 4-5's exact odd-H degree lower bounds, while relaxing
    upper even-H degree bounds (other than H_2). It is therefore a sound relaxation
    of the full 50-mer S6 for proving UNSAT with formal DRAT/LRAT certificates.
    """
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

    cnf = CNFBuilder()
    p_var: dict[tuple[int, tuple[int, int]], int] = {}
    cell_to_res: dict[tuple[int, int], list[int]] = {c: [] for c in all_cells}
    for i in range(n):
        for c in valid_cells[i]:
            v = cnf.new_var()
            p_var[(i, c)] = v
            cell_to_res[c].append(i)

    for i in range(n):
        cnf.add_exactly_one([p_var[(i, c)] for c in valid_cells[i]])
    for c, rlist in cell_to_res.items():
        if len(rlist) > 1:
            cnf.add_at_most_one([p_var[(i, c)] for i in rlist])

    for i in range(n - 1):
        vc_i, vc_next = set(valid_cells[i]), set(valid_cells[i + 1])
        for c in valid_cells[i]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_next]
            cnf.add_clause([-p_var[(i, c)]] + [p_var[(i + 1, nb)] for nb in nbs])
        for c in valid_cells[i + 1]:
            nbs = [(c[0] + dx, c[1] + dy) for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in vc_i]
            cnf.add_clause([-p_var[(i + 1, c)]] + [p_var[(i, nb)] for nb in nbs])

    # H occupancy bits: h_c <=> OR_{i in hl} p(i, c)
    h_var: dict[tuple[int, int], int] = {}
    for c in h_cells:
        hl = [i for i in cell_to_res[c] if seq[i] == "H"]
        if hl:
            hv = cnf.new_var()
            h_var[c] = hv
            # h_c => OR p(i, c)
            cnf.add_clause([-hv] + [p_var[(i, c)] for i in hl])
            # p(i, c) => h_c
            for i in hl:
                cnf.add_clause([-p_var[(i, c)], hv])

    # Exact odd-H degree lower bounds in pure CNF (ZERO arithmetic!)
    odd_h_seq = [i for i in h_indices if i % 2 == 1]
    for i in odd_h_seq:
        req = _max_hcore_deg(full_seq, i + 2) - (1 if (i + 2) in slack_set else 0)
        assert 1 <= req <= 4
        for c in valid_cells[i]:
            nbs = [h_var[(c[0] + dx, c[1] + dy)] for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in h_var]
            m = len(nbs)
            if m < req:
                cnf.add_clause([-p_var[(i, c)]])
            else:
                # At least req of m booleans are true <=> every subset of size (m - req + 1) has >= 1 true
                sub_sz = m - req + 1
                for comb in itertools.combinations(nbs, sub_sz):
                    cnf.add_clause([-p_var[(i, c)]] + list(comb))

    # Cut 3 (full 50-mer soundness): enforce deg_H(H_2) <= 2 at residue i = 0 (since P1, P3 occupy 2 neighbors)
    for c in valid_cells[0]:
        nbs = [h_var[(c[0] + dx, c[1] + dy)] for dx, dy in NBRS if (c[0] + dx, c[1] + dy) in h_var]
        if len(nbs) >= 3:
            for comb in itertools.combinations(nbs, 3):
                cnf.add_clause([-p_var[(0, c)]] + [-v for v in comb])

    # Cut 1 (exact unary order encoding): (max_x(H) - min_x(H)) + (max_y(H) - min_y(H)) <= max_wh_minus_2
    D = max_wh_minus_2
    left_le = {x: cnf.new_var() for x in range(-D, 1)}
    right_ge = {x: cnf.new_var() for x in range(1, D + 1)}
    bot_le = {y: cnf.new_var() for y in range(-D, 1)}
    top_ge = {y: cnf.new_var() for y in range(0, D + 1)}
    cnf.add_clause([left_le[0]])
    cnf.add_clause([right_ge[1]])
    cnf.add_clause([bot_le[0]])
    cnf.add_clause([top_ge[0]])
    for x in range(-D, 0):
        cnf.add_clause([-left_le[x], left_le[x + 1]])
    for x in range(2, D + 1):
        cnf.add_clause([-right_ge[x], right_ge[x - 1]])
    for y in range(-D, 0):
        cnf.add_clause([-bot_le[y], bot_le[y + 1]])
    for y in range(1, D + 1):
        cnf.add_clause([-top_ge[y], top_ge[y - 1]])
    for (x, y), hv in h_var.items():
        if x < 0:
            cnf.add_clause([-hv, left_le[x]])
        if x > 1:
            cnf.add_clause([-hv, right_ge[x]])
        if y < 0:
            cnf.add_clause([-hv, bot_le[y]])
        if y > 0:
            cnf.add_clause([-hv, top_ge[y]])

    xspan_ge = {w: cnf.new_var() for w in range(1, D + 2)}
    yspan_ge = {h: cnf.new_var() for h in range(0, D + 2)}
    cnf.add_clause([yspan_ge[0]])
    for w in range(2, D + 2):
        cnf.add_clause([-xspan_ge[w], xspan_ge[w - 1]])
    for h in range(1, D + 2):
        cnf.add_clause([-yspan_ge[h], yspan_ge[h - 1]])
    for x_lo in range(-D, 1):
        for x_hi in range(1, D + 1):
            w = x_hi - x_lo
            if w <= D + 1:
                cnf.add_clause([-left_le[x_lo], -right_ge[x_hi], xspan_ge[w]])
            else:
                cnf.add_clause([-left_le[x_lo], -right_ge[x_hi]])
    for y_lo in range(-D, 1):
        for y_hi in range(0, D + 1):
            h = y_hi - y_lo
            if h <= D + 1:
                cnf.add_clause([-bot_le[y_lo], -top_ge[y_hi], yspan_ge[h]])
            else:
                cnf.add_clause([-bot_le[y_lo], -top_ge[y_hi]])
    for w in range(1, D + 2):
        cnf.add_clause([-xspan_ge[w], -yspan_ge[D + 1 - w]])

    return cnf, p_var, valid_cells


def main() -> None:
    import pycosat

    export_dir = None
    if "--export-dimacs" in sys.argv:
        export_dir = Path(sys.argv[sys.argv.index("--export-dimacs") + 1])

    print("=== Independent Pure-CNF SAT Verification via PicoSAT (ZERO OR-Tools!) ===", flush=True)

    # 1. S8 (6x7 rectangle, pad=3)
    s8_seq = BENCHMARKS["S8"].sequence
    rect67 = {(x, y) for x in range(6) for y in range(7)}
    st8, dt8, v8, c8 = verify_hcore_picosat(
        s8_seq, rect67, pad=3, dimacs_prefix=(export_dir / "s8_6x7") if export_dir else None
    )
    print(f"  S8 (6x7, pad=3) PicoSAT: {st8} in {dt8:.3f}s (vars={v8}, clauses={c8})", flush=True)
    assert st8 == "UNSAT"

    # 2. S7 (all 4/4 parity-ok minimum-perimeter 43-ominoes, pad=5)
    s7_seq = BENCHMARKS["S7"].sequence
    s7_all = enumerate_min_perimeter_polyominoes(43, max_ie_harary_harborth(43))
    h_even = sum(1 for i, c in enumerate(s7_seq) if c == "H" and i % 2 == 0)
    h_odd = sum(1 for i, c in enumerate(s7_seq) if c == "H" and i % 2 == 1)
    s7_ok = [
        set(hc) for hc in sorted(s7_all)
        if (sum(1 for x, y in hc if (x + y) % 2 == 0), len(hc) - sum(1 for x, y in hc if (x + y) % 2 == 0))
        in ((h_even, h_odd), (h_odd, h_even))
    ]
    assert len(s7_all) == 147 and len(s7_ok) == 4
    t_s7 = time.time()
    for idx, hc in enumerate(s7_ok):
        st7, dt7, v7, c7 = verify_hcore_picosat(
            s7_seq, hc, pad=5, dimacs_prefix=(export_dir / f"s7_cand{idx+1}") if export_dir else None
        )
        print(f"  S7 cand {idx+1}/4 (pad=5) PicoSAT: {st7} in {dt7:.3f}s (vars={v7}, clauses={c7})", flush=True)
        assert st7 == "UNSAT"
    print(f"  S7 ALL 4/4 UNSAT in PicoSAT ({time.time() - t_s7:.3f}s total)", flush=True)

    # 3. S6 C=19 sanity check in PicoSAT: verify transformed submissions/S6.json satisfies EVERY clause!
    d21 = verify_exact_z2_diameter_bound(21)
    t0 = time.time()
    cnf19, p_var19, vc19 = build_s6_cnf(slack_set={31, 37}, max_wh_minus_2=d21)
    s6_data = json.loads((REPO / "submissions" / "S6.json").read_text())
    raw_c = [tuple(c) for c in s6_data["coords"]]
    p0, p1 = raw_c[38], raw_c[39]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    tr = [((x - p0[0]) * dx + (y - p0[1]) * dy, -(x - p0[0]) * dy + (y - p0[1]) * dx) for x, y in raw_c]
    if tr[40][1] < 0:
        tr = [(x, -y) for x, y in tr]
    sub_tr = tr[2:]
    assump = [[p_var19[(i, sub_tr[i])]] for i in range(len(sub_tr))]
    res19 = pycosat.solve(cnf19.clauses + assump)
    dt19 = time.time() - t0
    assert isinstance(res19, list), f"Known 19-contact fold rejected by CNF: {res19}"
    print(
        f"  S6 C=19 known-fold CNF check (PicoSAT): SAT in {dt19:.3f}s "
        f"(vars={cnf19.nvars}, clauses={len(cnf19.clauses)}, 0 violated clauses)",
        flush=True,
    )

    # 4. S6 C=21 & C=20 DIMACS CNF export (and optional PicoSAT solve via --solve-s6-c21)
    d20 = verify_exact_z2_diameter_bound(20)
    cnf21, _, _ = build_s6_cnf(slack_set=set(), max_wh_minus_2=d21)
    if export_dir is not None:
        cnf21.write_dimacs(export_dir / "s6_c21_unconditional_z2.cnf")
        odd_h_full = [i for i, ch in enumerate(BENCHMARKS["S6"].sequence) if ch == "H" and i % 2 == 1]
        for k in odd_h_full:
            cnfk, _, _ = build_s6_cnf(slack_set={k}, max_wh_minus_2=d20)
            cnfk.write_dimacs(export_dir / f"s6_c20_slack_H{k}_unconditional_z2.cnf")
        print(f"  Exported DIMACS CNF files (S8, S7 4/4, S6 C=21, S6 C=20 10/10) to {export_dir}", flush=True)

    if "--solve-s6-c21" in sys.argv:
        t0 = time.time()
        print(f"  Solving S6 C=21 unconditional Z^2 in PicoSAT (vars={cnf21.nvars}, clauses={len(cnf21.clauses)})...", flush=True)
        res21 = pycosat.solve(cnf21.clauses)
        dt21 = time.time() - t0
        print(f"  S6 C=21 unconditional Z^2 PicoSAT: {res21} in {dt21:.3f}s", flush=True)
        assert res21 == "UNSAT"
    print("ALL PICOSAT INDEPENDENT CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
