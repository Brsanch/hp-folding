"""Zero-dependency pure-Python RUP/DRAT proof-certificate verifier (and optional
Z3 DRAT generator via --generate) for the unconditional Z^2 refutations of
S8 (6x7 rectangle, both parity alignments) and S7 (all 4/4 parity-compatible
minimum-perimeter 43-ominoes).

Default mode (`python scripts/verify_drat_certificates.py`) imports ZERO solvers
(pure Python standard library + our pure-CNF DIMACS builder) and verifies every
derived clause in docs/certificates/*.drat by Reverse Unit Propagation (RUP)
down to the empty clause `0`. The archived .drat files are standard DRAT format
and can also be checked directly by `drat-trim` against the DIMACS CNF files
exported by `scripts/verify_hp_sat.py --export-dimacs`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prove_optimality import enumerate_min_perimeter_polyominoes, max_ie_harary_harborth
from verify_hp_sat import build_hcore_cnf

from hp_folding.sequences import BENCHMARKS

REPO = Path(__file__).resolve().parents[1]
CERT_DIR = REPO / "docs" / "certificates"


def verify_rup_certificate(orig_clauses: list[list[int]], rup_steps: list[list[int]]) -> tuple[bool, int]:
    """Zero-dependency RUP (Reverse Unit Propagation) checker.

    Returns (True, n_steps) iff every clause in rup_steps is implied by unit
    propagation from orig_clauses plus previously verified clauses, terminating
    in the empty clause [].
    """
    db = [list(c) for c in orig_clauses]
    for step_idx, cl in enumerate(rup_steps):
        assign: dict[int, bool] = {}
        conflict = False
        for lit in cl:
            v = abs(lit)
            val = lit < 0  # assume negation of every literal in cl
            if v in assign and assign[v] != val:
                conflict = True
                break
            assign[v] = val
        while not conflict:
            changed = False
            for c in db:
                unassigned: list[int] = []
                sat = False
                for lit in c:
                    v = abs(lit)
                    if v in assign:
                        if assign[v] == (lit > 0):
                            sat = True
                            break
                    else:
                        unassigned.append(lit)
                if sat:
                    continue
                if not unassigned:
                    conflict = True
                    break
                if len(unassigned) == 1:
                    u = unassigned[0]
                    assign[abs(u)] = u > 0
                    changed = True
            if not changed:
                break
        if not conflict:
            return False, step_idx
        db.append(list(cl))
    return (len(rup_steps) > 0 and rup_steps[-1] == []), len(rup_steps)


def read_drat(path: Path) -> list[list[int]]:
    steps: list[list[int]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        assert not line.startswith("d "), "Deletion steps not expected in pure-RUP certificates"
        nums = [int(tok) for tok in line.split()]
        assert nums and nums[-1] == 0, f"Malformed DRAT line: {line!r}"
        steps.append(nums[:-1])
    return steps


def write_drat(path: Path, steps: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for cl in steps:
            if cl:
                f.write(" ".join(str(x) for x in cl) + " 0\n")
            else:
                f.write("0\n")


def _all_s8_s7_instances() -> list[tuple[str, str, set[tuple[int, int]], int, int]]:
    out: list[tuple[str, str, set[tuple[int, int]], int, int]] = []
    s8_seq = BENCHMARKS["S8"].sequence
    rect67 = {(x, y) for x in range(6) for y in range(7)}
    out.append(("s8_6x7_par0", s8_seq, rect67, 3, 0))
    out.append(("s8_6x7_par1", s8_seq, rect67, 3, 1))

    s7_seq = BENCHMARKS["S7"].sequence
    s7_all = enumerate_min_perimeter_polyominoes(43, max_ie_harary_harborth(43))
    he = sum(1 for i, c in enumerate(s7_seq) if c == "H" and i % 2 == 0)
    ho = sum(1 for i, c in enumerate(s7_seq) if c == "H" and i % 2 == 1)
    s7_ok = [
        set(hc)
        for hc in sorted(s7_all)
        if (
            sum(1 for x, y in hc if (x + y) % 2 == 0),
            len(hc) - sum(1 for x, y in hc if (x + y) % 2 == 0),
        )
        in ((he, ho), (ho, he))
    ]
    assert len(s7_all) == 147 and len(s7_ok) == 4
    for idx, hc in enumerate(s7_ok):
        c0 = sum(1 for x, y in hc if (x + y) % 2 == 0)
        par = 0 if (c0, 43 - c0) == (he, ho) else 1
        out.append((f"s7_cand{idx + 1}_par{par}", s7_seq, hc, 5, par))
    return out


def generate_drat_with_z3() -> None:
    import z3

    def _expr_to_clause(e) -> list[int]:
        if z3.is_false(e):
            return []
        lits = e.children() if z3.is_or(e) else [e]
        res: list[int] = []
        for l in lits:
            if z3.is_not(l):
                res.append(-int(l.arg(0).decl().name()[1:]))
            else:
                res.append(int(l.decl().name()[1:]))
        return res

    for name, seq, hc, pad, par in _all_s8_s7_instances():
        cnf = build_hcore_cnf(seq, hc, pad, par)
        ctx = z3.Context(proof=True)
        s = z3.Solver(ctx=ctx)
        b = [None] + [z3.Bool(f"x{i}", ctx=ctx) for i in range(1, cnf.nvars + 1)]
        for cl in cnf.clauses:
            s.add(z3.Or([b[v] if v > 0 else z3.Not(b[-v]) for v in cl]))
        assert s.check() == z3.unsat
        pf = s.proof()
        visited: set[int] = set()
        lemmas: list[list[int]] = []

        def dfs(node) -> None:
            nid = node.get_id()
            if nid in visited:
                return
            visited.add(nid)
            for i in range(node.num_args() - 1):
                ch = node.arg(i)
                if z3.is_app(ch) and ch.sort().name() == "Proof":
                    dfs(ch)
            if node.decl().name() == "lemma":
                lemmas.append(_expr_to_clause(node.arg(node.num_args() - 1)))

        dfs(pf)
        lemmas.append([])
        write_drat(CERT_DIR / f"{name}.drat", lemmas)
        print(f"Generated {CERT_DIR / f'{name}.drat'} ({len(lemmas)} RUP clauses)", flush=True)


def verify_all() -> dict[str, dict]:
    print("=== Zero-Solver-Trust RUP/DRAT Certificate Verification (S8 & S7 4/4) ===", flush=True)
    t_all = time.time()
    summary: dict[str, dict] = {}
    for name, seq, hc, pad, par in _all_s8_s7_instances():
        t0 = time.time()
        cnf = build_hcore_cnf(seq, hc, pad, par)
        cert_path = CERT_DIR / f"{name}.drat"
        steps = read_drat(cert_path)
        ok, n_steps = verify_rup_certificate(cnf.clauses, steps)
        dt = time.time() - t0
        assert ok, f"RUP verification FAILED for {name} at step {n_steps}"
        print(f"  {name:14s}: RUP VERIFIED ({n_steps:2d} clauses, {dt:.3f}s)", flush=True)
        summary[name] = {"rup_clauses": n_steps, "elapsed_s": round(dt, 4), "verified": True}
    print(f"ALL 6/6 DRAT/RUP CERTIFICATES VERIFIED IN {time.time() - t_all:.3f}s (ZERO SOLVERS INVOKED).", flush=True)
    return summary


if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_drat_with_z3()
    verify_all()
