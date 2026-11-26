"""Tests for grid CP-SAT solver and unconditional optimality proofs (S5, S7, S8, S9)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ortools = pytest.importorskip("ortools")

from hp_folding.grid_cpsat import solve_grid_cpsat
from hp_folding.sequences import BENCHMARKS
from hp_folding.spatial_lns import lns_step_cpsat, select_spatial_window
from hp_folding.verifier import verify


def _load_prove_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prove_optimality.py"
    spec = importlib.util.spec_from_file_location("prove_optimality", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_cpsat_solves_s1_optimal():
    b = BENCHMARKS["S1"]
    coords, contacts, status, _ = solve_grid_cpsat(
        b.sequence, grid_w=9, grid_h=9, time_budget_seconds=15.0, n_workers=4
    )
    assert status == "OPTIMAL"
    assert contacts == b.best_known == 9
    assert coords is not None and len(coords) == b.length


def test_unconditional_optimality_s5_s7_s8_s9():
    mod = _load_prove_module()
    res = mod.prove_all()
    assert res["S5"]["contacts"] == 23
    assert res["S7"]["contacts"] == 36
    assert res["S7"]["candidates_total"] == 147
    assert res["S7"]["candidates_checked"] == 4
    assert res["S8"]["contacts"] == 42
    assert res["S8"]["candidates_total"] == 1
    assert res["S8"]["candidates_checked"] == 1
    assert res["S9"]["contacts"] == 53


def test_spatial_lns_preserves_validity_and_monotonicity():
    import random

    b = BENCHMARKS["S1"]
    coords, c0, st, _ = solve_grid_cpsat(
        b.sequence, grid_w=9, grid_h=9, time_budget_seconds=10.0, n_workers=4
    )
    assert st == "OPTIMAL" and coords is not None
    free = select_spatial_window(b.sequence, coords, random.Random(0), min_free=8, max_free=12)
    new_coords, c1, _, _ = lns_step_cpsat(
        b.sequence, coords, free, pad=2, time_budget_seconds=5.0, n_workers=4
    )
    assert new_coords is not None
    assert verify(b.sequence, new_coords).valid
    assert c1 >= c0


def test_s6_unconditional_z2_diameter_bounds_and_ledger():
    import json

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("prove_s6_unconditional_z2", repo / "scripts" / "prove_s6_unconditional_z2.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.verify_exact_z2_diameter_bound(21) == 19
    assert mod.verify_exact_z2_diameter_bound(20) == 22

    ledger = json.loads((repo / "results" / "s6_unconditional_z2_proof.json").read_text())
    assert ledger["d21_max_wh_minus_2"] == 19
    assert ledger["d20_max_wh_minus_2"] == 22
    assert ledger["sanity_c19"]["verified_contacts"] == 19
    assert ledger["c21"]["status"] == "INFEASIBLE"
    assert ledger["c20_all_infeasible"] is True
    assert len(ledger["c20_cases"]) == 10
    assert all(v["status"] == "INFEASIBLE" for v in ledger["c20_cases"].values())

    lrat = json.loads((repo / "results" / "s6_lrat_verification.json").read_text())
    assert lrat["all_11_lrat_verified"] is True
    assert lrat["sanity_c19"]["status"] == "SATISFIABLE"
    assert lrat["sanity_c19"]["verified_contacts"] == 19
    assert lrat["c21"]["lrat_verified"] is True
    assert len(lrat["c20_cases"]) == 10
    assert all(v["lrat_verified"] is True for v in lrat["c20_cases"].values())


def test_independent_solvers_highs_and_picosat():
    highspy = pytest.importorskip("highspy")
    pycosat = pytest.importorskip("pycosat")
    _ = (highspy, pycosat)
    repo = Path(__file__).resolve().parents[1]

    for script_name in ("verify_hp_highs.py", "verify_hp_sat.py"):
        spec = importlib.util.spec_from_file_location(script_name[:-3], repo / "scripts" / script_name)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()


def test_drat_rup_certificates_zero_solver_trust():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "verify_drat_certificates", repo / "scripts" / "verify_drat_certificates.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod.verify_all()
    assert len(summary) == 6
    assert all(v["verified"] for v in summary.values())
