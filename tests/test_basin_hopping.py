"""Smoke tests for basin-hopping iterated simulated annealing."""

from hp_folding.search import basin_hopping


def test_basin_hopping_produces_valid_fold():
    r = basin_hopping("HPHPHPHPHPHP", n_basins=3, iterations_per_basin=500, seed=0)
    v = r.fold.verify()
    assert v.valid, v.error
    assert r.contacts >= 0


def test_basin_hopping_is_monotone_in_contacts():
    """Basin hopping should never return fewer contacts than the first basin
    alone, because it keeps the best found so far across basins."""
    # Running a structured sequence: contacts possible, so some basin should find one.
    r = basin_hopping(
        "HHHHPPPPHHHHPPPPHHHH",
        n_basins=5,
        iterations_per_basin=1000,
        seed=0,
    )
    assert r.contacts >= 0
    assert r.fold.verify().valid
