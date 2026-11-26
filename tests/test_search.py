"""Smoke tests for the simulated-annealing baseline.

These tests are fast (small iteration counts) and verify the solver
produces valid output, not that it achieves any particular score.
"""

from hp_folding.search import simulated_annealing


def test_sa_produces_valid_fold_for_short_sequence():
    seq = "HPHPHPHP"
    result = simulated_annealing(seq, iterations=2000, seed=0)
    assert result.fold.verify().valid
    assert result.contacts >= 0
    assert result.iterations == 2000


def test_sa_straight_line_baseline_improves():
    """On a sequence with potential contacts, SA should find at least one."""
    seq = "HHHHPPPPHHHHPPPPHHHH"
    result = simulated_annealing(seq, iterations=5000, seed=0)
    assert result.fold.verify().valid
    # Straight-line starting fold has 0 contacts; any positive outcome is an
    # improvement. We don't assert a tight number because the schedule here
    # is intentionally short.
    assert result.contacts >= 1


def test_sa_reproducible_with_seed():
    seq = "HPHPHPHP"
    r1 = simulated_annealing(seq, iterations=1000, seed=123)
    r2 = simulated_annealing(seq, iterations=1000, seed=123)
    assert r1.contacts == r2.contacts
    assert r1.fold.coords == r2.fold.coords
