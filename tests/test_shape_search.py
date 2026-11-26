"""Tests for shape-constrained SAW search."""

import pytest

from hp_folding.shape_search import (
    anchor_positions,
    inner_positions,
    rectangle_shape,
    search_shape_constrained,
    shape_5x5,
)


def test_rectangle_shape_sizes():
    assert len(rectangle_shape(5, 5)) == 25
    assert len(rectangle_shape(3, 4)) == 12


def test_5x5_has_25_cells():
    s = shape_5x5()
    assert len(s) == 25


def test_anchor_positions_surround_shape():
    s = shape_5x5()
    a = anchor_positions(s)
    # Perimeter of a 5x5 grid: 4*(5+5)-4 = 36 anchors? Actually it's the lattice
    # neighbours of the shape that are outside — for a 5x5 solid block that's
    # 5 on each side = 20 anchors total.
    assert len(a) == 20
    # All anchors are adjacent to at least one shape cell.
    for p in a:
        has_nbr = any(
            (p[0] + dx, p[1] + dy) in s
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        assert has_nbr


def test_inner_positions_are_the_shape():
    s = shape_5x5()
    assert set(inner_positions(s)) == set(s)


def test_search_rejects_mismatched_h_count():
    # 4-cell shape but 10-residue all-H sequence — strict mode must reject.
    with pytest.raises(ValueError):
        search_shape_constrained(
            "HHHHHHHHHH",
            rectangle_shape(2, 2),
            timeout_seconds=0.1,
        )


def test_search_finds_trivial_small_case():
    """HHHH in a 2x2 shape: a valid Hamiltonian cycle of the shape."""
    coords, contacts, nodes = search_shape_constrained(
        "HHHH",
        rectangle_shape(2, 2),
        timeout_seconds=2.0,
    )
    assert coords is not None
    assert len(coords) == 4
    # 2x2 block with all H — 4 internal edges, 3 sequential, so 1 non-sequential.
    assert contacts == 1


def test_search_handles_empty_shape_mismatch():
    """Sequence has 2 H's, shape has 1 cell — should raise in strict mode."""
    with pytest.raises(ValueError):
        search_shape_constrained("HH", rectangle_shape(1, 1), timeout_seconds=0.1)


def test_search_timeout_reports_partial():
    """Very short timeout should return None or a partial best."""
    # Large search: 48-residue S5 into 5x5, timeout 0.05s.
    from hp_folding.sequences import BENCHMARKS
    coords, best, nodes = search_shape_constrained(
        BENCHMARKS["S5"].sequence,
        shape_5x5(),
        timeout_seconds=0.05,
    )
    # Either we didn't finish a fold, or we found one. Nodes should have grown.
    assert nodes >= 1
    assert best >= -1  # sentinel
