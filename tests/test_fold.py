"""Tests for the Fold data structure."""

import random

from hp_folding.fold import Fold


def test_linear_fold_is_valid():
    f = Fold.linear("HHHHHH")
    assert f.verify().valid
    assert f.contacts() == 0


def test_random_saw_is_valid():
    rng = random.Random(0)
    for _ in range(10):
        f = Fold.random_saw("HPHPHPHPHPHPHPHP", rng)
        v = f.verify()
        assert v.valid, v.error
        assert len(f) == 16


def test_random_saw_longer_sequences():
    rng = random.Random(42)
    # The growth algorithm can trap itself; it should either produce a valid
    # SAW or fall back to the straight line.
    f = Fold.random_saw("H" * 40, rng)
    v = f.verify()
    assert v.valid, v.error
    assert len(f) == 40


def test_fold_copy_is_deep():
    f = Fold(sequence="HH", coords=[(0, 0), (1, 0)])
    g = f.copy()
    g.coords[0] = (9, 9)
    assert f.coords[0] == (0, 0)


def test_fold_len_and_iter():
    f = Fold.linear("HPHP")
    assert len(f) == 4
    assert list(f) == [(0, 0), (1, 0), (2, 0), (3, 0)]
