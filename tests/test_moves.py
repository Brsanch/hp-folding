"""Tests for move operators."""

import random

from hp_folding.fold import Fold
from hp_folding.moves import end_flip, pivot, pull_move, random_move
from hp_folding.verifier import verify


def _assert_saw(seq: str, coords) -> None:
    v = verify(seq, coords)
    assert v.valid, v.error


def test_end_flip_produces_saw():
    seq = "HPHPHP"
    f = Fold.linear(seq)
    rng = random.Random(0)
    for _ in range(50):
        proposal = end_flip(f.coords, rng)
        if proposal is not None:
            _assert_saw(seq, proposal)


def test_end_flip_preserves_length():
    seq = "HHHHH"
    f = Fold.linear(seq)
    rng = random.Random(7)
    proposal = end_flip(f.coords, rng)
    assert proposal is None or len(proposal) == len(seq)


def test_pivot_none_for_short_chains():
    rng = random.Random(0)
    assert pivot([(0, 0), (1, 0)], rng) is None
    assert pivot([(0, 0), (1, 0), (2, 0)], rng) is None


def test_pivot_can_produce_valid_folds():
    seq = "HPHPHPHP"
    f = Fold.linear(seq)
    rng = random.Random(1)
    got_one = False
    for _ in range(200):
        proposal = pivot(f.coords, rng)
        if proposal is not None:
            _assert_saw(seq, proposal)
            got_one = True
    assert got_one, "pivot should occasionally succeed on a straight line of length 8"


def test_pull_move_produces_saw_or_none():
    seq = "HHHHHHHH"
    f = Fold.linear(seq)
    rng = random.Random(2)
    for _ in range(200):
        proposal = pull_move(f.coords, rng)
        if proposal is not None:
            _assert_saw(seq, proposal)
            assert len(proposal) == len(seq)


def test_random_move_returns_saw_or_none():
    seq = "HPHPHPHPHP"
    coords = Fold.linear(seq).coords
    rng = random.Random(3)
    for _ in range(300):
        proposal = random_move(coords, rng)
        if proposal is not None:
            _assert_saw(seq, proposal)
