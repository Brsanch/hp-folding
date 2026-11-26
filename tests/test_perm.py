"""Tests for PERM."""

from hp_folding.perm import perm
from hp_folding.sequences import BENCHMARKS


def test_perm_produces_valid_fold():
    r = perm("HPHPHPHPHPHP", n_tours=500, seed=0)
    assert r.fold.verify().valid


def test_perm_finds_contacts_on_structured_sequence():
    r = perm("HHHHPPPPHHHHPPPPHHHH", n_tours=1000, seed=0)
    assert r.fold.verify().valid
    # Structured sequence should yield at least a few contacts
    assert r.contacts >= 2


def test_perm_seeded_reproducible():
    r1 = perm("HPHPHPHPHPHP", n_tours=200, seed=42)
    r2 = perm("HPHPHPHPHPHP", n_tours=200, seed=42)
    assert r1.contacts == r2.contacts


def test_perm_s5_smoke():
    """Smoke test on S5 — does not need to beat SA baseline, just be valid."""
    seq = BENCHMARKS["S5"].sequence
    r = perm(seq, n_tours=500, seed=0, temperature=0.5)
    assert r.fold.verify().valid
