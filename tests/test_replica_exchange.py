"""Smoke tests for replica-exchange Monte Carlo."""

import pytest

from hp_folding.search import replica_exchange


def test_replica_exchange_produces_valid_fold():
    result = replica_exchange("HPHPHPHPHPHP", iterations=500, seed=0)
    v = result.fold.verify()
    assert v.valid, v.error
    assert result.contacts >= 0


def test_replica_exchange_seeded_reproducible():
    r1 = replica_exchange("HPHPHPHPHPHP", iterations=500, seed=42)
    r2 = replica_exchange("HPHPHPHPHPHP", iterations=500, seed=42)
    assert r1.contacts == r2.contacts
    assert r1.fold.coords == r2.fold.coords


def test_replica_exchange_rejects_single_temperature():
    with pytest.raises(ValueError):
        replica_exchange("HH", iterations=10, temperatures=(1.0,), seed=0)


def test_replica_exchange_finds_contacts_on_structured_sequence():
    seq = "HHHHPPPPHHHHPPPPHHHH"
    result = replica_exchange(
        seq, iterations=2000, temperatures=(0.2, 0.7, 2.0, 5.0), seed=0
    )
    assert result.fold.verify().valid
    # Structured seq has potential contacts; should find at least 1
    assert result.contacts >= 1
