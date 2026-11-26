"""Tests for beam search."""

from hp_folding.beam_search import beam_search, multi_start_beam
from hp_folding.sequences import BENCHMARKS


def test_beam_search_returns_valid_fold():
    f = beam_search("HPHPHPHPHP", beam_width=50)
    assert f.verify().valid


def test_beam_search_finds_contacts_on_structured_sequence():
    f = beam_search("HHHHPPPPHHHHPPPPHHHH", beam_width=200)
    v = f.verify()
    assert v.valid
    # This sequence has clear H-run structure; beam should find contacts
    assert v.contacts >= 2


def test_beam_search_matches_s5_on_small_beam():
    """Smoke: S5 on a tiny beam. Doesn't need to match record, just valid."""
    seq = BENCHMARKS["S5"].sequence
    f = beam_search(seq, beam_width=100)
    assert f.verify().valid


def test_multi_start_beam_returns_best():
    f = multi_start_beam(
        "HHHHPPPPHHHHPPPPHHHH",
        beam_widths=(50, 100),
        compactness_weights=(0.1, 0.5),
    )
    assert f.verify().valid
