"""Tests for submission serialization."""

import json
from pathlib import Path

from hp_folding.fold import Fold
from hp_folding.io import (
    fold_to_submission,
    read_submission,
    submission_from_dict,
    write_submission,
)


def test_round_trip(tmp_path: Path) -> None:
    seq = "HHHHHH"
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    f = Fold(sequence=seq, coords=coords)
    path = tmp_path / "sub.json"
    write_submission(f, "S5", path)
    sid, got = read_submission(path)
    assert sid == "S5"
    assert got == coords


def test_submission_from_dict_rejects_3d() -> None:
    bad = {"sequence_id": "S5", "lattice": "3D", "coords": []}
    try:
        submission_from_dict(bad)
    except ValueError as e:
        assert "2D" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_submission_from_dict_missing_key() -> None:
    bad = {"sequence_id": "S5", "lattice": "2D"}
    try:
        submission_from_dict(bad)
    except ValueError as e:
        assert "coords" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_fold_to_submission_shape() -> None:
    f = Fold(sequence="HH", coords=[(0, 0), (1, 0)])
    sub = fold_to_submission(f, "S5")
    assert sub["sequence_id"] == "S5"
    assert sub["lattice"] == "2D"
    assert sub["coords"] == [[0, 0], [1, 0]]
    # JSON round-trip
    assert json.loads(json.dumps(sub)) == sub
