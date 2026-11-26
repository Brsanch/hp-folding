"""JSON serialization for CAISc 2026 HP Protein Folding submissions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hp_folding.fold import Fold


def fold_to_submission(fold: Fold, sequence_id: str) -> dict[str, Any]:
    """Convert a :class:`Fold` to the CAISc submission schema.

    Parameters
    ----------
    fold
        The fold to serialise.
    sequence_id
        Benchmark identifier (``"S5"`` through ``"S10"``).

    Returns
    -------
    dict
        Dictionary matching the CAISc schema::

            {
                "sequence_id": "S5",
                "lattice": "2D",
                "coords": [[0, 0], [1, 0], ...]
            }
    """
    return {
        "sequence_id": sequence_id,
        "lattice": "2D",
        "coords": [list(c) for c in fold.coords],
    }


def submission_from_dict(data: Mapping[str, Any]) -> tuple[str, list[tuple[int, int]]]:
    """Parse a CAISc submission dictionary and return (sequence_id, coords).

    Raises ``ValueError`` if the input is missing required keys or has the
    wrong shape.
    """
    for key in ("sequence_id", "lattice", "coords"):
        if key not in data:
            raise ValueError(f"submission missing required key {key!r}")
    if data["lattice"] != "2D":
        raise ValueError(
            f"only 2D lattice is supported; got {data['lattice']!r}"
        )
    coords = [(int(c[0]), int(c[1])) for c in data["coords"]]
    return str(data["sequence_id"]), coords


def write_submission(fold: Fold, sequence_id: str, path: str | Path) -> None:
    """Write a submission JSON file to ``path``."""
    payload = fold_to_submission(fold, sequence_id)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def read_submission(path: str | Path) -> tuple[str, list[tuple[int, int]]]:
    """Read a submission JSON file and return (sequence_id, coords)."""
    data = json.loads(Path(path).read_text())
    return submission_from_dict(data)
