"""Standard 2D HP-lattice benchmark sequences.

Sources: Unger & Moult (1993), Yue & Dill (1995), Beutler & Dill (1996),
Ramakrishnan, Ramachandran & Pekny (1997), Bastolla et al. (1998),
Liang & Wong (2001), Hsu et al. (2003), Chen & Huang (2005),
Zhang, Kou & Liu (2007), Wust & Landau (2012), and Giaquinta & Pozzi (2013).

Sequences S1-S4 are proven-optimal short reference instances; S5, S6
(PP-prefix variant from Chen & Huang 2005 Table 1, C* = 19), S6_canonical
(Unger & Moult 1993 HH-prefix 2D50, C* = 21), S7, S8, and S9 are proven
globally optimal on Z^2; S10 (2D100b, Ramakrishnan et al. 1997) remains
open in [50, 56].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    """A benchmark HP sequence with its best-known H-H contact count."""

    id: str
    length: int
    sequence: str
    best_known: int
    proven_optimal: bool
    note: str


BENCHMARKS: dict[str, Benchmark] = {
    "S1": Benchmark(
        "S1", 20, "HPHPPHHPHPPHPHHPPHPH", 9, True, "Proven optimal",
    ),
    "S2": Benchmark(
        "S2", 24, "HHPPHPPHPPHPPHPPHPPHPPHH", 9, True, "Proven optimal",
    ),
    "S3": Benchmark(
        "S3", 25, "PPHPPHHPPPPHHPPPPHHPPPPHH", 8, True, "Proven optimal",
    ),
    "S4": Benchmark(
        "S4", 36, "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
        14, True, "Proven optimal",
    ),
    "S5": Benchmark(
        "S5", 48, "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
        23, True, "Proven optimal (Harary-Harborth upper bound)",
    ),
    "S6": Benchmark(
        "S6", 50, "PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
        19, True,
        "PP-prefix 50-mer variant (differs in first 4 residues from canonical HH-prefix 2D50). "
        "Proven globally optimal on Z^2 (C*(S6) = 19) via exact spanning-tree H-core diameter "
        "bounds (D = 19 at C = 21, D = 22 at C = 20) and 11 unconditional CP-SAT UNSAT proofs "
        "(scripts/prove_s6_unconditional_z2.py).",
    ),
    "S6_canonical": Benchmark(
        "S6_canonical", 50,
        "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
        21, True,
        "Canonical HH-prefix 2D50 50-mer (Unger & Moult 1993; optimal value 21 established "
        "by Giaquinta & Pozzi 2013, J. Comput. Biol. 20(8):593-609; matched by submissions/2D50.json).",
    ),
    "S7": Benchmark(
        "S7", 60,
        "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP",
        36, True,
        "Proven optimal (all 147 min-perimeter 43-ominoes at ie=72 ruled out: "
        "4 parity-valid shapes proven CP-SAT infeasible with unconditional pad)",
    ),
    "S8": Benchmark(
        "S8", 64,
        "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH",
        42, True,
        "Proven optimal (unique ie=71 42-cell subset is 6x7 rectangle; "
        "proven CP-SAT infeasible with unconditional pad)",
    ),
    "S9": Benchmark(
        "S9", 85,
        "HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH",
        53, True,
        "Proven optimal (Harary-Harborth upper bound; achieved via CP-SAT routability)",
    ),
    "S10": Benchmark(
        "S10", 100,
        "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH",
        50, False,
        "Open; longest standard benchmark (Wust & Landau 2012)",
    ),
}

SUBMITTABLE_IDS: tuple[str, ...] = ("S5", "S6", "S7", "S8", "S9", "S10")
REFERENCE_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4")


def get(sequence_id: str) -> Benchmark:
    """Return the Benchmark with the given id, or raise KeyError."""
    try:
        return BENCHMARKS[sequence_id]
    except KeyError as err:
        known = ", ".join(sorted(BENCHMARKS))
        raise KeyError(
            f"unknown sequence_id {sequence_id!r}; known ids: {known}"
        ) from err
