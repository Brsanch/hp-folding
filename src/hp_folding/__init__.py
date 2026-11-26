"""HP lattice protein folding on the 2D square lattice.

A clean implementation of the hydrophobic-polar lattice model: given a
sequence of H (hydrophobic) and P (polar) residues, find a self-avoiding
walk on the 2D integer lattice that maximises the number of non-sequential
H-H contacts.

Problem reference: CAISc 2026 Verifiable Track, HP Protein Folding.
"""

from hp_folding.fold import Fold
from hp_folding.io import fold_to_submission, submission_from_dict
from hp_folding.sequences import BENCHMARKS, REFERENCE_IDS, SUBMITTABLE_IDS, Benchmark
from hp_folding.verifier import Verdict, verify

__all__ = [
    "Benchmark",
    "BENCHMARKS",
    "SUBMITTABLE_IDS",
    "REFERENCE_IDS",
    "Verdict",
    "verify",
    "Fold",
    "fold_to_submission",
    "submission_from_dict",
]

__version__ = "0.1.0"
