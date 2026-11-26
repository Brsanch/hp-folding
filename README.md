# hp-folding

Certified global optimality proofs, independent DRAT/LRAT clausal verification pipelines, and machine-verified lattice coordinates for two-dimensional hydrophobic--polar (2D HP) lattice protein folding benchmarks on $\mathbb{Z}^2$.

In the 2D HP lattice model (Dill 1985; Lau & Dill 1989), a sequence $s \in \{\mathrm{H}, \mathrm{P}\}^n$ is embedded as a self-avoiding walk on $\mathbb{Z}^2$ to maximize the number of non-sequential H--H lattice contacts. Computing the global optimum $C^*(s)$ is NP-complete in 2D (Crescenzi et al. 1998) and 3D (Berger & Leighton 1998).

## Manuscript

- **Paper:** [`paper/hp_folding_cpsat_optimality.pdf`](paper/hp_folding_cpsat_optimality.pdf) ([LaTeX source](paper/hp_folding_cpsat_optimality.tex)) — *Certified Optimality and a Spanning-Tree Core Diameter Bound for 2D HP Lattice Protein Folding* (Bryan Sanchez, ORCID [`0009-0000-6080-5407`](https://orcid.org/0009-0000-6080-5407)).
- **Pinned Release:** [`v1.0.0`](https://github.com/Brsanch/hp-folding/releases/tag/v1.0.0) (Apache-2.0 [`LICENSE`](LICENSE)).
- **Prior-Art Context:** Optimal contact counts and exact degeneracies for canonical 2D benchmarks up to length 85 (S5 = 23, HH-prefix `2D50` = 21, S7 = 36, S8 = 42, S9 = 53) were first established via the specialized exact branch-and-bound algorithm `minwalk` (Giaquinta & Pozzi, *J. Comput. Biol.* 20(8):593--609, 2013). Meanwhile, Chen & Huang (*Genomics Proteomics Bioinformatics* 3(4):225--230, 2005, Table 1) paired a four-residue PP-prefix variant (`PPHP...`, $n_H = 22$, our S6) with target energy $-21$. This repository provides:
  1. Unconditional optimality certification on $\mathbb{Z}^2$ for S5 ($C^* = 23$) and S9 ($C^* = 53$) via the analytic Harary--Harborth isoperimetric upper bound, and for S7 ($C^* = 36$) and S8 ($C^* = 42$) across four independent solvers (OR-Tools CP-SAT, HiGHS MILP, PicoSAT, and Z3) backed by six archived DRAT refutation proofs checked by the unverified C reference checker `drat-trim` and a standalone pure-Python RUP verifier.
  2. An exact spanning-tree H-core diameter bound over all of $\mathbb{Z}^2$ (valid for disconnected hydrophobic cores, paired with small-component bounds under at most one unit of parity degree slack) proving $C^*(\mathrm{S6}) = 19$ unconditionally via both OR-Tools CP-SAT and CaDiCaL 3.1.0 with LRAT proofs checked by the unverified C reference checker `lrat-check` (distinct from formally verified Coq/ACL2 LRAT checkers and CakeML's `cake_lpr`) across all 11 unsatisfiable branches, resolving the $n = 50$ discrepancy with canonical HH-prefix `2D50` ($C^* = 21$).
  3. Machine-verified public coordinate files for S5--S10 and `2D50`.

## Benchmark Summary

| ID | $n$ | $n_H$ | $\mathrm{adj}_{HH}$ | Harary--Harborth | Parity UB | Achieved $C$ | Unconditional $\mathbb{Z}^2$ Optimality Status |
|---|---:|---:|---:|---:|---:|---:|---|
| S5 | 48 | 25 | 17 | 23 | 25 | **23** | **Proven optimal on $\mathbb{Z}^2$** (Harary--Harborth; confirms Giaquinta & Pozzi 2013) |
| S6 | 50 | 22 | 7 | 27 | 21 | **19** | **Proven optimal on $\mathbb{Z}^2$** (PP-prefix, Chen & Huang 2005 Table 1; CP-SAT + 11 LRAT proofs) |
| `2D50` | 50 | 24 | 8 | 30 | 25 | **21** | Matches canonical optimum (Unger & Moult 1993 HH-prefix; Giaquinta & Pozzi 2013) |
| S7 | 60 | 43 | 35 | 37 | 40 | **36** | **Proven optimal on $\mathbb{Z}^2$** (147 $\to$ 4 $\to$ 0; 4 solvers + DRAT; confirms Giaquinta & Pozzi 2013) |
| S8 | 64 | 42 | 28 | 43 | 43 | **42** | **Proven optimal on $\mathbb{Z}^2$** (unique $6\times 7$ core; 4 solvers + DRAT; confirms Giaquinta & Pozzi 2013) |
| S9 | 85 | 59 | 49 | 53 | 58 | **53** | **Proven optimal on $\mathbb{Z}^2$** (Harary--Harborth; confirms Giaquinta & Pozzi 2013) |
| S10 | 100 | 56 | 39 | 58 | 56 | **50** | Matches record (Hsu et al. 2003; Zhang et al. 2007; Wüst & Landau 2012; `2D100b`, Ramakrishnan et al. 1997; open $[50, 56]$) |

Short reference instances S1--S4 ($n \in \{20, 24, 25, 36\}$, proven optima $9, 9, 8, 14$) are also included in `src/hp_folding/sequences.py` for unit testing.

## Installation

Python 3.10+ is required. Core verification and heuristic search have no third-party runtime dependencies; exact proof and cross-verification scripts require the `[proofs]` extra (`ortools`, `highspy`, `pycosat`), and figure generation requires `[viz]` (`matplotlib`, `numpy`).

```bash
pip install -e ".[proofs,viz,dev]"
```

Optional external tools for clausal proof generation and checking: `cadical` (tested with v3.1.0), and the unverified C reference checkers `drat-trim` and `lrat-check` (distinct from the formally verified Coq/ACL2 LRAT checkers of Cruz-Filipe et al. 2017 and CakeML's `cake_lpr`; any standard formally verified LRAT checker can re-check the generated LRAT traces drop-in).

## Reproducing the Optimality Proofs and Figures

All timings below were measured on an Apple M3 Ultra workstation (28 cores, 96 GiB unified memory, macOS) using Python 3.13.9, OR-Tools CP-SAT 9.15.6755, HiGHS 1.14.0, `pycosat` 0.6.6, Z3 4.16.0, and CaDiCaL 3.1.0.

```bash
# 1. Unconditional Z^2 optimality for S5, S7, S8, and S9 in OR-Tools CP-SAT (~0.19 s)
PYTHONPATH=src python scripts/prove_optimality.py

# 2. Independent cross-verification in HiGHS MILP (~0.45 s for S8 + S7, ~0.35 s S6 C=19 check)
PYTHONPATH=src python scripts/verify_hp_highs.py

# 3. Independent pure-CNF cross-verification in PicoSAT (~0.09 s for S8 + S7) + DIMACS export
PYTHONPATH=src python scripts/verify_hp_sat.py --export-dimacs /tmp/hp_dimacs

# 4. Independent DRAT/RUP verification of all 6 archived certificates for S8 & S7 (~0.58 s pure Python)
PYTHONPATH=src python scripts/verify_drat_certificates.py

#    (Optional) Verify the same 6 DRAT files with drat-trim (~0.024-0.027 s each):
for f in docs/certificates/*.drat; do
  base=$(basename "$f" .drat)
  drat-trim "/tmp/hp_dimacs/${base}.cnf" "$f"
done

# 5. Unconditional Z^2 optimality C*(S6) = 19 in OR-Tools CP-SAT (~1059.7 s across 11 UNSAT branches)
PYTHONPATH=src python scripts/prove_s6_unconditional_z2.py --resume

# 6. Independent CaDiCaL 3.1.0 + lrat-check pipeline for S6 (~42 min cumulative CPU; pass --keep-lrat [DIR] to retain traces)
PYTHONPATH=src python scripts/verify_s6_lrat.py --workers 3

# 7. Regenerate publication figures (paper/figures/fig1_*.pdf, fig2_*.pdf)
python paper/scripts/make_hp_figures.py

# 8. Run the automated optimality and verifier test suite
pytest tests/test_optimality_proofs.py tests/test_verifier.py
```

For S6, the eleven uncompressed LRAT traces total 64.92 GB (60.46 GiB) and are deleted by default after verification (use `--keep-lrat [DIR]` to retain them). Exact CNF and LRAT SHA-256 digests, conflict counts, and timings are archived in `results/s6_lrat_verification.json` and `results/s6_unconditional_z2_proof.json`.

## Repository Layout

```text
.
├── paper/
│   ├── hp_folding_cpsat_optimality.tex    Manuscript LaTeX source
│   ├── hp_folding_cpsat_optimality.pdf    Compiled manuscript PDF
│   ├── figures/                           Vector PDF figures (Fig. 1 and Fig. 2)
│   └── scripts/make_hp_figures.py         Reproducible figure generator
├── docs/
│   └── certificates/                      Archived DRAT/RUP certificates for S7 & S8 (6 files, 1,309 B)
├── results/
│   ├── s6_unconditional_z2_proof.json     OR-Tools CP-SAT unconditional Z^2 ledger for S6
│   └── s6_lrat_verification.json          CaDiCaL 3.1.0 + lrat-check ledger & SHA-256 digests for S6
├── submissions/                           Verified lattice coordinate JSON files (S5--S10, 2D50)
├── scripts/
│   ├── prove_optimality.py                S5, S7, S8, S9 unconditional Z^2 CP-SAT proof script
│   ├── prove_s6_unconditional_z2.py       S6 spanning-tree bound & 11-branch CP-SAT proof script
│   ├── verify_hp_highs.py                 Independent HiGHS MILP cross-verifier
│   ├── verify_hp_sat.py                   Independent PicoSAT CNF verifier & DIMACS CNF exporter
│   ├── verify_drat_certificates.py        Pure-Python RUP/DRAT checker (+ Z3 DRAT generator)
│   └── verify_s6_lrat.py                  Independent CaDiCaL + lrat-check pipeline for S6
├── src/hp_folding/                        Core verifier, sequences, CP-SAT routability, polyomino & search modules
├── tests/                                 Pytest unit and exact optimality verification test suite
└── pyproject.toml                         Package metadata and optional dependency groups
```

## Coordinate JSON Format and Verification Rules

Each archived fold in `submissions/*.json` stores:

```json
{
  "sequence_id": "S5",
  "sequence": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
  "lattice": "2D",
  "coords": [[0, 0], [1, 0], [1, 1], ...]
}
```

A fold is valid if and only if: (1) `len(coords) == len(sequence)`, (2) all lattice coordinates are distinct (self-avoidance), and (3) consecutive residues satisfy $\|c_{i+1} - c_i\|_1 = 1$. The objective is the count of non-sequential H--H pairs at Manhattan distance 1.

## Selected References

- Backofen, R., & Will, S. (2006). A constraint-based approach to fast and exact structure prediction in three-dimensional protein models. *Constraints*, 11(1), 5--30.
- Berger, B., & Leighton, T. (1998). Protein folding in the hydrophobic-hydrophilic (HP) model is NP-complete. *J. Comput. Biol.*, 5(1), 27--40.
- Biere, A., Faller, T., Fazekas, K., Fleury, M., Froleyks, N., & Pollitt, F. (2024). CaDiCaL 2.0. *CAV 2024*, LNCS 14681, 133--152.
- Chen, M., & Huang, W. Q. (2005). A branch and bound algorithm for the protein folding problem in the HP lattice model. *Genomics Proteomics Bioinformatics*, 3(4), 225--230.
- Crescenzi, P., Goldman, D., Papadimitriou, C., Piccolboni, A., & Yannakakis, M. (1998). On the complexity of protein folding. *J. Comput. Biol.*, 5(3), 423--465.
- Cruz-Filipe, L., Heule, M. J. H., Hunt, W. A., Jr., Kaufmann, M., & Schneider-Kamp, P. (2017). Efficient certified RAT verification. *CADE-26*, LNCS 10395, 220--236.
- Dill, K. A. (1985). Theory for the folding and stability of globular proteins. *Biochemistry*, 24(6), 1501--1509.
- Giaquinta, E., & Pozzi, L. (2013). An effective exact algorithm and a new upper bound for the number of contacts in the hydrophobic-polar two-dimensional lattice model. *J. Comput. Biol.*, 20(8), 593--609.
- Harary, F., & Harborth, H. (1976). Extremal animals. *J. Combin. Inform. System Sci.*, 1(1), 1--8.
- Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Math. Program. Comput.*, 10(1), 119--142.
- Perron, L., Didier, F., & Gay, S. (2023). The CP-SAT-LP solver. *CP 2023*, LIPIcs 280, 3:1--3:2.
- Ramakrishnan, R., Ramachandran, B., & Pekny, J. F. (1997). A dynamic Monte Carlo algorithm for exploration of dense conformational spaces in heteropolymers. *J. Chem. Phys.*, 106(6), 2418--2425.
- Unger, R., & Moult, J. (1993). Genetic algorithms for protein folding simulations. *J. Mol. Biol.*, 231(1), 75--81.
- Wetzler, N., Heule, M. J. H., & Hunt, W. A., Jr. (2014). DRAT-trim: Efficient checking and trimming using expressive clausal proofs. *SAT 2014*, LNCS 8561, 422--429.
- Wüst, T., & Landau, D. P. (2012). Optimized Wang--Landau sampling of lattice polymers: Ground state search and folding thermodynamics of HP model proteins. *J. Chem. Phys.*, 137(6), 064903.

## License

Apache-2.0. See [LICENSE](LICENSE).
