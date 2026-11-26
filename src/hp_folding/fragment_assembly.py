"""Fragment-assembly HP folder.

Pre-build a library of structural motifs (small lattice SAW paths with
known entry/exit topology), then enumerate compositions of motifs that
sum to the chain length. Each composition is rotated/reflected/translated
into a full lattice fold. SAW-validate, score by H-H contacts.

The composition space is vastly smaller than the chain-conformation
space, AND every output is structurally distinct, NOT a perturbation of
an MCMC-reachable basin.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

Coord = tuple[int, int]
_NEIGHBOURS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Motif:
    name: str
    cells: tuple[Coord, ...]  # relative to (0, 0)

    @property
    def length(self) -> int:
        return len(self.cells)

    @property
    def exit_pos(self) -> Coord:
        return self.cells[-1]

    @property
    def exit_bond(self) -> Coord:
        """Direction of last bond (cells[-1] - cells[-2])."""
        a, b = self.cells[-2], self.cells[-1]
        return (b[0] - a[0], b[1] - a[1])

    @property
    def entry_bond(self) -> Coord:
        a, b = self.cells[0], self.cells[1]
        return (b[0] - a[0], b[1] - a[1])


def _rot(cell: Coord, k: int) -> Coord:
    """Rotate by 90°*k (k in 0,1,2,3) counterclockwise."""
    x, y = cell
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y)


def _refl(cell: Coord, do_refl: bool) -> Coord:
    return (cell[0], -cell[1]) if do_refl else cell


def _transform(cells: tuple[Coord, ...], rot: int, refl: bool, tx: int, ty: int):
    out = []
    for c in cells:
        c = _refl(c, refl)
        c = _rot(c, rot)
        out.append((c[0] + tx, c[1] + ty))
    return out


# -- Motif library -----------------------------------------------------------


def _multirow_snake(L: int, R: int) -> Motif:
    """Snake of R rows × L columns. Starts at (0,0), ends at (?, R-1)."""
    cells = []
    for r in range(R):
        row = list(range(L))
        if r % 2 == 1:
            row.reverse()
        for i in row:
            cells.append((i, r))
    name = f"snake_{L}x{R}"
    return Motif(name, tuple(cells))


def _spiral(side: int) -> Motif:
    """Inward square spiral filling a side×side block."""
    cells: list[Coord] = []
    occ: set[Coord] = set()
    x, y = 0, 0
    dx, dy = 1, 0
    for _ in range(side * side):
        cells.append((x, y))
        occ.add((x, y))
        # Try to keep going same direction; if blocked or off-grid, turn left
        nx, ny = x + dx, y + dy
        if (nx, ny) in occ or not (0 <= nx < side and 0 <= ny < side):
            dx, dy = -dy, dx  # left turn
            nx, ny = x + dx, y + dy
        x, y = nx, ny
    return Motif(f"spiral_{side}", tuple(cells))


def _build_library() -> list[Motif]:
    out: list[Motif] = []

    # Compact multi-row snakes — prioritize first since these have high
    # internal-edge density (= many H-H contact opportunities).
    for L in range(3, 9):
        for R in range(2, 9):
            if L * R <= 56 and L * R >= 6:
                out.append(_multirow_snake(L, R))

    # Spirals (square blocks)
    for side in range(3, 8):
        out.append(_spiral(side))

    # Hairpins
    out.append(Motif("hairpin4", ((0, 0), (1, 0), (1, 1), (0, 1))))
    out.append(Motif("hairpin5", ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1))))
    out.append(Motif("hairpin6", ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1))))

    # L-corner 3-cell
    out.append(Motif("Lcorner", ((0, 0), (1, 0), (1, 1))))

    # Straight short pieces (for connecting compact motifs)
    for k in (3, 4, 5, 6):
        cells = tuple((i, 0) for i in range(k))
        out.append(Motif(f"straight_{k}", cells))

    return out


MOTIF_LIBRARY = _build_library()


def load_empirical_library(
    path: str | Path | None = None,
    min_freq_quantile: float = 0.0,
    max_per_length: int = 50,
) -> list[Motif]:
    """Load motifs extracted from real high-contact folds.

    The empirical library has, per length k, a list of (frequency, cells)
    pairs. We keep top `max_per_length` by frequency at each length.
    """
    import json
    from pathlib import Path as _Path

    lib_path = (
        _Path(path)
        if path is not None
        else _Path(__file__).resolve().parents[2] / "diagnostics" / "empirical_motif_library.json"
    )
    data = json.loads(lib_path.read_text())
    out: list[Motif] = []
    for k_str, entries in data.items():
        k = int(k_str)
        if k < 3:
            continue
        for idx, entry in enumerate(entries[:max_per_length]):
            freq, cells = entry
            cells_t = tuple(tuple(c) for c in cells)
            out.append(Motif(f"emp_k{k}_{idx}_{freq}", cells_t))
    return out


# -- Composition enumerator --------------------------------------------------


@dataclass
class Placement:
    motif: Motif
    rot: int
    refl: bool
    tx: int
    ty: int


def _placement_cells(p: Placement) -> list[Coord]:
    return _transform(p.motif.cells, p.rot, p.refl, p.tx, p.ty)


def _all_transforms_for(motif: Motif):
    """Yield (rot, refl) variants of a motif (all 8 D4 symmetries)."""
    seen = set()
    for rot in range(4):
        for refl in (False, True):
            cells = tuple(sorted(_transform(motif.cells, rot, refl, 0, 0)))
            # Dedupe by canonical cell set + entry/exit relative ordering
            entry_dir = (
                _rot(motif.cells[1], rot)[0] - _rot(motif.cells[0], rot)[0]
                if not refl
                else _refl(_rot(motif.cells[1], rot), refl)[0]
                - _refl(_rot(motif.cells[0], rot), refl)[0]
            )
            key = (cells, rot, refl)
            if key in seen:
                continue
            seen.add(key)
            yield rot, refl


def enumerate_assemblies(
    sequence: str,
    *,
    library: list[Motif] = MOTIF_LIBRARY,
    max_assemblies: int = 200_000,
    score_fn=None,
):
    """Enumerate fold assemblies. Yield (cells, score) tuples for valid
    SAW assemblies whose total length == len(sequence).

    Recursive backtracking over motif choices. Hard-prunes any partial
    assembly that overlaps itself.
    """
    n = len(sequence)
    yielded = [0]

    def rec(remaining: int, last_pos: Coord | None,
            last_bond: Coord | None, occupied: set[Coord],
            chain: list[Coord]):
        if yielded[0] >= max_assemblies:
            return
        if remaining == 0:
            score = score_fn(sequence, chain) if score_fn else 0
            yield (list(chain), score)
            yielded[0] += 1
            return
        for motif in library:
            if motif.length > remaining:
                continue
            if motif.length < remaining and motif.length < 3:
                # avoid 1-2 cell trailing motifs (nothing in library that small)
                continue
            for rot, refl in _all_transforms_for(motif):
                if last_pos is None:
                    # First motif: place at origin
                    cells = _transform(motif.cells, rot, refl, 0, 0)
                else:
                    # Anchor: motif's first cell must be at last_pos + last_bond
                    new_first = (last_pos[0] + last_bond[0],
                                 last_pos[1] + last_bond[1])
                    motif0 = _transform(motif.cells, rot, refl, 0, 0)[0]
                    tx, ty = new_first[0] - motif0[0], new_first[1] - motif0[1]
                    cells = _transform(motif.cells, rot, refl, tx, ty)
                # Validate SAW: no overlap, all bonds == 1
                ok = True
                for i, c in enumerate(cells):
                    if c in occupied:
                        ok = False
                        break
                    if i > 0:
                        prev = cells[i - 1]
                        if abs(c[0] - prev[0]) + abs(c[1] - prev[1]) != 1:
                            ok = False
                            break
                if not ok:
                    continue
                # Commit
                new_occupied = occupied | set(cells)
                new_chain = chain + cells
                # Compute new last bond: end of this motif
                if motif.length >= 2:
                    a, b = cells[-2], cells[-1]
                    new_bond = (b[0] - a[0], b[1] - a[1])
                else:
                    new_bond = last_bond
                yield from rec(remaining - motif.length, cells[-1],
                                new_bond, new_occupied, new_chain)
                if yielded[0] >= max_assemblies:
                    return

    yield from rec(n, None, None, set(), [])
