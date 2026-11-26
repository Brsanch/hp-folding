"""Telemetry for shape-constrained DFS: sensors that record what the
search sees so we can understand where its compute is going.

Three classes:

SearchTelemetry
    Counters and histograms populated during a single DFS run.

AnchorStats
    Per-anchor depth and success counts.

CanonicalForm
    D4 + translation canonicalisation for detecting duplicate folds.

The telemetry is opt-in (pass ``telemetry=SearchTelemetry()`` to
``search_shape_constrained``). It adds a few-percent overhead because
counter updates happen in the inner loop.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

Coord = tuple[int, int]


# ---- counters ---------------------------------------------------------------

@dataclass
class SearchTelemetry:
    """Per-run diagnostic counters for shape-constrained DFS."""

    # depth -> number of branches that backtracked at exactly this depth
    fail_depth_hist: Counter[int] = field(default_factory=Counter)

    # depth -> [branch factors observed at this depth]
    # A branch factor is the count of neighbour moves that passed constraints
    # before the loop terminated.
    valid_moves_per_depth: list[list[int]] = field(default_factory=list)

    # Dead-end cause counters.
    # An "out-of-moves" event is when a DFS node finds no valid neighbour.
    out_of_moves_at_depth: Counter[int] = field(default_factory=Counter)

    # Per-anchor tracking: (anchor_coord) -> maximum depth reached from it
    anchor_max_depth: dict[Coord, int] = field(default_factory=dict)
    # Per-anchor: number of *completed* folds rooted at this anchor
    anchor_successes: Counter[Coord] = field(default_factory=Counter)

    # Canonicalised successful solutions (distinct modulo D4 + translation)
    distinct_solutions: set[tuple[Coord, ...]] = field(default_factory=set)
    # Total successful solutions found (pre-canonicalisation)
    total_solutions: int = 0

    # Max contacts seen across all successful folds
    best_contacts: int = -1

    def record_fail(self, depth: int) -> None:
        self.fail_depth_hist[depth] += 1

    def record_out_of_moves(self, depth: int) -> None:
        self.out_of_moves_at_depth[depth] += 1

    def record_valid_moves(self, depth: int, count: int) -> None:
        while len(self.valid_moves_per_depth) <= depth:
            self.valid_moves_per_depth.append([])
        self.valid_moves_per_depth[depth].append(count)

    def record_anchor_depth(self, anchor: Coord, depth: int) -> None:
        prev = self.anchor_max_depth.get(anchor, -1)
        if depth > prev:
            self.anchor_max_depth[anchor] = depth

    def record_success(self, anchor: Coord, coords: list[Coord], contacts: int) -> None:
        self.anchor_successes[anchor] += 1
        self.total_solutions += 1
        canonical = canonicalise(coords)
        self.distinct_solutions.add(canonical)
        if contacts > self.best_contacts:
            self.best_contacts = contacts


# ---- canonicalisation (D4 + translation) ------------------------------------

_D4_TRANSFORMS: tuple[callable, ...] = (
    lambda p: (p[0], p[1]),                    # identity
    lambda p: (p[1], -p[0]),                   # 90 deg CCW
    lambda p: (-p[0], -p[1]),                  # 180 deg
    lambda p: (-p[1], p[0]),                   # 270 deg CCW
    lambda p: (p[0], -p[1]),                   # reflect y=0
    lambda p: (-p[0], p[1]),                   # reflect x=0
    lambda p: (p[1], p[0]),                    # reflect y=x
    lambda p: (-p[1], -p[0]),                  # reflect y=-x
)


def _translate_to_origin(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    pts = list(coords)
    min_x = min(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    return tuple((p[0] - min_x, p[1] - min_y) for p in pts)


def canonicalise(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    """Return the lex-smallest representation of a fold under the
    8 symmetries of the 2D lattice (D4) plus translation to origin.

    Two folds that represent the same physical configuration up to
    rotation, reflection, and translation map to the same tuple.
    """
    pts = list(coords)
    best: tuple[Coord, ...] | None = None
    for tr in _D4_TRANSFORMS:
        transformed = [tr(p) for p in pts]
        candidate = _translate_to_origin(transformed)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


# ---- pretty report ----------------------------------------------------------

def format_report(tel: SearchTelemetry, *, max_depth: int | None = None) -> str:
    """Render a human-readable summary of a telemetry run."""
    lines: list[str] = []
    lines.append("=== Shape-DFS Telemetry Report ===")
    lines.append(f"total_solutions       : {tel.total_solutions}")
    lines.append(f"distinct (D4+trans)   : {len(tel.distinct_solutions)}")
    lines.append(f"best_contacts         : {tel.best_contacts}")
    lines.append("")

    if tel.anchor_max_depth:
        anchors = sorted(tel.anchor_max_depth.items(), key=lambda kv: -kv[1])
        lines.append(f"-- Anchor survival ({len(tel.anchor_max_depth)} anchors) --")
        lines.append(f"{'anchor':>12s}  {'max depth':>9s}  {'successes':>9s}")
        for anchor, d in anchors:
            s = tel.anchor_successes.get(anchor, 0)
            lines.append(f"{str(anchor):>12s}  {d:>9d}  {s:>9d}")
        lines.append("")

    if tel.fail_depth_hist:
        lines.append("-- Failure-depth distribution --")
        total_fails = sum(tel.fail_depth_hist.values())
        max_d = max(tel.fail_depth_hist)
        show = max_depth if max_depth is not None else max_d
        bucket = max(1, (show + 1) // 20)
        for d_start in range(0, show + 1, bucket):
            d_end = d_start + bucket - 1
            count = sum(
                tel.fail_depth_hist[d] for d in range(d_start, min(d_end + 1, show + 1))
            )
            if count == 0:
                continue
            pct = 100 * count / max(1, total_fails)
            bar = "█" * int(pct / 2)
            lines.append(f"  depth [{d_start:>3d}..{d_end:>3d}]: {count:>9d} "
                         f"({pct:>5.1f}%) {bar}")
        lines.append(f"  total failures: {total_fails}")
        lines.append(f"  mean fail depth: "
                     f"{sum(d*n for d,n in tel.fail_depth_hist.items())/total_fails:.1f}")
        lines.append(f"  max  fail depth: {max_d}")
        lines.append("")

    if tel.valid_moves_per_depth:
        lines.append("-- Branching factor by depth (mean valid moves / branch) --")
        for d, samples in enumerate(tel.valid_moves_per_depth):
            if not samples:
                continue
            mean = sum(samples) / len(samples)
            if d % max(1, len(tel.valid_moves_per_depth) // 25) == 0:
                lines.append(f"  depth {d:>3d}: mean={mean:.2f} n={len(samples):,}")
        lines.append("")

    if tel.out_of_moves_at_depth:
        total = sum(tel.out_of_moves_at_depth.values())
        lines.append(f"-- Out-of-moves events: {total} total --")
        items = sorted(tel.out_of_moves_at_depth.items(), key=lambda kv: -kv[1])
        for d, c in items[:10]:
            pct = 100 * c / total
            lines.append(f"  depth {d:>3d}: {c:>9d}  ({pct:>5.1f}%)")
        lines.append("")

    return "\n".join(lines)
