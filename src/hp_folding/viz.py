"""Visualization utilities for HP folds.

This module is intentionally kept optional — it depends on matplotlib and
numpy, which are listed in the ``viz`` extra of ``pyproject.toml``. The
core library does not import this module, so ``hp_folding`` still works
on installs without matplotlib.

Typical use from a script::

    from hp_folding.viz import plot_fold, diagnostic_panel

    plot_fold(fold, path="results/s5_best.pdf")
    diagnostic_panel(fold, best_known=23, path="results/s5_diag.pdf")
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt

from hp_folding.fold import Fold

# ---- colour constants --------------------------------------------------------

H_COLOR = "#d43d3d"        # hydrophobic
P_COLOR = "#3d6dd4"        # polar
CONTACT_COLOR = "#d43d3d"  # H-H contact line
BACKBONE_COLOR = "#777777"
NEAR_MISS_COLOR = "#ff9900"


# ---- fold analysis (pure python, no I/O) ------------------------------------

def h_indices(sequence: str) -> list[int]:
    return [i for i, c in enumerate(sequence.upper()) if c == "H"]


def non_sequential_pair_distances(
    sequence: str, coords: list[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    """Return (i, j, manhattan_distance) for every H-H pair with |i-j| > 1."""
    hs = h_indices(sequence)
    out: list[tuple[int, int, int]] = []
    for a_idx, i in enumerate(hs):
        xi, yi = coords[i]
        for j in hs[a_idx + 1 :]:
            if j - i <= 1:
                continue
            xj, yj = coords[j]
            out.append((i, j, abs(xi - xj) + abs(yi - yj)))
    return out


def h_radius_of_gyration_squared(
    sequence: str, coords: list[tuple[int, int]]
) -> float:
    hs = h_indices(sequence)
    if not hs:
        return 0.0
    pts = [coords[i] for i in hs]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts) / len(pts)


def optimal_h_rg_squared(h_count: int) -> float:
    """Minimum radius-of-gyration squared for placing ``h_count`` distinct
    lattice points as compactly as possible, approximated by a square spiral."""
    pts: list[tuple[int, int]] = [(0, 0)]
    occupied = {(0, 0)}
    # BFS-grow: always pick the free lattice neighbour closest to the centroid.
    frontier = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while len(pts) < h_count:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        frontier.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        nxt = frontier.pop(0)
        while nxt in occupied:
            if not frontier:
                break
            nxt = frontier.pop(0)
        pts.append(nxt)
        occupied.add(nxt)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neigh = (nxt[0] + dx, nxt[1] + dy)
            if neigh not in occupied and neigh not in frontier:
                frontier.append(neigh)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts) / len(pts)


# ---- primitive plots ---------------------------------------------------------

def draw_fold(ax, fold: Fold, *, title: str | None = None) -> None:
    """Draw a fold on a given matplotlib axis."""
    coords = fold.coords
    seq = fold.sequence.upper()
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Backbone
    ax.plot(xs, ys, "-", color=BACKBONE_COLOR, linewidth=1.3, zorder=1)

    # Contacts
    idx = {c: i for i, c in enumerate(coords)}
    contact_lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, ci in enumerate(coords):
        if seq[i] != "H":
            continue
        for dx, dy in ((1, 0), (0, 1)):  # only count each pair once
            j = idx.get((ci[0] + dx, ci[1] + dy))
            if j is None or abs(i - j) <= 1:
                continue
            if seq[j] == "H":
                contact_lines.append((ci, coords[j]))
    for a, b in contact_lines:
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=CONTACT_COLOR, linewidth=2.4, alpha=0.7, zorder=2)

    # Residues
    for i, (x, y) in enumerate(coords):
        c = H_COLOR if seq[i] == "H" else P_COLOR
        ax.scatter([x], [y], c=c, s=140, edgecolors="black",
                   linewidth=0.6, zorder=3)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_fold(fold: Fold, *, path: str | Path, title: str | None = None) -> None:
    """Save a standalone picture of a fold."""
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_fold(ax, fold, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- near-miss analysis ------------------------------------------------------

def draw_near_miss_histogram(
    ax, fold: Fold, *, max_distance: int = 10
) -> None:
    dists = [d for (_, _, d) in non_sequential_pair_distances(fold.sequence, fold.coords)]
    bins = list(range(1, max_distance + 2))
    ax.hist(dists, bins=bins, align="left",
            color="#445577", edgecolor="black", rwidth=0.85)
    ax.set_xticks(list(range(1, max_distance + 1)))
    ax.set_xlabel("Manhattan distance between non-sequential H-H pair")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Near-miss contact histogram")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    at_1 = sum(1 for d in dists if d == 1)
    at_2 = sum(1 for d in dists if d == 2)
    total = len(dists)
    ax.text(
        0.98, 0.95,
        f"H pairs:        {total}\n"
        f"In contact (=1): {at_1}\n"
        f"Distance 2:      {at_2}\n"
        f"Distance >= 3:   {total - at_1 - at_2}",
        transform=ax.transAxes,
        ha="right", va="top",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#888", alpha=0.85),
    )


# ---- compactness gauge -------------------------------------------------------

def draw_compactness_bar(ax, fold: Fold) -> None:
    seq = fold.sequence.upper()
    hs = h_indices(seq)
    rg2 = h_radius_of_gyration_squared(seq, fold.coords)
    rg2_min = optimal_h_rg_squared(len(hs))
    ratio = (rg2 / rg2_min) if rg2_min > 0 else 1.0
    ax.barh(
        ["Rg² (H subset)"],
        [rg2],
        color="#445577", edgecolor="black", zorder=3,
    )
    ax.axvline(rg2_min, color="#d43d3d", linestyle="--", linewidth=2,
               label=f"lattice-optimum Rg² = {rg2_min:.2f}")
    ax.set_xlim(0, max(rg2, rg2_min) * 1.25)
    ax.set_title(f"H-core compactness (ratio to optimum = {ratio:.2f}×)")
    ax.set_xlabel("Squared radius of gyration of H-subset")
    ax.legend(loc="lower right")


# ---- contact count history ---------------------------------------------------

def draw_history(ax, history: Iterable[int], *, best_known: int | None = None) -> None:
    h = list(history)
    if not h:
        ax.text(0.5, 0.5, "no history recorded", transform=ax.transAxes,
                ha="center", va="center", color="#888")
        ax.set_axis_off()
        return
    ax.plot(h, color="#445577", linewidth=1.3)
    if best_known is not None:
        ax.axhline(best_known, color="#d43d3d", linestyle="--", linewidth=1.5,
                   label=f"best known = {best_known}")
        ax.legend(loc="lower right")
    ax.set_xlabel("log-step index")
    ax.set_ylabel("best contacts so far")
    ax.set_title("SA trajectory")
    ax.grid(True, linestyle=":", alpha=0.4)


# ---- top-level diagnostic panel ---------------------------------------------

def diagnostic_panel(
    fold: Fold,
    *,
    best_known: int | None = None,
    history: Iterable[int] | None = None,
    path: str | Path,
    subtitle: str | None = None,
) -> None:
    """4-panel diagnostic figure for a single fold."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    draw_fold(axes[0, 0], fold,
              title=f"Fold (contacts = {fold.contacts()}"
              + (f" / best known = {best_known}" if best_known else "")
              + ")")
    draw_near_miss_histogram(axes[0, 1], fold)
    draw_compactness_bar(axes[1, 0], fold)
    draw_history(axes[1, 1], history or [], best_known=best_known)
    suptitle = "HP-folding diagnostic panel"
    if subtitle:
        suptitle += f" — {subtitle}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fold_grid(
    folds: list[tuple[Fold, str]],
    *,
    path: str | Path,
    cols: int = 3,
) -> None:
    """Grid of folds with labels. Useful for basin-coverage visualization."""
    n = len(folds)
    if n == 0:
        return
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if rows * cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[a] for a in axes]
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        if k < n:
            fold, label = folds[k]
            draw_fold(ax, fold, title=label)
        else:
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
