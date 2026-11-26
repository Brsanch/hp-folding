"""Generate publication-grade vector figures for the 2D HP lattice optimality paper.

Outputs:
  paper/figures/fig1_optimal_folds_gallery.pdf
  paper/figures/fig2_optimality_bounds_and_s6.pdf
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hp_folding.verifier import verify

OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.0,
    "axes.titlesize": 9.6,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def _plot_fold(ax, seq: str, coords: list[tuple[int, int]], title: str) -> None:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    pos = {c: i for i, c in enumerate(coords)}

    # Non-sequential H–H contacts
    for i, c in enumerate(coords):
        if seq[i] != "H":
            continue
        x, y = c
        for dx, dy in ((1, 0), (0, 1)):
            nb = (x + dx, y + dy)
            j = pos.get(nb)
            if j is not None and seq[j] == "H" and abs(i - j) > 1:
                ax.plot(
                    [x, nb[0]], [y, nb[1]],
                    color="#D95F02", linestyle="--", linewidth=1.15, alpha=0.84, zorder=1,
                )

    # Backbone chain
    ax.plot(xs, ys, color="#333333", linewidth=1.35, zorder=2)

    # Residues
    hx = [c[0] for i, c in enumerate(coords) if seq[i] == "H"]
    hy = [c[1] for i, c in enumerate(coords) if seq[i] == "H"]
    px = [c[0] for i, c in enumerate(coords) if seq[i] == "P"]
    py = [c[1] for i, c in enumerate(coords) if seq[i] == "P"]
    ax.scatter(hx, hy, s=26, color="#1F5A92", edgecolor="#0D2C4A", linewidth=0.6, zorder=3)
    ax.scatter(px, py, s=20, color="#FFFFFF", edgecolor="#555555", linewidth=0.75, zorder=3)
    ax.scatter([xs[0]], [ys[0]], s=44, facecolors="none", edgecolors="#1B7837", linewidth=1.35, zorder=4)

    ax.set_aspect("equal")
    pad = 0.9
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.7)
    ax.set_title(title, fontsize=9.0, pad=6)


def make_fig1() -> None:
    specs = [
        ("S5", "S5 ($n = 48$, $C^* = 23$)"),
        ("S6", "S6, PP-prefix ($n = 50$, $C^* = 19$)"),
        ("S7", "S7 ($n = 60$, $C^* = 36$)"),
        ("S8", "S8 ($n = 64$, $C^* = 42$)"),
        ("S9", "S9 ($n = 85$, $C^* = 53$)"),
        ("S10", "S10 ($n = 100$, $C = 50$, best known)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), dpi=300)
    for ax, (sid, title) in zip(axes.flat, specs):
        d = json.loads((REPO / "submissions" / f"{sid}.json").read_text())
        coords = [tuple(c) for c in d["coords"]]
        v = verify(d["sequence"], coords)
        assert v.valid, f"Invalid fold for {sid}: {v.error}"
        _plot_fold(ax, d["sequence"], coords, title)

    legend_handles = [
        mlines.Line2D([], [], color="none", marker="o", markersize=5.8,
                      markerfacecolor="#1F5A92", markeredgecolor="#0D2C4A", markeredgewidth=0.7,
                      label="H residue"),
        mlines.Line2D([], [], color="none", marker="o", markersize=5.2,
                      markerfacecolor="#FFFFFF", markeredgecolor="#555555", markeredgewidth=0.8,
                      label="P residue"),
        mlines.Line2D([], [], color="none", marker="o", markersize=7.2,
                      markerfacecolor="none", markeredgecolor="#1B7837", markeredgewidth=1.35,
                      label="Residue 0"),
        mlines.Line2D([], [], color="#333333", linewidth=1.4, label="Chain backbone"),
        mlines.Line2D([], [], color="#D95F02", linestyle="--", linewidth=1.25,
                      label="Non-sequential H\u2013H contact"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.005),
    )
    plt.tight_layout(rect=(0.0, 0.055, 1.0, 1.0), w_pad=1.0, h_pad=1.25)
    fig.savefig(OUT / "fig1_optimal_folds_gallery.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig2() -> None:
    cpsat_ledger = json.loads((REPO / "results" / "s6_unconditional_z2_proof.json").read_text())
    lrat_ledger = json.loads((REPO / "results" / "s6_lrat_verification.json").read_text())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 3.85), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.18]}
    )

    # Panel (a): Achieved vs proven/literature upper bounds and Harary–Harborth ceilings
    seqs = [
        "S5\n(n=48)",
        "S6\n(PP-50)",
        "2D50\n(HH-50)",
        "S7\n(n=60)",
        "S8\n(n=64)",
        "S9\n(n=85)",
        "S10\n(n=100)",
    ]
    achieved = [23, 19, 21, 36, 42, 53, 50]
    hh_ub = [23, 27, 30, 37, 43, 53, 58]
    # Tightest known UB: S5, S6, S7, S8, S9 certified in this work; 2D50 proven C*=21 by
    # Giaquinta & Pozzi (2013); S10 parity bound = 56.
    tight_ub = [23, 19, 21, 36, 42, 53, 56]

    x = np.arange(len(seqs))
    w = 0.34
    ax1.bar(
        x - w / 2, achieved, width=w,
        color="#1F5A92", edgecolor="#0D2C4A", linewidth=0.7,
        label="Verified public fold ($C$)",
    )
    ub_bars = ax1.bar(
        x + w / 2, tight_ub, width=w,
        color="#D95F02", alpha=0.80, edgecolor="#7F3600", linewidth=0.7,
        label="Certified / parity upper bound",
    )
    # Distinguish 2D50 literature optimum (Giaquinta & Pozzi 2013) with hatch
    ub_bars[2].set_facecolor("#F1A340")
    ub_bars[2].set_hatch("///")
    ub_bars[2].set_edgecolor("#7F3600")

    ax1.plot(x + w / 2, hh_ub, "k^", markersize=4.8, zorder=5, label="Harary\u2013Harborth ceiling")

    # Place OPT badges with zero collision against triangles
    for i in (0, 3, 4, 5):
        ax1.text(
            x[i], max(achieved[i], tight_ub[i], hh_ub[i]) + 1.6, "OPT",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1B7837",
        )
    ax1.text(
        x[1], achieved[1] + 1.1, "OPT",
        ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1B7837",
    )
    ax1.text(
        x[2], achieved[2] + 1.1, "$C^*\\!=\\!21$\n(G&P '13)",
        ha="center", va="bottom", fontsize=6.2, fontweight="bold", color="#7F3600",
        linespacing=1.05,
    )
    ax1.text(
        x[6], hh_ub[6] + 1.6, "[50, 56]",
        ha="center", va="bottom", fontsize=6.7, color="#444444",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(seqs, fontsize=7.4)
    ax1.set_ylabel("Non-sequential H\u2013H contacts")
    ax1.set_ylim(0, 66)
    ax1.set_title("(a) Verified folds vs. upper bounds", pad=7)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.2)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): S6 unconditional Z^2 branch-by-branch CP-SAT, CaDiCaL, and lrat-check runtimes
    odd_keys = ["H5", "H7", "H9", "H11", "H27", "H31", "H35", "H37", "H39", "H49"]
    branch_labels = ["$C\\!=\\!21$"] + [f"$\\mathrm{{H}}_{{{k[1:]}}}$" for k in odd_keys]

    cpsat_times = [cpsat_ledger["c21"]["elapsed_s"]] + [
        cpsat_ledger["c20_cases"][k]["elapsed_s"] for k in odd_keys
    ]
    cadical_times = [lrat_ledger["c21"]["cadical_s"]] + [
        lrat_ledger["c20_cases"][k]["cadical_s"] for k in odd_keys
    ]
    lrat_times = [lrat_ledger["c21"]["lrat_check_s"]] + [
        lrat_ledger["c20_cases"][k]["lrat_check_s"] for k in odd_keys
    ]

    xb = np.arange(len(branch_labels))
    wb = 0.27
    ax2.bar(
        xb - wb, cpsat_times, width=wb,
        color="#1F5A92", edgecolor="#0D2C4A", linewidth=0.55,
        label="OR-Tools CP-SAT (6 workers)",
    )
    ax2.bar(
        xb, cadical_times, width=wb,
        color="#D95F02", alpha=0.85, edgecolor="#7F3600", linewidth=0.55,
        label="CaDiCaL 3.1.0 (1 thread, LRAT)",
    )
    ax2.bar(
        xb + wb, lrat_times, width=wb,
        color="#1B7837", alpha=0.85, edgecolor="#0E441E", linewidth=0.55,
        label="lrat-check (proof checking)",
    )

    ax2.axvline(0.5, color="#666666", linestyle="--", linewidth=0.85, alpha=0.75)
    ax2.text(
        -0.02, 128, "$D \\leq 19$",
        ha="center", va="bottom", fontsize=6.7, color="#444444",
    )
    ax2.text(
        4.5, 412, "$C = 20$ single-slack branches ($D \\leq 22$)",
        ha="center", va="top", fontsize=7.5, color="#444444",
    )

    ax2.set_xticks(xb)
    ax2.set_xticklabels(branch_labels, fontsize=7.6)
    ax2.set_ylabel("Wall-clock runtime (s)")
    ax2.set_ylim(0, 435)
    ax2.set_title("(b) S6 unconditional $\\mathbb{Z}^2$ UNSAT branch runtimes", pad=7)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.1, bbox_to_anchor=(0.14, 0.90))
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    plt.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig2_optimality_bounds_and_s6.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    print("Wrote fig1_optimal_folds_gallery.pdf and fig2_optimality_bounds_and_s6.pdf")
