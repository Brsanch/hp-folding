"""Visual deep-dive across all session diagnostics.

Generates PNGs into figures/ that show:
  1. Tail-decay per sequence (log-scale histograms from crossover runs).
  2. Stepping-stone library shape (basins per contact level).
  3. S6 19-fold ensemble (all 14 unique 19-folds overlaid).
  4. S8 four distinct 42-folds + Wüst reference.
  5. S9 52-fold + parents.
  6. S10 best 49-fold and stepping-stone basin diversity.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS
from hp_folding.verifier import verify

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SEQS = ("S5", "S6", "S7", "S8", "S9", "S10")
COLORS = {"S5": "#1f77b4", "S6": "#d62728", "S7": "#2ca02c",
          "S8": "#9467bd", "S9": "#ff7f0e", "S10": "#17becf"}


def parse_distribution(log_path: Path):
    """Pull 'contact distribution: {...}' line from a crossover log."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    m = re.search(r"contact distribution:\s*(\{[^}]+\})", text)
    if not m:
        return None
    try:
        d = eval(m.group(1))
        return {int(k): int(v) for k, v in d.items()}
    except Exception:
        return None


def fig1_tail_decay():
    """Aggregate distributions across all crossover runs per sequence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    agg = defaultdict(lambda: defaultdict(int))
    for log in ROOT.glob("crossover_*.log"):
        m = re.search(r"crossover_(S\d+)_lvl", log.name)
        if not m:
            continue
        sid = m.group(1)
        dist = parse_distribution(log)
        if dist:
            for k, v in dist.items():
                agg[sid][k] += v

    # Linear y
    ax = axes[0]
    for sid in sorted(agg.keys()):
        rec = BENCHMARKS[sid].best_known
        cur = verify(BENCHMARKS[sid].sequence,
                     [tuple(c) for c in
                      json.loads((ROOT / f"submissions/{sid}.json").read_text())["coords"]]
                     ).contacts
        levels = sorted(agg[sid].keys())
        counts = [agg[sid][l] for l in levels]
        ax.plot(levels, counts, "-o", color=COLORS.get(sid, "k"),
                label=f"{sid} (cur={cur}, rec={rec})", linewidth=2, markersize=5)
        ax.axvline(rec, color=COLORS.get(sid, "k"), linestyle=":", alpha=0.4)
    ax.set_xlabel("Contact count")
    ax.set_ylabel("Hits in crossover trials")
    ax.set_title("Crossover tail (linear)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Log y
    ax = axes[1]
    for sid in sorted(agg.keys()):
        rec = BENCHMARKS[sid].best_known
        levels = sorted(agg[sid].keys())
        counts = [agg[sid][l] for l in levels]
        ax.plot(levels, counts, "-o", color=COLORS.get(sid, "k"),
                label=f"{sid} (rec={rec})", linewidth=2, markersize=5)
        ax.axvline(rec, color=COLORS.get(sid, "k"), linestyle=":", alpha=0.4)
    ax.set_xlabel("Contact count")
    ax.set_ylabel("Hits (log scale)")
    ax.set_yscale("log")
    ax.set_title("Crossover tail (log) — slope = decay rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Figure 1: Crossover tail decay per sequence\n"
                 "Vertical dotted line = published record. "
                 "Cliff vs gradual slope tells us 'wall' vs 'tail'.",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "01_tail_decay.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, dict(agg)


def fig2_stepping_stones():
    """Library shape: how many distinct basins per contact level for each seq."""
    ss = json.loads((ROOT / "diagnostics/stepping_stones.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, sid in zip(axes.flat, SEQS):
        sd = ss.get(sid, {})
        levels = sorted([int(k) for k in sd.keys()])
        counts = [len(sd[str(l)]) for l in levels]
        rec = BENCHMARKS[sid].best_known
        cur = verify(BENCHMARKS[sid].sequence,
                     [tuple(c) for c in
                      json.loads((ROOT / f"submissions/{sid}.json").read_text())["coords"]]
                     ).contacts
        ax.bar(levels, counts, color=COLORS.get(sid, "k"), alpha=0.8)
        ax.axvline(rec, color="red", linestyle="--", linewidth=1.5,
                   label=f"record={rec}")
        ax.axvline(cur, color="green", linestyle=":", linewidth=1.5,
                   label=f"current={cur}")
        ax.set_title(f"{sid}: {sum(counts)} stepping-stones, "
                     f"max lvl={max(levels) if levels else 0}")
        ax.set_xlabel("Contact level")
        ax.set_ylabel("# distinct stones")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Figure 2: Stepping-stone library shape\n"
                 "Sparse top-end = WL/SA can't sample near record. "
                 "Distance from green to red = our gap.")
    fig.tight_layout()
    out = FIG / "02_stepping_stones.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def draw_fold(ax, seq, coords, title="", show_indices=False):
    """Render one HP-lattice fold."""
    coords = list(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    # backbone
    ax.plot(xs, ys, "-", color="#888", linewidth=1.2, zorder=1)
    # H/P beads
    for i, (x, y) in enumerate(coords):
        if seq[i] == "H":
            ax.scatter([x], [y], s=80, c="#222", edgecolors="black",
                       linewidths=0.6, zorder=3)
        else:
            ax.scatter([x], [y], s=60, facecolors="white",
                       edgecolors="#888", linewidths=0.6, zorder=3)
        if show_indices:
            ax.annotate(str(i), (x, y), fontsize=5, color="#444",
                        ha="center", va="center", zorder=4)
    # H-H non-sequential contacts in red
    pos = {tuple(c): i for i, c in enumerate(coords)}
    drawn = set()
    n_contacts = 0
    for i, (x, y) in enumerate(coords):
        if seq[i] != "H":
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1 or seq[j] != "H":
                continue
            key = tuple(sorted((i, j)))
            if key in drawn:
                continue
            drawn.add(key)
            n_contacts += 1
            ax.plot([coords[i][0], coords[j][0]],
                    [coords[i][1], coords[j][1]],
                    color="red", linewidth=1.0, alpha=0.6, zorder=2)
    ax.set_aspect("equal")
    ax.set_title(f"{title}  c={n_contacts}", fontsize=9)
    ax.axis("off")


def fig3_s6_basin_attractor():
    """Show all distinct S6 19-folds we've found — the 'wall'."""
    ss = json.loads((ROOT / "diagnostics/stepping_stones.json").read_text())
    pool = ss["S6"]["19"]  # 14 stones at level 19
    seq = BENCHMARKS["S6"].sequence
    # Dedup by contact set
    distinct = []
    seen = set()
    for f in pool:
        coords = [tuple(c) for c in f]
        pos = {c: i for i, c in enumerate(coords)}
        cs = []
        for i, (x, y) in enumerate(coords):
            if seq[i] != "H":
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = pos.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == "H":
                    cs.append((i, j))
        key = frozenset(cs)
        if key not in seen:
            seen.add(key)
            distinct.append(coords)
    n = len(distinct)
    cols = min(7, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5 + 0.6))
    if rows == 1:
        axes = np.array([axes])
    for i, coords in enumerate(distinct):
        r, c = divmod(i, cols)
        draw_fold(axes[r, c], seq, coords, title=f"basin #{i}")
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")
    fig.suptitle(f"Figure 3: S6 wall — all {n} distinct 19-fold basins.\n"
                 f"Black=H, white=P, red=H-H contact. "
                 f"60k+ trials never broke past 19.\n"
                 f"Look for: shared topology that traps every method.",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "03_s6_basin_attractor.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, distinct


def fig4_s8_42_folds():
    """Our 4 distinct 42-folds vs Wüst's reference."""
    seq = BENCHMARKS["S8"].sequence
    folds = []
    folds_dir = ROOT / "results/multibasin_S8_lvl38"
    if folds_dir.exists():
        for fp in sorted(folds_dir.glob("*c42.json")):
            d = json.loads(fp.read_text())
            folds.append((fp.stem, [tuple(c) for c in d["coords"]]))
    # Add Wüst reference
    wpath = ROOT / "diagnostics/wust_S8_reference.json"
    if wpath.exists():
        d = json.loads(wpath.read_text())
        if "coords" in d:
            folds.append(("Wust reference", [tuple(c) for c in d["coords"]]))
    n = len(folds)
    if n == 0:
        return None, []
    cols = n
    fig, axes = plt.subplots(1, cols, figsize=(cols * 3.5, 4))
    if cols == 1:
        axes = [axes]
    for ax, (name, coords) in zip(axes, folds):
        draw_fold(ax, seq, coords, title=name[:24])
    fig.suptitle(f"Figure 4: S8 four independent 42-folds + Wüst reference.\n"
                 f"Different topologies = our discovery is genuinely independent.",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "04_s8_42_folds.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, folds


def fig5_s9_top_folds():
    """S9 52-fold + best 51 stepping-stones for context."""
    seq = BENCHMARKS["S9"].sequence
    sub = json.loads((ROOT / "submissions/S9.json").read_text())
    cur_coords = [tuple(c) for c in sub["coords"]]
    folds = [("S9 submission (52)", cur_coords)]
    # Add up to 5 distinct 51-folds from stepping_stones
    ss = json.loads((ROOT / "diagnostics/stepping_stones.json").read_text())
    pool51 = ss["S9"].get("51", [])[:8]
    seen = set()
    added = 0
    for f in pool51:
        coords = [tuple(c) for c in f]
        pos = {c: i for i, c in enumerate(coords)}
        cs = []
        for i, (x, y) in enumerate(coords):
            if seq[i] != "H":
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = pos.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == "H":
                    cs.append((i, j))
        key = frozenset(cs)
        if key in seen:
            continue
        seen.add(key)
        folds.append((f"51-stone #{added}", coords))
        added += 1
        if added >= 5:
            break
    n = len(folds)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0 + 0.6))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    for i, (name, coords) in enumerate(folds):
        r, c = divmod(i, cols)
        draw_fold(axes[r, c], seq, coords, title=name)
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")
    fig.suptitle("Figure 5: S9 52-fold (current best) + nearby 51-stones.\n"
                 "Topology comparison: where might a 53rd contact emerge?",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "05_s9_top_folds.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, folds


def fig6_basin_diversity_heatmap():
    """For each seq, show 'distinct contact-set count vs total stones' per level.
    Wide gap = many redundant copies of same basin, low diversity.
    """
    ss = json.loads((ROOT / "diagnostics/stepping_stones.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, sid in zip(axes.flat, SEQS):
        sd = ss.get(sid, {})
        levels = sorted([int(k) for k in sd.keys()])
        seq = BENCHMARKS[sid].sequence
        totals = []
        distincts = []
        for l in levels:
            stones = sd[str(l)]
            totals.append(len(stones))
            seen = set()
            for f in stones:
                coords = [tuple(c) for c in f]
                pos = {c: i for i, c in enumerate(coords)}
                cs = []
                for i, (x, y) in enumerate(coords):
                    if seq[i] != "H":
                        continue
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = pos.get((x + dx, y + dy))
                        if j is not None and j > i + 1 and seq[j] == "H":
                            cs.append((i, j))
                seen.add(frozenset(cs))
            distincts.append(len(seen))
        ax.plot(levels, totals, "-o", color="#888", label="total stones",
                markersize=4)
        ax.plot(levels, distincts, "-s", color=COLORS.get(sid, "k"),
                label="distinct basins", markersize=4)
        rec = BENCHMARKS[sid].best_known
        ax.axvline(rec, color="red", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_title(f"{sid}")
        ax.set_xlabel("Contact level")
        ax.set_ylabel("Count (log)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Figure 6: Basin diversity per level (distinct basins vs total stones).\n"
                 "Gap = redundancy. Few-distinct at top = exhausted basin space.")
    fig.tight_layout()
    out = FIG / "06_basin_diversity.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig7_s6_overlay():
    """Overlay all 19-folds onto one canvas, centered on H-core, to find shared structure."""
    ss = json.loads((ROOT / "diagnostics/stepping_stones.json").read_text())
    pool = ss["S6"]["19"]
    seq = BENCHMARKS["S6"].sequence

    # For each fold, center on H-centroid
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    for f in pool:
        coords = np.array(f, dtype=float)
        h_mask = np.array([1 if c == "H" else 0 for c in seq], dtype=bool)
        cen = coords[h_mask].mean(axis=0)
        coords -= cen
        ax.plot(coords[:, 0], coords[:, 1], "-", alpha=0.10, color="#1f77b4",
                linewidth=0.7)
        ax.scatter(coords[h_mask, 0], coords[h_mask, 1], s=15, c="#222",
                   alpha=0.20, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(f"All {len(pool)} S6 19-folds overlaid (H-centroid centered)")
    ax.grid(alpha=0.3)

    # Heatmap of H positions
    ax = axes[1]
    pos_list = []
    for f in pool:
        coords = np.array(f, dtype=float)
        h_mask = np.array([1 if c == "H" else 0 for c in seq], dtype=bool)
        cen = coords[h_mask].mean(axis=0)
        coords -= cen
        pos_list.append(coords[h_mask])
    all_h = np.concatenate(pos_list, axis=0)
    H, xe, ye = np.histogram2d(all_h[:, 0], all_h[:, 1],
                               bins=np.arange(-6, 7))
    im = ax.imshow(H.T, origin="lower",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   cmap="hot")
    fig.colorbar(im, ax=ax, label="H-bead count")
    ax.set_aspect("equal")
    ax.set_title("H-bead density heatmap (across all 19-folds)")
    fig.suptitle("Figure 7: S6 wall — does every 19-fold share a shape?\n"
                 "Bright cluster = the basin attractor we can't escape.")
    fig.tight_layout()
    out = FIG / "07_s6_overlay.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    print("=== Visual deep-dive ===")
    out1, agg = fig1_tail_decay()
    print(f"  {out1}")
    out2 = fig2_stepping_stones()
    print(f"  {out2}")
    out3, s6_distinct = fig3_s6_basin_attractor()
    print(f"  {out3} — {len(s6_distinct)} distinct S6 19-folds")
    out4, s8_folds = fig4_s8_42_folds()
    if out4:
        print(f"  {out4} — {len(s8_folds)} S8 42-folds")
    out5, _ = fig5_s9_top_folds()
    print(f"  {out5}")
    out6 = fig6_basin_diversity_heatmap()
    print(f"  {out6}")
    out7 = fig7_s6_overlay()
    print(f"  {out7}")
    print("\nAggregated tail counts:")
    for sid, d in sorted(agg.items()):
        total = sum(d.values())
        peak_at_rec = d.get(BENCHMARKS[sid].best_known, 0)
        print(f"  {sid}: {total} trials, {peak_at_rec} hits at record")


if __name__ == "__main__":
    main()
