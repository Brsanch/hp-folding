"""Extract empirical motifs from all diagnostic fold data.

Walks every diagnostic JSON, every results/*.json, every submissions/*.json.
Pulls fold coordinates. For each fold, slices into k-cell sub-segments via a
sliding window. Canonicalizes each via D4 + translation. Counts frequencies.

The most-frequent sub-motifs are the empirical building blocks our successful
folds use. They become the motif library — no guessing.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canon_translate(cells):
    """Translate so min coord is at (0, 0)."""
    minx = min(x for x, _ in cells)
    miny = min(y for _, y in cells)
    return tuple((x - minx, y - miny) for x, y in cells)


def all_d4_transforms(cells):
    """Yield 8 D4 transforms (4 rotations × 2 reflections) of cells."""
    out = set()
    pts = list(cells)
    for refl in (False, True):
        cur = [(x, -y) if refl else (x, y) for x, y in pts]
        for _ in range(4):
            canon = canon_translate(cur)
            out.add(canon)
            cur = [(-y, x) for x, y in cur]
    return out


def canonical_form(cells):
    """Pick the D4-canonical form: lex smallest among all 8 transforms.
    Preserves chain ORDER (not a set canonical) — we use tuple of cells
    in chain order, transformed."""
    best = None
    pts = list(cells)
    for refl in (False, True):
        cur = [(x, -y) if refl else (x, y) for x, y in pts]
        for _ in range(4):
            canon = canon_translate(cur)
            t = tuple(canon)
            if best is None or t < best:
                best = t
            cur = [(-y, x) for x, y in cur]
    return best


def fold_contacts(sequence, coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    c = 0
    for i, (x, y) in enumerate(coords):
        if sequence[i] != "H":
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1:
                continue
            if sequence[j] == "H":
                c += 1
    return c // 2


def gather_all_folds():
    """Yield (sequence_id, coords, contacts, source) for every fold we've saved."""
    BENCH_SEQS = {
        "S1": "HPHPPHHPHPPHPHHPPHPH",
        "S2": "HHPPHPPHPPHPPHPPHPPHPPHH",
        "S3": "PPHPPHHPPPPHHPPPPHHPPPPHH",
        "S4": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
        "S5": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
        "S6": "PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
        "S7": "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP",
        "S8": "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH",
        "S9": "HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH",
        "S10": "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH",
    }
    # diagnostics/*/*.json — per-trial fold coords
    for jpath in (ROOT / "diagnostics").rglob("*.json"):
        try:
            d = json.loads(jpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sid = d.get("sequence_id")
        coords = d.get("best_coords")
        if not sid or not coords:
            continue
        coords_t = [tuple(c) for c in coords]
        # Verify length matches
        if sid in BENCH_SEQS and len(coords_t) == len(BENCH_SEQS[sid]):
            c = fold_contacts(BENCH_SEQS[sid], coords_t)
            yield sid, coords_t, c, str(jpath)
    # results/*.json and submissions/*.json
    for d in ("results", "submissions"):
        for jp in (ROOT / d).glob("*.json"):
            try:
                payload = json.loads(jp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            sid = payload.get("sequence_id") or payload.get("id") or jp.stem.split("_")[0]
            if sid not in BENCH_SEQS:
                continue
            coords = payload.get("coords")
            if not coords:
                continue
            coords_t = [tuple(c) for c in coords]
            if len(coords_t) != len(BENCH_SEQS[sid]):
                continue
            c = fold_contacts(BENCH_SEQS[sid], coords_t)
            yield sid, coords_t, c, str(jp)


def extract_motifs(folds, k_range=(3, 4, 5, 6, 7, 8, 9, 10, 12)):
    """For each fold, slide a length-k window across the chain and record
    the canonical form of each window. Returns frequency Counter per k."""
    by_k: dict[int, Counter] = defaultdict(Counter)
    for sid, coords, contacts, src in folds:
        n = len(coords)
        for k in k_range:
            if k > n:
                continue
            for i in range(n - k + 1):
                window = coords[i: i + k]
                c = canonical_form(window)
                by_k[k][c] += 1
    return by_k


def main():
    folds = list(gather_all_folds())
    print(f"gathered {len(folds)} folds total")

    # Filter to top-quality folds (high contact count per sequence)
    by_seq = defaultdict(list)
    for sid, coords, c, src in folds:
        by_seq[sid].append((c, coords, src))
    print()
    print("Folds per sequence (count, max contacts, n unique coords):")
    for sid in sorted(by_seq.keys()):
        bs = by_seq[sid]
        max_c = max(b[0] for b in bs)
        unique_chains = len({tuple(map(tuple, b[1])) for b in bs})
        print(f"  {sid}: {len(bs)} folds, max={max_c}, "
              f"unique chains={unique_chains}")

    # Take top-quartile folds per sequence for motif extraction
    quality_folds = []
    for sid, bs in by_seq.items():
        bs.sort(reverse=True)
        cutoff_idx = max(1, len(bs) // 4)  # top 25%
        quality_folds.extend((sid, b[1], b[0], b[2]) for b in bs[:cutoff_idx])
    print(f"\nUsing top-25% folds for motif extraction: {len(quality_folds)} folds")

    motifs = extract_motifs(quality_folds, k_range=(3, 4, 5, 6, 7, 8, 10, 12))
    print()
    print("Top-10 most frequent motifs per length:")
    for k in sorted(motifs.keys()):
        ctr = motifs[k]
        total = sum(ctr.values())
        print(f"\nlength k={k} ({total} total occurrences, {len(ctr)} unique motifs):")
        for motif_cells, count in ctr.most_common(10):
            frac = count / total * 100
            print(f"  count={count:>5} ({frac:>5.1f}%): {motif_cells}")

    # Write the empirical library to a file
    out = {}
    for k, ctr in motifs.items():
        # Keep all motifs above a frequency threshold
        threshold = max(1, max(ctr.values()) // 100)  # 1% of max
        kept = [(c, list(cells)) for cells, c in ctr.most_common() if c >= threshold]
        out[str(k)] = kept[:100]  # top 100 per length
    out_path = ROOT / "diagnostics" / "empirical_motif_library.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nWrote {out_path}")
    print(f"  total motifs: {sum(len(v) for v in out.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
