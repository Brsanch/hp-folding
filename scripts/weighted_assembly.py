"""Probability-weighted random fragment assembly.

Per-length, motifs have empirical frequencies (from extract_motifs.py).
We sample assemblies by:
  1. Pick a target length partition for the chain (e.g., [12, 8, 8, 12, 10, 10] for n=60)
  2. For each slot, draw a motif with probability proportional to its empirical count
  3. Draw an orientation (rot, refl) uniformly from the 8 D4 symmetries
  4. Place sequentially; abort early on SAW violation

Run K parallel samplers. Each sample is independent. Tail probability
of finding a high-contact assembly scales with K.

Per-sequence weighting: we also accumulate per-sequence motif frequencies
so each sequence gets its own preferred library.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.verifier import count_contacts, verify

Coord = tuple[int, int]
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _rot(c, k):
    x, y = c
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y)


def _transform(cells, rot, refl, tx, ty):
    out = []
    for c in cells:
        x, y = c
        if refl:
            y = -y
        for _ in range(rot):
            x, y = -y, x
        out.append((x + tx, y + ty))
    return out


def load_motif_library_with_weights(path):
    """Returns dict[length] -> list of (weight, cells_tuple)."""
    data = json.loads(open(path).read())
    library = {}
    for k_str, entries in data.items():
        k = int(k_str)
        if k < 3:
            continue
        # entries are list of [count, cells]
        weighted = []
        for entry in entries:
            count, cells = entry
            cells_t = tuple(tuple(c) for c in cells)
            weighted.append((count, cells_t))
        library[k] = weighted
    return library


def extract_per_sequence_motifs(target_seq_id, max_per_length=50):
    """Extract motifs from folds OF THIS SEQUENCE ONLY, weighted by their
    contact count (better folds contribute proportionally more)."""
    ROOT = Path(__file__).resolve().parents[1]
    BENCH_SEQS = {sid: BENCHMARKS[sid].sequence for sid in BENCHMARKS}
    if target_seq_id not in BENCH_SEQS:
        return {}
    target_seq = BENCH_SEQS[target_seq_id]
    target_n = len(target_seq)

    folds = []
    for jpath in (ROOT / "diagnostics").rglob("*.json"):
        try:
            d = json.loads(jpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("sequence_id") != target_seq_id:
            continue
        coords = d.get("best_coords")
        if not coords or len(coords) != target_n:
            continue
        coords_t = [tuple(c) for c in coords]
        c = count_contacts(target_seq, coords_t)
        folds.append((c, coords_t))
    for sub_dir in ("results", "submissions"):
        for jp in (ROOT / sub_dir).glob(f"{target_seq_id}*.json"):
            try:
                payload = json.loads(jp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            coords = payload.get("coords")
            if not coords or len(coords) != target_n:
                continue
            coords_t = [tuple(c) for c in coords]
            c = count_contacts(target_seq, coords_t)
            folds.append((c, coords_t))

    if not folds:
        return {}

    # Take top-50% by contacts
    folds.sort(reverse=True)
    folds = folds[: max(5, len(folds) // 2)]

    # Extract motifs, weighted by fold's contact count
    library: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    for c, coords in folds:
        for k in (3, 4, 5, 6, 7, 8, 10, 12):
            if k > target_n:
                continue
            for i in range(target_n - k + 1):
                window = coords[i: i + k]
                # canonicalize
                best = None
                pts = list(window)
                for refl in (False, True):
                    cur = [(x, -y) if refl else (x, y) for x, y in pts]
                    for _ in range(4):
                        minx = min(x for x, _ in cur)
                        miny = min(y for _, y in cur)
                        canon = tuple((x - minx, y - miny) for x, y in cur)
                        if best is None or canon < best:
                            best = canon
                        cur = [(-y, x) for x, y in cur]
                # Weight by fold contacts
                library[k][best] += c

    out = {}
    for k, motifs in library.items():
        # Sort by weight desc
        sorted_m = sorted(motifs.items(), key=lambda x: -x[1])[:max_per_length]
        out[k] = [(w, m) for m, w in sorted_m]
    return out


def random_assembly(sequence, library, rng, max_attempts=20):
    """Sample one assembly. Returns (cells, contacts) or (None, -1) if failed."""
    n = len(sequence)
    for _ in range(max_attempts):
        # Pick a length partition (sequence of motif lengths summing to n)
        lengths = []
        remaining = n
        while remaining > 0:
            # Available lengths in library that fit
            avail = [k for k in library.keys() if k <= remaining and k >= 3]
            if not avail:
                # Try to fit a length-3 if remaining < 3 fail
                if remaining >= 3:
                    avail = [3]
                else:
                    break
            # Prefer mid-sized motifs when remaining is large
            k = rng.choice(avail)
            # If remaining - k is in [1, 2], we'd be stuck — pick smaller
            if 0 < remaining - k < 3 and remaining >= 3 + 3:
                # need at least 3 left, so adjust k
                k = remaining - 3
                if k < 3 or k not in library:
                    k = rng.choice([x for x in avail if remaining - x == 0
                                     or remaining - x >= 3] or avail)
            lengths.append(k)
            remaining -= k
        if remaining != 0:
            continue

        # For each length slot, draw a motif weighted by frequency
        chosen = []
        for k in lengths:
            entries = library[k]
            total_w = sum(w for w, _ in entries)
            if total_w == 0:
                break
            r = rng.random() * total_w
            acc = 0
            picked = None
            for w, m in entries:
                acc += w
                if r < acc:
                    picked = m
                    break
            if picked is None:
                picked = entries[-1][1]
            chosen.append(picked)
        if len(chosen) != len(lengths):
            continue

        # Place sequentially with random orientations
        chain = []
        occupied = set()
        last_pos = None
        last_bond = None
        ok = True
        for motif_cells in chosen:
            rot = rng.randrange(4)
            refl = rng.random() < 0.5
            if last_pos is None:
                cells = _transform(motif_cells, rot, refl, 0, 0)
            else:
                target_first = (last_pos[0] + last_bond[0],
                                last_pos[1] + last_bond[1])
                m0 = _transform(motif_cells, rot, refl, 0, 0)[0]
                tx, ty = target_first[0] - m0[0], target_first[1] - m0[1]
                cells = _transform(motif_cells, rot, refl, tx, ty)
            # Validate: no overlap, all bonds == 1
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
                break
            chain.extend(cells)
            occupied.update(cells)
            last_pos = cells[-1]
            if len(cells) >= 2:
                a, b = cells[-2], cells[-1]
                last_bond = (b[0] - a[0], b[1] - a[1])
        if ok and len(chain) == n:
            return chain, count_contacts(sequence, chain)
    return None, -1


def worker(args):
    seq, library, n_samples, seed = args
    rng = random.Random(seed)
    best_c = -1
    best_chain = None
    for _ in range(n_samples):
        chain, c = random_assembly(seq, library, rng)
        if c > best_c:
            best_c = c
            best_chain = chain
    return seed, best_c, best_chain


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--total-samples", type=int, default=2_000_000)
    p.add_argument("--use-per-seq-library", action="store_true",
                   help="Build sequence-specific empirical library")
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    bench = BENCHMARKS[args.sequence]
    sub = json.loads(open(f"submissions/{args.sequence}.json").read())
    init_contacts = verify(bench.sequence, [tuple(c) for c in sub["coords"]]).contacts
    print(f"=== Weighted assembly on {args.sequence} ===")
    print(f"  current: {init_contacts} (record {bench.best_known})")

    if args.use_per_seq_library:
        library = extract_per_sequence_motifs(args.sequence, max_per_length=80)
        print(f"  per-sequence library: {sum(len(v) for v in library.values())} motifs across {len(library)} length classes")
    else:
        library = load_motif_library_with_weights(
            "diagnostics/empirical_motif_library.json"
        )
        print(f"  global empirical library: {sum(len(v) for v in library.values())} motifs")

    samples_per_proc = args.total_samples // args.processes
    work = [(bench.sequence, library, samples_per_proc, seed)
            for seed in range(args.processes)]

    t0 = time.monotonic()
    best_contacts = init_contacts
    best_chain = None
    with mp.Pool(args.processes) as pool:
        for seed, contacts, chain in pool.imap_unordered(worker, work):
            tag = " [BEST]" if contacts > best_contacts else ""
            print(f"  seed={seed} best_contacts={contacts}{tag}", flush=True)
            if contacts > best_contacts:
                best_contacts = contacts
                best_chain = chain
                if args.target is not None and best_contacts >= args.target:
                    pool.terminate()
                    break

    wall = time.monotonic() - t0
    print(f"\nTotal wall: {wall:.1f}s, {args.total_samples} samples ({args.total_samples/wall:.0f} samples/s)")
    print(f"Best: {best_contacts} (record {bench.best_known}, was {init_contacts})")

    if best_contacts > init_contacts and best_chain:
        out = Path(f"results/{args.sequence}_weighted_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=bench.sequence, coords=best_chain),
                         args.sequence, out)
        if args.update_submission:
            shutil.copy(out, f"submissions/{args.sequence}.json")
            print(f"PROMOTED submissions/{args.sequence}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
