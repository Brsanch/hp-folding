"""Deep mining of all diagnostic data.

Untapped data per diagnostic JSON:
  - best_fold_at_energy: ~38 folds per trial at different energy levels
  - trace: ~107 time-series snapshots
  - log_g, hist: WL state per energy bin
  - move_stats: per-move-class accept rates

What we extract:
  1. ESSENTIAL CONTACTS — pairs in nearly all best folds (structural invariants)
  2. STEPPING-STONE LIBRARY — all best_fold_at_energy folds saved per
     (sequence, energy_level), giving search starting points at every level
  3. TIME-TO-BEST distribution across all trials per sequence
  4. MOVE-RATE vs CONTACT-LEVEL correlation
  5. RESIDUE BURIAL — per-residue frequency of being H-H contacted
  6. ANOMALY signals — trials whose best emerged late (= tail events)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fold_contacts(seq, coords):
    if len(coords) != len(seq):
        return set()
    pos = {tuple(c): i for i, c in enumerate(coords)}
    pairs = set()
    n = len(seq)
    for i, (x, y) in enumerate(coords):
        if seq[i] != "H": continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos.get((x + dx, y + dy))
            if j is None or abs(j - i) <= 1 or j >= n: continue
            if seq[j] == "H":
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    return pairs


def main():
    BENCHMARKS = json.loads('''{
        "S5": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
        "S6": "PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
        "S7": "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP",
        "S8": "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH",
        "S9": "HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH",
        "S10": "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH"
    }''')

    # Aggregate all diagnostic folds per sequence with their contact counts
    folds_by_seq: dict[str, list[tuple[int, list, dict]]] = defaultdict(list)
    stepping_stones: dict[tuple[str, int], list[list]] = defaultdict(list)
    trace_data: dict[str, list[dict]] = defaultdict(list)
    move_stats_by_seq: dict[str, list[dict]] = defaultdict(list)

    for jpath in (ROOT / "diagnostics").rglob("*.json"):
        if jpath.name == "empirical_motif_library.json":
            continue
        try:
            d = json.loads(jpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sid = d.get("sequence_id")
        if sid not in BENCHMARKS:
            continue
        if d.get("best_coords"):
            coords = [tuple(c) for c in d["best_coords"]]
            folds_by_seq[sid].append((d.get("best_contacts", -1), coords, d))
        # stepping stones
        for e_str, fold_coords in d.get("best_fold_at_energy", {}).items():
            e = int(e_str)
            contacts = -e  # since energy = -contacts
            stepping_stones[(sid, contacts)].append(
                [tuple(c) for c in fold_coords]
            )
        # trace + moves
        if d.get("trace"):
            trace_data[sid].append(d)
        if d.get("move_stats"):
            move_stats_by_seq[sid].append(d)

    print("=" * 80)
    print("MINED DIAGNOSTIC INSIGHTS")
    print("=" * 80)

    # 1. ESSENTIAL CONTACTS PER SEQUENCE
    print("\n--- Essential contact pairs (≥95% of top folds) ---")
    for sid in sorted(folds_by_seq.keys()):
        seq = BENCHMARKS[sid]
        # Take top 25% of folds by contact count
        bs = sorted(folds_by_seq[sid], key=lambda x: -x[0])
        cutoff = max(1, len(bs) // 4)
        top = bs[:cutoff]
        if not top:
            continue
        max_c = top[0][0]
        # Only consider folds at max contacts
        max_folds = [b for b in bs if b[0] == max_c]
        if len(max_folds) < 2:
            continue
        all_pair_counts = Counter()
        for c, coords, _ in max_folds:
            pairs = fold_contacts(seq, coords)
            for p in pairs:
                all_pair_counts[p] += 1
        n_max = len(max_folds)
        essential = [(p, cnt) for p, cnt in all_pair_counts.items() if cnt / n_max >= 0.95]
        rare = sorted(
            [(p, cnt) for p, cnt in all_pair_counts.items()
             if 0.1 <= cnt / n_max < 0.95],
            key=lambda x: -x[1],
        )[:5]
        print(f"  {sid} ({n_max} folds at {max_c} contacts):")
        print(f"    {len(essential)} contacts are ESSENTIAL (in ≥95% of top folds)")
        print(f"    variable contacts (10-94% frequency): {len(rare)} examples")
        for p, cnt in rare[:3]:
            print(f"      pair {p}: in {cnt}/{n_max} ({cnt/n_max*100:.0f}%)")

    # 2. STEPPING STONES
    print("\n--- Stepping-stone library ---")
    for sid in sorted(BENCHMARKS.keys()):
        levels = sorted([(c, len(folds))
                         for (s, c), folds in stepping_stones.items() if s == sid])
        if not levels:
            continue
        # Show coverage
        n_stones = sum(c for _, c in levels)
        n_levels = len(levels)
        max_c = max(c for c, _ in levels)
        print(f"  {sid}: {n_stones} stepping-stone folds across {n_levels} energy levels "
              f"(min={levels[0][0]}, max={max_c})")

    # 3. TIME-TO-BEST distribution
    print("\n--- Time-to-best signatures (when did peak occur?) ---")
    for sid in sorted(trace_data.keys()):
        trials = trace_data[sid]
        late_signals: list[tuple[int, float]] = []
        for d in trials:
            tr = d.get("trace", [])
            if not tr:
                continue
            best_seen = -1
            t_best = 0.0
            for row in tr:
                # row format: [step, wall_s, current, best, log_f, flatness, acc]
                if len(row) >= 4 and row[3] > best_seen:
                    best_seen = row[3]
                    t_best = row[1]
            if best_seen > 0:
                late_signals.append((best_seen, t_best))
        if not late_signals:
            continue
        late_signals.sort(key=lambda x: -x[0])
        # Identify "late tail" trials: best contact found in last 25% of run
        late_tail = [
            (b, t) for b, t in late_signals
            if t > max(t_ for _, t_ in late_signals) * 0.75
            and b >= max(b_ for b_, _ in late_signals[:5]) - 1
        ]
        print(f"  {sid}: {len(late_signals)} traced trials")
        print(f"    top contacts found late (last 25% of run, near max): {len(late_tail)}")
        if late_tail:
            print(f"    e.g., contacts={late_tail[0][0]} at t={late_tail[0][1]:.0f}s")

    # 4. MOVE-RATE vs CONTACT-LEVEL
    print("\n--- Move-rate vs final contact level (per sequence) ---")
    for sid in sorted(move_stats_by_seq.keys()):
        ds = move_stats_by_seq[sid]
        # Bucket by best_contacts
        buckets: dict[int, list[dict]] = defaultdict(list)
        for d in ds:
            buckets[d.get("best_contacts", -1)].append(d.get("move_stats", {}))
        if not buckets:
            continue
        # Show top contact bucket vs lower
        sorted_levels = sorted(buckets.keys(), reverse=True)
        print(f"  {sid}: ", end="")
        for level in sorted_levels[:3]:
            ms = buckets[level]
            avg_rates = {}
            for cls in ("corner_flip", "crankshaft", "pivot"):
                p_total = sum(m.get(cls, [0, 0])[0] for m in ms)
                a_total = sum(m.get(cls, [0, 0])[1] for m in ms)
                avg_rates[cls] = a_total / p_total if p_total else 0
            print(f"contacts={level}: cf={avg_rates['corner_flip']:.2f}/cs={avg_rates['crankshaft']:.2f}/pv={avg_rates['pivot']:.2f}",
                  end=" | ")
        print()

    # 5. PER-RESIDUE CONTACT FREQUENCY (HOTSPOTS / COLD SPOTS)
    print("\n--- Per-residue contact-degree distribution ---")
    for sid in sorted(folds_by_seq.keys()):
        seq = BENCHMARKS[sid]
        bs = sorted(folds_by_seq[sid], key=lambda x: -x[0])
        cutoff = max(1, len(bs) // 4)
        top_folds = bs[:cutoff]
        # For each H residue, count contact incidences across top folds
        n = len(seq)
        deg_sum = [0] * n
        for c, coords, _ in top_folds:
            if len(coords) != n: continue
            pos = {tuple(co): i for i, co in enumerate(coords)}
            for i, (x, y) in enumerate(coords):
                if seq[i] != "H": continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = pos.get((x + dx, y + dy))
                    if j is None or abs(j - i) <= 1 or j >= n: continue
                    if seq[j] == "H":
                        deg_sum[i] += 1
        n_folds = len(top_folds)
        # Mean degree per H residue
        h_indices = [i for i in range(n) if seq[i] == "H"]
        # Identify low-degree H's (boundary/under-utilized)
        low = sorted(h_indices, key=lambda i: deg_sum[i] / max(1, n_folds))
        print(f"  {sid} ({n_folds} top folds): "
              f"least-contacted H residues (across all top folds):")
        for h in low[:5]:
            mean_d = deg_sum[h] / max(1, n_folds)
            print(f"    H{h}: mean contacts in top folds = {mean_d:.2f}")

    # 6. SAVE STEPPING-STONE LIBRARY for reuse
    out_path = ROOT / "diagnostics" / "stepping_stones.json"
    payload = {}
    for (sid, c), folds in stepping_stones.items():
        # Dedupe
        unique = {tuple(map(tuple, f)) for f in folds}
        payload.setdefault(sid, {})[str(c)] = [list(f) for f in unique]
    out_path.write_text(json.dumps(payload))
    print(f"\nSaved stepping-stone library: {out_path}")
    n_total = sum(len(level) for sd in payload.values() for level in sd.values())
    print(f"  total unique stepping-stone folds: {n_total}")


if __name__ == "__main__":
    main()
