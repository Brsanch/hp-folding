"""Aggregate every diagnostic produced this session into one synthesis.

Reads:
- All diagnostics/*/*.json files (per-trial WL telemetry)
- All top-level *.log files (run summaries)

Produces:
- Per-sequence best-contact distribution across ALL methods
- Per-method comparison table
- Total trials and compute estimate
- Any rare-event tails worth investigating
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    # 1. JSON diagnostics
    method_seq_best: dict[tuple[str, str], list[int]] = defaultdict(list)
    method_seq_sweeps: dict[tuple[str, str], list[int]] = defaultdict(list)
    method_seq_logf: dict[tuple[str, str], list[float]] = defaultdict(list)
    method_seq_movestats: dict[tuple[str, str], dict] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    for diag_dir in sorted((ROOT / "diagnostics").iterdir()):
        method = diag_dir.name
        for jpath in diag_dir.glob("*.json"):
            try:
                d = json.loads(jpath.read_text())
            except (json.JSONDecodeError, IOError):
                continue
            seq = d.get("sequence_id")
            if not seq:
                continue
            method_seq_best[(method, seq)].append(d.get("best_contacts", -1))
            method_seq_sweeps[(method, seq)].append(d.get("sweeps", 0))
            method_seq_logf[(method, seq)].append(d.get("log_f_final", 1.0))
            for mc, (p, a) in d.get("move_stats", {}).items():
                method_seq_movestats[(method, seq)][mc][0] += p
                method_seq_movestats[(method, seq)][mc][1] += a

    # 2. Log files - extract per-trial bests
    log_method_seq_best: dict[tuple[str, str], list[int]] = defaultdict(list)
    for logp in ROOT.glob("*.log"):
        name = logp.name
        text = logp.read_text(errors="ignore")
        # Identify method + sequence from filename
        method = name.replace(".log", "")
        # Parse all "contacts=N" or "best=N" or "Best: N"
        for m in re.finditer(r"contacts=(\d+)", text):
            seq_match = re.match(
                r"(?:wl_|bh_|hose_|tabu_|lateral_|ga_|mass_gpu_|"
                r"hybrid_|long_hcore_|map_|targeted_|csp_)(S\d+)", name
            )
            if seq_match:
                seq = seq_match.group(1)
                log_method_seq_best[(method, seq)].append(int(m.group(1)))

    print("=" * 80)
    print("PER-SEQUENCE BEST-CONTACT DISTRIBUTION ACROSS ALL METHODS")
    print("=" * 80)
    seqs = ("S5", "S6", "S7", "S8", "S9", "S10")
    records = {"S5": 23, "S6": 21, "S7": 36, "S8": 42, "S9": 53, "S10": 50}
    submissions = {"S5": 23, "S6": 19, "S7": 35, "S8": 39, "S9": 50, "S10": 46}

    for seq in seqs:
        print(f"\n--- {seq} (record {records[seq]}, current submission {submissions[seq]}) ---")
        # Combine all trials for this sequence
        all_bests: list[int] = []
        per_method: dict[str, list[int]] = {}
        for (m, s), bests in method_seq_best.items():
            if s == seq:
                per_method.setdefault(f"diag:{m}", []).extend(bests)
                all_bests.extend(bests)
        for (m, s), bests in log_method_seq_best.items():
            if s == seq:
                per_method.setdefault(f"log:{m}", []).extend(bests)
                all_bests.extend(bests)

        if not all_bests:
            print("  no data")
            continue

        # Distribution histogram
        c = Counter(all_bests)
        max_c = max(c)
        median_c = sorted(all_bests)[len(all_bests) // 2]
        n = len(all_bests)
        print(f"  total observations: {n}, max={max_c}, median={median_c}")
        # Tail
        print(f"  tail (top contacts):")
        for ck in sorted(c.keys(), reverse=True)[:6]:
            cnt = c[ck]
            frac = cnt / n
            bar = "#" * min(50, int(frac * 200))
            print(f"    {ck:>3d}: {cnt:>5d} ({frac*100:>5.1f}%) {bar}")

        # Per-method top
        print(f"  per-method bests (top-5):")
        for m, bs in sorted(per_method.items(), key=lambda x: -max(x[1])):
            mx, n_above = max(bs), sum(1 for b in bs if b == max(bs))
            mc = max(c)
            n_at_global = sum(1 for b in bs if b == mc)
            print(f"    {m:<35s}: max={mx:>3d} (n={len(bs):>5d}) "
                  f"reached_global_max={n_at_global}")

    # 3. Move-class effectiveness across all WL diagnostics
    print()
    print("=" * 80)
    print("MOVE-CLASS EFFECTIVENESS (aggregated across all WL diagnostics)")
    print("=" * 80)
    for (m, s), stats in sorted(method_seq_movestats.items()):
        if not stats:
            continue
        line = f"{m} {s}: "
        parts = []
        for cls in ("corner_flip", "crankshaft", "segment_rot", "pivot",
                    "end_flip", "pull"):
            p, a = stats.get(cls, [0, 0])
            if p > 0:
                rate = a / p
                parts.append(f"{cls}={rate:.2f}({a/1e6:.1f}M acc)")
        if parts:
            print(f"  {line}{' | '.join(parts)}")


if __name__ == "__main__":
    main()
