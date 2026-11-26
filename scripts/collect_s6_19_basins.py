"""Collect all distinct S6 19-fold basins from:
  1. diagnostics/stepping_stones.json (the original 4)
  2. results/S6_fresh_rebridge_*.json (newly discovered via bond_rebridge)
  3. The current submission

Inject them into stepping_stones.json at level 19 so basin_crossover can see them.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS
from hp_folding.verifier import count_contacts

SEQ = BENCHMARKS["S6"].sequence
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def contact_set(coords):
    pos = {tuple(c): i for i, c in enumerate(coords)}
    cs = []
    for i, (x, y) in enumerate(coords):
        if SEQ[i] != "H":
            continue
        for dx, dy in NB:
            j = pos.get((x + dx, y + dy))
            if j is not None and j > i + 1 and SEQ[j] == "H":
                cs.append((i, j))
    return frozenset(cs)


def main():
    # Load existing stepping stones
    ss_path = ROOT / "diagnostics/stepping_stones.json"
    ss = json.loads(ss_path.read_text())
    s6 = ss.setdefault("S6", {})
    cur_19 = s6.get("19", [])
    cur_keys = set()
    for f in cur_19:
        coords = [tuple(c) for c in f]
        if count_contacts(SEQ, coords) == 19:
            cur_keys.add(contact_set(coords))
    print(f"existing 19-fold basins in stones: {len(cur_keys)}")

    new_folds = []

    # Source 2a: dumped basins from fresh SA + bond_rebridge (per-basin)
    basins_dir = ROOT / "results/s6_fresh_19_basins"
    if basins_dir.exists():
        for src in sorted(basins_dir.glob("basin_*_c19.json")):
            d = json.loads(src.read_text())
            coords = [tuple(c) for c in d["coords"]]
            if count_contacts(SEQ, coords) == 19:
                k = contact_set(coords)
                if k not in cur_keys:
                    cur_keys.add(k)
                    cur_19.append([list(c) for c in coords])
                    new_folds.append(("fresh_basin", coords))

    # Source 2b: results from any earlier global-best dumps
    for src in (ROOT / "results").glob("S6_fresh_rebridge_*.json"):
        d = json.loads(src.read_text())
        coords = [tuple(c) for c in d["coords"]]
        if count_contacts(SEQ, coords) == 19:
            k = contact_set(coords)
            if k not in cur_keys:
                cur_keys.add(k)
                cur_19.append([list(c) for c in coords])
                new_folds.append(("results", coords))

    print(f"new 19-fold basins added: {len(new_folds)}")
    print(f"total 19-fold basins now: {len(cur_keys)}")

    # Persist
    s6["19"] = cur_19
    ss_path.write_text(json.dumps(ss))
    print(f"updated {ss_path}")


if __name__ == "__main__":
    main()
