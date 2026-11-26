"""Cross-basin targeted CSP.

Take the published Wüst-Landau 42-fold for S8 (their basin) and try to add
one of OUR 40-fold's 24 unique contact pairs (our basin) to construct a
43-contact SAW. If any pair works, that's a record-BREAK.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.verifier import count_contacts, verify
from scripts.targeted_csp import solve_target_pairs, _h_contact_pairs  # type: ignore


def main():
    seq = BENCHMARKS["S8"].sequence
    ours = [tuple(c) for c in json.loads(open("results/S8_megasweep40.json").read())["coords"]]
    ref = json.loads(open("diagnostics/wust_S8_reference.json").read())
    theirs = [tuple(c) for c in ref["42"][0]]

    ours_pairs = _h_contact_pairs(seq, ours)
    theirs_pairs = _h_contact_pairs(seq, theirs)
    ours_unique = ours_pairs - theirs_pairs

    print(f"base: their 42 pairs")
    print(f"candidates: our {len(ours_unique)} unique pairs")
    print(f"target: 43 (record-BREAK)")

    best_total = 42
    best_coords = theirs
    t0 = time.time()
    for k, candidate in enumerate(sorted(ours_unique)):
        target = theirs_pairs | {candidate}
        result, nodes = solve_target_pairs(seq, target, time_budget=60.0)
        if result is not None:
            total = count_contacts(seq, result)
            v = verify(seq, result)
            tag = " [RECORD-BREAK]" if total > 42 else ""
            print(f"  [{k+1}/{len(ours_unique)}] candidate {candidate}: "
                  f"valid={bool(v)} total={total} nodes={nodes}{tag}",
                  flush=True)
            if v and total > best_total:
                best_total = total
                best_coords = result
                if best_total >= 43:
                    print("STOPPING - record-break found")
                    break
        else:
            print(f"  [{k+1}/{len(ours_unique)}] candidate {candidate}: "
                  f"no SAW realization (nodes={nodes})", flush=True)

    wall = time.time() - t0
    print(f"\nWall: {wall:.1f}s")
    print(f"Best: {best_total}")
    if best_total > 42:
        out = Path("results/S8_cross_basin_best.json")
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=seq, coords=best_coords), "S8", out)
        import shutil
        shutil.copy(out, "submissions/S8.json")
        print(f"PROMOTED submissions/S8.json: 42 -> {best_total}")


if __name__ == "__main__":
    main()
