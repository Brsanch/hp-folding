"""Unconditional global optimality proofs for 2D HP benchmarks S5, S7, S8, S9.

Verifies both:
  1. Lower bound: the submitted fold in submissions/S{5,7,8,9}.json is a valid SAW
     achieving contacts C*.
  2. Upper bound (unconditional over all of Z^2, connected or disconnected H-cores):
     - S5 (nH=25, adjHH=17): Harary-Harborth max_ie(25) = 40 => C <= 23.
     - S9 (nH=59, adjHH=49): Harary-Harborth max_ie(59) = 102 => C <= 53.
     - S8 (nH=42, adjHH=28): C >= 43 requires ie >= 71 <=> P(H)/2 <= 13.
       Since w+h <= 13 and w*h >= 42, the UNIQUE subset of Z^2 (connected or not)
       is the 6x7 rectangle. Since max P-run is 2, every P is at distance <= 1 from H,
       so bbox_pad=3 (= max_p_run + 1) is unconditional. CP-SAT proves 6x7 INFEASIBLE (~0.03s).
     - S7 (nH=43, adjHH=35): C >= 37 requires ie >= 72 <=> P(H)/2 <= 14.
       No disconnected 43-cell set has P/2 <= 14 (all have >= 15).
       Every connected 43-omino with P/2 <= 14 is HV-convex in {5x9, 6x8, 7x7}
       (147 canonical shapes). Checkerboard parity |E-O|=|23-20|=3 filters to 4.
       Since max P-run is 4, every P is at distance <= 2 from H, so bbox_pad=5
       (= max_p_run + 1) is unconditional. CP-SAT proves all 4 INFEASIBLE (~0.15s).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from hp_folding.cpsat_routability import is_routable_cpsat
from hp_folding.polyomino_enum import canonicalize, internal_edges
from hp_folding.sequences import BENCHMARKS
from hp_folding.verifier import verify

REPO = Path(__file__).resolve().parents[1]


def max_ie_harary_harborth(n: int) -> int:
    """Harary-Harborth (1976) exact upper bound on internal edges of any n-cell subset of Z^2."""
    return 2 * n - math.ceil(2 * math.sqrt(n))


def ferrers_diagrams(k: int, max_rows: int, max_cols: int) -> list[tuple[int, ...]]:
    """All integer partitions of k with at most max_rows parts and largest part <= max_cols."""
    if k == 0:
        return [()]
    res: list[tuple[int, ...]] = []

    def rec(rem: int, max_part: int, cur: list[int]) -> None:
        if rem == 0:
            res.append(tuple(cur))
            return
        if len(cur) == max_rows:
            return
        for v in range(min(rem, max_part), 0, -1):
            rec(rem - v, v, cur + [v])

    rec(k, max_cols, [])
    return res


def enumerate_min_perimeter_polyominoes(n: int, min_ie: int) -> set[tuple[tuple[int, int], ...]]:
    """Enumerate ALL canonical connected n-ominoes with internal edges >= min_ie.

    Valid when min_ie = max_ie_harary_harborth(n), so P(H)/2 = 2n - min_ie is minimal
    and every achieving polyomino must be HV-convex (perimeter == 2*(W + H)).
    """
    assert min_ie == max_ie_harary_harborth(n), (
        f"HV-convex minimum-perimeter enumeration requires min_ie == max_ie_harary_harborth({n})"
    )
    max_half_perim = 2 * n - min_ie
    # Verify no disconnected set can reach min_ie
    for a in range(1, n // 2 + 1):
        b = n - a
        p_half = math.ceil(2 * math.sqrt(a)) + math.ceil(2 * math.sqrt(b))
        if p_half <= max_half_perim:
            raise ValueError(
                f"Disconnected split ({a}, {b}) has P/2={p_half} <= {max_half_perim}; "
                "connected HV-convex enumeration is not sufficient."
            )

    found: set[tuple[tuple[int, int], ...]] = set()
    for W in range(1, max_half_perim):
        for H in range(W, max_half_perim - W + 1):
            if W * H < n:
                continue
            k = W * H - n
            f_by_k = {m: ferrers_diagrams(m, H - 1, W - 1) for m in range(k + 1)}
            for a in range(k + 1):
                for b in range(k - a + 1):
                    for c in range(k - a - b + 1):
                        d = k - a - b - c
                        for fa in f_by_k[a]:
                            for fb in f_by_k[b]:
                                for fc in f_by_k[c]:
                                    for fd in f_by_k[d]:
                                        removed: set[tuple[int, int]] = set()
                                        for r, L in enumerate(fa):
                                            for col in range(L):
                                                removed.add((col, r))
                                        for r, L in enumerate(fb):
                                            for col in range(W - L, W):
                                                removed.add((col, r))
                                        for r, L in enumerate(fc):
                                            for col in range(L):
                                                removed.add((col, H - 1 - r))
                                        for r, L in enumerate(fd):
                                            for col in range(W - L, W):
                                                removed.add((col, H - 1 - r))
                                        if len(removed) != k:
                                            continue
                                        cells = {
                                            (x, y) for x in range(W) for y in range(H)
                                        } - removed
                                        if len({x for x, _ in cells}) != W or len(
                                            {y for _, y in cells}
                                        ) != H:
                                            continue
                                        if internal_edges(cells) >= min_ie:
                                            found.add(canonicalize(cells))
    return found


def prove_all() -> dict[str, dict]:
    results: dict[str, dict] = {}
    for sid in ("S5", "S7", "S8", "S9"):
        t0 = time.time()
        b = BENCHMARKS[sid]
        seq = b.sequence
        sub = json.loads((REPO / "submissions" / f"{sid}.json").read_text())
        coords = [tuple(c) for c in sub["coords"]]
        v = verify(seq, coords)
        assert v.valid, f"{sid} submission invalid: {v.error}"

        n_h = seq.count("H")
        adj_hh = sum(1 for i in range(len(seq) - 1) if seq[i] == "H" and seq[i + 1] == "H")
        hh_ie = max_ie_harary_harborth(n_h)
        hh_ub = hh_ie - adj_hh

        if v.contacts == hh_ub:
            results[sid] = {
                "contacts": v.contacts,
                "method": "Harary-Harborth analytic",
                "candidates_checked": 0,
                "elapsed": round(time.time() - t0, 4),
            }
            continue

        assert v.contacts + 1 == hh_ub, f"{sid} gap > 1 not handled here"
        target_ie = v.contacts + 1 + adj_hh
        all_cands = enumerate_min_perimeter_polyominoes(n_h, target_ie)

        ev_seq = sum(1 for i, c in enumerate(seq) if c == "H" and i % 2 == 0)
        od_seq = n_h - ev_seq
        target_diff = abs(ev_seq - od_seq)
        parity_cands = [
            c
            for c in sorted(all_cands)
            if abs(2 * sum(1 for x, y in c if (x + y) % 2 == 0) - n_h) == target_diff
        ]

        # Exact unconditional padding: any P in a run of length L is at distance <= ceil(L/2)
        # from an H (except leading/trailing P runs, which sit at distance <= L).
        # Use max(L) + 1 for generous unconditional margin.
        max_p_run = max((len(r) for r in seq.split("H")), default=0)
        pad = max_p_run + 1

        for canon in parity_cands:
            c_out, st = is_routable_cpsat(
                seq, frozenset(canon), time_budget_seconds=30.0, bbox_pad=pad, n_workers=8
            )
            assert c_out is None and st == "INFEASIBLE", (
                f"{sid}: expected INFEASIBLE, got {st}"
            )

        results[sid] = {
            "contacts": v.contacts,
            "method": f"HV-convex min-perim ({len(all_cands)} total, {len(parity_cands)} parity-ok) + CP-SAT (pad={pad})",
            "candidates_total": len(all_cands),
            "candidates_checked": len(parity_cands),
            "elapsed": round(time.time() - t0, 4),
        }
    return results


if __name__ == "__main__":
    res = prove_all()
    print("UNCONDITIONAL GLOBAL OPTIMALITY PROOFS (Z^2):")
    for sid, info in res.items():
        print(f"  {sid} = {info['contacts']:2d}  PROVEN OPTIMAL  [{info['method']}] ({info['elapsed']}s)")
