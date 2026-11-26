"""Designability pilot: do "easy" sequences inform the limits for hard ones?

Hypothesis: designing HP sequences with theoretically maximal contact ceilings
(parity-balanced, low sequential-HH count, optimal H-count for compact polyominos)
produces folds that approach theoretical max — and the *features* of those
sequences (parity_pairs, H-run distribution, etc.) predict achievable fraction
of theoretical max for ANY sequence.

Designed pilot sequences:
  DESIGN_9_K0   : 9 H's at chain-distance 3, parity-balanced, K=0       (theor max ≤ 12 for 3x3 polyomino)
  DESIGN_16_K0  : 16 H's at chain-distance 3, parity-balanced, K=0      (theor max ≤ 24 for 4x4 polyomino)
  DESIGN_16_K8  : 16 H's split into 8 HH-pairs (K=8)                    (theor max ≤ 16)
  DESIGN_25_K0  : 25 H's at chain-distance 3, parity-balanced, K=0      (theor max ≤ 40 for 5x5 polyomino)
  DESIGN_22_K0  : 22 H's at chain-distance 3 (matching S6 H-count, but K=0 vs S6 K=7)

Plus the 6 canonical sequences for cross-comparison.

For each sequence we measure:
  - theoretical max contacts (using H-core packing UB)
  - parity-feasible H-pair count
  - sequential-HH count K
  - achieved best contacts via our SA methodology
  - fraction = achieved / theoretical_max

We then check whether sequence features predict the achieved fraction.
"""
from __future__ import annotations
import sys, math, random, time, json, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, "src")
from hp_folding import BENCHMARKS
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip,
    local_rebridge, bond_rebridge, end_rebridge, lmw_pull,
    fragment_random_walk,
)
from hp_folding.verifier import count_contacts, verify

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ---- Sequence design ----

def make_h_at_distance_3(nH: int, ends_pad_p: int = 0) -> str:
    """nH H's at positions 0, 3, 6, 9, ..., 3*(nH-1).
    K = 0 (no sequential HH). Parity: alternates E/O => balanced.
    Length = 3*(nH-1) + 1 + 2*ends_pad_p chars.
    """
    pos = [3 * i for i in range(nH)]
    L = pos[-1] + 1
    s = ['P'] * L
    for p in pos:
        s[p] = 'H'
    return 'P' * ends_pad_p + ''.join(s) + 'P' * ends_pad_p


def make_h_pairs_at_distance_4(n_pairs: int, ends_pad_p: int = 0) -> str:
    """n_pairs HH-pairs separated by single P. K = n_pairs (each pair has 1 internal HH).
    Pattern: HH P HH P HH P ... — chain distances within pair = 1, between pairs = 3.
    Parity within pair: HH at (0,1) — E,O. Mixed.
    Length = 3*n_pairs - 1 + 2*ends_pad_p.
    """
    s = []
    for i in range(n_pairs):
        s.append('HH')
        if i < n_pairs - 1:
            s.append('P')
    base = ''.join(s)
    return 'P' * ends_pad_p + base + 'P' * ends_pad_p


PILOT_SEQUENCES = {
    "DESIGN_9_K0":   make_h_at_distance_3(9),         # 25 chars, 9H K=0
    "DESIGN_16_K0":  make_h_at_distance_3(16),        # 46 chars, 16H K=0
    "DESIGN_16_K8":  make_h_pairs_at_distance_4(8),   # 23 chars, 16H K=8
    "DESIGN_25_K0":  make_h_at_distance_3(25),        # 73 chars, 25H K=0
    "DESIGN_22_K0":  make_h_at_distance_3(22),        # 64 chars, 22H K=0 (matches S6 H-count)
}


# ---- Sequence feature analysis ----

def parity_feasible_pairs(seq: str) -> int:
    """H pairs (i, j) with j - i odd (≥3)."""
    h_positions = [i for i, c in enumerate(seq) if c == 'H']
    even = [p for p in h_positions if p % 2 == 0]
    odd = [p for p in h_positions if p % 2 == 1]
    # Each (e, o) pair has odd chain-distance
    # But filter j - i ≥ 3
    count = 0
    for e in even:
        for o in odd:
            if abs(o - e) >= 3:
                count += 1
    return count


def sequential_hh_count(seq: str) -> int:
    """K = number of (i, i+1) pairs where both H."""
    return sum(1 for i in range(len(seq) - 1) if seq[i] == 'H' and seq[i+1] == 'H')


def h_run_lengths(seq: str) -> list[int]:
    """Lengths of consecutive H-runs."""
    runs = []
    cur = 0
    for c in seq:
        if c == 'H':
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def theoretical_max_contacts(seq: str) -> int:
    """H-core packing upper bound: (4*nH - 2*P_min)/2 - K
    where P_min = perimeter of compact bounding shape, K = sequential HH bonds.
    This is the classical upper bound from Yue & Dill style analysis.
    """
    nH = seq.count('H')
    K = sequential_hh_count(seq)
    # P_min for nH cells: minimal-perimeter polyomino
    # For an a×b rectangle with a*b ≥ nH, perimeter = 2(a+b)
    # Find minimal P_min
    p_min = float('inf')
    for a in range(1, nH + 1):
        for b in range(a, nH + 1):
            if a * b >= nH:
                p_min = min(p_min, 2 * (a + b))
                break
    # Adjust for non-rectangular: the actual min perimeter for n cells
    # Detailed: floor(2 * sqrt(n)) * 2 if n is a perfect square, etc.
    # Simple bound: 2*(ceil(sqrt(n)) + ceil(n/ceil(sqrt(n))))
    import math
    a = math.isqrt(nH)
    if a * a == nH:
        p_min = min(p_min, 4 * a)
    else:
        # Try various near-square shapes
        p_min = min(p_min, 2 * (a + math.ceil(nH / a)))
    # Each H residue in the polyomino has 4 lattice-adjacent positions
    # Each position is either internal (counted as edge) or perimeter
    # Total H-H edges in polyomino = (4*nH - perimeter)/2
    edges = (4 * nH - p_min) // 2
    contacts_max = edges - K
    return max(contacts_max, 0)


# ---- SA attack ----

def random_saw(seq, rng):
    n = len(seq)
    for _ in range(500):
        coords = [(0, 0)]
        occ = {(0, 0)}
        ok = True
        for _ in range(n - 1):
            last = coords[-1]
            avail = [(last[0]+dx, last[1]+dy) for dx, dy in NB
                     if (last[0]+dx, last[1]+dy) not in occ]
            if not avail:
                ok = False
                break
            ch = rng.choice(avail)
            coords.append(ch)
            occ.add(ch)
        if ok and len(coords) == n:
            return coords
    return [(i, 0) for i in range(n)]


def davis_propose(c, rng):
    u = rng.random()
    if u < 0.55: return lmw_pull(c, rng)
    if u < 0.68: return bond_rebridge(c, rng)
    if u < 0.74: return end_rebridge(c, rng)
    if u < 0.84: return fragment_random_walk(c, rng)
    if u < 0.88: return pivot(c, rng)
    if u < 0.90: return corner_flip(c, rng)
    if u < 0.93: return crankshaft(c, rng)
    if u < 0.96: return segment_rotate(c, rng)
    if u < 0.99: return local_rebridge(c, rng, seg_len=rng.choice([4, 6, 8]))
    return end_flip(c, rng)


def chain_run(args):
    seq, n_steps, T_start, T_end, seed = args
    rng = random.Random(seed)
    cur = random_saw(seq, rng)
    cur_c = count_contacts(seq, cur)
    best = cur_c
    best_coords = list(cur)
    for step in range(n_steps):
        T = T_start * (1 - step/n_steps) + T_end * (step/n_steps)
        new = davis_propose(cur, rng)
        if new is None: continue
        c = count_contacts(seq, new)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE/max(T, 1e-6)):
            cur = new
            cur_c = c
            if c > best:
                best = c
                best_coords = list(cur)
    return best, best_coords


def attack_sequence(name, seq, n_chains=14, n_steps=4_000_000):
    n = len(seq)
    nH = seq.count('H')
    K = sequential_hh_count(seq)
    parity_pairs = parity_feasible_pairs(seq)
    runs = h_run_lengths(seq)
    theor_max = theoretical_max_contacts(seq)

    print(f"\n=== {name} ===", flush=True)
    print(f"  seq (n={n}, nH={nH}): {seq}", flush=True)
    print(f"  H-runs: {runs}, K={K}, parity_pairs={parity_pairs}", flush=True)
    print(f"  theoretical max contacts: {theor_max}", flush=True)
    sys.stdout.flush()

    t0 = time.monotonic()
    with mp.Pool(n_chains) as pool:
        results = list(pool.imap_unordered(chain_run,
                       [(seq, n_steps, 2.5, 0.05, s) for s in range(n_chains)]))
    contacts = sorted([r[0] for r in results], reverse=True)
    best_idx = max(range(len(results)), key=lambda i: results[i][0])
    best_c, best_coords = results[best_idx]
    fraction = best_c / theor_max if theor_max > 0 else 0.0
    wall = time.monotonic() - t0
    print(f"  Wall: {wall:.0f}s, top contacts: {contacts[:6]}", flush=True)
    print(f"  Best: {best_c} / theoretical {theor_max} = {fraction:.2%}", flush=True)

    # Save
    v = verify(seq, best_coords)
    out = Path(f"submissions/{name}.json")
    out.write_text(json.dumps({
        "sequence_id": name,
        "sequence": seq,
        "n_residues": n,
        "n_h_residues": nH,
        "K_sequential_hh": K,
        "parity_feasible_pairs": parity_pairs,
        "h_runs": runs,
        "theoretical_max_contacts": theor_max,
        "achieved_contacts": v.contacts,
        "achieved_fraction_of_theoretical": fraction,
        "coords": [list(c) for c in best_coords],
        "search_methodology": "SA + Davis 2016 move proportions + FRW + LMW pull",
        "search_compute": f"{n_chains} parallel chains × {n_steps:,} MC steps",
    }, indent=2))
    return {
        "name": name, "n": n, "nH": nH, "K": K,
        "parity_pairs": parity_pairs, "h_runs": runs,
        "theor_max": theor_max, "achieved": v.contacts,
        "fraction": fraction,
    }


def main():
    print("=== Designability pilot: features vs achievable contacts ===", flush=True)

    summary = []

    # 5 designed sequences
    for name, seq in PILOT_SEQUENCES.items():
        r = attack_sequence(name, seq, n_chains=14, n_steps=4_000_000)
        summary.append(r)

    # 6 canonical sequences (use existing best from submissions)
    for sid in ["S5", "S6", "S7", "S8", "S9", "S10"]:
        seq = BENCHMARKS[sid].sequence
        nH = seq.count('H')
        K = sequential_hh_count(seq)
        runs = h_run_lengths(seq)
        parity_pairs = parity_feasible_pairs(seq)
        theor_max = theoretical_max_contacts(seq)
        # Use our existing best
        sub = json.loads(open(f"submissions/{sid}.json").read())
        from hp_folding.verifier import verify as _v
        coords = [tuple(c) for c in sub['coords']]
        v = _v(seq, coords)
        achieved = v.contacts
        fraction = achieved / theor_max if theor_max > 0 else 0.0
        summary.append({
            "name": sid, "n": len(seq), "nH": nH, "K": K,
            "parity_pairs": parity_pairs, "h_runs": runs,
            "theor_max": theor_max, "achieved": achieved,
            "fraction": fraction,
        })

    # Print summary table
    print("\n\n=== SUMMARY: designed + canonical sequences ===")
    print(f"{'Name':<14} | {'n':<4} | {'nH':<4} | {'K':<3} | {'TheorMax':<8} | {'Achieved':<8} | {'Fraction':<8}")
    print("-" * 70)
    for s in summary:
        print(f"{s['name']:<14} | {s['n']:<4} | {s['nH']:<4} | {s['K']:<3} | "
              f"{s['theor_max']:<8} | {s['achieved']:<8} | {s['fraction']:<.2%}")

    # Save summary
    out = Path("docs/designability_pilot_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")

    # Quick correlation: K vs fraction
    print("\n=== Quick stats ===")
    K_low = [s for s in summary if s['K'] <= 1]
    K_mid = [s for s in summary if 2 <= s['K'] <= 5]
    K_high = [s for s in summary if s['K'] > 5]
    for label, group in [("K=0-1", K_low), ("K=2-5", K_mid), ("K>5", K_high)]:
        if group:
            avg_frac = sum(s['fraction'] for s in group) / len(group)
            print(f"  {label}: {len(group)} sequences, avg fraction = {avg_frac:.2%}")


if __name__ == "__main__":
    main()
