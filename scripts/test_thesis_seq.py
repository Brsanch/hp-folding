"""Quick: can we reach -21 on the THESIS 2D50 sequence (24 H's)?
If yes, our 'S6' was wrong all along — we were chasing a record for a different sequence.
"""
import sys, math, random, time, multiprocessing as mp
sys.path.insert(0, "src")
from hp_folding.moves import (corner_flip, crankshaft, pivot, segment_rotate,
                              end_flip, local_rebridge, bond_rebridge,
                              end_rebridge, lmw_pull)
from hp_folding.verifier import count_contacts

THESIS_SEQ = "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH"
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


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


def propose(c, rng):
    u = rng.random()
    if u < 0.20: return bond_rebridge(c, rng)
    if u < 0.35: return lmw_pull(c, rng)
    if u < 0.42: return end_rebridge(c, rng)
    if u < 0.55: return local_rebridge(c, rng, seg_len=rng.choice([3, 4, 5, 6, 8]))
    if u < 0.72: return corner_flip(c, rng)
    if u < 0.85: return crankshaft(c, rng)
    if u < 0.92: return segment_rotate(c, rng)
    if u < 0.98: return pivot(c, rng)
    return end_flip(c, rng)


def chain_run(seed):
    rng = random.Random(seed)
    cur = random_saw(THESIS_SEQ, rng)
    cur_c = count_contacts(THESIS_SEQ, cur)
    best = cur_c
    n_steps = 1_000_000
    T_start, T_end = 2.5, 0.05
    for step in range(n_steps):
        T = T_start * (1 - step/n_steps) + T_end * (step/n_steps)
        new = propose(cur, rng)
        if new is None: continue
        c = count_contacts(THESIS_SEQ, new)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE/max(T, 1e-6)):
            cur = new
            cur_c = c
            if c > best: best = c
    return best


def main():
    print(f"Thesis 2D50: {THESIS_SEQ}")
    print(f"  length={len(THESIS_SEQ)}, H-count={THESIS_SEQ.count('H')}")
    t0 = time.monotonic()
    print("Running 14 SA chains × 1M steps each...", flush=True)
    with mp.Pool(14) as pool:
        results = list(pool.imap_unordered(chain_run, range(14)))
    print(f"Wall: {time.monotonic()-t0:.0f}s")
    print(f"Best per chain (sorted): {sorted(results, reverse=True)}")
    print(f"Max overall: {max(results)} (target=21)")


if __name__ == "__main__":
    main()
