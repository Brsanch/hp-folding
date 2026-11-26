"""Save a clean 21-fold for the thesis 2D50 sequence."""
import sys, math, random, time, json, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, "src")
from hp_folding.moves import (corner_flip, crankshaft, pivot, segment_rotate,
                              end_flip, local_rebridge, bond_rebridge,
                              end_rebridge, lmw_pull)
from hp_folding.verifier import count_contacts, verify

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
    best_coords = list(cur)
    n_steps = 2_000_000
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
            if c > best:
                best = c
                best_coords = list(cur)
                if best >= 22:  # in case we go above record
                    return best, best_coords
    return best, best_coords


def main():
    print(f"Sequence: {THESIS_SEQ}  (length {len(THESIS_SEQ)}, H-count {THESIS_SEQ.count('H')})")
    print("Running 28 SA chains × 2M steps each, target ≥21 (record)...")
    t0 = time.monotonic()
    with mp.Pool(14) as pool:
        results = list(pool.imap_unordered(chain_run, range(28)))

    contacts = [r[0] for r in results]
    contacts.sort(reverse=True)
    print(f"Wall: {time.monotonic()-t0:.0f}s")
    print(f"Best per chain: {contacts}")
    print(f"Max overall: {max(contacts)}")

    # Find best result and save
    best_idx = max(range(len(results)), key=lambda i: results[i][0])
    best_c, best_coords = results[best_idx]
    v = verify(THESIS_SEQ, best_coords)
    if not v:
        print(f"ERROR: best fold doesn't verify: {v}")
        return
    print(f"Verified: {v.contacts} contacts on thesis sequence")

    out = Path("submissions/S6_thesis.json")
    payload = {
        "sequence_id": "S6_thesis_2D50",
        "sequence": THESIS_SEQ,
        "n_residues": len(THESIS_SEQ),
        "n_h_residues": THESIS_SEQ.count("H"),
        "contacts": v.contacts,
        "coords": [list(c) for c in best_coords],
        "note": "Davis 2016 thesis 2D50 (S1-6); record -21 cited from SISPER/FRESS/EES/ACO. Wüst-Landau 2012 did NOT include this sequence. Different from our codebase 'S6' which has 22 H residues.",
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
