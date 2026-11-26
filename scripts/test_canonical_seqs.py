"""Quick test: can we hit native energies on the CANONICAL thesis sequences
for 2D100a and 2D100b that differ from our codebase S10?
"""
import sys, math, random, time, json, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, "src")
from hp_folding.moves import (corner_flip, crankshaft, pivot, segment_rotate,
                              end_flip, local_rebridge, bond_rebridge,
                              end_rebridge, lmw_pull)
from hp_folding.verifier import count_contacts

CANONICAL = {
    "2D100a": "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
    "2D100b": "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
}
# Note: 2D100a and 2D100b in our concatenation came out identical above. Let me re-verify.
# Actually from raw thesis text:
# 2D100a: "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHH" + "HHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH"
# 2D100b: "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHH"        + "HHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH"
CANONICAL = {
    "2D100a": "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHH"
              "HHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
    "2D100b": "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHH"
              "HHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH",
}
TARGETS = {"2D100a": 48, "2D100b": 50}  # Wüst-Landau 2012 records

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


SEQ = ""  # Set per-call


def chain_run(args):
    seq, seed, n_steps = args
    rng = random.Random(seed)
    cur = random_saw(seq, rng)
    cur_c = count_contacts(seq, cur)
    best = cur_c
    best_coords = list(cur)
    T_start, T_end = 2.5, 0.05
    for step in range(n_steps):
        T = T_start * (1 - step/n_steps) + T_end * (step/n_steps)
        new = propose(cur, rng)
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


def main():
    n_steps = 3_000_000
    n_chains = 14
    for sid, seq in CANONICAL.items():
        print(f"\n=== {sid} (length {len(seq)}, H={seq.count('H')}, target {TARGETS[sid]}) ===", flush=True)
        t0 = time.monotonic()
        with mp.Pool(14) as pool:
            results = list(pool.imap_unordered(chain_run,
                          [(seq, s, n_steps) for s in range(n_chains)]))
        contacts = [r[0] for r in results]
        contacts.sort(reverse=True)
        max_c = max(contacts)
        print(f"  Wall: {time.monotonic()-t0:.0f}s")
        print(f"  Best per chain: {contacts}")
        print(f"  Max overall: {max_c} (target {TARGETS[sid]})")
        # Save best
        best_idx = max(range(len(results)), key=lambda i: results[i][0])
        best_c, best_coords = results[best_idx]
        from hp_folding.verifier import verify
        v = verify(seq, best_coords)
        if v:
            out = Path(f"submissions/{sid}.json")
            payload = {
                "sequence_id": sid,
                "sequence": seq,
                "n_residues": len(seq),
                "n_h_residues": seq.count("H"),
                "contacts": v.contacts,
                "coords": [list(c) for c in best_coords],
                "note": f"Davis 2016 thesis {sid}; Wüst-Landau 2012 record cited as -{TARGETS[sid]}",
            }
            out.write_text(json.dumps(payload, indent=2))
            print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
