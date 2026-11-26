"""Fresh SA on S6 with bond_rebridge added to the move proposer.

Hypothesis: bond_rebridge is a topology-rewiring move our prior runs lacked.
It reaches different basins than corner_flip / pivot / local_rebridge alone.
If S6 has 20-fold basins, bond_rebridge has a real chance of finding them.

Run many parallel chains with random restarts. Hope that at least one
chain finds a 20+ fold.
"""
from __future__ import annotations
import argparse, json, math, multiprocessing as mp, random, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding import BENCHMARKS, Fold
from hp_folding.io import write_submission
from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip,
    local_rebridge, bond_rebridge, end_rebridge,
)
from hp_folding.verifier import count_contacts, verify

SEQ = BENCHMARKS["S6"].sequence
N = len(SEQ)
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def random_saw(rng):
    """Random valid SAW of length N (with retries)."""
    for _ in range(500):
        coords = [(0, 0)]
        occ = {(0, 0)}
        ok = True
        for _ in range(N - 1):
            last = coords[-1]
            avail = []
            for dx, dy in NB:
                nb = (last[0] + dx, last[1] + dy)
                if nb not in occ:
                    avail.append(nb)
            if not avail:
                ok = False
                break
            choice = rng.choice(avail)
            coords.append(choice)
            occ.add(choice)
        if ok and len(coords) == N:
            return coords
    return [(i, 0) for i in range(N)]


def propose(c, rng):
    u = rng.random()
    if u < 0.20:
        return bond_rebridge(c, rng)  # NEW: 20% topology-rewiring
    if u < 0.30:
        return end_rebridge(c, rng)   # NEW: 10% end-rewiring
    if u < 0.50:
        return local_rebridge(c, rng, seg_len=rng.choice([3, 4, 5, 6, 8]))
    if u < 0.70:
        return corner_flip(c, rng)
    if u < 0.85:
        return crankshaft(c, rng)
    if u < 0.93:
        return segment_rotate(c, rng)
    if u < 0.99:
        return pivot(c, rng)
    return end_flip(c, rng)


def chain(args):
    n_steps, T_start, T_end, seed = args
    rng = random.Random(seed)
    cur = random_saw(rng)
    cur_c = count_contacts(SEQ, cur)
    best_c = cur_c
    best = list(cur)
    for step in range(n_steps):
        T = T_start * (1 - step / n_steps) + T_end * (step / n_steps)
        new = propose(cur, rng)
        if new is None:
            continue
        c = count_contacts(SEQ, new)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new
            cur_c = c
            if c > best_c:
                best_c = c
                best = list(cur)
    return best_c, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-chains", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=2_000_000)
    p.add_argument("--T-start", type=float, default=2.5)
    p.add_argument("--T-end", type=float, default=0.05)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--target", type=int, default=21)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    sub = json.loads((ROOT / "submissions/S6.json").read_text())
    init_c = verify(SEQ, [tuple(c) for c in sub["coords"]]).contacts
    print(f"=== S6 fresh SA + bond_rebridge ===")
    print(f"  current: {init_c}, target: {args.target}")
    print(f"  chains: {args.n_chains} × {args.n_steps:,} steps")
    print(f"  T: {args.T_start} → {args.T_end}")

    work = [(args.n_steps, args.T_start, args.T_end, seed)
            for seed in range(args.n_chains)]

    t0 = time.monotonic()
    best_overall = init_c
    best_coords = None
    n_done = 0
    contact_dist = {}
    distinct_19_basins: dict[frozenset, list] = {}  # contact-set -> coords

    out_basins_dir = ROOT / "results/s6_fresh_19_basins"
    out_basins_dir.mkdir(parents=True, exist_ok=True)

    with mp.Pool(args.processes) as pool:
        for c, coords in pool.imap_unordered(chain, work):
            n_done += 1
            contact_dist[c] = contact_dist.get(c, 0) + 1
            if c >= 19:
                # Track distinct 19+ basin contact-sets
                pos = {tuple(p): i for i, p in enumerate(coords)}
                cs = []
                for i, (x, y) in enumerate(coords):
                    if SEQ[i] != "H":
                        continue
                    for dx, dy in NB:
                        j = pos.get((x + dx, y + dy))
                        if j is not None and j > i + 1 and SEQ[j] == "H":
                            cs.append((i, j))
                key = frozenset(cs)
                if c == 19 and key not in distinct_19_basins:
                    distinct_19_basins[key] = [list(p) for p in coords]
                    # Persist to disk
                    bn = len(distinct_19_basins) - 1
                    out = out_basins_dir / f"basin_{bn:03d}_c19.json"
                    out.write_text(json.dumps({"sequence": SEQ,
                                               "coords": [list(p) for p in coords]}))
            if c > best_overall:
                best_overall = c
                best_coords = coords
                print(f"  [{n_done}/{len(work)}] NEW BEST: {c}", flush=True)
                if best_overall >= args.target:
                    print(f"  RECORD MATCH at chain {n_done}", flush=True)
                    pool.terminate()
                    break
            if n_done % 25 == 0:
                top = sorted(contact_dist.items(), reverse=True)[:6]
                elapsed = time.monotonic() - t0
                print(f"  [{n_done}/{len(work)}] best={best_overall} "
                      f"19-basins={len(distinct_19_basins)} "
                      f"top: {dict(top)} elapsed={elapsed:.0f}s", flush=True)

    print(f"\n{len(distinct_19_basins)} distinct 19-basins saved to {out_basins_dir}")

    print(f"\nFinal: best={best_overall} (was {init_c})")
    print(f"distinct 19+ basins discovered: {len(distinct_19_basins)} (prior: 4)")
    if best_coords and best_overall > init_c:
        out = ROOT / f"results/S6_fresh_rebridge_c{best_overall}.json"
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=SEQ, coords=best_coords), "S6", out)
        if args.update_submission:
            shutil.copy(out, ROOT / "submissions/S6.json")
            print(f"PROMOTED S6: {init_c} -> {best_overall}")


if __name__ == "__main__":
    main()
