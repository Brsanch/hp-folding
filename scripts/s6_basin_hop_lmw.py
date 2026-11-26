"""S6 Basin Hopping with LMW pull + bond_rebridge.

Strategy: from each of our 15 known 19-fold basins as SEED:
  1. Apply K random "macro-perturbation" moves (lmw_pull + bond_rebridge
     + segment_rotate + pivot — each forces non-local rearrangement).
  2. SA-cool from that perturbed state.
  3. If SA reaches 19+, save as discovery.

This is option 3 of the 3-method strategy: USE our existing 19-fold
findings as starting points but make BIG jumps to escape the basin
attractor. Independent discovery: any 21-fold found has a different
contact set than what we started with.
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
    local_rebridge, bond_rebridge, end_rebridge, lmw_pull,
)
from hp_folding.verifier import count_contacts, verify

SEQ = BENCHMARKS["S6"].sequence
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))


def sa_propose(c, rng):
    u = rng.random()
    if u < 0.18: return bond_rebridge(c, rng)
    if u < 0.30: return lmw_pull(c, rng)
    if u < 0.36: return end_rebridge(c, rng)
    if u < 0.50: return local_rebridge(c, rng, seg_len=rng.choice([3, 4, 5, 6, 8]))
    if u < 0.68: return corner_flip(c, rng)
    if u < 0.82: return crankshaft(c, rng)
    if u < 0.91: return segment_rotate(c, rng)
    if u < 0.97: return pivot(c, rng)
    return end_flip(c, rng)


def macro_perturb(coords, rng, n_moves):
    """Apply n_moves consecutive non-local moves with NO acceptance criterion.
    This forces the chain into a distant region of fold space."""
    cur = list(coords)
    for _ in range(n_moves):
        # Heavy bias toward topology-rewiring moves
        u = rng.random()
        new = None
        if u < 0.35:
            new = bond_rebridge(cur, rng)
        elif u < 0.60:
            new = lmw_pull(cur, rng)
        elif u < 0.80:
            new = pivot(cur, rng)
        elif u < 0.92:
            new = segment_rotate(cur, rng)
        else:
            new = local_rebridge(cur, rng, seg_len=rng.choice([6, 8, 10, 12]))
        if new is not None:
            cur = new
    return cur


def hop(args):
    seed_coords, n_perturb_moves, n_sa_steps, T_start, T_end, seed = args
    rng = random.Random(seed)
    perturbed = macro_perturb(seed_coords, rng, n_perturb_moves)
    cur = list(perturbed)
    cur_c = count_contacts(SEQ, cur)
    best_c = cur_c
    best = list(cur)
    for step in range(n_sa_steps):
        T = T_start * (1 - step / n_sa_steps) + T_end * (step / n_sa_steps)
        new = sa_propose(cur, rng)
        if new is None: continue
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
    p.add_argument("--n-hops-per-seed", type=int, default=20)
    p.add_argument("--n-perturb-moves", type=int, default=10)
    p.add_argument("--n-sa-steps", type=int, default=3_000_000)
    p.add_argument("--T-start", type=float, default=2.0)
    p.add_argument("--T-end", type=float, default=0.05)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--target", type=int, default=21)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    # Load all 15 known 19-fold basins (4 original + 11 from fresh SA + bond_rebridge)
    seeds = []
    seen = set()
    NB_local = NB
    for src_dir in [ROOT / "results/s6_fresh_19_basins", ROOT / "results/s6_pt_basins"]:
        if not src_dir.exists(): continue
        for fp in sorted(src_dir.glob("*.json")):
            d = json.loads(fp.read_text())
            coords = [tuple(c) for c in d["coords"]]
            c = count_contacts(SEQ, coords)
            if c < 19: continue
            pos = {p: i for i, p in enumerate(coords)}
            cs = []
            for i, (x, y) in enumerate(coords):
                if SEQ[i] != "H": continue
                for dx, dy in NB_local:
                    j = pos.get((x + dx, y + dy))
                    if j is not None and j > i + 1 and SEQ[j] == "H":
                        cs.append((i, j))
            key = frozenset(cs)
            if key not in seen:
                seen.add(key)
                seeds.append(coords)

    print(f"=== S6 basin hopping (LMW + bond_rebridge) ===")
    print(f"  seed basins (≥19 contacts, distinct): {len(seeds)}")
    print(f"  hops per seed: {args.n_hops_per_seed}")
    print(f"  perturbation moves: {args.n_perturb_moves}")
    print(f"  SA steps after perturb: {args.n_sa_steps:,}")
    print(f"  T: {args.T_start} → {args.T_end}")

    sub = json.loads((ROOT / "submissions/S6.json").read_text())
    init_c = verify(SEQ, [tuple(c) for c in sub["coords"]]).contacts
    print(f"  init S6 = {init_c}, target = {args.target}\n")

    work = []
    for s_idx, seed_coords in enumerate(seeds):
        for h in range(args.n_hops_per_seed):
            seed = s_idx * 10000 + h
            work.append((seed_coords, args.n_perturb_moves,
                         args.n_sa_steps, args.T_start, args.T_end, seed))

    t0 = time.monotonic()
    best_overall = init_c
    best_coords = None
    n_done = 0
    contact_dist = {}

    with mp.Pool(args.processes) as pool:
        for c, coords in pool.imap_unordered(hop, work):
            n_done += 1
            contact_dist[c] = contact_dist.get(c, 0) + 1
            if c > best_overall:
                best_overall = c
                best_coords = coords
                print(f"  [{n_done}/{len(work)}] NEW BEST: {c}", flush=True)
                if best_overall >= args.target:
                    out = ROOT / f"results/S6_basin_hop_c{best_overall}.json"
                    write_submission(Fold(sequence=SEQ, coords=best_coords),
                                     "S6", out)
                    if args.update_submission:
                        shutil.copy(out, ROOT / "submissions/S6.json")
                        print(f"  PROMOTED S6: {init_c} -> {best_overall}",
                              flush=True)
                    pool.terminate()
                    break
            if n_done % 25 == 0:
                top = sorted(contact_dist.items(), reverse=True)[:6]
                elapsed = time.monotonic() - t0
                print(f"  [{n_done}/{len(work)}] best={best_overall} "
                      f"top: {dict(top)} elapsed={elapsed:.0f}s", flush=True)

    print(f"\nFinal: best={best_overall} (was {init_c})")
    if best_coords and best_overall > init_c:
        out = ROOT / f"results/S6_basin_hop_c{best_overall}.json"
        write_submission(Fold(sequence=SEQ, coords=best_coords), "S6", out)
        if args.update_submission:
            shutil.copy(out, ROOT / "submissions/S6.json")
            print(f"PROMOTED S6: {init_c} -> {best_overall}")


if __name__ == "__main__":
    main()
