"""Replica exchange (parallel tempering) on S6.

K replicas at geometrically-spaced temperatures share information via
periodic Metropolis swap attempts between adjacent temperature chains.
High-T replicas explore widely (can drop to ~5 contacts and reorganize
topology); low-T replicas exploit basins they descend into.

Hypothesis: PT can reach cluster-B (20+ fold) basins that random restarts
+ SA cannot, because high-T replicas escape cluster-A traps and propagate
basin info downward.
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
    if u < 0.20: return bond_rebridge(c, rng)
    if u < 0.30: return end_rebridge(c, rng)
    if u < 0.50: return local_rebridge(c, rng, seg_len=rng.choice([3, 4, 5, 6, 8]))
    if u < 0.70: return corner_flip(c, rng)
    if u < 0.85: return crankshaft(c, rng)
    if u < 0.93: return segment_rotate(c, rng)
    if u < 0.99: return pivot(c, rng)
    return end_flip(c, rng)


def run_chunk(args):
    """Run n_steps of fixed-T SA on one replica. Return (final_coords, best_coords_found, best_c, final_c)."""
    coords, T, n_steps, seed = args
    rng = random.Random(seed)
    cur = list(coords)
    cur_c = count_contacts(SEQ, cur)
    best_c = cur_c
    best = list(cur)
    for _ in range(n_steps):
        new = propose(cur, rng)
        if new is None: continue
        c = count_contacts(SEQ, new)
        dE = -(c - cur_c)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-6)):
            cur = new
            cur_c = c
            if c > best_c:
                best_c = c
                best = list(cur)
    return cur, best, best_c, cur_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-replicas", type=int, default=14)
    p.add_argument("--T-min", type=float, default=0.05)
    p.add_argument("--T-max", type=float, default=3.0)
    p.add_argument("--n-swap-rounds", type=int, default=300)
    p.add_argument("--n-steps-per-chunk", type=int, default=30000)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--target", type=int, default=21)
    p.add_argument("--update-submission", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-systems", type=int, default=1,
                   help="Independent PT systems run sequentially")
    args = p.parse_args()

    # Geometric temperature ladder
    K = args.n_replicas
    T_ladder = [args.T_min * (args.T_max / args.T_min) ** (k / (K - 1))
                for k in range(K)]
    print(f"=== S6 Replica Exchange ({args.n_systems} systems) ===")
    print(f"  K={K} replicas, T = [{args.T_min:.3f} .. {args.T_max:.3f}]")
    print(f"  swap rounds: {args.n_swap_rounds}, steps/chunk: {args.n_steps_per_chunk:,}")
    print(f"  total steps per replica: {args.n_swap_rounds * args.n_steps_per_chunk:,}")
    print(f"  T ladder: {[f'{t:.2f}' for t in T_ladder]}")
    sys.stdout.flush()

    sub = json.loads((ROOT / "submissions/S6.json").read_text())
    init_c = verify(SEQ, [tuple(c) for c in sub["coords"]]).contacts
    print(f"  init S6 = {init_c}, target = {args.target}\n")

    rng_main = random.Random(args.seed)
    best_overall = init_c
    best_overall_coords = None
    distinct_19plus_basins = set()

    out_basins_dir = ROOT / "results/s6_pt_basins"
    out_basins_dir.mkdir(parents=True, exist_ok=True)

    for sys_id in range(args.n_systems):
        # Initialize K replicas
        sys_rng = random.Random(args.seed + sys_id * 10000)
        replicas = [random_saw(sys_rng) for _ in range(K)]
        replica_counts = [count_contacts(SEQ, r) for r in replicas]
        best_per_replica = [c for c in replica_counts]
        best_coords_per_replica = [list(r) for r in replicas]

        n_swap_accepts = [0] * (K - 1)
        n_swap_attempts = [0] * (K - 1)

        t0 = time.monotonic()
        pool = mp.Pool(args.processes)
        try:
          for swap_round in range(args.n_swap_rounds):
            # Parallel chunk evaluation
            work = [(replicas[k], T_ladder[k], args.n_steps_per_chunk,
                     args.seed + sys_id * 1_000_000 + swap_round * 10000 + k)
                    for k in range(K)]
            results = pool.map(run_chunk, work)
            for k in range(K):
                final_coords, best_c_in_chunk, best_c, _ = results[k]
                replicas[k] = final_coords
                replica_counts[k] = count_contacts(SEQ, final_coords)
                if best_c > best_per_replica[k]:
                    best_per_replica[k] = best_c
                    best_coords_per_replica[k] = best_c_in_chunk

            # Track best overall
            for k in range(K):
                bc = best_per_replica[k]
                if bc > best_overall:
                    best_overall = bc
                    best_overall_coords = best_coords_per_replica[k]
                    print(f"  sys{sys_id} round {swap_round} replica {k} "
                          f"(T={T_ladder[k]:.3f}): NEW BEST {bc}", flush=True)
                    if best_overall >= args.target:
                        # Persist immediately
                        out = ROOT / f"results/S6_PT_c{best_overall}.json"
                        write_submission(Fold(sequence=SEQ, coords=best_overall_coords),
                                         "S6", out)
                        if args.update_submission:
                            shutil.copy(out, ROOT / "submissions/S6.json")
                            print(f"  PROMOTED submissions/S6.json: {init_c} -> {best_overall}",
                                  flush=True)
                        return
                if bc >= 19:
                    pos = {tuple(p): i for i, p in enumerate(best_coords_per_replica[k])}
                    cs = []
                    for i, (x, y) in enumerate(best_coords_per_replica[k]):
                        if SEQ[i] != "H": continue
                        for dx, dy in NB:
                            j = pos.get((x + dx, y + dy))
                            if j is not None and j > i + 1 and SEQ[j] == "H":
                                cs.append((i, j))
                    key = (bc, frozenset(cs))
                    if bc >= 19 and key not in distinct_19plus_basins:
                        distinct_19plus_basins.add(key)
                        if bc >= 19:
                            bn = len(distinct_19plus_basins) - 1
                            out = out_basins_dir / f"sys{sys_id}_basin_{bn:03d}_c{bc}.json"
                            out.write_text(json.dumps({
                                "sequence": SEQ,
                                "coords": [list(p) for p in best_coords_per_replica[k]],
                            }))

            # Attempt swaps between adjacent temperature chains
            # Alternate even/odd to avoid bias
            order = list(range(0, K - 1, 2)) if swap_round % 2 == 0 else list(range(1, K - 1, 2))
            for k in order:
                beta_k = 1.0 / T_ladder[k]
                beta_k1 = 1.0 / T_ladder[k + 1]
                # Energy = -contacts
                E_k = -replica_counts[k]
                E_k1 = -replica_counts[k + 1]
                log_p = (beta_k - beta_k1) * (E_k1 - E_k)
                n_swap_attempts[k] += 1
                if log_p >= 0 or rng_main.random() < math.exp(log_p):
                    replicas[k], replicas[k + 1] = replicas[k + 1], replicas[k]
                    replica_counts[k], replica_counts[k + 1] = (
                        replica_counts[k + 1], replica_counts[k])
                    best_per_replica[k], best_per_replica[k + 1] = (
                        best_per_replica[k + 1], best_per_replica[k])
                    best_coords_per_replica[k], best_coords_per_replica[k + 1] = (
                        best_coords_per_replica[k + 1], best_coords_per_replica[k])
                    n_swap_accepts[k] += 1

            if swap_round % 10 == 0:
                elapsed = time.monotonic() - t0
                # Top contacts across all replicas
                top_replica_contacts = sorted(replica_counts, reverse=True)[:5]
                top_best = sorted(best_per_replica, reverse=True)[:5]
                swap_rate = [
                    f"{(n_swap_accepts[k]/max(n_swap_attempts[k],1)*100):.0f}%"
                    for k in range(K - 1)
                ]
                print(f"  sys{sys_id} round {swap_round}: best={best_overall} "
                      f"distinct_19+={len(distinct_19plus_basins)} "
                      f"replica_best={top_best} "
                      f"current={top_replica_contacts} "
                      f"elapsed={elapsed:.0f}s", flush=True)

        finally:
            pool.close()
            pool.join()
        print(f"\n  System {sys_id} done. Swap acceptance: {swap_rate}\n")

    print(f"\nFinal best: {best_overall} (init {init_c})")
    print(f"distinct 19+ basins discovered via PT: {len(distinct_19plus_basins)}")
    if best_overall_coords and best_overall > init_c:
        out = ROOT / f"results/S6_PT_c{best_overall}.json"
        write_submission(Fold(sequence=SEQ, coords=best_overall_coords), "S6", out)
        if args.update_submission:
            shutil.copy(out, ROOT / "submissions/S6.json")
            print(f"PROMOTED S6: {init_c} -> {best_overall}")


if __name__ == "__main__":
    main()
