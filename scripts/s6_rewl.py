"""S6 Replica-Exchange Wang-Landau (REWL) — proper implementation.

Energy range divided into K overlapping windows. One walker per window.
Each walker does WL within its window only (rejecting moves that exit).
At the end of each iteration, attempt swap between walkers in adjacent
windows IF both their current energies lie in the overlap region.

This is the algorithmic piece our prior plain-WL was missing.
"""
from __future__ import annotations
import argparse, json, math, multiprocessing as mp, random, shutil, sys, time
from collections import defaultdict
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
    if u < 0.35: return lmw_pull(c, rng)
    if u < 0.42: return end_rebridge(c, rng)
    if u < 0.55: return local_rebridge(c, rng, seg_len=rng.choice([3, 4, 5, 6, 8]))
    if u < 0.72: return corner_flip(c, rng)
    if u < 0.85: return crankshaft(c, rng)
    if u < 0.92: return segment_rotate(c, rng)
    if u < 0.98: return pivot(c, rng)
    return end_flip(c, rng)


def initialize_in_window(rng, e_low, e_high, max_attempts=10000):
    """Generate a SAW with energy E in [e_low, e_high] (energy = -contacts)."""
    # Try random SAWs first
    for _ in range(50):
        coords = random_saw(rng)
        E = -count_contacts(SEQ, coords)
        if e_low <= E <= e_high:
            return coords, E

    # If not in window, drive there with biased SA
    coords = random_saw(rng)
    E = -count_contacts(SEQ, coords)
    for _ in range(max_attempts):
        new = propose(coords, rng)
        if new is None:
            continue
        new_E = -count_contacts(SEQ, new)
        # Greedy push toward window: prefer moves that bring E closer
        cur_dist = max(0, max(e_low - E, E - e_high))
        new_dist = max(0, max(e_low - new_E, new_E - e_high))
        if new_dist <= cur_dist:
            coords = new
            E = new_E
            if e_low <= E <= e_high:
                return coords, E
    return coords, E


def wl_walker_chunk(args):
    """Run n_steps of WL within window [e_low, e_high].
    Returns updated state for swap attempt + DoS data.
    """
    (coords, E, log_g_dict, e_low, e_high, log_f, n_steps, seed,
     best_E_so_far, best_coords_so_far) = args
    rng = random.Random(seed)
    log_g = defaultdict(float, log_g_dict)
    H: dict[int, int] = defaultdict(int)
    best_E = best_E_so_far
    best = list(best_coords_so_far) if best_coords_so_far else list(coords)
    for _ in range(n_steps):
        new = propose(coords, rng)
        if new is None: continue
        new_E = -count_contacts(SEQ, new)
        # Reject if new_E outside window
        if not (e_low <= new_E <= e_high):
            # Still update H/log_g for current E
            log_g[E] += log_f
            H[E] += 1
            continue
        # WL acceptance
        log_p = log_g[E] - log_g[new_E]
        if log_p >= 0 or rng.random() < math.exp(log_p):
            coords = new
            E = new_E
            if E < best_E:
                best_E = E
                best = list(coords)
        log_g[E] += log_f
        H[E] += 1
    return coords, E, dict(log_g), dict(H), best_E, best


def is_flat(H: dict, flatness: float) -> bool:
    if not H:
        return False
    h_values = list(H.values())
    if len(h_values) < 2:
        return False
    h_mean = sum(h_values) / len(h_values)
    h_min = min(h_values)
    return h_mean > 0 and h_min / h_mean >= flatness


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-windows", type=int, default=5)
    p.add_argument("--e-min", type=int, default=-21,
                   help="Lowest energy (best fold target). Energy = -contacts.")
    p.add_argument("--e-max", type=int, default=0,
                   help="Highest energy (random fold).")
    p.add_argument("--overlap", type=int, default=3,
                   help="Energy levels of overlap between adjacent windows.")
    p.add_argument("--n-steps-per-iter", type=int, default=300_000)
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--log-f-min", type=float, default=1e-6)
    p.add_argument("--flatness", type=float, default=0.7)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--target", type=int, default=21)
    p.add_argument("--update-submission", action="store_true")
    args = p.parse_args()

    # Build energy windows
    K = args.n_windows
    span = args.e_max - args.e_min
    win_size = (span + (K - 1) * args.overlap) // K
    windows = []
    cur_low = args.e_min
    for k in range(K):
        cur_high = min(args.e_max, cur_low + win_size)
        windows.append((cur_low, cur_high))
        cur_low = cur_high - args.overlap
    print(f"=== S6 REWL ===")
    print(f"  windows: {windows}")
    print(f"  steps/iter: {args.n_steps_per_iter:,}")
    print(f"  max_iters: {args.max_iters}, log_f_min: {args.log_f_min}, flatness: {args.flatness}")

    sub = json.loads((ROOT / "submissions/S6.json").read_text())
    init_c = verify(SEQ, [tuple(c) for c in sub["coords"]]).contacts
    print(f"  init S6 = {init_c}, target = {args.target}")
    sys.stdout.flush()

    # Initialize one walker per window
    rng = random.Random(0)
    walkers = []  # list of (coords, E)
    for k, (e_low, e_high) in enumerate(windows):
        coords, E = initialize_in_window(random.Random(1000 + k), e_low, e_high)
        walkers.append({"coords": coords, "E": E,
                        "log_g": {}, "log_f": 1.0,
                        "best_E": E, "best_coords": list(coords)})
        print(f"  walker {k} init at E={E} (target window {windows[k]})", flush=True)
    sys.stdout.flush()

    best_overall = init_c
    best_overall_coords = None
    iter_count = 0
    n_swap_attempts = [0] * (K - 1)
    n_swap_accepts = [0] * (K - 1)

    pool = mp.Pool(args.processes)
    try:
        t0 = time.monotonic()
        while iter_count < args.max_iters:
            # Run all walkers in parallel
            work = [(w["coords"], w["E"], w["log_g"],
                     windows[k][0], windows[k][1],
                     w["log_f"], args.n_steps_per_iter,
                     iter_count * 100000 + k * 17,
                     w["best_E"], w["best_coords"])
                    for k, w in enumerate(walkers)]
            results = pool.map(wl_walker_chunk, work)
            for k, (coords, E, log_g, H, best_E, best_coords) in enumerate(results):
                walkers[k]["coords"] = coords
                walkers[k]["E"] = E
                walkers[k]["log_g"] = log_g
                walkers[k]["best_E"] = best_E
                walkers[k]["best_coords"] = best_coords
                walkers[k]["H"] = H
                # Check global best
                if -best_E > best_overall:
                    best_overall = -best_E
                    best_overall_coords = best_coords
                    print(f"  *** NEW BEST: contacts={-best_E} from window {k}",
                          flush=True)
                    if best_overall >= args.target:
                        out = ROOT / f"results/S6_REWL_c{best_overall}.json"
                        write_submission(Fold(sequence=SEQ, coords=best_overall_coords),
                                         "S6", out)
                        if args.update_submission:
                            shutil.copy(out, ROOT / "submissions/S6.json")
                            print(f"  PROMOTED S6: {init_c} -> {best_overall}",
                                  flush=True)
                        return

            # Attempt swaps between adjacent windows (alternating even/odd order)
            order = list(range(0, K - 1, 2)) if iter_count % 2 == 0 else list(range(1, K - 1, 2))
            for k in order:
                w_a = walkers[k]
                w_b = walkers[k + 1]
                # Both walkers must be in the overlap region
                e_overlap_low = max(windows[k][0], windows[k + 1][0])
                e_overlap_high = min(windows[k][1], windows[k + 1][1])
                if not (e_overlap_low <= w_a["E"] <= e_overlap_high):
                    continue
                if not (e_overlap_low <= w_b["E"] <= e_overlap_high):
                    continue
                n_swap_attempts[k] += 1
                # WL swap acceptance:
                # P_acc = min(1, g_a(E_a) g_b(E_b) / (g_a(E_b) g_b(E_a)))
                # = min(1, exp(log_g_a[E_a] + log_g_b[E_b] - log_g_a[E_b] - log_g_b[E_a]))
                la_a = w_a["log_g"].get(w_a["E"], 0.0)
                la_b = w_a["log_g"].get(w_b["E"], 0.0)
                lb_a = w_b["log_g"].get(w_a["E"], 0.0)
                lb_b = w_b["log_g"].get(w_b["E"], 0.0)
                log_p = (la_a + lb_b) - (la_b + lb_a)
                if log_p >= 0 or rng.random() < math.exp(log_p):
                    # Swap configurations between walkers
                    w_a["coords"], w_b["coords"] = w_b["coords"], w_a["coords"]
                    w_a["E"], w_b["E"] = w_b["E"], w_a["E"]
                    n_swap_accepts[k] += 1

            # Per-walker flatness checks → reduce log_f
            for w in walkers:
                if is_flat(w.get("H", {}), args.flatness):
                    w["log_f"] /= 2
                    w["H"] = {}

            iter_count += 1
            if iter_count % 5 == 0:
                elapsed = time.monotonic() - t0
                log_fs = [f"{w['log_f']:.2e}" for w in walkers]
                bests = [-w["best_E"] for w in walkers]
                cur_E = [w["E"] for w in walkers]
                swap_rates = [
                    f"{(n_swap_accepts[k]/max(n_swap_attempts[k],1)*100):.0f}%"
                    for k in range(K - 1)]
                print(f"  iter {iter_count} | best={best_overall} "
                      f"window_bests={bests} cur_E={cur_E} "
                      f"log_f={log_fs} swaps={swap_rates} "
                      f"elapsed={elapsed:.0f}s", flush=True)

            # Check if all walkers have converged
            if all(w["log_f"] < args.log_f_min for w in walkers):
                print(f"  All walkers converged (log_f < {args.log_f_min})", flush=True)
                break
    finally:
        pool.close()
        pool.join()

    print(f"\nFinal: best={best_overall} (init {init_c})")
    if best_overall_coords and best_overall > init_c:
        out = ROOT / f"results/S6_REWL_c{best_overall}.json"
        write_submission(Fold(sequence=SEQ, coords=best_overall_coords), "S6", out)
        if args.update_submission:
            shutil.copy(out, ROOT / "submissions/S6.json")
            print(f"PROMOTED S6: {init_c} -> {best_overall}")


if __name__ == "__main__":
    main()
