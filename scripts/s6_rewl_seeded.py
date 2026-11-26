"""S6 REWL with deep-window seeded from our 15 known 19-fold basins.

The single algorithmic piece we never tried: REWL flat-histogram pressure
applied to a walker that already sits at E = -19 (not initialized via
greedy push from random). Each system seeds walker 0 with a different
19-fold basin; remaining walkers initialize normally for their windows.
4 systems sequentially × 4 different seeds = comprehensive coverage.
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
    for _ in range(50):
        coords = random_saw(rng)
        E = -count_contacts(SEQ, coords)
        if e_low <= E <= e_high:
            return coords, E
    coords = random_saw(rng)
    E = -count_contacts(SEQ, coords)
    for _ in range(max_attempts):
        new = propose(coords, rng)
        if new is None:
            continue
        new_E = -count_contacts(SEQ, new)
        cur_dist = max(0, max(e_low - E, E - e_high))
        new_dist = max(0, max(e_low - new_E, new_E - e_high))
        if new_dist <= cur_dist:
            coords = new
            E = new_E
            if e_low <= E <= e_high:
                return coords, E
    return coords, E


def wl_walker_chunk(args):
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
        if not (e_low <= new_E <= e_high):
            log_g[E] += log_f
            H[E] += 1
            continue
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


def is_flat(H, flatness):
    if not H: return False
    h_values = list(H.values())
    if len(h_values) < 2: return False
    h_mean = sum(h_values) / len(h_values)
    h_min = min(h_values)
    return h_mean > 0 and h_min / h_mean >= flatness


def load_seeds():
    """Load all 15 distinct 19-fold basins from prior session work."""
    NB_local = NB
    seeds = []
    seen = set()
    for src_dir in [ROOT / "results/s6_fresh_19_basins",
                    ROOT / "results/s6_pt_basins"]:
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
    return seeds


def run_one_system(args, seed_coords):
    """Run one REWL system with deep window seeded by seed_coords."""
    K = args.n_windows
    span = args.e_max - args.e_min
    win_size = (span + (K - 1) * args.overlap) // K
    windows = []
    cur_low = args.e_min
    for k in range(K):
        cur_high = min(args.e_max, cur_low + win_size)
        windows.append((cur_low, cur_high))
        cur_low = cur_high - args.overlap

    rng = random.Random(args.seed_base)
    walkers = []
    # Seed window 0 with seed_coords (assuming E in window 0)
    seed_E = -count_contacts(SEQ, seed_coords)
    if not (windows[0][0] <= seed_E <= windows[0][1]):
        # Drive seed into window 0 if it's not there
        seed_coords, seed_E = initialize_in_window(
            random.Random(args.seed_base), windows[0][0], windows[0][1])
    walkers.append({"coords": list(seed_coords), "E": seed_E,
                    "log_g": {}, "log_f": 1.0,
                    "best_E": seed_E, "best_coords": list(seed_coords)})
    # Other windows: standard init
    for k in range(1, K):
        coords, E = initialize_in_window(
            random.Random(args.seed_base + 1000 * k), windows[k][0], windows[k][1])
        walkers.append({"coords": coords, "E": E,
                        "log_g": {}, "log_f": 1.0,
                        "best_E": E, "best_coords": list(coords)})

    print(f"  windows: {windows}", flush=True)
    print(f"  walker 0 SEEDED at E={seed_E} (target window {windows[0]})", flush=True)
    for k in range(1, K):
        print(f"  walker {k} init at E={walkers[k]['E']} (target window {windows[k]})",
              flush=True)
    sys.stdout.flush()

    best_overall_E = walkers[0]["best_E"]
    best_overall_coords = walkers[0]["best_coords"]
    iter_count = 0
    n_swap_attempts = [0] * (K - 1)
    n_swap_accepts = [0] * (K - 1)

    pool = mp.Pool(args.processes)
    try:
        t0 = time.monotonic()
        while iter_count < args.max_iters:
            work = [(w["coords"], w["E"], w["log_g"],
                     windows[k][0], windows[k][1],
                     w["log_f"], args.n_steps_per_iter,
                     args.seed_base + iter_count * 1000 + k,
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
                if best_E < best_overall_E:
                    best_overall_E = best_E
                    best_overall_coords = best_coords
                    print(f"  *** NEW BEST: contacts={-best_E} from window {k}",
                          flush=True)
                    if -best_E >= args.target:
                        return -best_overall_E, best_overall_coords

            order = list(range(0, K - 1, 2)) if iter_count % 2 == 0 else list(range(1, K - 1, 2))
            for k in order:
                w_a = walkers[k]
                w_b = walkers[k + 1]
                e_overlap_low = max(windows[k][0], windows[k + 1][0])
                e_overlap_high = min(windows[k][1], windows[k + 1][1])
                if not (e_overlap_low <= w_a["E"] <= e_overlap_high): continue
                if not (e_overlap_low <= w_b["E"] <= e_overlap_high): continue
                n_swap_attempts[k] += 1
                la_a = w_a["log_g"].get(w_a["E"], 0.0)
                la_b = w_a["log_g"].get(w_b["E"], 0.0)
                lb_a = w_b["log_g"].get(w_a["E"], 0.0)
                lb_b = w_b["log_g"].get(w_b["E"], 0.0)
                log_p = (la_a + lb_b) - (la_b + lb_a)
                if log_p >= 0 or rng.random() < math.exp(log_p):
                    w_a["coords"], w_b["coords"] = w_b["coords"], w_a["coords"]
                    w_a["E"], w_b["E"] = w_b["E"], w_a["E"]
                    n_swap_accepts[k] += 1

            for w in walkers:
                if is_flat(w.get("H", {}), args.flatness):
                    w["log_f"] /= 2
                    w["H"] = {}

            iter_count += 1
            if iter_count % 5 == 0:
                elapsed = time.monotonic() - t0
                bests = [-w["best_E"] for w in walkers]
                cur_E = [w["E"] for w in walkers]
                log_fs = [f"{w['log_f']:.2e}" for w in walkers]
                print(f"  iter {iter_count} | best={-best_overall_E} "
                      f"window_bests={bests} cur_E={cur_E} "
                      f"log_f={log_fs} elapsed={elapsed:.0f}s", flush=True)

            if all(w["log_f"] < args.log_f_min for w in walkers):
                print(f"  All walkers converged", flush=True)
                break
    finally:
        pool.close()
        pool.join()

    return -best_overall_E, best_overall_coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-windows", type=int, default=5)
    p.add_argument("--e-min", type=int, default=-21)
    p.add_argument("--e-max", type=int, default=0)
    p.add_argument("--overlap", type=int, default=3)
    p.add_argument("--n-steps-per-iter", type=int, default=1_000_000)
    p.add_argument("--max-iters", type=int, default=60)
    p.add_argument("--log-f-min", type=float, default=1e-6)
    p.add_argument("--flatness", type=float, default=0.7)
    p.add_argument("--processes", type=int, default=5)
    p.add_argument("--target", type=int, default=21)
    p.add_argument("--update-submission", action="store_true")
    p.add_argument("--n-seed-systems", type=int, default=4,
                   help="How many of the 15 basins to seed with (sequential systems)")
    p.add_argument("--seed-base", type=int, default=1000)
    args = p.parse_args()

    sub = json.loads((ROOT / "submissions/S6.json").read_text())
    init_c = verify(SEQ, [tuple(c) for c in sub["coords"]]).contacts

    seeds = load_seeds()
    print(f"=== S6 Seeded REWL ({args.n_seed_systems} systems) ===")
    print(f"  init: {init_c}, target: {args.target}")
    print(f"  loaded {len(seeds)} distinct 19-fold seed basins")
    if args.n_seed_systems > len(seeds):
        args.n_seed_systems = len(seeds)
    # Pick first N (could randomize)
    chosen_seeds = seeds[:args.n_seed_systems]

    best_overall = init_c
    best_overall_coords = None
    for sys_id, seed_coords in enumerate(chosen_seeds):
        print(f"\n=== System {sys_id + 1}/{len(chosen_seeds)} "
              f"(seed basin {sys_id}) ===")
        args.seed_base = args.seed_base + sys_id * 100000
        c, coords = run_one_system(args, seed_coords)
        if c > best_overall:
            best_overall = c
            best_overall_coords = coords
            print(f"  *** SYSTEM {sys_id + 1} produced new best: {c}")
            if best_overall >= args.target:
                out = ROOT / f"results/S6_seeded_REWL_c{best_overall}.json"
                write_submission(Fold(sequence=SEQ, coords=best_overall_coords),
                                 "S6", out)
                if args.update_submission:
                    shutil.copy(out, ROOT / "submissions/S6.json")
                    print(f"PROMOTED S6: {init_c} -> {best_overall}")
                return

    print(f"\n=== Final: best={best_overall} (init {init_c}) ===")
    if best_overall_coords and best_overall > init_c:
        out = ROOT / f"results/S6_seeded_REWL_c{best_overall}.json"
        write_submission(Fold(sequence=SEQ, coords=best_overall_coords), "S6", out)
        if args.update_submission:
            shutil.copy(out, ROOT / "submissions/S6.json")
            print(f"PROMOTED S6: {init_c} -> {best_overall}")


if __name__ == "__main__":
    main()
