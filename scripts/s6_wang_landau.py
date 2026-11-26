"""Wang-Landau sampling on S6 — Wüst's actual record-setting algorithm.

Maintain log(g(E)) (density of states) and H(E) (histogram).
Acceptance: P(old→new) = min(1, exp(log_g[E_old] - log_g[E_new])).
After move: log_g[E_new] += log_f; H[E_new] += 1.
When H is "flat" (min/mean >= flatness threshold), log_f /= 2; reset H.
Continue until log_f < log_f_min.

Key advantage over SA: forces visits to all energy levels uniformly,
escapes basin attractors at low energy by walking back to higher energy,
then re-descending into possibly DIFFERENT basins. This is the
canonical algorithm for finding Wüst-Landau record folds.

Multi-replica with mp.Pool: each independent WL run gets its own
density-of-states estimate. They independently search for ground states.
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


def wang_landau_run(args):
    """One WL run. Returns (best_contacts, best_coords, n_steps, n_iters)."""
    seed, n_steps_per_iter, log_f_min, flatness, max_iters, target_contacts = args
    rng = random.Random(seed)

    coords = random_saw(rng)
    E = -count_contacts(SEQ, coords)
    log_g: dict[int, float] = defaultdict(float)
    H: dict[int, int] = defaultdict(int)
    log_f = 1.0  # log(modification factor); start with f = e^1 ≈ 2.718

    best_E = E
    best_coords = list(coords)
    n_total_steps = 0

    for it in range(max_iters):
        # Run n_steps_per_iter MC steps
        for _ in range(n_steps_per_iter):
            new = propose(coords, rng)
            if new is None: continue
            new_E = -count_contacts(SEQ, new)
            # WL acceptance: min(1, exp(log_g[E] - log_g[new_E]))
            log_p = log_g[E] - log_g[new_E]
            if log_p >= 0 or rng.random() < math.exp(log_p):
                coords = new
                E = new_E
                if E < best_E:
                    best_E = E
                    best_coords = list(coords)
                    if -best_E >= target_contacts:
                        return -best_E, best_coords, n_total_steps + 1, it
            # Update DoS and histogram (always, regardless of acceptance)
            log_g[E] += log_f
            H[E] += 1
            n_total_steps += 1

        # Flatness check (only consider energies we've visited)
        if not H:
            continue
        h_values = list(H.values())
        h_mean = sum(h_values) / len(h_values)
        h_min = min(h_values)
        if h_mean > 0 and h_min / h_mean >= flatness:
            # Histogram is flat; reduce log_f
            log_f /= 2
            H.clear()
            if log_f < log_f_min:
                return -best_E, best_coords, n_total_steps, it

    return -best_E, best_coords, n_total_steps, max_iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-replicas", type=int, default=14)
    p.add_argument("--n-steps-per-iter", type=int, default=500_000,
                   help="MC steps before checking flatness")
    p.add_argument("--log-f-min", type=float, default=1e-6,
                   help="Stop when log_f drops below this")
    p.add_argument("--flatness", type=float, default=0.7,
                   help="Histogram flatness threshold (min/mean)")
    p.add_argument("--max-iters", type=int, default=200,
                   help="Max histogram-flatness checks per replica")
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--target", type=int, default=21)
    p.add_argument("--update-submission", action="store_true")
    p.add_argument("--seed-base", type=int, default=10000)
    args = p.parse_args()

    sub = json.loads((ROOT / "submissions/S6.json").read_text())
    init_c = verify(SEQ, [tuple(c) for c in sub["coords"]]).contacts

    print(f"=== S6 Wang-Landau ({args.n_replicas} replicas) ===")
    print(f"  init: {init_c}, target: {args.target}")
    print(f"  steps/iter: {args.n_steps_per_iter:,}")
    print(f"  log_f_min: {args.log_f_min}, flatness: {args.flatness}")
    print(f"  max iters: {args.max_iters}")
    sys.stdout.flush()

    work = [(args.seed_base + r, args.n_steps_per_iter, args.log_f_min,
             args.flatness, args.max_iters, args.target)
            for r in range(args.n_replicas)]

    t0 = time.monotonic()
    best_overall = init_c
    best_coords = None
    n_done = 0

    with mp.Pool(args.processes) as pool:
        for c, coords, n_steps, n_iters in pool.imap_unordered(wang_landau_run, work):
            n_done += 1
            elapsed = time.monotonic() - t0
            print(f"  [{n_done}/{args.n_replicas}] best={c} "
                  f"steps={n_steps:,} iters={n_iters} elapsed={elapsed:.0f}s",
                  flush=True)
            if c > best_overall:
                best_overall = c
                best_coords = coords
                print(f"    NEW BEST: {c}", flush=True)
                if best_overall >= args.target:
                    print(f"  RECORD MATCH", flush=True)
                    pool.terminate()
                    break

    print(f"\nFinal: best={best_overall} (init {init_c})")
    if best_coords and best_overall > init_c:
        out = ROOT / f"results/S6_wang_landau_v2_c{best_overall}.json"
        out.parent.mkdir(exist_ok=True)
        write_submission(Fold(sequence=SEQ, coords=best_coords), "S6", out)
        if args.update_submission:
            shutil.copy(out, ROOT / "submissions/S6.json")
            print(f"PROMOTED S6: {init_c} -> {best_overall}")


if __name__ == "__main__":
    main()
