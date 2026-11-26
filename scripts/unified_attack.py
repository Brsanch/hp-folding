"""Unified attack on all 6 canonical thesis benchmark sequences.

Implements Davis 2016's move-set proportions + the new FRW move. Two phases:
  Phase 1: long SA with Davis proportions (~5 min per sequence × 6 = ~30 min)
  Phase 2: REWL with energy windows + Davis proportions (~30 min per sequence)

Targets:
  2D50:  -22 (push beyond record -21)
  2D60:  -37 (push beyond record -36)
  2D64:  -43 (push beyond record -42)
  2D85:  -53 (match WL2012 record; Davis got -52, we got -52)
  2D100a:-48 (match WL2012 record; Davis got -47)
  2D100b:-50 (match WL2012 record; Davis got -49)
"""
from __future__ import annotations
import sys, math, random, time, json, argparse, multiprocessing as mp
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hp_folding.moves import (
    corner_flip, crankshaft, pivot, segment_rotate, end_flip,
    local_rebridge, bond_rebridge, end_rebridge, lmw_pull,
    fragment_random_walk,
)
from hp_folding.verifier import count_contacts, verify
from hp_folding.fold import Fold
from hp_folding.io import write_submission

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))

CANONICAL = {
    "2D50":   ("HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH", 22),  # target -22
    "2D60":   ("PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHH"
               "HHHHHHPPPPHHHHHHPHHPHP", 37),
    "2D64":   ("HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHP"
               "PHPHPHHHHHHHHHHHH", 43),
    "2D85":   ("HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHH"
               "HHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH", 53),
    # 2D100a: line1 = 49 chars (PPHHHHH at end), line2 = 51 chars (HHHHHPHH at start)
    "2D100a": ("PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHH"
               "HHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH", 48),
    # 2D100b: line1 = 41 chars (HHHHH end of line1), line2 = 59 chars
    "2D100b": ("PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHH"
               "HHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH", 50),
}

# Critical temperatures from Davis 2016 Appendix D
DAVIS_TC = {
    "2D50": 0.576, "2D60": 0.42, "2D64": 0.39,
    "2D85": 0.545, "2D100a": 0.535, "2D100b": 0.577,
}


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
    """Davis 2016 move set: 70% pull, 19% bond rebridging (split: 30% chain-terminal,
    70% type II), 10% FRW, 4% pivot, 2% kink-flip (corner_flip).
    Approximated as: 70% lmw_pull, 13% bond_rebridge, 6% end_rebridge,
                     10% FRW, 4% pivot, 2% corner_flip.
    Local moves (crankshaft, segment_rotate, local_rebridge) interleaved at low rates
    for diversity.
    """
    u = rng.random()
    if u < 0.55:
        return lmw_pull(c, rng)
    if u < 0.68:
        return bond_rebridge(c, rng)
    if u < 0.74:
        return end_rebridge(c, rng)
    if u < 0.84:
        return fragment_random_walk(c, rng)
    if u < 0.88:
        return pivot(c, rng)
    if u < 0.90:
        return corner_flip(c, rng)
    if u < 0.93:
        return crankshaft(c, rng)
    if u < 0.96:
        return segment_rotate(c, rng)
    if u < 0.99:
        return local_rebridge(c, rng, seg_len=rng.choice([4, 6, 8]))
    return end_flip(c, rng)


def sa_chain(args):
    seq, target_id, tc, n_steps, seed = args
    rng = random.Random(seed)
    cur = random_saw(seq, rng)
    cur_c = count_contacts(seq, cur)
    best = cur_c
    best_coords = list(cur)
    # Anneal from above TC to well below TC
    T_start = max(2.0, 4 * tc)
    T_end = max(0.05, 0.1 * tc)
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


def initialize_in_window(seq, rng, e_low, e_high, max_attempts=10000):
    coords = random_saw(seq, rng)
    E = -count_contacts(seq, coords)
    for _ in range(max_attempts):
        if e_low <= E <= e_high:
            return coords, E
        new = davis_propose(coords, rng)
        if new is None: continue
        new_E = -count_contacts(seq, new)
        cur_dist = max(0, max(e_low - E, E - e_high))
        new_dist = max(0, max(e_low - new_E, new_E - e_high))
        if new_dist <= cur_dist:
            coords = new
            E = new_E
    return coords, E


def wl_walker_chunk(args):
    (seq, coords, E, log_g_dict, e_low, e_high, log_f, n_steps, seed,
     best_E_so_far, best_coords_so_far) = args
    rng = random.Random(seed)
    log_g = defaultdict(float, log_g_dict)
    H: dict[int, int] = defaultdict(int)
    best_E = best_E_so_far
    best = list(best_coords_so_far) if best_coords_so_far else list(coords)
    for _ in range(n_steps):
        new = davis_propose(coords, rng)
        if new is None: continue
        new_E = -count_contacts(seq, new)
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


def rewl_run(seq, target_id, target_contacts, e_min, e_max, n_windows, overlap,
             n_steps_per_iter, max_iters, log_f_min, flatness, processes, seed_base):
    """One REWL run: K windows × walkers, swap between adjacent windows."""
    K = n_windows
    span = e_max - e_min
    win_size = (span + (K - 1) * overlap) // K
    windows = []
    cur_low = e_min
    for k in range(K):
        cur_high = min(e_max, cur_low + win_size)
        windows.append((cur_low, cur_high))
        cur_low = cur_high - overlap

    rng = random.Random(seed_base)
    walkers = []
    for k in range(K):
        coords, E = initialize_in_window(seq, random.Random(seed_base + 1000*k),
                                          windows[k][0], windows[k][1])
        walkers.append({"coords": coords, "E": E, "log_g": {}, "log_f": 1.0,
                        "best_E": E, "best_coords": list(coords)})
    print(f"  [{target_id}] REWL windows: {windows}", flush=True)
    for k, w in enumerate(walkers):
        print(f"    walker {k} init E={w['E']} window {windows[k]}", flush=True)

    best_overall_E = walkers[0]["best_E"]
    best_overall_coords = walkers[0]["best_coords"]
    for k in range(K):
        if walkers[k]["best_E"] < best_overall_E:
            best_overall_E = walkers[k]["best_E"]
            best_overall_coords = walkers[k]["best_coords"]

    pool = mp.Pool(processes)
    try:
        t0 = time.monotonic()
        for it in range(max_iters):
            work = [(seq, w["coords"], w["E"], w["log_g"],
                     windows[k][0], windows[k][1],
                     w["log_f"], n_steps_per_iter,
                     seed_base + it * 1000 + k,
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
                    print(f"    iter {it} window {k}: NEW BEST -E={-best_E}",
                          flush=True)
                    if -best_E >= target_contacts:
                        return -best_overall_E, best_overall_coords

            order = list(range(0, K-1, 2)) if it % 2 == 0 else list(range(1, K-1, 2))
            for k in order:
                w_a, w_b = walkers[k], walkers[k+1]
                e_lo = max(windows[k][0], windows[k+1][0])
                e_hi = min(windows[k][1], windows[k+1][1])
                if not (e_lo <= w_a["E"] <= e_hi): continue
                if not (e_lo <= w_b["E"] <= e_hi): continue
                la_a = w_a["log_g"].get(w_a["E"], 0.0)
                la_b = w_a["log_g"].get(w_b["E"], 0.0)
                lb_a = w_b["log_g"].get(w_a["E"], 0.0)
                lb_b = w_b["log_g"].get(w_b["E"], 0.0)
                log_p = (la_a + lb_b) - (la_b + lb_a)
                if log_p >= 0 or rng.random() < math.exp(log_p):
                    w_a["coords"], w_b["coords"] = w_b["coords"], w_a["coords"]
                    w_a["E"], w_b["E"] = w_b["E"], w_a["E"]

            for w in walkers:
                if is_flat(w.get("H", {}), flatness):
                    w["log_f"] /= 2
                    w["H"] = {}

            if (it + 1) % 10 == 0:
                elapsed = time.monotonic() - t0
                bests = [-w["best_E"] for w in walkers]
                cur_E = [w["E"] for w in walkers]
                print(f"    iter {it+1}/{max_iters} best={-best_overall_E} "
                      f"window_bests={bests} cur_E={cur_E} elapsed={elapsed:.0f}s",
                      flush=True)
            if all(w["log_f"] < log_f_min for w in walkers):
                break
    finally:
        pool.close(); pool.join()

    return -best_overall_E, best_overall_coords


def attack_one(target_id, n_sa_chains, n_sa_steps, processes,
               run_rewl, rewl_iters, rewl_steps_per_iter):
    seq, target = CANONICAL[target_id]
    tc = DAVIS_TC[target_id]

    print(f"\n=== {target_id} (n={len(seq)}, H={seq.count('H')}, target={target}) ===",
          flush=True)

    # Phase 1: SA
    print(f"  Phase 1: {n_sa_chains} SA chains × {n_sa_steps:,} steps "
          f"(TC={tc})", flush=True)
    t0 = time.monotonic()
    with mp.Pool(processes) as pool:
        results = list(pool.imap_unordered(sa_chain,
                       [(seq, target_id, tc, n_sa_steps, s)
                        for s in range(n_sa_chains)]))
    contacts = [r[0] for r in results]
    contacts.sort(reverse=True)
    print(f"  SA wall: {time.monotonic()-t0:.0f}s, "
          f"top: {contacts[:8]}, max: {contacts[0]}", flush=True)
    best_idx = max(range(len(results)), key=lambda i: results[i][0])
    best_c, best_coords = results[best_idx]

    # Phase 2: REWL (only if SA didn't already break record)
    if run_rewl and best_c < target:
        print(f"  Phase 2: REWL (target={target})", flush=True)
        # Window range based on SA's best
        e_min = -(target + 1)  # try to push 1 beyond target
        e_max = max(0, -(best_c - 8))  # allow 8 levels above SA best
        rewl_c, rewl_coords = rewl_run(
            seq, target_id, target,
            e_min=e_min, e_max=e_max,
            n_windows=4, overlap=3,
            n_steps_per_iter=rewl_steps_per_iter,
            max_iters=rewl_iters,
            log_f_min=1e-6, flatness=0.7,
            processes=processes, seed_base=hash(target_id) % 1000000,
        )
        if rewl_c > best_c:
            best_c = rewl_c
            best_coords = rewl_coords
            print(f"  REWL improved to {best_c}", flush=True)

    # Save (only if improving over existing submission)
    v = verify(seq, best_coords)
    if v:
        out = ROOT / f"submissions/{target_id}.json"
        existing_c = -1
        if out.exists():
            try:
                d = json.loads(out.read_text())
                if d.get("sequence") == seq:  # same sequence
                    existing_c = d.get("contacts", -1)
            except Exception:
                pass
        if v.contacts > existing_c:
            payload = {
                "sequence_id": target_id,
                "sequence": seq,
                "n_residues": len(seq),
                "n_h_residues": seq.count("H"),
                "contacts": v.contacts,
                "coords": [list(c) for c in best_coords],
            }
            out.write_text(json.dumps(payload, indent=2))
            print(f"  Saved {out} with {v.contacts} contacts (was {existing_c})",
                  flush=True)
        else:
            print(f"  Not saved: existing {existing_c} >= our {v.contacts}",
                  flush=True)
    return best_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sa-chains", type=int, default=14)
    p.add_argument("--n-sa-steps", type=int, default=5_000_000)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--no-rewl", action="store_true",
                   help="Skip REWL phase (faster)")
    p.add_argument("--rewl-iters", type=int, default=40)
    p.add_argument("--rewl-steps-per-iter", type=int, default=1_000_000)
    p.add_argument("--targets", nargs="+",
                   default=["2D50", "2D60", "2D64", "2D85", "2D100a", "2D100b"])
    args = p.parse_args()

    print("=== Unified attack on canonical thesis sequences ===", flush=True)
    print(f"  Targets: {args.targets}")
    print(f"  Phase 1: {args.n_sa_chains} chains × {args.n_sa_steps:,} steps each")
    print(f"  Phase 2 (REWL): {'enabled' if not args.no_rewl else 'disabled'}")
    print()

    summary = {}
    for tid in args.targets:
        if tid not in CANONICAL:
            print(f"  unknown target: {tid}, skipping")
            continue
        c = attack_one(tid, args.n_sa_chains, args.n_sa_steps, args.processes,
                       run_rewl=not args.no_rewl,
                       rewl_iters=args.rewl_iters,
                       rewl_steps_per_iter=args.rewl_steps_per_iter)
        summary[tid] = c

    print("\n=== FINAL SUMMARY ===", flush=True)
    for tid, c in summary.items():
        target = CANONICAL[tid][1]
        delta = c - target
        marker = "✓" if c >= target else f"({delta:+d})"
        print(f"  {tid}: {c} (target {target}) {marker}")


if __name__ == "__main__":
    main()
